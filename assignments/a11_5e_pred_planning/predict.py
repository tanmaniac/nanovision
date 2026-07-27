"""Multimodal motion prediction: RoI-align agent features, a mode-query decoder, and a
winner-take-all loss.

The future of an agent is multimodal: the same observed state (position, speed) is consistent
with several distinct intentions (turn left, go straight, turn right). A single trajectory
regressor trained on mean error is pulled to the conditional mean of those futures, a path
through the middle of every option that matches none of them. This is mode averaging. The fix
is to predict K trajectory hypotheses and supervise only the closest one per sample, the
min-of-N (winner-take-all) loss.

Module layout: this is an assignment-LOCAL file. Nothing in nanovision imports it, so there is
no nanovision shim; the tests import it bare through conftest (top-level holed copy by default,
the solution copy under NANOVISION_IMPL=solution). Attention comes from nanovision.attention
(the A1 MultiHeadAttention), never from torch.nn high-level modules.

Shapes used throughout:
- bev_feat:   (C, nx, ny)   one shared BEV feature grid; nx along x/forward, ny along y/left.
- centers:    (N, 2)        per-agent fractional cell coords (x_cell, y_cell).
- roi tokens: (N, roi_size**2, C)  the sampled per-agent token set.
- trajs:      (B, K, T, 2)   K predicted agent-centric position trajectories, B = N agents.
- scores:     (B, K)         per-mode logits.
- gt:         (B, T, 2)      one observed agent-centric future per agent.
"""

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from nanovision.attention import MultiHeadAttention
from nanovision.primitives import MLP


def roi_align_bev(
    bev_feat: Tensor, centers: Tensor, out_size: int = 3, radius: float = 1.0
) -> Tensor:
    """Bilinearly sample an out_size x out_size grid of BEV features around each agent.

    Turns a dense BEV map into a fixed-length per-agent token set, the RoI-align that stands in
    for what a motion decoder pools around each tracked agent. One shared bev_feat is broadcast
    over the N agents; each agent gets an out_size x out_size grid of sample points spanning
    +/- radius cells in each axis, centered on its (fractional) cell.

    grid_sample axis convention (the single highest-risk line). bev_feat is (C, nx, ny), fed to
    F.grid_sample as (1, C, H, W) with H = nx, W = ny. grid_sample reads the last grid dimension
    as (x = width, y = height), so the grid's last dim must be (g_w, g_h) = (normalized ny-coord,
    normalized nx-coord). centers are (x_cell, y_cell): x_cell indexes nx (height), y_cell indexes
    ny (width). So the width coordinate comes from y_cell and the height coordinate from x_cell -
    a SWAP relative to centers' order. Sample bilinearly with padding_mode="border" (an edge agent
    samples the boundary feature) and align_corners=False. See the pooling BEV features around an agent section of the README
    for the +/- radius sampling window and the per-axis normalization.

    Args:
        bev_feat: (C, nx, ny) shared BEV feature grid.
        centers: (N, 2) fractional cell coords (x_cell, y_cell) per agent.
        out_size: side length of the sampling grid; returns out_size**2 tokens per agent.
        radius: half-extent of the sampling window in cell units.

    Returns:
        (N, out_size**2, C) sampled features, row-major over the (x_cell, y_cell) grid.
    """
    raise NotImplementedError(
        "implement roi_align_bev: build +/- radius cell offsets, add to centers, normalize with "
        "g = 2*(cell+0.5)/S - 1, SWAP to (g_w=ny, g_h=nx) for grid_sample, bilinear/border/"
        "align_corners=False over the N-expanded bev_feat; return (N, out_size**2, C)"
    )


class MultimodalTrajectoryHead(nn.Module):
    """K-mode trajectory decoder: mode queries attend over an agent's RoI tokens, then each mode
    regresses a full future and a confidence score.

    K learned mode-query embeddings are initialized DISTINCT (randn * 0.02, not zeros or all
    equal). If the queries start identical and the trajectory MLP is shared, all K outputs are
    identical at step 0, the min-of-N winner is arbitrary, and the modes collapse; distinct init
    is necessary just to break that symmetry. Each decoder layer runs mode-query self-attention
    (modes see each other and can spread out), then cross-attention of the mode queries over the
    RoI tokens, then an MLP. No causal mask: neither the modes nor the trajectory are
    autoregressive. Each mode's trajectory head predicts per-step displacements that are
    cumsum-ed to absolute agent-centric POSITIONS, which keeps the regression targets small and
    well-scaled.

    Args:
        in_ch: channel count C of the BEV feature / RoI tokens.
        dim: model width.
        n_modes: number of trajectory hypotheses K.
        horizon: future length T.
        n_layers: number of decoder layers.
        n_heads: attention heads.
        roi_size: side length of the RoI grid (roi_size**2 tokens per agent).
        radius: RoI half-extent in cell units.
    """

    def __init__(
        self,
        in_ch: int,
        dim: int = 64,
        n_modes: int = 6,
        horizon: int = 12,
        n_layers: int = 2,
        n_heads: int = 4,
        roi_size: int = 3,
        radius: float = 1.0,
    ):
        super().__init__()
        self.in_ch = in_ch
        self.dim = dim
        self.n_modes = n_modes
        self.horizon = horizon
        self.roi_size = roi_size
        self.radius = radius

        # Distinct mode queries: break the K-way symmetry so the modes can specialize.
        self.mode_queries = nn.Parameter(torch.randn(n_modes, dim) * 0.02)
        self.in_proj = nn.Linear(in_ch, dim)

        self.self_attn = nn.ModuleList(
            [MultiHeadAttention(dim, n_heads) for _ in range(n_layers)]
        )
        self.cross_attn = nn.ModuleList(
            [MultiHeadAttention(dim, n_heads) for _ in range(n_layers)]
        )
        self.mlps = nn.ModuleList([MLP(dim, dim * 2) for _ in range(n_layers)])
        self.norm_sa = nn.ModuleList([nn.LayerNorm(dim) for _ in range(n_layers)])
        self.norm_ca = nn.ModuleList([nn.LayerNorm(dim) for _ in range(n_layers)])
        self.norm_mlp = nn.ModuleList([nn.LayerNorm(dim) for _ in range(n_layers)])

        self.traj_head = nn.Linear(dim, horizon * 2)
        self.score_head = nn.Linear(dim, 1)

    def forward(self, bev_feat: Tensor, centers: Tensor) -> tuple[Tensor, Tensor]:
        """Map a shared BEV grid and N agent centers to K trajectories and K scores per agent.

        RoI-align the per-agent tokens, expand the distinct mode queries over the B = N agents,
        run the decoder layers (mode self-attention, cross-attention over the RoI tokens, MLP),
        then map each mode to a trajectory (per-step displacements integrated to absolute
        positions) and a score logit. See the mode queries and the decoder section of the README.

        Args:
            bev_feat: (C, nx, ny) shared BEV feature grid.
            centers: (N, 2) fractional cell coords per agent. B = N is the batch.

        Returns:
            trajs: (B, K, T, 2) absolute agent-centric position trajectories.
            scores: (B, K) per-mode logits.
        """
        raise NotImplementedError(
            "implement the forward: roi_align_bev -> in_proj -> expand mode_queries over B -> "
            "per layer (q + self_attn(norm), q + cross_attn(norm, kv=tokens), q + mlp(norm)) -> "
            "traj_head -> view (B,K,T,2) -> cumsum over T -> absolute positions; score_head -> (B,K)"
        )


def wta_loss(
    trajs: Tensor,
    scores: Tensor,
    gt: Tensor,
    *,
    temperature: float | None = None,
    cls_weight: float = 1.0,
    return_components: bool = False,
):
    """Winner-take-all (min-of-N) trajectory loss with a hard and a soft/annealed path.

    The committed (winner) mode per sample is the one whose endpoint is closest to the GT
    endpoint (minFDE selection by Euclidean distance; the endpoint carries most of the
    uncertainty). See the winner-take-all loss section of the README for both paths.

    The regression term is selected by temperature:
    - temperature is None (hard WTA, the canonical min-of-N): regress only the winner's full
      trajectory, so only the winning mode carries regression gradient. Spare modes that never win
      get no gradient and stay dead (the dead-mode risk the soft path fixes).
    - temperature is a float (soft / annealed min-of-N): every mode gets gradient, weighted toward
      the closer ones; annealing temperature -> 0 recovers the hard loss.

    Contracts: compute the winner index once and detach it (argmin has no gradient); selection is
    by Euclidean FDE while the regression is squared error (the mismatch is intentional - squared
    error is smooth at 0 for gradcheck). The classification term is a cross-entropy of the score
    logits against the hard winner index in both paths, with scores as RAW logits (cross_entropy
    applies log-softmax internally - do not pre-softmax).

    Args:
        trajs: (B, K, T, 2) predicted absolute agent-centric positions.
        scores: (B, K) raw mode logits.
        gt: (B, T, 2) observed agent-centric future per agent.
        temperature: None for hard WTA; a float for the soft/annealed path.
        cls_weight: weight on the classification term.
        return_components: if True, also return (reg, cls) as a dict.

    Returns:
        scalar total loss, or (total, {"reg": ..., "cls": ...}) if return_components.
    """
    raise NotImplementedError(
        "implement wta_loss: fde = ||traj[...,-1] - gt[-1]||; winner = fde.argmin(1).detach(); "
        "hard (temperature is None): MSE of gathered winner traj; soft: w = softmax(-fde/temperature), "
        "reg = sum_k w_k * mean-sq-error_k; cls = cross_entropy(scores, winner); "
        "total = reg + cls_weight*cls"
    )


@torch.no_grad()
def _best_by_fde(trajs: Tensor, gt: Tensor) -> Tensor:
    """Per-sample best mode index by minFDE (consistent with the WTA winner). Returns (B,)."""
    fde = torch.linalg.norm(trajs[:, :, -1, :] - gt[:, None, -1, :], dim=-1)  # (B, K)
    return fde.argmin(dim=1)


@torch.no_grad()
def min_ade(trajs: Tensor, gt: Tensor) -> Tensor:
    """Mean-over-time L2 of the best-of-K trajectory (minADE_K). Best mode chosen by minFDE."""
    raise NotImplementedError(
        "implement min_ade: pick best mode by minFDE, then mean-over-time L2 of that mode vs gt, "
        "averaged over the batch"
    )


@torch.no_grad()
def min_fde(trajs: Tensor, gt: Tensor) -> Tensor:
    """Endpoint L2 of the best-of-K trajectory (minFDE_K)."""
    raise NotImplementedError(
        "implement min_fde: min over K of the endpoint L2 to gt, averaged over the batch"
    )


@torch.no_grad()
def miss_rate(trajs: Tensor, gt: Tensor, thresh: float = 2.0) -> Tensor:
    """Fraction of agents whose best-of-K (by minFDE) endpoint error exceeds thresh meters.

    The best mode is selected by minFDE (the same selection the WTA winner uses), then a miss is
    counted when that mode's endpoint error is above thresh. Do not pick a different best-of-K for
    the miss count than FDE uses.
    """
    raise NotImplementedError(
        "implement miss_rate: best-of-K endpoint error by minFDE, count > thresh, mean over batch"
    )
