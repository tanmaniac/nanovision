"""A tiny autoregressive prior over the discrete token grid.

After the VQ-VAE is trained, each image is a 4x4 grid of code indices. Flattened row-major,
that is a length-16 sequence over a vocabulary of K codes, which a causal transformer (the
one built in A1) models exactly like text: predict each token from the ones before it. A
learned BOS token provides the input for position 0. Sampling autoregressively from the
prior and decoding through the codebook produces new images.
"""

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from nanovision.transformer import TransformerDecoder, build_causal_mask


class TokenPrior(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.bos = cfg.num_codes                                  # BOS index = K (vocab K+1)
        self.num_codes = cfg.num_codes
        self.embed = nn.Embedding(cfg.num_codes + 1, cfg.prior_dim)
        self.decoder = TransformerDecoder(cfg.prior_dim, cfg.prior_heads, cfg.prior_depth)
        self.head = nn.Linear(cfg.prior_dim, cfg.num_codes)       # predict the K real codes

    def forward(self, tokens: Tensor) -> Tensor:
        """tokens (B, S) of indices in [0, K] -> next-token logits (B, S, K)."""
        x = self.embed(tokens)
        mask = build_causal_mask(tokens.shape[1]).to(tokens.device)
        x = self.decoder(x, mask=mask)
        return self.head(x)


def ar_nll(prior: TokenPrior, indices: Tensor) -> Tensor:
    """Teacher-forced next-token cross-entropy over the flattened token grid.

    Input is [BOS, t_0, ..., t_{L-2}], targets are [t_0, ..., t_{L-1}]. Provided: this is the
    same next-token loss built in A1's char-LM.
    """
    B = indices.shape[0]
    tokens = indices.reshape(B, -1)                              # (B, L)
    bos = torch.full((B, 1), prior.bos, dtype=torch.long, device=tokens.device)
    inp = torch.cat([bos, tokens[:, :-1]], dim=1)                # [BOS, t_0..t_{L-2}]
    logits = prior(inp)                                          # (B, L, K)
    return F.cross_entropy(logits.reshape(-1, prior.num_codes), tokens.reshape(-1))


def ar_sample(prior: TokenPrior, n: int, grid_hw: tuple[int, int], num_codes: int,
              generator: torch.Generator | None = None, device: str = "cpu") -> Tensor:
    """Autoregressively sample a token grid (n, H, W) from the prior.

    Start from [BOS]; at each of the L = H*W steps run the prior, take the last position's
    logits over the K codes, sample one token (multinomial), and append. Drop BOS and reshape
    to the grid.
    """
    raise NotImplementedError("implement autoregressive token sampling from the prior")
