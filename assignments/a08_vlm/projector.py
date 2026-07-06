"""The MLP connector that maps ViT patch features into the LM embedding space.

This is the LLaVA-1.5 connector. The original LLaVA used a single linear layer; LLaVA-1.5
replaced it with a 2-layer MLP (Linear -> GELU -> Linear) and the controlled ablation in
that paper showed the MLP improves the vision-language alignment. Each patch feature is
projected independently, so the projector is applied per token and does not mix patches.
"""

import torch.nn as nn
from torch import Tensor


class MLPProjector(nn.Module):
    """Per-patch 2-layer MLP: (B, N, dim_v) -> (B, N, dim_l).

    dim_v is the ViT feature dim, dim_l the LM embedding dim. The two Linears and the GELU
    are built here; forward applies them per patch token.
    """

    def __init__(self, dim_v: int, dim_l: int):
        super().__init__()
        self.fc1 = nn.Linear(dim_v, dim_l)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim_l, dim_l)

    def forward(self, feats: Tensor) -> Tensor:
        """feats (B, N, dim_v) -> (B, N, dim_l), applied per patch token.

        See the visual-token interface section of the README.
        """
        raise NotImplementedError("implement the 2-layer MLP projector forward")
