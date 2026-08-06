"""Bottleneck residual adapters + per-joint module bundles for cross-joint BioPM."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .model import D_MODEL, N_LAYERS

JOINTS: List[str] = ["wrist", "hip", "thigh", "ankle"]
JOINT2ID: Dict[str, int] = {j: i for i, j in enumerate(JOINTS)}
NON_WRIST: List[str] = ["hip", "thigh", "ankle"]


class BottleneckAdapter(nn.Module):
    """down-proj -> ReLU -> up-proj + residual."""

    def __init__(self, d_model: int = D_MODEL, bottleneck: int = 64):
        super().__init__()
        self.down = nn.Linear(d_model, bottleneck)
        self.up = nn.Linear(bottleneck, d_model)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.up(F.relu(self.down(x)))


class PerJointModules(nn.Module):
    """Trainable modules for one joint (input + layer adapters + joint emb)."""

    def __init__(
        self,
        d_model: int = D_MODEL,
        bottleneck: int = 64,
        n_layers: int = N_LAYERS,
        use_input_adapter: bool = True,
        use_layer_adapters: bool = True,
        use_joint_embedding: bool = True,
    ):
        super().__init__()
        self.use_input_adapter = bool(use_input_adapter)
        self.use_layer_adapters = bool(use_layer_adapters)
        self.use_joint_embedding = bool(use_joint_embedding)

        self.input_adapter = (
            BottleneckAdapter(d_model, bottleneck) if use_input_adapter else None
        )
        self.layer_adapters = (
            nn.ModuleList(
                [BottleneckAdapter(d_model, bottleneck) for _ in range(n_layers)]
            )
            if use_layer_adapters
            else None
        )
        if use_joint_embedding:
            self.joint_embedding = nn.Parameter(torch.zeros(d_model))
            nn.init.normal_(self.joint_embedding, std=0.02)
        else:
            self.register_parameter("joint_embedding", None)

    def apply_input(self, h: Tensor, bypass: bool = False) -> Tensor:
        if bypass or not self.use_input_adapter or self.input_adapter is None:
            return h
        return self.input_adapter(h)

    def apply_joint_emb(self, h: Tensor, bypass: bool = False) -> Tensor:
        if bypass or not self.use_joint_embedding or self.joint_embedding is None:
            return h
        return h + self.joint_embedding

    def apply_layer(self, h: Tensor, layer_idx: int, bypass: bool = False) -> Tensor:
        if (
            bypass
            or not self.use_layer_adapters
            or self.layer_adapters is None
        ):
            return h
        return self.layer_adapters[layer_idx](h)


def build_joint_modules(
    joints: Iterable[str] = JOINTS,
    d_model: int = D_MODEL,
    bottleneck: int = 64,
    n_layers: int = N_LAYERS,
    use_input_adapter: bool = True,
    use_layer_adapters: bool = True,
    use_joint_embedding: bool = True,
    input_bottleneck: Optional[int] = None,
) -> nn.ModuleDict:
    """Build a ModuleDict of PerJointModules.

    ``input_bottleneck`` overrides bottleneck width for the *input* adapter
    only (used by D-wide to match D's trainable param count).
    """
    ib = bottleneck if input_bottleneck is None else int(input_bottleneck)
    out = nn.ModuleDict()
    for j in joints:
        # Custom: input adapter may use different width than layer adapters.
        mod = PerJointModules(
            d_model=d_model,
            bottleneck=bottleneck,
            n_layers=n_layers,
            use_input_adapter=False,
            use_layer_adapters=use_layer_adapters,
            use_joint_embedding=use_joint_embedding,
        )
        if use_input_adapter:
            mod.use_input_adapter = True
            mod.input_adapter = BottleneckAdapter(d_model, ib)
        out[j] = mod
    return out


def count_trainable(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
