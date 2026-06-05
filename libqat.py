#!/usr/bin/env python3
"""
Quantization-Aware Training (QAT) with Overflow Simulation.

Trains neural networks that are aware of:
1. 2-bit weight quantization: weights in {-2, -1, 0, +1}
2. 24-bit signed accumulator overflow during matmul (eZ80 native)
3. Fixed-point activation scaling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# eZ80 constraints (24-bit signed accumulator)
MAX_ACCUM = 8388607      # 24-bit signed max
MIN_ACCUM = -8388608     # 24-bit signed min
ACTIVATION_SCALE = 32    # Fixed-point scale factor


class StraightThroughEstimator(torch.autograd.Function):
    """Straight-through estimator for non-differentiable ops."""

    @staticmethod
    def forward(ctx, x, x_quantized):
        return x_quantized

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


# Weight grid (2-bit, 4 codes). Default {-2,-1,0,+1}. A zero-free grid {-2,-1,+1,+2}
# (experiment) removes the 0 code so every weight carries a sign — addresses the
# measured ~64% deadzone (weights trapped at 0 contribute nothing). NeochatModel sets
# this module flag from spec['weight_grid'] at construction so both the QAT path
# (here) and the reference integer path (train._forward_int) stay consistent.
_ZERO_FREE_GRID = False


def _grid_round(w_scaled: torch.Tensor) -> torch.Tensor:
    """Round scaled weights to the active 2-bit code grid."""
    if _ZERO_FREE_GRID:
        s = torch.sign(w_scaled)
        s = torch.where(s == 0, torch.ones_like(s), s)
        m = torch.clamp(torch.round(w_scaled.abs()), 1, 2)   # {1,2}
        return s * m                                          # {-2,-1,+1,+2}
    return torch.clamp(torch.round(w_scaled), -2, 1)          # {-2,-1,0,+1}


def quantize_weights_2bit(w: torch.Tensor, hard: bool = True, temperature: float = 1.0) -> torch.Tensor:
    """Quantize weights to a 2-bit grid (4 codes; grid set by _ZERO_FREE_GRID).

    Args:
        w: Weights tensor
        hard: If True, use STE for gradients
        temperature: 0.0 = float weights, 1.0 = fully quantized
    """
    if temperature <= 0:
        return w

    scale = torch.quantile(w.abs().flatten(), 0.85).clamp(min=1e-6)
    w_scaled = w / scale
    w_quant = _grid_round(w_scaled) * scale

    if temperature >= 1.0:
        if hard:
            return StraightThroughEstimator.apply(w, w_quant)
        else:
            return w_quant
    else:
        w_blend = (1 - temperature) * w + temperature * w_quant
        if hard:
            return StraightThroughEstimator.apply(w, w_blend)
        else:
            return w_blend


def quantization_friendly_loss(w: torch.Tensor) -> torch.Tensor:
    """Loss that encourages weights to be close to the active quantization grid."""
    scale = torch.quantile(w.abs().flatten(), 0.85).clamp(min=1e-6)
    w_scaled = w / scale
    w_rounded = _grid_round(w_scaled)
    distance = (w_scaled - w_rounded).abs()
    return distance.mean()


class OverflowAwareLinear(nn.Module):
    """Linear layer with 2-bit weight quantization (STE).

    NOTE: the 24-bit accumulator never gets close to overflowing for the trained
    models we ship (peak |accum| is ~thousands vs the ~6.7M safe threshold), so
    the old overflow-penalty / max-accum tracking was a measured no-op (it added
    exactly 0.0 to the loss) and was removed. The real range guard now lives in
    test_faithfulness.py, which fails if any activation exceeds the int16 storage
    the device uses. MAX_ACCUM/MIN_ACCUM remain documented for reference.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.randn(out_features, in_features) * np.sqrt(2.0 / (in_features + out_features)))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor, quant_temp: float = 1.0) -> torch.Tensor:
        w_quant = quantize_weights_2bit(self.weight, hard=True, temperature=quant_temp)
        return F.linear(x, w_quant, self.bias)

    def get_quantization_loss(self) -> torch.Tensor:
        return quantization_friendly_loss(self.weight)
