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


def quantize_weights_2bit(w: torch.Tensor, hard: bool = True, temperature: float = 1.0) -> torch.Tensor:
    """Quantize weights to 2-bit: {-2, -1, 0, +1} (4 values for 2 bits)

    Args:
        w: Weights tensor
        hard: If True, use STE for gradients
        temperature: 0.0 = float weights, 1.0 = fully quantized
    """
    if temperature <= 0:
        return w

    scale = torch.quantile(w.abs().flatten(), 0.90).clamp(min=1e-6)
    w_scaled = w / scale
    w_quant = torch.clamp(torch.round(w_scaled), -2, 1) * scale

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
    """Loss that encourages weights to be close to quantization grid {-2,-1,0,+1}."""
    scale = torch.quantile(w.abs().flatten(), 0.90).clamp(min=1e-6)
    w_scaled = w / scale
    w_rounded = torch.clamp(torch.round(w_scaled), -2, 1)
    distance = (w_scaled - w_rounded).abs()
    return distance.mean()


def quantize_activations(x: torch.Tensor, scale: int = ACTIVATION_SCALE) -> torch.Tensor:
    """Quantize activations to simulated fixed-point."""
    x_scaled = x * scale
    x_quant = torch.round(x_scaled)
    return StraightThroughEstimator.apply(x_scaled, x_quant)


class OverflowAwareLinear(nn.Module):
    """
    Linear layer with 2-bit weight quantization and overflow-aware regularization.

    Uses efficient matmul but adds regularization to prevent overflow:
    1. Quantize weights to {-2,-1,0,+1} using STE
    2. Compute worst-case accumulator magnitude
    3. Penalize if it would exceed 24-bit signed range
    """

    def __init__(self, in_features: int, out_features: int,
                 simulate_overflow: bool = True,
                 overflow_penalty: float = 0.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.simulate_overflow = simulate_overflow

        self.weight = nn.Parameter(torch.randn(out_features, in_features) * np.sqrt(2.0 / (in_features + out_features)))
        self.bias = nn.Parameter(torch.zeros(out_features))

        self.register_buffer('max_accum_seen', torch.tensor(0.0))

    def forward(self, x: torch.Tensor, quant_temp: float = 1.0) -> torch.Tensor:
        w_quant = quantize_weights_2bit(self.weight, hard=True, temperature=quant_temp)
        out = F.linear(x, w_quant, self.bias)

        if self.training and self.simulate_overflow:
            with torch.no_grad():
                w_hard = quantize_weights_2bit(self.weight, hard=False, temperature=1.0)
                worst_case = (w_hard.abs() @ x.abs().T).max()
                self.max_accum_seen = max(self.max_accum_seen, worst_case)

        return out

    def get_quantization_loss(self) -> torch.Tensor:
        return quantization_friendly_loss(self.weight)

    def get_overflow_risk(self) -> float:
        return (self.max_accum_seen / MAX_ACCUM).item()

    def compute_overflow_penalty(self, x: torch.Tensor) -> torch.Tensor:
        w_quant = quantize_weights_2bit(self.weight)
        accum_estimate = (w_quant.abs() @ x.abs().T)
        safe_threshold = MAX_ACCUM * 0.8
        overflow = F.relu(accum_estimate - safe_threshold)
        return overflow.mean()

    def reset_overflow_stats(self):
        self.max_accum_seen.zero_()
