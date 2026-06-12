#!/usr/bin/env python3
"""
modelspec.py — the single source of truth for a NEOCHAT model.

This is the ONE place the research agent changes to define a model. Every stage
of the pipeline (train.py, exportmodel.py, ez80research/evaluate.py,
buildchat84.py, the faithfulness gate) reads the SAME spec, so a knob set here
propagates everywhere instead of having to be hand-synced across files (the old
hazard: `weight_quantile` lived in 6 spots, `ACTIVATION_SCALE` in 3).

An experiment = editing DEFAULT_SPEC below (plus, rarely, adding a genuinely new
mechanism to BOTH the reference integer kernel and the eZ80 codegen).

ANTI-SKEW INVARIANT (learned the hard way — a divergent build once shipped
on-calc gibberish): the *resolved* spec is baked into the .pt checkpoint and the
.npz at train/export time, and every downstream stage reads that FROZEN baked
spec from the artifact — never this live file. So a model is always built/eval'd
with the exact spec it was trained with, even if DEFAULT_SPEC is edited between
train and build.

What stays FIXED (the task, not a model knob; validated, never configured):
  * the 43-char output charset incl. EOS (train.CHARSET) → num_classes == 43
  * integer-only inference, the calculator memory budgets (flash-resident
    weights vs sizes.FLASH_BUDGET_BYTES, program vs RAM), the in-sample IntAcc
    metric

Bucket counts must be powers of two: the eZ80 tokenizer masks with `and (n-1)`.
"""

import copy
import json

# The output space is the task definition, not a tunable.
EXPECTED_NUM_CLASSES = 43

# Defaults reproduce the released baseline EXACTLY. Changing a value here is how
# the agent runs an experiment. Keep this dict the canonical schema: every field
# a stage may read must exist here with a sensible default.
DEFAULT_SPEC = {
    # --- topology ---
    "hidden_sizes": [1600, 1408, 896],
    "num_classes": EXPECTED_NUM_CLASSES,     # validated == len(CHARSET)
    "activation": "relu",                    # enum: relu (others need kernel+codegen)

    # --- per-layer integer path ---
    # weight_bits / inter_layer_shift may be a scalar (broadcast to every layer)
    # or an explicit per-layer list of length len(hidden_sizes)+1 (output last).
    # Scalars are the default so changing hidden_sizes "just works".
    "weight_bits": 2,                        # 2-bit today; mixed precision allowed
    "weight_quantile": 0.85,                 # per-layer scale = this pctile of |W|
    # right-shift after each layer; scalar broadcasts, list is per-layer (output
    # last). Output shift is 4 (not 2): logits are stored int16 on device, and
    # the ~5M-param archs swing raw output accumulators to ~±82k>>2 — two extra
    # bits keep every logit inside int16 (measured headroom ~37%).
    "inter_layer_shift": [2, 2, 2, 4],
    "activation_scale": 32,                  # fixed-point input scale
    "rounding": "trunc",                     # 'trunc' (sim today) | 'floor' (device)
    "weight_grid": "default",                # 'default' {-2,-1,0,1} | 'zero_free' {-2,-1,1,2}
    "accum_bits": 24,                        # eZ80 native accumulator width

    # --- output bias ---
    "dual_bias": True,
    "dual_bias_threshold": 3,                # first N gen positions use bias_start
    "n_bias_buckets": 2,                     # 2 == dual; >2 reserved for later

    # --- encoding (a real DOF; input_size is DERIVED from the two bucket counts) ---
    "query_buckets": 512,                    # power of two
    "context_buckets": 512,                  # power of two
    "context_len": 8,
    "query_ngram_orders": [3],               # trigram query today
    "context_ngram_orders": [1, 2, 3],
    "query_hash": {"mult": 31, "mask": 0xFFFF, "pos_offset_mult": 0},
    "context_hash": {"mult": 31, "mask": 0xFFFF, "pos_offset_mult": 7},
    "signed_hash": False,                    # signed count-sketch (±1 per n-gram)

    # --- training hyperparameters ---
    "lr": 0.002,
    "batch_size": 3072,
    "epochs": 300,
    "weight_decay": 1e-4,                    # LOAD-BEARING for quantization (do not 0)
    "optimizer": "adam",
    "quant_loss_weight": 0.25,
    "qt_start": 0.3,
    "qt_ramp_factor": 0.3,                   # QT reaches 1.0 by qt_ramp_factor*horizon
    "lr_schedule": "cosine",
    "eta_min_frac": 0.02,

    # --- training-loop speed knobs (float-path ONLY: never touch _forward_int /
    # intkernel / the exported artifact, whose quantization always recomputes the
    # exact full quantile). Defaults reproduce the legacy loop exactly. ---
    "train_compile": True,        # torch.compile the model body (quantile stays eager)
    "train_bf16": True,           # bf16 autocast on matmuls (XMX units); eval stays fp32
    "scale_refresh_every": 1,      # recompute weight-scale quantile every k steps
                                   # (1 = every step, exact incl. quantile gradient;
                                   #  k>1 = detached stale scale, drops that gradient)

    # --- compute budget (P6); 'wall_s' keeps today's wall-clock semantics) ---
    "compute_budget": {"mode": "wall_s", "limit": 600},
}


class SpecError(ValueError):
    """Raised when a model spec is internally inconsistent or infeasible."""


def _is_pow2(n):
    return isinstance(n, int) and n > 0 and (n & (n - 1)) == 0


def n_layers(spec):
    """Number of weight layers = hidden layers + the output layer."""
    return len(spec["hidden_sizes"]) + 1


def layer_dims(spec):
    """[input_size, *hidden_sizes, num_classes] — the I/O width of every layer."""
    return [spec["input_size"]] + list(spec["hidden_sizes"]) + [spec["num_classes"]]


def _broadcast(value, n):
    """Allow a scalar to stand in for a per-layer list of length n."""
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value] * n


def resolve(spec):
    """Fill derived fields and broadcast scalars to per-layer lists. Pure (returns
    a new dict). `input_size` is ALWAYS derived from the bucket counts so it cannot
    drift from the encoder."""
    s = copy.deepcopy(spec)
    s["input_size"] = s["query_buckets"] + s["context_buckets"]
    nl = len(s["hidden_sizes"]) + 1
    s["weight_bits"] = _broadcast(s.get("weight_bits", 2), nl)
    s["inter_layer_shift"] = _broadcast(s.get("inter_layer_shift", 2), nl)
    return s


def validate(spec):
    """Raise SpecError on any inconsistency. Call on a RESOLVED spec."""
    nl = len(spec["hidden_sizes"]) + 1

    if spec["num_classes"] != EXPECTED_NUM_CLASSES:
        raise SpecError(
            f"num_classes must be {EXPECTED_NUM_CLASSES} (the fixed task charset), "
            f"got {spec['num_classes']}")
    if spec["input_size"] != spec["query_buckets"] + spec["context_buckets"]:
        raise SpecError(
            f"input_size {spec['input_size']} != query_buckets+context_buckets "
            f"{spec['query_buckets']}+{spec['context_buckets']}")
    for name in ("query_buckets", "context_buckets"):
        if not _is_pow2(spec[name]):
            raise SpecError(f"{name} must be a power of two (asm masks with `and`), "
                            f"got {spec[name]}")
    for name in ("weight_bits", "inter_layer_shift"):
        if len(spec[name]) != nl:
            raise SpecError(f"{name} must have length {nl} (one per weight layer), "
                            f"got {len(spec[name])}")
    for b in spec["weight_bits"]:
        if b not in (2, 3, 4):
            raise SpecError(f"weight_bits entries must be in {{2,3,4}}, got {b}")
    if any(s < 0 for s in spec["inter_layer_shift"]):
        raise SpecError("inter_layer_shift entries must be >= 0")
    if spec["rounding"] not in ("trunc", "floor"):
        raise SpecError(f"rounding must be 'trunc' or 'floor', got {spec['rounding']}")
    if spec.get("weight_grid", "default") not in ("default", "zero_free"):
        raise SpecError(f"weight_grid must be 'default' or 'zero_free', got {spec.get('weight_grid')}")
    if spec["activation"] != "relu":
        raise SpecError(f"activation '{spec['activation']}' not implemented "
                        f"(needs a kernel + codegen addition)")
    if spec["n_bias_buckets"] < 1:
        raise SpecError("n_bias_buckets must be >= 1")
    if not (0.0 < spec["weight_quantile"] < 1.0):
        raise SpecError("weight_quantile must be in (0,1)")
    if spec["weight_decay"] == 0:
        raise SpecError("weight_decay is load-bearing for quantization; must be > 0")
    return spec


def load_spec(path=None):
    """Return a fully-defaulted, resolved, validated spec.

    With no path: the canonical DEFAULT_SPEC (what the agent edits in this file).
    With a path: a JSON spec on disk, layered over DEFAULT_SPEC so partial specs
    are allowed. This is the live-file entry point used at TRAIN time; downstream
    stages instead read the frozen spec baked into the artifact (load_baked)."""
    base = copy.deepcopy(DEFAULT_SPEC)
    if path is not None:
        with open(path) as f:
            base.update(json.load(f))
    return validate(resolve(base))


def from_legacy(arch, dual_bias_threshold=3):
    """Reconstruct a resolved spec from an OLD artifact that predates modelspec
    (only an `architecture` dict + a dual-bias threshold). Defaults fill the rest;
    by construction this reproduces the baseline behavior for legacy checkpoints."""
    base = copy.deepcopy(DEFAULT_SPEC)
    base["hidden_sizes"] = list(arch["hidden_sizes"])
    base["num_classes"] = arch["num_classes"]
    base["dual_bias_threshold"] = int(dual_bias_threshold)
    # Honor a legacy input_size by splitting it evenly across the two encoders
    # (the baseline is 128+128=256), so the derived input_size matches.
    in_size = arch.get("input_size", base["query_buckets"] + base["context_buckets"])
    if in_size != base["query_buckets"] + base["context_buckets"]:
        half = in_size // 2
        base["query_buckets"] = base["context_buckets"] = half
    return validate(resolve(base))


def to_json(spec):
    """Serialize a resolved spec to a compact JSON string for baking into a .pt /
    .npz artifact."""
    return json.dumps(spec, sort_keys=True, separators=(",", ":"))


def from_json(s):
    """Parse a baked JSON spec string and re-validate it."""
    return validate(resolve(json.loads(s)))


if __name__ == "__main__":
    # `python3 modelspec.py` prints the resolved default spec (handy sanity check).
    print(to_json(load_spec()))
