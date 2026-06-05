#!/usr/bin/env python3
"""
intkernel.py — the ONE definition of NEOCHAT's integer inference math.

GRADER-OWNED. The research agent does not edit this; it edits modelspec.py.

Why this exists: the integer forward + the integer tokenizer used to be copied in
three places (train._forward_int, train.get_quantized_params' quantization, and
test_faithfulness). Copies drift, and on this project a drift once shipped on-calc
gibberish. This module is the single numpy reference that:

  * train._forward_int (torch) is held BIT-EXACT against (see test_intkernel.py),
  * the eZ80 faithfulness gate compares the emitted machine code against,
  * ez80research/evaluate.py builds its eval encoder from.

Pure numpy, no torch — so the faithfulness gate and CI can run without PyTorch.

ROUNDING (read this). The eZ80 divides between layers with `SRA A; RR D; RR E`,
which is an arithmetic shift = FLOOR toward -inf. The Python sim historically used
truncation toward zero (`torch.div(..., 'trunc')`); the two differ only on
negative accumulators. `spec['rounding']` selects which this kernel reproduces:
  * 'floor' = the true device contract (what the faithfulness gate checks against),
  * 'trunc' = the legacy sim/metric (kept the default so IntAcc history stays
    comparable). Setting spec['rounding']='floor' aligns the metric to the
    hardware and is a legitimate, logged experiment.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Integer arithmetic primitives (match the eZ80 exactly)
# ---------------------------------------------------------------------------

def wrap_accum(x, accum_bits=24):
    """Wrap into a signed `accum_bits` accumulator (eZ80 native register width)."""
    m = 1 << accum_bits
    half = m >> 1
    return ((x.astype(np.int64) + half) % m) - half


def rshift(x, s, rounding):
    """Right-shift by s bits with the chosen rounding.
    'floor'  -> arithmetic shift toward -inf (the device: SRA;RR;RR).
    'trunc'  -> toward zero (legacy torch.div trunc)."""
    x = x.astype(np.int64)
    if s == 0:
        return x
    if rounding == 'floor':
        return x >> s                      # numpy signed >> is arithmetic = floor
    # trunc toward zero: sign(x) * (|x| >> s)
    q = np.abs(x) >> s
    return np.where(x < 0, -q, q)


def quant_range(bits):
    """The asymmetric signed grid for `bits`-bit weights, matching the packer.
    2-bit -> (-2, 1); 4-bit -> (-8, 7)."""
    lo = -(1 << (bits - 1))
    hi = (1 << (bits - 1)) - 1
    return lo, hi


def quantize_layer(w, quantile, bits):
    """Float weights -> integer grid {lo..hi}. Scale = `quantile` of |w|.
    This is the SINGLE quantization recipe; exportmodel's packing and the torch
    integer path must produce identical integers."""
    w = np.asarray(w, dtype=np.float64)
    scale = max(float(np.quantile(np.abs(w).flatten(), quantile)), 1e-6)
    lo, hi = quant_range(bits)
    return np.clip(np.round(w / scale), lo, hi).astype(np.int64)


# ---------------------------------------------------------------------------
# Integer forward pass (mirrors train._forward_int, generalized over the spec)
# ---------------------------------------------------------------------------

def forward_int(weights, biases, bias_start, x, spec, positions=None):
    """Integer logits for already-quantized integer params.

    weights    : list of int arrays (out, in), one per layer (output layer last)
    biases     : list of int arrays (out,) per layer; the LAST is the output
                 layer's rest-bias (added AFTER the shift, as the dual bias)
    bias_start : int array (num_classes,) — output layer's start-bias
    x          : input counts (batch, input_size) or (input_size,)
    positions  : (batch,) generation positions; < dual_bias_threshold -> bias_start
    """
    acc = spec['accum_bits']
    rounding = spec['rounding']
    shifts = spec['inter_layer_shift']
    scale = spec['activation_scale']
    thr = spec['dual_bias_threshold']

    x = np.atleast_2d(np.asarray(x))
    h = np.round(x.astype(np.float64) * scale).astype(np.int64)

    n = len(weights)
    # Hidden layers: matmul + inline bias, wrap, shift, relu.
    for i in range(n - 1):
        h = h @ weights[i].astype(np.int64).T + biases[i].astype(np.int64)
        h = wrap_accum(h, acc)
        h = rshift(h, shifts[i], rounding)
        h = np.maximum(h, 0)

    # Output layer: matmul (no inline bias), wrap, shift, then dual bias.
    logits = h @ weights[-1].astype(np.int64).T
    logits = wrap_accum(logits, acc)
    logits = rshift(logits, shifts[-1], rounding)

    b_rest = biases[-1].astype(np.int64)
    b_start = bias_start.astype(np.int64)
    if positions is None:
        logits = logits + b_rest
    else:
        positions = np.asarray(positions).reshape(-1, 1)
        use_start = positions < thr
        logits = logits + np.where(use_start, b_start, b_rest)
    return logits


def layers_from_params(params):
    """Extract ordered (weights, biases, bias_start) from an npz-style param dict
    with keys fc1_weight/fc1_bias .. fcN_weight/fcN_bias (+ fcN_bias_start). The
    order is by the integer in the fc-name, matching buildchat84's discovery."""
    names = sorted((k[:-7] for k in params if k.endswith('_weight')),
                   key=lambda nm: int(''.join(ch for ch in nm if ch.isdigit())))
    weights = [np.asarray(params[f'{nm}_weight']) for nm in names]
    biases = [np.asarray(params[f'{nm}_bias']) for nm in names]
    start_key = f'{names[-1]}_bias_start'
    bias_start = np.asarray(params[start_key]) if start_key in params else biases[-1]
    return weights, biases, bias_start


def forward_int_params(params, x, spec, positions=None):
    """Convenience: integer forward straight from an npz-style param dict."""
    weights, biases, bias_start = layers_from_params(params)
    return forward_int(weights, biases, bias_start, x, spec, positions)


# ---------------------------------------------------------------------------
# Integer tokenizer (mirrors encoding.py, generalized over the spec)
# ---------------------------------------------------------------------------

def _hash_ngram(ngram, mult, mask, h0):
    h = h0
    for c in ngram:
        h = (h * mult + ord(c)) & mask
    return h


def tokenize_query(text, spec):
    """Query buckets. For the default trigram order this reproduces
    encoding.TrigramEncoder exactly (1-space padding both ends, window=n)."""
    nb = spec['query_buckets']
    hp = spec['query_hash']
    mult, mask, pom = hp['mult'], hp['mask'], hp['pos_offset_mult']
    vec = np.zeros(nb, dtype=np.int64)
    text = ' ' + text.lower() + ' '
    for n in spec['query_ngram_orders']:
        for i in range(len(text) - n + 1):
            h = _hash_ngram(text[i:i + n], mult, mask, (i * pom) & mask)
            vec[h & (nb - 1)] += 1
    return vec


def encode_context(recent, spec):
    """Context buckets. Reproduces encoding.ContextEncoder for the default config
    (left-pad to context_len, n in {1,2,3} with position offset)."""
    nb = spec['context_buckets']
    clen = spec['context_len']
    hp = spec['context_hash']
    mult, mask, pom = hp['mult'], hp['mask'], hp['pos_offset_mult']
    vec = np.zeros(nb, dtype=np.int64)
    recent = recent[-clen:].lower().rjust(clen)
    for n in spec['context_ngram_orders']:
        for i in range(len(recent) - n + 1):
            h = _hash_ngram(recent[i:i + n], mult, mask, (i * pom) & mask)
            vec[h & (nb - 1)] += 1
    return vec


def build_input(query, context, spec):
    """Full model input = [query buckets | context buckets]."""
    return np.concatenate([tokenize_query(query, spec),
                           encode_context(context, spec)]).astype(np.int64)
