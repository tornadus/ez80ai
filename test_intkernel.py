#!/usr/bin/env python3
"""
test_intkernel.py — hold the shared integer reference (intkernel.py) bit-exact
against the two things it is supposed to be the single source of truth for:

  1. the tokenizer: intkernel.tokenize_query / encode_context vs encoding.py
  2. the integer forward: intkernel.forward_int (numpy, exact int64) vs
     train.NeochatModel._forward_int (torch).

If either drifts, the faithfulness gate and the metric would be measuring
different things. Run: ./venv/bin/python test_intkernel.py
"""

import sys
import numpy as np
import torch

import modelspec
import intkernel
from encoding import TrigramEncoder, ContextEncoder
from train import NeochatModel, filter_legacy_state


def test_tokenizer(spec):
    qe = TrigramEncoder(num_buckets=spec['query_buckets'])
    ce = ContextEncoder(num_buckets=spec['context_buckets'],
                        context_len=spec['context_len'])
    samples = ["", "hello", "WHAT IS YOUR NAME", "a", "the quick brown fox",
               "12345", "?!,.-", "  spaces  "]
    ctx_samples = ["", "h", "he", "hello world", "abcdefghijklmnop", "  "]
    for s in samples:
        a = qe.encode(s).astype(np.int64)
        b = intkernel.tokenize_query(s, spec)
        assert np.array_equal(a, b), f"query mismatch on {s!r}"
    for s in ctx_samples:
        a = ce.encode(s).astype(np.int64)
        b = intkernel.encode_context(s, spec)
        assert np.array_equal(a, b), f"context mismatch on {s!r}"
    print(f"[tokenizer] OK ({len(samples)} query + {len(ctx_samples)} context cases)")


def test_forward(spec, model, n=4000, seed=7):
    rng = np.random.default_rng(seed)
    # Random inputs in the realistic small-count range.
    X = rng.integers(0, 6, size=(n, spec['input_size'])).astype(np.float32)
    pos = rng.integers(0, 8, size=n)
    params = model.get_quantized_params()

    with torch.no_grad():
        tl = model(torch.tensor(X), positions=torch.tensor(pos),
                   use_int=True).numpy().astype(np.int64)
    il = intkernel.forward_int_params(params, X, model.spec, positions=pos)

    diff = np.abs(tl - il).max()
    assert diff == 0, f"forward mismatch: max |logit diff| = {diff}"
    print(f"[forward] OK  bit-exact over {n} random inputs (max logit diff 0)")


def main():
    spec = modelspec.load_spec()
    test_tokenizer(spec)

    # Test against the real checkpoint if present, else a random-init model.
    try:
        cp = torch.load('neochat_model.pt', weights_only=False, map_location='cpu')
        model = NeochatModel()
        model.load_state_dict(filter_legacy_state(cp['model_state']))
        src = "real checkpoint"
    except FileNotFoundError:
        model = NeochatModel()
        src = "random-init model"
    model.eval()
    print(f"[forward] model = {src}")
    test_forward(spec, model)

    # Also exercise a random-init model (different weight distribution).
    test_forward(spec, NeochatModel().eval(), n=2000, seed=99)
    print("ALL OK")


if __name__ == '__main__':
    sys.exit(main())
