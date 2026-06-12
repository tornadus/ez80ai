#!/usr/bin/env python3
"""
gen_synth_probe.py — build a synthetic random-weight model .npz whose spec
exercises codegen paths the shipped model does not (pattern of 45bf2ea):

  * in_size > 1024 (query_buckets=1024 + context_buckets=512 -> 1536)
  * 512/1024 bucket tokenizer masks (>256 high-byte mask path)
  * a column-major 2-bit layer needing multiple input-range shards and a
    multi-page row counter (1088 outputs -> 272 packed bytes/column)
  * a 4-bit layer (legacy row-major path + MULADD_GEN) coexisting with the
    2-bit column-major path in one FORWARD
  * dual output bias

Usage: venv/bin/python gen_synth_probe.py [out.npz]
Then:  venv/bin/python faithgate.py --model synth_probe.npz
"""
import json
import sys

import numpy as np

import modelspec

HIDDEN = [1088, 64]
WBITS = [2, 4, 2]
SHIFTS = [2, 2, 4]
QB, CB = 1024, 512
CHARSET = ' 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ.?!,-\x00'  # 42 chars + EOS


def main(out='synth_probe.npz'):
    spec = dict(modelspec.DEFAULT_SPEC)
    spec.update(hidden_sizes=HIDDEN, weight_bits=WBITS, inter_layer_shift=SHIFTS,
                query_buckets=QB, context_buckets=CB)
    spec = modelspec.validate(modelspec.resolve(spec))

    rng = np.random.default_rng(20260612)
    dims = [spec['input_size']] + HIDDEN + [spec['num_classes']]
    data = {}
    for i in range(len(dims) - 1):
        n_in, n_out = dims[i], dims[i + 1]
        lo = -(1 << (WBITS[i] - 1))
        hi = (1 << (WBITS[i] - 1)) - 1
        # Mostly-sparse weights (like real QAT layers) but with every code
        # value present, so zero-skip paths AND all decode arms are exercised.
        w = rng.integers(lo, hi + 1, size=(n_out, n_in))
        mask = rng.random((n_out, n_in)) < 0.55
        w = np.where(mask, 0, w)
        data[f'fc{i+1}_weight'] = w.astype(np.int64)
        data[f'fc{i+1}_bias'] = rng.integers(-1500, 1500,
                                             size=n_out).astype(np.int64)
    data[f'fc{len(dims)-1}_bias_start'] = rng.integers(
        -1500, 1500, size=dims[-1]).astype(np.int64)

    arch = {'input_size': spec['input_size'], 'hidden_sizes': HIDDEN,
            'num_classes': spec['num_classes']}
    data['_architecture'] = np.array(json.dumps(arch).encode())
    data['_charset'] = np.array(CHARSET.encode())
    data['_dual_bias_threshold'] = np.array(spec['dual_bias_threshold'])
    data['_modelspec'] = np.array(modelspec.to_json(spec).encode())
    np.savez(out, **data)
    print(f"wrote {out}: dims {dims}, bits {WBITS}, qb={QB} cb={CB} "
          f"(in_size {spec['input_size']})")


if __name__ == '__main__':
    main(*sys.argv[1:])
