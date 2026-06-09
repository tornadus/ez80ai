#!/usr/bin/env python3
"""Measure interpreter steps for one emitted FORWARD pass (perf proxy).

Not a cycle-accurate benchmark, but interpreter step count tracks the number of
executed instructions, which is what the LAYER inner-loop optimization targets.
"""
import contextlib
import io

import faithgate
from ez80interp import CPU
from loadmodel import load_model_params, load_spec_from_model


def measure(model_path='model.npz', query='who are you', genpos=0):
    b, blobs = faithgate.build_resolved(model_path)
    spec = load_spec_from_model(model_path)
    meta = b.forward_meta
    L = b.labels
    mem = bytearray(faithgate.SIZE)
    mem[b.org - faithgate.BASE: b.org - faithgate.BASE + len(b.code)] = b.code
    cpu = CPU(mem, faithgate.BASE, b.org)

    addr = faithgate.AV_REGION
    for i, name in enumerate(meta['appvar_names']):
        payload = blobs[name]
        mem[addr - faithgate.BASE: addr - faithgate.BASE + len(payload)] = payload
        cpu.w24(L[f'AVPTR{i}'], addr)
        addr += len(payload) + 16

    for j, ch in enumerate(query):
        cpu.w8(L['INPBUF'] + j, ord(ch))
    cpu.w8(L['INPLEN'], len(query))
    cpu.run(L['TOKENIZE'])
    tok_steps = cpu.steps

    clen = spec['context_len']
    ctx = ''.rjust(clen)
    for j, ch in enumerate(ctx):
        cpu.w8(L['CTXCHARS'] + j, ord(ch))
    cpu.run(L['ENCODE_CTX'])

    cpu.w8(L['GENPOS'], genpos)
    cpu.run(L[meta['forward']])
    fwd_steps = cpu.steps
    cpu.run(L[meta['argmax']])
    arg_steps = cpu.steps

    print(f"program bytes: {len(b.code)}")
    print(f"TOKENIZE steps: {tok_steps:,}")
    print(f"FORWARD steps:  {fwd_steps:,}")
    print(f"ARGMAX steps:   {arg_steps:,}")
    return fwd_steps


if __name__ == '__main__':
    measure()
