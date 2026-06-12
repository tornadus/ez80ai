#!/usr/bin/env python3
"""
faithgate.py — the eZ80 faithfulness gate.

GRADER-OWNED. Builds the program in-process, executes the REAL emitted machine
code (ez80interp) over the packed weight AppVars on a fixed probe set, and
requires it to match intkernel.forward_device (the device contract) EXACTLY —
both the int16 logits and the argmax. This is what makes it safe to give the
agent build-coupled degrees of freedom: a codegen bug that diverges from the
integer reference fails the gate (FAITH_FAIL) instead of silently shipping
on-calc gibberish.

It complements test_faithfulness.py: that checks the device CONTRACT vs the sim
(math-spec drift); this checks the emitted BYTES vs the contract (codegen bugs).
"""

import contextlib
import io

import numpy as np

import buildchat84
import intkernel
from ez80interp import CPU
from loadmodel import load_model_params, load_spec_from_model

# Synthetic memory layout for the interpreter (addresses only need
# self-consistency). AppVars are placed at a FLASH-RANGE address (< 0xD00000),
# mirroring the device where weights are read in place from archive — so the
# emitted compute code is verified against flash-typical pointers. BASE must be
# 0: the interpreter indexes a flat bytearray as mem[addr - base], and a flash
# address below a 0xD00000 base would index NEGATIVE (silent wrong reads).
BASE = 0x000000
SIZE = 0xE40000           # flat window: flash AVs (0x3B0000) + program (~0xD1A881) + stack
AV_REGION = 0x3B0000      # where we drop the AppVar payloads (flash-range, like archive)


def build_resolved(model_path, quiet=True):
    """Build + resolve the program once; reused across probes. `quiet` swallows
    buildchat84's stdout so the gate doesn't pollute the harness output."""
    if quiet:
        with contextlib.redirect_stdout(io.StringIO()):
            b, blobs = buildchat84.build_autoreg(model_path)
    else:
        b, blobs = buildchat84.build_autoreg(model_path)
    b.resolve()
    return b, blobs


def run_emitted(b, blobs, params, spec, query, context, genpos):
    """Execute the emitted machine code for one (query, context, genpos) and
    return (logits int16 vector, argmax index) read straight from OUTBUF/RESULT."""
    meta = b.forward_meta
    L = b.labels
    mem = bytearray(SIZE)
    mem[b.org - BASE: b.org - BASE + len(b.code)] = b.code
    cpu = CPU(mem, BASE, b.org)

    # Drop AppVar payloads at flash-range addresses and point AVPTRn at them
    # (the runtime loader is skipped — we set the pointers directly). AVPTR
    # points at the blob start, i.e. AT the 8-byte magic header; every shard
    # offset already includes it, matching the device loader.
    addr = AV_REGION
    for i, name in enumerate(meta['appvar_names']):
        payload = blobs[name]
        mem[addr - BASE: addr - BASE + len(payload)] = payload
        cpu.w24(L[f'AVPTR{i}'], addr)
        addr += len(payload) + 16

    # Query -> INPBUF/INPLEN, then tokenize (fills TOKBUF[0:query_buckets]).
    for j, ch in enumerate(query):
        cpu.w8(L['INPBUF'] + j, ord(ch))
    cpu.w8(L['INPLEN'], len(query))
    cpu.run(L['TOKENIZE'])

    # Context -> CTXCHARS (lowercased, left-padded), then encode (fills the rest).
    clen = spec['context_len']
    ctx = context[-clen:].lower().rjust(clen)
    for j, ch in enumerate(ctx):
        cpu.w8(L['CTXCHARS'] + j, ord(ch))
    cpu.run(L['ENCODE_CTX'])

    cpu.w8(L['GENPOS'], genpos)

    # FORWARD's step count scales with the weight count (~7 steps/weight with
    # zero-skip; budget 16x + slack so big specs fit and a real hang still trips).
    dims = [spec['input_size']] + list(spec['hidden_sizes']) + [spec['num_classes']]
    n_weights = sum(a * b for a, b in zip(dims, dims[1:]))
    cpu.run(L[meta['forward']],      # FORWARD reads GENPOS and writes OUTBUF
            max_steps=max(20_000_000, 16 * n_weights))
    cpu.run(L[meta['argmax']])

    n = meta['output_size']
    ob = L['OUTBUF']
    logits = np.empty(n, dtype=np.int64)
    for i in range(n):
        v = cpu.r16(ob + 2 * i)
        logits[i] = v - 0x10000 if v & 0x8000 else v
    return logits, cpu.r8(L['RESULT'])


def default_probes():
    """A small fixed probe set. A codegen bug in the matmul/shift/bias/argmax
    shows up on essentially any input, so a handful covering the structural cases
    (genpos < and >= threshold, empty and non-empty context) is enough — and the
    interpreter is slow, so we keep it small."""
    queries = ["who are you", "hello", "what is your name", "how are you",
               "tell me a joke", "what can you do"]
    probes = []
    for q in queries:
        probes.append((q, "", 0))          # first char: start bias, empty context
        probes.append((q, "abc", 5))       # later char: rest bias, real context
    return probes


def check_faithfulness(model_path, probes=None, b=None, blobs=None):
    """Return (ok, reasons). ok == emitted bytes reproduce intkernel.forward_device
    EXACTLY on every probe."""
    if b is None:
        b, blobs = build_resolved(model_path)
    params, _arch, _charset = load_model_params(model_path)
    spec = load_spec_from_model(model_path)
    if probes is None:
        probes = default_probes()

    reasons = []
    for (query, context, genpos) in probes:
        emu_logits, emu_arg = run_emitted(b, blobs, params, spec, query, context, genpos)
        x = intkernel.build_input(query, context, spec)
        ref = intkernel.forward_device_params(params, x, spec, positions=[genpos])[0]
        if not np.array_equal(emu_logits, ref):
            d = int(np.abs(emu_logits - ref).max())
            reasons.append(f"LOGIT_DIVERGE(q={query!r},ctx={context!r},gp={genpos},maxdiff={d})")
        elif emu_arg != int(ref.argmax()):
            reasons.append(f"ARGMAX_DIVERGE(q={query!r},gp={genpos},"
                           f"emu={emu_arg},ref={int(ref.argmax())})")
    return (len(reasons) == 0, reasons)


if __name__ == '__main__':
    import argparse
    import time
    ap = argparse.ArgumentParser(description='eZ80 faithfulness gate')
    ap.add_argument('--model', default='model.npz')
    ap.add_argument('--probes', type=int, default=0,
                    help='limit to the first N probes (0 = all)')
    args = ap.parse_args()

    b, blobs = build_resolved(args.model)
    probes = default_probes()
    if args.probes:
        probes = probes[:args.probes]
    t0 = time.time()
    ok, reasons = check_faithfulness(args.model, probes=probes, b=b, blobs=blobs)
    dt = time.time() - t0
    print(f"probes={len(probes)} time={dt:.1f}s")
    if ok:
        print("PASS: emitted machine code is faithful to intkernel.forward_device")
        raise SystemExit(0)
    print("FAIL:")
    for r in reasons:
        print("  - " + r)
    raise SystemExit(1)
