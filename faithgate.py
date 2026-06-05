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

# Synthetic RAM layout for the interpreter (addresses only need self-consistency).
BASE = 0xD00000
SIZE = 0x200000           # 2 MB window: program (~0xD1A881) + weight AppVars
AV_REGION = 0xE00000      # where we drop the AppVar payloads


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

    # Drop AppVar payloads into RAM and point AVPTRn at them (the runtime loader
    # is skipped — we set the pointers directly).
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

    cpu.run(L[meta['forward']])      # FORWARD reads GENPOS and writes OUTBUF
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
