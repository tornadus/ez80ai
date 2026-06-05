#!/usr/bin/env python3
"""
Device <-> sim numeric faithfulness checks for NEOCHAT.

WHY THIS EXISTS
  CLAUDE.md notes the on-calc build's numerics are "not auto-verified" against the
  Python sim, so any divergence is silent. This script closes most of that gap on
  the host. It reimplements, from model.npz, BOTH:
    * sim path  -- a bit-exact transcription of train._forward_int (the declared
                   source of truth), validated here against the live torch model.
    * device path -- the eZ80 integer CONTRACT that buildchat84.py emits: 24-bit
                   accumulate, add bias, arithmetic-shift (floor) /4, store each
                   activation as int16, output bias pre-scaled x4, dual-bias
                   threshold read from the model, plain first-max argmax.
  and asserts they generate identical text. It also pins the input-range invariant
  the int16 device storage silently depends on, and the query-strip tokenization
  contract.

LIMITATION (be honest): this checks the device *contract*, not the literal eZ80
bytes -- it does not run an emulator. It catches sim-side drift (e.g. someone
changes the quantile or scaling in _forward_int without updating the build) and
range/encoding regressions; it cannot catch a bug that lives purely in the
hand-written assembly. For that, build the --debug program and compare its
per-layer checksums on real hardware.

Usage:  ./venv/bin/python test_faithfulness.py
Exits 0 on pass, 1 on divergence, and SKIPS (0) if model.npz is absent.
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NPZ = os.path.join(HERE, 'model.npz')
PT = os.path.join(HERE, 'neochat_model.pt')
LABELED = os.path.join(HERE, 'labeled_data.txt')

from loadmodel import load_model_params, load_dual_bias_threshold
from train import CHARSET, EOS_IDX, ACTIVATION_SCALE
from encoding import TrigramEncoder, ContextEncoder

INT16_MAX = 32767
M24, H24, M16, H16 = 1 << 24, 1 << 23, 1 << 16, 1 << 15

def w24(x): return ((np.asarray(x, dtype=np.int64) + H24) % M24) - H24
def w16(x): return ((np.asarray(x, dtype=np.int64) + H16) % M16) - H16
def floor4(x): return np.asarray(x, dtype=np.int64) >> 2                  # eZ80 SRA;RR;RR
def trunc4(x):
    x = np.asarray(x, dtype=np.int64)
    return np.where(x >= 0, x // 4, -((-x) // 4))


def main():
    if not os.path.exists(NPZ):
        print(f"SKIP: {NPZ} not found (build/export a model first).")
        return 0

    params, arch, charset = load_model_params(NPZ)
    W = {k: params[k].astype(np.int64) for k in params if k.endswith('_weight')}
    B = {k: params[k].astype(np.int64) for k in params if k.endswith('_bias')}
    bstart = params['fc4_bias_start'].astype(np.int64)
    brest = params['fc4_bias'].astype(np.int64)
    DBT = load_dual_bias_threshold(NPZ)
    hidden = ['fc1', 'fc2', 'fc3']

    qe = TrigramEncoder(128); ce = ContextEncoder(128, 8)
    def raw(q, c): return np.concatenate([qe.encode(q), ce.encode(c)]).astype(np.int64)

    act_peak = {'maxabs': 0}

    def sim(rawcounts, genpos):
        """Exact transcription of train._forward_int (trunc /4, full precision)."""
        x = np.rint(rawcounts * ACTIVATION_SCALE).astype(np.int64)
        for nm in hidden:
            x = np.maximum(trunc4(w24(x @ W[nm + '_weight'].T + B[nm + '_bias'])), 0)
        bias = bstart if genpos < DBT else brest
        return trunc4(w24(x @ W['fc4_weight'].T)) + bias

    def device(rawcounts, genpos):
        """eZ80 contract: floor /4, int16 activation storage, output bias x4."""
        x = np.rint(rawcounts * ACTIVATION_SCALE).astype(np.int64)
        for nm in hidden:
            acc = w24(x @ W[nm + '_weight'].T + B[nm + '_bias'])
            x = np.maximum(w16(floor4(acc)), 0)
            act_peak['maxabs'] = max(act_peak['maxabs'], int(np.abs(x).max()))
        bias = bstart if genpos < DBT else brest
        return w16(floor4(w24(x @ W['fc4_weight'].T + bias * 4)))

    def gen(fwd, q, ml=50):
        out = ""
        for pos in range(ml):
            nx = int(np.asarray(fwd(raw(q, out), pos)).argmax())
            if nx == EOS_IDX or CHARSET[nx] == '\x00':
                break
            out += CHARSET[nx]
        return out.strip()

    fails = []

    # ---- 1. sim transcription must be bit-exact vs the live torch _forward_int ----
    if os.path.exists(PT):
        import torch
        from train import NeochatModel, filter_legacy_state
        cp = torch.load(PT, weights_only=False, map_location='cpu')
        m = NeochatModel(); m.load_state_dict(filter_legacy_state(cp['model_state'])); m.eval()
        worst = 0
        for q in ["who are you", "hello", "what is your name"]:
            for gp in (0, 5):
                rc = raw(q, "ab" if gp else "")
                tl = m._forward_int(torch.tensor(rc, dtype=torch.float32).unsqueeze(0),
                                    torch.full((1,), gp, dtype=torch.long)
                                    ).detach().squeeze(0).numpy().astype(np.int64)
                worst = max(worst, int(np.abs(tl - sim(rc, gp)).max()))
        print(f"[1] sim transcription vs torch _forward_int: max|logit diff| = {worst}")
        if worst != 0:
            fails.append("sim transcription drifted from torch _forward_int "
                         "(update test_faithfulness.sim to match train._forward_int)")
    else:
        print("[1] SKIP torch cross-check (neochat_model.pt absent)")

    # ---- build a query set ----
    queries = ["who are you", "hello", "what is your name", "how are you",
               "what is love", "tell me a joke", "goodbye", "what can you do"]
    if os.path.exists(LABELED):
        seen = set(queries)
        with open(LABELED) as f:
            for line in f:
                if '|' in line:
                    q = line.split('|', 1)[0].strip()
                    if q and q not in seen:
                        seen.add(q); queries.append(q)
                if len(queries) >= 150:
                    break

    # ---- 2. device contract must generate identically to the sim ----
    div = [q for q in queries if gen(device, q) != gen(sim, q)]
    print(f"[2] device-contract vs sim generation: {len(div)}/{len(queries)} divergent")
    for q in div[:8]:
        print(f"      {q!r}: sim={gen(sim, q)!r}  device={gen(device, q)!r}")
    if div:
        fails.append(f"{len(div)} queries generate differently on device vs sim")

    # ---- 3. int16 activation-range invariant (the device stores activations in
    #         2 bytes; the sim keeps full precision. If activations exceed int16
    #         the device silently wraps. This is the guard the dead overflow
    #         penalty never provided). ----
    print(f"[3] peak |activation| = {act_peak['maxabs']} (int16 ceiling {INT16_MAX})")
    if act_peak['maxabs'] > INT16_MAX:
        fails.append(f"activation magnitude {act_peak['maxabs']} exceeds int16 -- "
                     f"device int16 storage would silently wrap")

    # ---- 4. query tokenization == TrigramEncoder.encode(query.strip()) ----
    # The device strips trailing spaces (READ_INPUT) and skips leading spaces
    # (TOKENIZE), so a typed query must hash to the same buckets as the stripped
    # query the sim sees. Probe leading/trailing/internal-space variants.
    def device_query_counts(typed):
        core = typed.rstrip(' ').lstrip(' ').lower()      # READ_INPUT rstrip + TOKENIZE lstrip
        padded = ' ' + core + ' '
        vec = np.zeros(128, dtype=np.int64)
        for i in range(len(padded) - 2):
            h = 0
            for chr_ in padded[i:i + 3]:
                h = (h * 31 + ord(chr_)) & 0xFFFF
            vec[h % 128] += ACTIVATION_SCALE              # device adds 32 per hit
        return vec
    tok_fail = 0
    for base in ["who are you", "hello", "a b", "what is 2 2"]:
        for variant in (base, base + " ", " " + base, "  " + base + "   "):
            sim_vec = np.rint(qe.encode(variant.strip()) * ACTIVATION_SCALE).astype(np.int64)
            if not np.array_equal(device_query_counts(variant), sim_vec):
                tok_fail += 1
                print(f"      tokenize mismatch: {variant!r}")
    print(f"[4] query tokenization vs encode(strip): {tok_fail} mismatches")
    if tok_fail:
        fails.append(f"{tok_fail} query-tokenization mismatches vs encode(query.strip())")

    print()
    if fails:
        print("FAIL:")
        for f in fails:
            print("  - " + f)
        return 1
    print("PASS: device contract is faithful to the sim.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
