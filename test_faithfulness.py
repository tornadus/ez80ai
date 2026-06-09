#!/usr/bin/env python3
"""
Device <-> sim numeric faithfulness checks for NEOCHAT.

WHY THIS EXISTS
  The on-calc build must generate character-for-character what the Python sim
  generates. This script checks the device CONTRACT against the sim on the host:
    * sim path    -- intkernel.forward_int (held bit-exact vs train._forward_int,
                     the declared source of truth, by test_intkernel.py).
    * device path -- intkernel.forward_device (24-bit accumulate, floor shift,
                     int16 activation/logit storage, output bias pre-scaled,
                     dual-bias threshold from the model spec).
  and asserts they generate identical text, pins the int16 activation-range
  invariant the device storage silently depends on, and pins the query-strip
  tokenization contract.

  Everything is SPEC-DRIVEN from the artifact's frozen baked spec (bucket counts,
  layer sizes/shifts, activation scale, dual-bias threshold), so this keeps
  working when modelspec knobs change. An earlier version hardcoded the baseline
  128/128 encoders and fc1..fc4 names and silently went stale when the encoding
  grew -- everything here now comes from load_spec_from_model/intkernel.

LIMITATION (be honest): this checks the device *contract*, not the literal eZ80
bytes. It catches sim-side drift (e.g. someone changes the quantile or scaling
in _forward_int without updating the contract) and range/encoding regressions;
it cannot catch a bug that lives purely in the emitted assembly. That gap is
closed by faithgate.py, which executes the real emitted machine code in
ez80interp and compares it against intkernel.forward_device exactly.

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

import intkernel
from loadmodel import load_model_params, load_spec_from_model
from encoding import TrigramEncoder

INT16_MAX = 32767


def main():
    if not os.path.exists(NPZ):
        print(f"SKIP: {NPZ} not found (build/export a model first).")
        return 0

    params, _arch, charset = load_model_params(NPZ)
    spec = load_spec_from_model(NPZ)
    weights, biases, bias_start = intkernel.layers_from_params(params)
    scale = spec['activation_scale']
    shifts = spec['inter_layer_shift']
    thr = spec['dual_bias_threshold']
    eos_idx = len(charset) - 1

    def raw(q, c):
        return intkernel.build_input(q, c, spec)

    def sim(rawcounts, genpos):
        """The sim/metric path (train._forward_int's reference)."""
        return intkernel.forward_int_params(params, rawcounts, spec,
                                            positions=[genpos])[0]

    act_peak = {'maxabs': 0}

    def device(rawcounts, genpos):
        """intkernel.forward_device with peak-activation tracking. This is the
        ONE local re-statement of the contract (forward_device does not expose
        intermediates); section [0] holds it equal to forward_device_params so
        it cannot drift silently."""
        h = np.round(np.atleast_2d(rawcounts).astype(np.float64)
                     * scale).astype(np.int64)
        for i in range(len(weights) - 1):
            acc = intkernel.wrap_accum(
                h @ weights[i].astype(np.int64).T + biases[i].astype(np.int64),
                spec['accum_bits'])
            h = np.maximum(intkernel.wrap16(acc >> shifts[i]), 0)
            act_peak['maxabs'] = max(act_peak['maxabs'], int(np.abs(h).max()))
        s = shifts[-1]
        bias = (bias_start if genpos < thr else biases[-1]).astype(np.int64)
        acc = intkernel.wrap_accum(
            h @ weights[-1].astype(np.int64).T + (bias << s), spec['accum_bits'])
        return intkernel.wrap16(acc >> s)[0]

    def gen(fwd, q, ml=50):
        out = ""
        for pos in range(ml):
            nx = int(np.asarray(fwd(raw(q, out), pos)).argmax())
            if nx == eos_idx or charset[nx] == '\x00':
                break
            out += charset[nx]
        return out.strip()

    fails = []

    # ---- 0. the local peak-tracking device copy == intkernel.forward_device ----
    probe = raw("who are you", "")
    dev_diverge = 0
    for gp in (0, thr - 1, thr, thr + 2):
        ref = intkernel.forward_device_params(params, probe, spec,
                                              positions=[gp])[0]
        if not np.array_equal(device(probe, gp), ref):
            dev_diverge += 1
    print(f"[0] local device tracker vs intkernel.forward_device: "
          f"{dev_diverge}/4 divergent")
    if dev_diverge:
        fails.append("local device() restatement drifted from "
                     "intkernel.forward_device (update it)")

    # ---- 1. sim reference must be bit-exact vs the live torch _forward_int ----
    if os.path.exists(PT):
        pt_spec = load_spec_from_model(PT)
        if (pt_spec['input_size'] != spec['input_size']
                or list(pt_spec['hidden_sizes']) != list(spec['hidden_sizes'])):
            print("[1] SKIP torch cross-check (neochat_model.pt arch "
                  f"{pt_spec['input_size']}->{pt_spec['hidden_sizes']} differs "
                  f"from model.npz {spec['input_size']}->{spec['hidden_sizes']} "
                  "-- stale checkpoint, re-export)")
        else:
            import torch
            from train import NeochatModel, filter_legacy_state
            cp = torch.load(PT, weights_only=False, map_location='cpu')
            m = NeochatModel(input_size=pt_spec['input_size'],
                             hidden_sizes=pt_spec['hidden_sizes'],
                             num_chars=pt_spec['num_classes'], spec=pt_spec)
            m.load_state_dict(filter_legacy_state(cp['model_state']))
            m.eval()
            worst = 0
            for q in ["who are you", "hello", "what is your name"]:
                for gp in (0, thr + 2):
                    rc = raw(q, "ab" if gp else "")
                    tl = m._forward_int(
                        torch.tensor(rc, dtype=torch.float32).unsqueeze(0),
                        torch.full((1,), gp, dtype=torch.long)
                    ).detach().squeeze(0).numpy().astype(np.int64)
                    worst = max(worst, int(np.abs(tl - sim(rc, gp)).max()))
            print(f"[1] sim reference vs torch _forward_int: "
                  f"max|logit diff| = {worst}")
            if worst != 0:
                fails.append("intkernel sim drifted from torch _forward_int, or "
                             "neochat_model.pt and model.npz are from different "
                             "runs (re-export with exportmodel.py)")
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
    # query the sim sees. Device side = intkernel.tokenize_query (which faithgate
    # holds equal to the emitted TOKENIZE); sim side = encoding.TrigramEncoder
    # (what training used), built from the same spec.
    qh = spec['query_hash']
    qe = TrigramEncoder(num_buckets=spec['query_buckets'],
                        ngram_orders=spec['query_ngram_orders'],
                        hash_mult=qh['mult'], hash_mask=qh['mask'],
                        pos_offset_mult=qh['pos_offset_mult'],
                        signed_hash=spec.get('signed_hash', False))
    tok_fail = 0
    for base in ["who are you", "hello", "a b", "what is 2 2"]:
        for variant in (base, base + " ", " " + base, "  " + base + "   "):
            core = variant.rstrip(' ').lstrip(' ')   # READ_INPUT rstrip + TOKENIZE lstrip
            dev_vec = intkernel.tokenize_query(core, spec) * scale
            sim_vec = np.rint(qe.encode(variant.strip()) * scale).astype(np.int64)
            if not np.array_equal(dev_vec, sim_vec):
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
