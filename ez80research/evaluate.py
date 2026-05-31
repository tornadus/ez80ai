#!/usr/bin/env python3
"""
ez80research feasibility + quality harness — the source of truth that keeps the
autonomous agent honest.

Given a freshly trained checkpoint, this:
  1. QUALITY   re-measures integer-inference accuracy (IntAcc) ITSELF, via the
               use_int=True path, never trusting the number train.py logged.
  2. CONTRACT  pins the I/O contract (43-char charset, 256-dim input) and
               cross-checks measured vs reported IntAcc to catch a faked metric.
  3. EXPORT    runs the real exportmodel.py (.pt -> .npz).
  4. BUILD     runs the real buildchat84.py (.npz -> .8xp + 4x .8xv).
  5. SIZE GATE applies the hard calculator budget to the REAL printed sizes.
  6. VERDICT   prints one grep-able line and exits 0 (PASS) / 1 (FAIL).

A model only passes if it actually exports, builds, fits the calculator, keeps
the contract, and has nonzero integer-path accuracy. Float-only "improvements",
unbuildable architectures, and over-budget models all FAIL.

Usage:
    python3 ez80research/evaluate.py --model neochat_model.pt \
        --npz model.npz --bin-dir bin [--samples 8000]
"""

import argparse
import os
import random
import subprocess
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import sizes  # noqa: E402
from train import (  # noqa: E402
    NeochatModel, CHARSET, INPUT_SIZE, NUM_CHARS,
    create_training_examples_with_pos,
)
from encoding import TrigramEncoder, ContextEncoder, parse_pair  # noqa: E402

PY = sys.executable  # the venv python running this harness
TRAINING_DATA = os.path.join(REPO_ROOT, 'training_data.txt')

# Anti-gaming / robustness thresholds
# Gaming = train.py logs a fake-high IntAcc while the real integer-path accuracy
# is low. So we only fail when the REPORTED number is inflated above what the
# harness MEASURES (one-directional); measured > reported just means the
# checkpoint was saved before/between train's periodic IntAcc evals, which is
# fine. Tolerance is generous because the two numbers use different samples.
INTACC_INFLATION_TOL = 0.10
EVAL_SEED = 1234          # fixed seed -> steadier, comparable IntAcc across runs


def get_device():
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        return torch.device('xpu')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def read_checkpoint_meta(model_path):
    """Read metadata without constructing the model."""
    cp = torch.load(model_path, weights_only=False, map_location='cpu')
    return {
        'architecture': cp.get('architecture', {}),
        'charset': cp.get('charset', ''),
        'total_epochs': cp.get('total_epochs', 0),
        'best_int_acc': float(cp.get('best_int_acc', 0.0)),
        'dual_bias_threshold': cp.get('dual_bias_threshold', None),
    }


def measure_intacc(model_path, device, n_samples=8000):
    """Independently re-measure integer-path next-char accuracy.

    Mirrors test_model.test_intacc internals (test_model.py:61-100) but returns
    the float accuracy instead of a pass/fail bool, and seeds the sampling so
    the number is comparable across experiments. The score the harness reports
    is computed HERE via use_int=True — not read from the checkpoint."""
    random.seed(EVAL_SEED)
    torch.manual_seed(EVAL_SEED)

    pairs = []
    with open(TRAINING_DATA) as f:
        for line in f:
            p = parse_pair(line)
            if p:
                pairs.append(p)
    random.shuffle(pairs)
    pairs = pairs[:n_samples]

    qe = TrigramEncoder(num_buckets=128)
    ce = ContextEncoder(num_buckets=128, context_len=8)

    examples = []
    for q, r in pairs:
        examples.extend(create_training_examples_with_pos(q, r, qe, ce))
    if len(examples) > 50000:
        random.shuffle(examples)
        examples = examples[:50000]

    cp = torch.load(model_path, weights_only=False, map_location='cpu')
    model = NeochatModel()
    model.load_state_dict(cp['model_state'])
    model.to(device)
    model.eval()

    X = torch.tensor(np.stack([e[0] for e in examples]),
                     dtype=torch.float32).to(device)
    y = torch.tensor(np.array([e[1] for e in examples]),
                     dtype=torch.long).to(device)
    pos = torch.tensor(np.array([e[2] for e in examples]),
                       dtype=torch.long).to(device)

    with torch.no_grad():
        int_out = model(X, positions=pos, use_int=True)
        int_acc = (int_out.argmax(dim=1) == y).float().mean().item()
    return int_acc


def check_contract(meta, measured_intacc):
    """Return list of contract VIOLATIONS (empty == OK). These pins stop the
    agent from 'winning' by shrinking the problem or faking the metric."""
    violations = []
    arch = meta['architecture']

    if arch.get('num_classes') != NUM_CHARS:
        violations.append(f'CONTRACT_NUMCLASSES({arch.get("num_classes")}!={NUM_CHARS})')
    if arch.get('input_size') != INPUT_SIZE:
        violations.append(f'CONTRACT_INPUTSIZE({arch.get("input_size")}!={INPUT_SIZE})')
    if meta['charset'] != CHARSET:
        violations.append('CONTRACT_CHARSET')

    reported = meta['best_int_acc']
    if reported - measured_intacc > INTACC_INFLATION_TOL:
        violations.append(
            f'CONTRACT_ACC_INFLATED(meas={measured_intacc:.3f},rep={reported:.3f})'
        )
    if measured_intacc <= 0.0:
        violations.append('CONTRACT_ZERO_INTACC')
    return violations


def run_export(model_path, npz_path):
    return subprocess.run(
        [PY, os.path.join(REPO_ROOT, 'exportmodel.py'),
         '-m', model_path, '-o', npz_path],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )


def run_build(npz_path, bin_dir):
    out_8xp = os.path.join(bin_dir, 'NEOCHAT.8xp')
    return subprocess.run(
        [PY, os.path.join(REPO_ROOT, 'buildchat84.py'),
         '-m', npz_path, '-o', out_8xp],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )


def parse_build_sizes(build_stdout):
    """Parse REAL printed byte counts from buildchat84.py stdout
    (buildchat84.py:2130-2143). Ground truth, not estimate."""
    program = 0
    appvars = {}
    total_weight = 0
    for line in build_stdout.splitlines():
        s = line.strip()
        if s.startswith('Program:'):
            program = int(s.split()[1])
        elif s.startswith('AppVar '):
            # "AppVar NEOA: 34,599 bytes -> bin/NEOA.8xv"
            parts = s.split()
            name = parts[1].rstrip(':')
            nbytes = int(parts[2].replace(',', ''))
            appvars[name] = nbytes
        elif s.startswith('Total weight data:'):
            total_weight = int(s.split()[3].replace(',', ''))
    return {
        'program': program,
        'appvars': appvars,
        'total_weight': total_weight,
        'n_appvars': len(appvars),
    }


def verdict_line(passed, intacc, sizes_d, reasons):
    if sizes_d.get('program') or sizes_d.get('total_weight'):
        ram_kb = sizes.total_ram(sizes_d.get('program', 0),
                                 sizes_d.get('total_weight', 0)) / 1024
    else:
        ram_kb = 0.0  # never reached the build stage
    n_av = sizes_d.get('n_appvars', 0)
    maxav_kb = (max(sizes_d['appvars'].values()) / 1024
                if sizes_d.get('appvars') else 0.0)
    reason_str = ';'.join(reasons) if reasons else '-'
    return (f"VERDICT pass={1 if passed else 0} intacc={intacc:.4f} "
            f"ram_kb={ram_kb:.1f} appvars={n_av} maxav_kb={maxav_kb:.1f} "
            f"reasons={reason_str}")


def _fail(intacc, sizes_d, reasons):
    print(verdict_line(False, intacc, sizes_d, reasons))
    sys.exit(1)


def main():
    ap = argparse.ArgumentParser(description='ez80research feasibility/quality harness')
    ap.add_argument('--model', default=os.path.join(REPO_ROOT, 'neochat_model.pt'))
    ap.add_argument('--npz', default=os.path.join(REPO_ROOT, 'model.npz'))
    ap.add_argument('--bin-dir', default=os.path.join(REPO_ROOT, 'bin'))
    ap.add_argument('--samples', type=int, default=8000)
    args = ap.parse_args()

    empty_sizes = {'appvars': {}, 'n_appvars': 0}

    try:
        if not os.path.exists(args.model):
            _fail(0.0, empty_sizes, ['NO_CHECKPOINT'])

        device = get_device()
        meta = read_checkpoint_meta(args.model)

        # 1. QUALITY (harness-computed, integer path)
        intacc = measure_intacc(args.model, device, n_samples=args.samples)
        print(f"[quality] measured IntAcc (integer path) = {intacc:.4f} "
              f"(checkpoint reported {meta['best_int_acc']:.4f})")

        # 2. CONTRACT
        violations = check_contract(meta, intacc)
        if violations:
            print(f"[contract] VIOLATIONS: {violations}")
            _fail(intacc, empty_sizes, violations)
        print("[contract] OK")

        # 3. EXPORT
        exp = run_export(args.model, args.npz)
        if exp.returncode != 0:
            print("[export] FAILED\n" + exp.stdout + "\n" + exp.stderr)
            _fail(intacc, empty_sizes, ['EXPORT_FAIL'])
        print("[export] OK")

        # 4. BUILD
        os.makedirs(args.bin_dir, exist_ok=True)
        bld = run_build(args.npz, args.bin_dir)
        if bld.returncode != 0:
            print("[build] FAILED\n" + bld.stdout + "\n" + bld.stderr)
            # A BudgetError raised by the build also lands here; surface it.
            reason = 'BUILD_OVER_BUDGET' if 'BudgetError' in bld.stderr else 'BUILD_CRASH'
            _fail(intacc, empty_sizes, [reason])
        print("[build] OK")

        # 5. SIZE GATE (against real printed sizes)
        build_sizes = parse_build_sizes(bld.stdout)
        ok, reasons = sizes.check_budget(build_sizes)
        ram_kb = sizes.total_ram(build_sizes['program'],
                                 build_sizes['total_weight']) / 1024
        if not ok:
            print(f"[size] OVER BUDGET: {reasons}")
            _fail(intacc, build_sizes, reasons)
        print(f"[size] OK  ram={ram_kb:.1f}KB appvars={build_sizes['n_appvars']}")

        # 6. VERDICT — PASS
        print(verdict_line(True, intacc, build_sizes, []))
        sys.exit(0)

    except SystemExit:
        raise
    except Exception as e:  # harness bug -> revert, never falsely keep
        print(f"[harness] ERROR: {type(e).__name__}: {e}")
        _fail(0.0, empty_sizes, ['HARNESS_ERROR'])


if __name__ == '__main__':
    main()
