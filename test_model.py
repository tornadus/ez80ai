#!/usr/bin/env python3
"""
Automated quality checks for NEOCHAT model.

Tests:
  1. IntAcc — integer vs float inference agreement
  2. Coherence — no 3+ repeated characters
  3. Charset compliance — all output chars valid
  4. EOS behavior — responses terminate before max_len
  5. Personality recall — exact match on labeled data
  6. Diversity — no empty responses

Usage:
    python3 test_model.py
    python3 test_model.py --model neochat_model.pt --samples 200
"""

import os
import random
import sys
import numpy as np
import torch

from train import (
    NeochatModel, CHARSET, EOS_IDX, NUM_CHARS, DUAL_BIAS_THRESHOLD,
    ACTIVATION_SCALE, create_training_examples_with_pos,
    generate_response, char_to_idx, idx_to_char, filter_legacy_state,
)
from encoding import TrigramEncoder, ContextEncoder, parse_pair

CHECKPOINT = os.path.join(os.path.dirname(__file__), 'neochat_model.pt')
TRAINING_DATA = os.path.join(os.path.dirname(__file__), 'training_data.txt')
LABELED_DATA = os.path.join(os.path.dirname(__file__), 'labeled_data.txt')

OUTPUT_CHARSET = set(CHARSET[:-1])  # Exclude EOS


def get_device():
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        return torch.device('xpu')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def load_model(model_path, device):
    cp = torch.load(model_path, weights_only=False, map_location='cpu')
    model = NeochatModel()
    model.load_state_dict(filter_legacy_state(cp['model_state']))
    model.to(device)
    model.eval()
    epochs = cp.get('total_epochs', 0)
    int_acc = cp.get('best_int_acc', 0)
    return model, epochs, int_acc


def test_intacc(model, device, n_samples=1000):
    """Test integer vs float inference agreement."""
    print(f"\n[1] IntAcc — sampling {n_samples} training examples...")

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

    all_examples = []
    for q, r in pairs:
        exs = create_training_examples_with_pos(q, r, qe, ce)
        all_examples.extend(exs)

    # Limit to avoid OOM
    if len(all_examples) > 50000:
        random.shuffle(all_examples)
        all_examples = all_examples[:50000]

    X = torch.tensor(np.stack([ex[0] for ex in all_examples]),
                     dtype=torch.float32).to(device)
    y = torch.tensor(np.array([ex[1] for ex in all_examples]),
                     dtype=torch.long).to(device)
    positions = torch.tensor(np.array([ex[2] for ex in all_examples]),
                             dtype=torch.long).to(device)

    with torch.no_grad():
        float_out = model(X, positions=positions)
        int_out = model(X, positions=positions, use_int=True)

        float_preds = float_out.argmax(dim=1)
        int_preds = int_out.argmax(dim=1)

        float_acc = (float_preds == y).float().mean().item()
        int_acc = (int_preds == y).float().mean().item()
        agreement = (float_preds == int_preds).float().mean().item()

    print(f"    Float accuracy: {float_acc:.1%}")
    print(f"    Int accuracy:   {int_acc:.1%}")
    print(f"    Float/Int agreement: {agreement:.1%}")

    # Trained models sit around ~55-60% IntAcc; 40% catches a real regression
    # while leaving headroom for sampling noise (a near-random model is ~1/43=2%).
    passed = int_acc > 0.40
    status = "PASS" if passed else "FAIL"
    print(f"    → {status} (IntAcc={int_acc:.1%}, threshold=40%)")
    return passed


def test_coherence(model, device, n_queries=100):
    """Test no 3+ repeated characters in responses."""
    print(f"\n[2] Coherence — generating {n_queries} responses...")

    test_queries = [
        "who are you", "what is your name", "how are you", "hello",
        "what is the meaning of life", "who made you", "tell me a joke",
        "what time is it", "where are you from", "what can you do",
        "are you smart", "do you like me", "goodbye", "thanks",
        "what is love", "who is the president", "how old are you",
        "what is pi", "are you real", "help me",
    ]
    # Pad with more queries
    with open(TRAINING_DATA) as f:
        lines = f.readlines()
    random.shuffle(lines)
    for line in lines:
        p = parse_pair(line)
        if p:
            test_queries.append(p[0])
        if len(test_queries) >= n_queries:
            break

    qe = TrigramEncoder()
    ce = ContextEncoder(num_buckets=128, context_len=8)
    failures = []

    for q in test_queries[:n_queries]:
        resp = generate_response(model, q, qe, ce, max_len=50, device=device)
        # Check for 3+ repeated chars
        for i in range(len(resp) - 2):
            if resp[i] == resp[i+1] == resp[i+2]:
                failures.append((q, resp, resp[i]))
                break

    fail_count = len(failures)
    passed = fail_count <= 5  # Allow a few
    status = "PASS" if passed else "FAIL"
    print(f"    Failures: {fail_count}/{n_queries}")
    for q, r, c in failures[:5]:
        print(f"      '{q}' → '{r}' (repeated '{c}')")
    print(f"    → {status}")
    return passed


def test_charset(model, device, n_queries=100):
    """Test all generated characters are in output charset."""
    print(f"\n[3] Charset compliance — {n_queries} responses...")

    with open(TRAINING_DATA) as f:
        lines = f.readlines()
    random.shuffle(lines)

    qe = TrigramEncoder()
    ce = ContextEncoder(num_buckets=128, context_len=8)
    failures = []

    count = 0
    for line in lines:
        p = parse_pair(line)
        if not p:
            continue
        resp = generate_response(model, p[0], qe, ce, max_len=50, device=device)
        bad = [c for c in resp if c not in OUTPUT_CHARSET]
        if bad:
            failures.append((p[0], resp, bad))
        count += 1
        if count >= n_queries:
            break

    passed = len(failures) == 0
    status = "PASS" if passed else "FAIL"
    print(f"    Failures: {len(failures)}/{count}")
    for q, r, bad in failures[:5]:
        print(f"      '{q}' → '{r}' (bad: {bad})")
    print(f"    → {status}")
    return passed


def test_eos(model, device, n_queries=100):
    """Test responses terminate before max_len."""
    print(f"\n[4] EOS behavior — {n_queries} responses...")

    with open(TRAINING_DATA) as f:
        lines = f.readlines()
    random.shuffle(lines)

    qe = TrigramEncoder()
    ce = ContextEncoder(num_buckets=128, context_len=8)
    max_len = 50
    hit_max = 0
    count = 0

    for line in lines:
        p = parse_pair(line)
        if not p:
            continue
        resp = generate_response(model, p[0], qe, ce, max_len=max_len, device=device)
        if len(resp) >= max_len:
            hit_max += 1
        count += 1
        if count >= n_queries:
            break

    pct_terminated = (count - hit_max) / count if count else 0
    passed = pct_terminated >= 0.90
    status = "PASS" if passed else "FAIL"
    print(f"    Terminated before max_len: {count - hit_max}/{count} ({pct_terminated:.1%})")
    print(f"    → {status} (threshold=90%)")
    return passed


def test_personality(model, device, n_samples=20):
    """Test exact match on personality data."""
    print(f"\n[5] Personality recall — {n_samples} labeled queries...")

    pairs = []
    with open(LABELED_DATA) as f:
        for line in f:
            line = line.strip()
            if '|' not in line:
                continue
            q, r = line.split('|', 1)
            pairs.append((q.strip(), r.strip().upper()))

    # Deduplicate by response (test each unique response once)
    by_response = {}
    for q, r in pairs:
        if r not in by_response:
            by_response[r] = q

    test_pairs = list(by_response.items())
    random.shuffle(test_pairs)
    test_pairs = test_pairs[:n_samples]

    qe = TrigramEncoder()
    ce = ContextEncoder(num_buckets=128, context_len=8)
    matches = 0

    for expected, query in test_pairs:
        resp = generate_response(model, query, qe, ce, max_len=50, device=device)
        if resp == expected:
            matches += 1
        else:
            print(f"      '{query}' → '{resp}' (expected '{expected}')")

    pct = matches / len(test_pairs) if test_pairs else 0
    # Personality pairs are 10x oversampled, so a healthy model recalls most of
    # them (~85% observed); 60% catches a regression without flaking on the few
    # genuinely ambiguous prompts.
    passed = pct >= 0.60
    status = "PASS" if passed else "FAIL"
    print(f"    Exact matches: {matches}/{len(test_pairs)} ({pct:.1%})")
    print(f"    → {status} (threshold=60%)")
    return passed


def test_diversity(model, device, n_queries=50):
    """Test no empty responses."""
    print(f"\n[6] Diversity — {n_queries} responses...")

    queries = [
        "hello", "who are you", "what is 2+2", "tell me something",
        "how are you", "what can you do", "are you smart", "goodbye",
        "what is your name", "where are you from",
    ]
    with open(TRAINING_DATA) as f:
        lines = f.readlines()
    random.shuffle(lines)
    for line in lines:
        p = parse_pair(line)
        if p:
            queries.append(p[0])
        if len(queries) >= n_queries:
            break

    qe = TrigramEncoder()
    ce = ContextEncoder(num_buckets=128, context_len=8)
    empty = 0

    for q in queries[:n_queries]:
        resp = generate_response(model, q, qe, ce, max_len=50, device=device)
        if not resp.strip():
            empty += 1

    passed = empty <= 2
    status = "PASS" if passed else "FAIL"
    print(f"    Empty responses: {empty}/{n_queries}")
    print(f"    → {status}")
    return passed


def main():
    import argparse
    parser = argparse.ArgumentParser(description='NEOCHAT automated tests')
    parser.add_argument('--model', default=CHECKPOINT, help='Model checkpoint')
    parser.add_argument('--samples', type=int, default=100, help='Samples per test')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = get_device()

    print("=" * 60)
    print("NEOCHAT Automated Tests")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Model: {args.model}")

    if not os.path.exists(args.model):
        print(f"\nModel not found: {args.model}\n"
              f"Train one first:  python3 train.py -f training_data.txt --save-best")
        sys.exit(0)

    model, epochs, int_acc = load_model(args.model, device)
    print(f"Epochs: {epochs}, Best IntAcc: {int_acc:.1%}")

    results = {}
    results['IntAcc'] = test_intacc(model, device, n_samples=args.samples)
    results['Coherence'] = test_coherence(model, device, n_queries=args.samples)
    results['Charset'] = test_charset(model, device, n_queries=args.samples)
    results['EOS'] = test_eos(model, device, n_queries=args.samples)
    results['Personality'] = test_personality(model, device, n_samples=20)
    results['Diversity'] = test_diversity(model, device, n_queries=50)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print("=" * 60)
    all_pass = True
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}  {name}")
        if not passed:
            all_pass = False

    print(f"\n{'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")
    sys.exit(0 if all_pass else 1)


if __name__ == '__main__':
    main()
