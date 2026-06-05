#!/usr/bin/env python3
"""
Data preparation for NEOCHAT.

Downloads google-research-datasets/nq_open from HuggingFace, filters and
normalizes answers to fit our output charset, combines with labeled
personality data, and outputs a shuffled training file.

Usage:
    python3 prepare_data.py
    python3 prepare_data.py --personality-mult 10 --max-response-len 50
    python3 prepare_data.py --output custom_training_data.txt
"""

import os
import sys
import random
import argparse
import unicodedata
from collections import Counter

# Output charset: space + digits + letters + punctuation + EOS
OUTPUT_CHARSET = set(" 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ.?!,-")
MAX_RESPONSE_LEN = 50

LABELED_DATA = os.path.join(os.path.dirname(__file__), 'labeled_data.txt')
DEFAULT_OUTPUT = os.path.join(os.path.dirname(__file__), 'training_data.txt')


def normalize_answer(text):
    """Normalize a raw answer string to fit our output charset."""
    # Unicode normalization (handles non-breaking spaces, accented chars, etc.)
    text = unicodedata.normalize('NFKD', text)
    # Replace common unicode variants before ASCII conversion
    text = text.replace('\u2013', '-')  # en-dash
    text = text.replace('\u2014', '-')  # em-dash
    text = text.replace('\u2018', "'")  # left single quote
    text = text.replace('\u2019', "'")  # right single quote
    text = text.replace('\u201c', '"')  # left double quote
    text = text.replace('\u201d', '"')  # right double quote
    # Drop to ASCII
    text = text.encode('ascii', 'ignore').decode('ascii')
    text = text.upper().strip()
    # Strip characters not in our charset
    text = ''.join(c for c in text if c in OUTPUT_CHARSET)
    # Collapse multiple spaces
    while '  ' in text:
        text = text.replace('  ', ' ')
    return text.strip()


def normalize_query(text):
    """Normalize a query string (input charset — no digits needed)."""
    text = unicodedata.normalize('NFKD', text)
    text = text.encode('ascii', 'ignore').decode('ascii')
    text = text.upper().strip()
    # Truncate long queries
    if len(text) > 60:
        text = text[:60].rsplit(' ', 1)[0] if ' ' in text[40:60] else text[:60]
    return text


def load_nq_open(max_response_len, min_response_len=3):
    """Download and filter nq_open dataset."""
    from datasets import load_dataset

    print("Loading nq_open from HuggingFace...")
    ds = load_dataset('google-research-datasets/nq_open', split='train')
    total = len(ds)
    print(f"  Raw examples: {total:,}")
    print(f"  Min response length: {min_response_len}")

    pairs = []
    skipped_charset = 0
    skipped_length = 0
    skipped_empty = 0
    multi_answer = 0

    for row in ds:
        question = row['question']
        answers = row['answer']

        if len(answers) > 1:
            multi_answer += 1

        # Normalize all answers, filter valid ones
        valid = []
        for a in answers:
            norm = normalize_answer(a)
            if not norm:
                continue
            if len(norm) < min_response_len:
                continue
            if len(norm) > max_response_len:
                continue
            if not all(c in OUTPUT_CHARSET for c in norm):
                continue
            valid.append(norm)

        if not valid:
            # Check why it failed for stats
            for a in answers:
                norm = normalize_answer(a)
                if not norm:
                    skipped_empty += 1
                elif len(norm) > max_response_len:
                    skipped_length += 1
                else:
                    skipped_charset += 1
            continue

        # Pick shortest valid answer
        best = min(valid, key=len)
        query = normalize_query(question)
        if query and best:
            pairs.append((query, best))

    print(f"  Valid pairs: {len(pairs):,}")
    print(f"  Multi-answer questions: {multi_answer:,}")
    print(f"  Skipped (charset): {skipped_charset:,}")
    print(f"  Skipped (too long): {skipped_length:,}")
    print(f"  Skipped (empty after norm): {skipped_empty:,}")

    return pairs


def load_personality(path):
    """Load pipe-separated personality data."""
    pairs = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if '|' not in line:
                continue
            query, response = line.split('|', 1)
            query = query.strip().upper()
            response = response.strip().upper()
            if query and response:
                pairs.append((query, response))
    print(f"  Personality pairs: {len(pairs):,}")
    print(f"  Unique responses: {len(set(r for _, r in pairs)):,}")
    return pairs


def main():
    parser = argparse.ArgumentParser(description='Prepare NEOCHAT training data')
    parser.add_argument('-o', '--output', default=DEFAULT_OUTPUT, help='Output file')
    parser.add_argument('--personality-mult', type=int, default=10,
                        help='Oversampling multiplier for personality data')
    parser.add_argument('--max-response-len', type=int, default=MAX_RESPONSE_LEN,
                        help='Max response length in characters')
    parser.add_argument('--min-response-len', type=int, default=1,
                        help='Min response length for QA answers')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    random.seed(args.seed)

    print("=" * 60)
    print("NEOCHAT Data Preparation")
    print("=" * 60)
    print(f"Output charset ({len(OUTPUT_CHARSET)} chars): {sorted(OUTPUT_CHARSET)}")
    print(f"Max response length: {args.max_response_len}")
    print()

    # Load QA datasets
    print("--- nq_open ---")
    nq_pairs = load_nq_open(args.max_response_len, args.min_response_len)

    # Load personality
    print("\n--- Personality ---")
    personality_pairs = load_personality(LABELED_DATA)

    # Oversample personality. This is DELIBERATE: the personality pairs are the
    # chatbot's "character" and we want the tiny model to recall them strongly, so
    # they are intentionally over-represented (with the default 10x they become a
    # large share of the corpus). The model is therefore part-recall, part-QA by
    # design — turn --personality-mult down to weight general QA more heavily.
    personality_expanded = personality_pairs * args.personality_mult
    print(f"  After {args.personality_mult}x oversampling: {len(personality_expanded):,}")

    # Combine and shuffle
    all_pairs = nq_pairs + personality_expanded
    random.shuffle(all_pairs)

    print(f"\n--- Combined ---")
    print(f"  nq_open: {len(nq_pairs):,}")
    print(f"  Personality (oversampled): {len(personality_expanded):,}")
    print(f"  Total: {len(all_pairs):,}")

    # Validate all responses fit charset
    bad = 0
    for q, r in all_pairs:
        for c in r:
            if c not in OUTPUT_CHARSET:
                bad += 1
                break
    if bad:
        print(f"  WARNING: {bad} pairs have invalid charset characters!")

    # Response length distribution
    lengths = [len(r) for _, r in all_pairs]
    print(f"  Response lengths: min={min(lengths)}, max={max(lengths)}, "
          f"median={sorted(lengths)[len(lengths)//2]}, mean={sum(lengths)/len(lengths):.1f}")

    # Write output
    with open(args.output, 'w', encoding='utf-8') as f:
        for query, response in all_pairs:
            f.write(f"{query}|{response}\n")

    print(f"\nWrote {len(all_pairs):,} pairs to {args.output}")

    # Sample output
    print("\n--- Sample entries ---")
    for i in range(min(10, len(all_pairs))):
        q, r = all_pairs[i]
        source = "personality" if (q, r) in set(personality_pairs) else "nq_open"
        print(f"  [{source}] {q} | {r}")


if __name__ == '__main__':
    main()
