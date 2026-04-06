#!/usr/bin/env python3
"""
Interactive tool to assign responses to each intent category.

Reads Small_talk_Intent.en.csv, groups by intent, and for each category
shows the example inputs and prompts for a response. Outputs pipe-separated
training data compatible with feedme.py / train.py.

Usage:
    python3 label_intents.py
    python3 label_intents.py -o training_data.txt       # custom output file
    python3 label_intents.py --resume                    # skip already-labeled intents
"""

import csv
import os
import sys
import argparse
from collections import OrderedDict

CSV_FILE = os.path.join(os.path.dirname(__file__), 'Small_talk_Intent.en.csv')
DEFAULT_OUTPUT = os.path.join(os.path.dirname(__file__), 'labeled_data.txt')


def load_intents(csv_path):
    """Load CSV and group examples by intent category."""
    intents = OrderedDict()
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=';')
        header = next(reader)  # skip header
        for row in reader:
            if len(row) < 2:
                continue
            text, intent = row[0].strip(), row[1].strip()
            if intent == 'intent':  # skip malformed
                continue
            if intent not in intents:
                intents[intent] = []
            intents[intent].append(text)
    return intents


def load_existing(output_path):
    """Load already-labeled intents from output file."""
    labeled = {}
    if not os.path.exists(output_path):
        return labeled
    with open(output_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if '|' not in line:
                continue
            query, response = line.split('|', 1)
            labeled[query.strip()] = response.strip()
    return labeled


def main():
    parser = argparse.ArgumentParser(description='Label intent categories with responses')
    parser.add_argument('-o', '--output', default=DEFAULT_OUTPUT, help='Output file')
    parser.add_argument('--resume', action='store_true', help='Skip intents already in output')
    args = parser.parse_args()

    intents = load_intents(CSV_FILE)
    existing = load_existing(args.output) if args.resume else {}

    # Find which intents are already fully labeled
    already_done = set()
    if args.resume:
        for intent, examples in intents.items():
            if all(ex in existing for ex in examples):
                already_done.add(intent)

    remaining = [(k, v) for k, v in intents.items() if k not in already_done]
    total = len(intents)
    done = len(already_done)

    print(f"Loaded {sum(len(v) for v in intents.values())} examples across {total} categories")
    if done:
        print(f"Resuming: {done} already labeled, {len(remaining)} remaining")
    print(f"Output: {args.output}")
    print()
    print("For each category, type the response you want the model to give.")
    print("  (empty)  = skip this category")
    print("  !q       = save and quit")
    print("  !b       = go back to previous category")
    print()

    # Open output file in append mode
    out = open(args.output, 'a', encoding='utf-8')
    labeled_this_session = []
    i = 0

    while i < len(remaining):
        intent, examples = remaining[i]

        print(f"\033[1m[{done + i + 1}/{total}] {intent}\033[0m")
        print(f"  {len(examples)} example inputs:")
        for ex in examples:
            print(f"    \033[36m{ex}\033[0m")
        print()

        try:
            response = input("  Response: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSaving and exiting.")
            break

        if response == '!q':
            print("Saving and exiting.")
            break

        if response == '!b':
            if labeled_this_session:
                # Undo last — we can't un-write the file easily,
                # but we can let the user re-enter
                i -= 1
                print(f"  (going back to previous category)\n")
            else:
                print("  (nothing to go back to)\n")
            continue

        if not response:
            print("  (skipped)\n")
            i += 1
            continue

        # Write all examples with this response
        for ex in examples:
            line = f"{ex}|{response}"
            out.write(line + '\n')
        out.flush()

        labeled_this_session.append((intent, response))
        print(f"  \033[32m✓ Wrote {len(examples)} pairs\033[0m\n")
        i += 1

    out.close()

    total_written = sum(len(intents[intent]) for intent, _ in labeled_this_session)
    print(f"\nDone! Wrote {total_written} training pairs to {args.output}")
    if i < len(remaining):
        print(f"Run with --resume to continue where you left off.")


if __name__ == '__main__':
    main()
