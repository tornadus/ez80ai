#!/usr/bin/env python3
"""
Interactive tool to assign responses to each intent category.

Reads Small_talk_Intent.en.csv, groups by intent, and for each category
shows the example inputs and prompts for a response. Outputs pipe-separated
training data compatible with encoding.py / train.py (the "QUERY|RESPONSE"
format consumed by prepare_data.load_personality and encoding.parse_pair).

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

    # Buffer labels in memory (intent -> response, in labeling order) and write
    # them all once on exit. The old design appended+flushed per category, so '!b'
    # could not actually undo an already-written line; buffering lets '!b' clear
    # the previous label cleanly. Trade-off: a hard kill (not Ctrl-C, which is
    # caught) loses this session's unsaved labels.
    responses = {}
    i = 0
    try:
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
                if i > 0:
                    i -= 1
                    responses.pop(remaining[i][0], None)  # truly undo previous label
                    print(f"  (going back — previous label cleared)\n")
                else:
                    print("  (nothing to go back to)\n")
                continue

            if not response:
                responses.pop(intent, None)  # clear if revisiting a category
                print("  (skipped)\n")
                i += 1
                continue

            responses[intent] = response
            print(f"  \033[32m✓ Recorded response for {len(examples)} inputs\033[0m\n")
            i += 1
    finally:
        with open(args.output, 'a', encoding='utf-8') as out:
            written = 0
            for intent, response in responses.items():
                for ex in intents[intent]:
                    out.write(f"{ex}|{response}\n")
                    written += 1
        print(f"\nDone! Wrote {written} training pairs to {args.output}")
        if i < len(remaining):
            print(f"Run with --resume to continue where you left off.")


if __name__ == '__main__':
    main()
