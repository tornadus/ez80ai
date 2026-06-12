#!/usr/bin/env python3
"""
Interactive NEOCHAT chat — GPU-accelerated inference for fast testing.

Usage:
    python3 chat.py                      # Interactive chat
    python3 chat.py -q "whats your name" # Single query
    python3 chat.py --cpu                # Force CPU
    python3 chat.py --float              # Use float model (not int simulation)
"""

import os
import argparse
import torch

from train import (
    NeochatModel, ACTIVATION_SCALE, DUAL_BIAS_THRESHOLD,
    CHARSET, EOS_IDX, IDX_TO_CHAR, NUM_CHARS,
    generate_response, filter_legacy_state,
)
from encoding import TrigramEncoder, ContextEncoder
from loadmodel import load_spec_from_model

CHECKPOINT = os.path.join(os.path.dirname(__file__), 'neochat_model.pt')


def get_device(force_cpu=False):
    if force_cpu:
        return torch.device('cpu')
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        return torch.device('xpu')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def main():
    parser = argparse.ArgumentParser(description='NEOCHAT interactive chat')
    parser.add_argument('-q', '--query', type=str, help='Single query (non-interactive)')
    parser.add_argument('--cpu', action='store_true', help='Force CPU')
    parser.add_argument('--float', action='store_true', help='Use float model instead of int simulation')
    parser.add_argument('--model', default=CHECKPOINT, help='Model checkpoint path')
    parser.add_argument('--max-len', type=int, default=50, help='Max response length')
    args = parser.parse_args()

    device = get_device(args.cpu)
    use_int = not args.float

    # Load model + the spec baked into the checkpoint (arch and encoding are
    # DOFs; hardcoded defaults here would skew vs what the model was trained on)
    cp = torch.load(args.model, weights_only=False, map_location='cpu')
    spec = load_spec_from_model(args.model)
    model = NeochatModel(input_size=spec['input_size'],
                         hidden_sizes=spec['hidden_sizes'],
                         num_chars=spec['num_classes'], spec=spec)
    model.load_state_dict(filter_legacy_state(cp['model_state']))
    model.to(device)
    model.eval()

    signed = spec.get('signed_hash', False)
    qe = TrigramEncoder(num_buckets=spec['query_buckets'],
                        ngram_orders=spec['query_ngram_orders'],
                        hash_mult=spec['query_hash']['mult'],
                        hash_mask=spec['query_hash']['mask'],
                        pos_offset_mult=spec['query_hash']['pos_offset_mult'],
                        signed_hash=signed)
    ce = ContextEncoder(num_buckets=spec['context_buckets'],
                        context_len=spec['context_len'],
                        ngram_orders=spec['context_ngram_orders'],
                        hash_mult=spec['context_hash']['mult'],
                        hash_mask=spec['context_hash']['mask'],
                        pos_offset_mult=spec['context_hash']['pos_offset_mult'],
                        signed_hash=signed)

    int_acc = cp.get('best_int_acc', 0)
    epochs = cp.get('total_epochs', 0)

    if args.query:
        response = generate_response(model, args.query, qe, ce,
                                     max_len=args.max_len, use_int=use_int,
                                     device=device)
        print(response)
        return

    # Interactive mode
    mode = "int" if use_int else "float"
    print(f"NEOCHAT Chat ({device}, {mode} mode, {epochs} epochs, IntAcc {int_acc:.1%})")
    print(f"Charset: {repr(CHARSET[:-1])}")
    print(f"Type '!' to quit\n")

    while True:
        try:
            query = input("> ").strip()
            if not query:
                continue
            if query == '!':
                break
            response = generate_response(model, query, qe, ce,
                                         max_len=args.max_len, use_int=use_int,
                                         device=device)
            print(response)
        except (EOFError, KeyboardInterrupt):
            break

    print("\nBye!")


if __name__ == '__main__':
    main()
