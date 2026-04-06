#!/usr/bin/env python3
"""
Export NEOCHAT model checkpoint to NumPy .npz format.

Handles dual bias export (fc4_bias + fc4_bias_start).

Usage:
    python3 exportmodel.py
    python3 exportmodel.py --model neochat_model.pt --output model.npz
"""

import argparse
import json
import os
import numpy as np
import torch

from train import NeochatModel, CHARSET, NUM_CHARS, HIDDEN_SIZES, INPUT_SIZE


def export_model(model_path, output_path):
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, weights_only=False, map_location='cpu')
    arch = checkpoint['architecture']
    charset = checkpoint.get('charset', CHARSET)
    num_chars = arch['num_classes']

    print(f"Architecture: input={arch['input_size']}, hidden={arch['hidden_sizes']}, output={num_chars}")
    print(f"Charset ({num_chars} chars): {repr(charset[:-1])} + EOS")

    model = NeochatModel(
        input_size=arch['input_size'],
        hidden_sizes=arch['hidden_sizes'],
        num_chars=num_chars,
    )
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    params = model.get_quantized_params()

    export_data = {}
    for key, value in params.items():
        export_data[key] = value

    export_data['_architecture'] = np.array(json.dumps(arch).encode('utf-8'))
    export_data['_charset'] = np.array(charset.encode('utf-8'))
    export_data['_dual_bias_threshold'] = np.array(
        checkpoint.get('dual_bias_threshold', 3))

    np.savez(output_path, **export_data)
    print(f"Exported to {output_path}")

    layer_names = sorted(set(k.replace('_weight', '').replace('_bias', '').replace('_start', '')
                             for k in params.keys()))
    for name in layer_names:
        w_key = f'{name}_weight'
        b_key = f'{name}_bias'
        if w_key in params:
            w = params[w_key]
            b = params[b_key]
            extra = ""
            if f'{name}_bias_start' in params:
                extra = f" + bias_start {params[f'{name}_bias_start'].shape}"
            print(f"  {name}: weight {w.shape}, bias {b.shape}{extra}")


def main():
    default_model = os.path.join(os.path.dirname(__file__), 'neochat_model.pt')
    default_output = os.path.join(os.path.dirname(__file__), 'model.npz')

    parser = argparse.ArgumentParser(description='Export NEOCHAT model to NumPy format')
    parser.add_argument('--model', '-m', default=default_model, help='Input .pt checkpoint')
    parser.add_argument('--output', '-o', default=default_output, help='Output .npz file')
    args = parser.parse_args()

    export_model(args.model, args.output)


if __name__ == '__main__':
    main()
