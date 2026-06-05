#!/usr/bin/env python3
"""
Training script for NEOCHAT — 512->512->256 architecture with dual output biases.

  - Output charset: 43 chars (space, digits, letters, punctuation + EOS)
  - Dual bias sets: separate biases for first 3 chars vs rest of response
  - 24-bit integer simulation matching eZ80 hardware

Usage:
    python3 train.py -f training_data.txt --epochs 300 --save-best --chat

Resuming is automatic: if neochat_model.pt exists with a matching architecture it
is loaded and training continues (pass a high --quant-target so the QT ramp / LR
schedule keep advancing instead of restarting). Delete the checkpoint to start
fresh.
"""

import sys
import os
import time
import numpy as np
import torch
import torch.nn as nn

# Force unbuffered output so background runs are monitorable
sys.stdout.reconfigure(line_buffering=True)

from libqat import OverflowAwareLinear, quantize_weights_2bit
from encoding import (
    TrigramEncoder, ContextEncoder,
    create_training_examples, parse_pair, load_chunk,
)
import modelspec

# ============================================================
# Configuration
# ============================================================

# Output charset: space + digits + letters + punctuation + EOS
CHARSET = " 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ.?!,-\x00"
CHAR_TO_IDX = {c: i for i, c in enumerate(CHARSET)}
IDX_TO_CHAR = {i: c for i, c in enumerate(CHARSET)}
EOS_IDX = len(CHARSET) - 1
NUM_CHARS = len(CHARSET)

# Model spec — the single source of truth (modelspec.py). The architecture and
# integer-path constants below are SOURCED from it so they can no longer drift
# from what the build/eval/faithfulness stages read. Defaults reproduce the
# released baseline exactly; an experiment edits modelspec.DEFAULT_SPEC.
SPEC = modelspec.load_spec()
assert SPEC['num_classes'] == NUM_CHARS, (
    f"spec num_classes {SPEC['num_classes']} != charset size {NUM_CHARS}")

# Architecture (from SPEC)
HIDDEN_SIZES = SPEC['hidden_sizes']
INPUT_SIZE = SPEC['input_size']  # query_buckets + context_buckets

# 24-bit integer simulation constants (from SPEC)
ACTIVATION_SCALE = SPEC['activation_scale']

CHECKPOINT_FILE = os.path.join(os.path.dirname(__file__), 'neochat_model.pt')

# Dual bias: first N characters use bias_start, rest use bias_rest (from SPEC)
DUAL_BIAS_THRESHOLD = SPEC['dual_bias_threshold']


def filter_legacy_state(state):
    """Drop buffers from removed instrumentation (e.g. `*.max_accum_seen`, the old
    overflow tracker) so checkpoints saved by an earlier train.py still load
    strictly into the current architecture instead of erroring on unexpected keys."""
    return {k: v for k, v in state.items() if not k.endswith('max_accum_seen')}


def char_to_idx(c):
    return CHAR_TO_IDX.get(c, 0)  # Unknown → space


def idx_to_char(i):
    return IDX_TO_CHAR.get(i, ' ')


# ============================================================
# Training example wrapper — adds position info for dual bias
# ============================================================

def create_training_examples_with_pos(query, response, query_encoder, context_encoder):
    """Wrap encoding.create_training_examples to add position index.

    Returns list of (input_vec, target_idx, position) tuples.
    """
    examples = create_training_examples(query, response, query_encoder, context_encoder,
                                        char_to_idx=char_to_idx, eos_idx=EOS_IDX)
    return [(vec, target, pos) for pos, (vec, target) in enumerate(examples)]


# ============================================================
# Model with dual bias and 24-bit integer simulation
# ============================================================

class NeochatModel(nn.Module):
    """Autoregressive model with dual output biases and 24-bit integer simulation.

    The output layer has two bias vectors:
    - bias_start: used for first DUAL_BIAS_THRESHOLD characters of response
    - bias_rest: used for remaining characters
    """

    def __init__(self, input_size=INPUT_SIZE, hidden_sizes=HIDDEN_SIZES,
                 num_chars=NUM_CHARS, spec=None):
        super().__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.num_chars = num_chars
        # The integer path reads its constants (quantile, per-layer bit-width and
        # shift, rounding, activation scale, dual-bias threshold) from the spec, so
        # the sim stays config-driven and in lockstep with the build. Defaults to
        # the module spec; eval can pass a model's frozen baked spec instead.
        self.spec = spec if spec is not None else SPEC

        # Hidden layers
        self.layers = nn.ModuleList()
        prev_size = input_size
        for size in hidden_sizes:
            self.layers.append(OverflowAwareLinear(prev_size, size))
            prev_size = size

        # Output layer (weight only — bias handled separately)
        self.output_layer = OverflowAwareLinear(prev_size, num_chars)

        # Dual bias sets
        self.bias_start = nn.Parameter(torch.zeros(num_chars))
        self.bias_rest = nn.Parameter(torch.zeros(num_chars))

        self.relu = nn.ReLU()

    def forward(self, x, positions=None, use_int=False, quant_temp=1.0):
        if use_int:
            return self._forward_int(x, positions)

        # Hidden layers
        for layer in self.layers:
            x = layer(x, quant_temp=quant_temp)
            x = self.relu(x)

        # Output layer (without its own bias)
        w = self.output_layer.weight
        w_quant = quantize_weights_2bit(w, hard=True, temperature=quant_temp)
        logits = x @ w_quant.T  # No bias from the linear layer

        # Apply dual bias
        if positions is not None:
            # Vectorized: select bias per example
            mask_start = (positions < DUAL_BIAS_THRESHOLD).unsqueeze(1)  # [B, 1]
            bias = torch.where(mask_start, self.bias_start, self.bias_rest)  # [B, num_chars]
            logits = logits + bias
        else:
            # Default to rest bias (generation mode, position tracked externally)
            logits = logits + self.bias_rest

        return logits

    def forward_with_bias(self, x, use_start_bias=False, use_int=False, quant_temp=1.0):
        """Forward pass with explicit bias selection (for generation)."""
        if use_int:
            pos = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
            if not use_start_bias:
                pos[:] = DUAL_BIAS_THRESHOLD  # use rest bias
            return self._forward_int(x, pos)

        for layer in self.layers:
            x = layer(x, quant_temp=quant_temp)
            x = self.relu(x)

        w = self.output_layer.weight
        w_quant = quantize_weights_2bit(w, hard=True, temperature=quant_temp)
        logits = x @ w_quant.T

        if use_start_bias:
            logits = logits + self.bias_start
        else:
            logits = logits + self.bias_rest

        return logits

    def _forward_int(self, x, positions=None):
        """Simulate the eZ80 integer path. The torch MIRROR of intkernel.forward_int
        (held bit-exact by test_intkernel.py); all constants come from self.spec."""
        spec = self.spec
        q = spec['weight_quantile']
        bits = spec['weight_bits']
        shifts = spec['inter_layer_shift']
        rounding = spec['rounding']
        thr = spec['dual_bias_threshold']
        ascale = spec['activation_scale']
        half = 1 << (spec['accum_bits'] - 1)
        mod = 1 << spec['accum_bits']

        def quant(w, b):
            scale = torch.quantile(w.abs().flatten(), q).clamp(min=1e-6)
            lo, hi = -(1 << (b - 1)), (1 << (b - 1)) - 1
            return torch.clamp(torch.round(w / scale), lo, hi)

        def wrap(t):
            return ((t + half) % mod) - half

        def shift(t, s):
            if s == 0:
                return t
            if rounding == 'floor':
                return torch.floor(t / (1 << s))
            return torch.div(t, 1 << s, rounding_mode='trunc')

        x = (x * ascale).round()

        for i, layer in enumerate(self.layers):
            w_quant = quant(layer.weight, bits[i])
            b_quant = torch.round(layer.bias * ascale)
            x = x @ w_quant.T + b_quant
            x = shift(wrap(x), shifts[i])
            x = torch.relu(x)

        # Output layer (no built-in bias)
        w_quant = quant(self.output_layer.weight, bits[-1])
        logits = shift(wrap(x @ w_quant.T), shifts[-1])

        # Add dual bias (quantized)
        b_start_q = torch.round(self.bias_start * ascale)
        b_rest_q = torch.round(self.bias_rest * ascale)
        if positions is not None:
            mask_start = (positions < thr).unsqueeze(1)
            logits = logits + torch.where(mask_start, b_start_q, b_rest_q)
        else:
            logits = logits + b_rest_q

        return logits

    def compute_quantization_loss(self):
        loss = sum(layer.get_quantization_loss() for layer in self.layers)
        loss += self.output_layer.get_quantization_loss()
        return loss

    def get_quantized_params(self):
        """Extract 2-bit quantized weights and biases for export.

        Returns dict with fc1-fc3 (hidden) + fc4 (output) weights/biases,
        plus fc4_bias_start for dual bias.
        """
        params = {}
        q = self.spec['weight_quantile']
        bits = self.spec['weight_bits']
        ascale = self.spec['activation_scale']

        def quant_w(w, b):
            scale = torch.quantile(w.abs().flatten(), q).clamp(min=1e-6)
            lo, hi = -(1 << (b - 1)), (1 << (b - 1)) - 1
            return torch.clamp(torch.round(w / scale), lo, hi).cpu().numpy().astype(np.int8)

        # Hidden layers
        for i, layer in enumerate(self.layers):
            name = f'fc{i+1}'
            with torch.no_grad():
                params[f'{name}_weight'] = quant_w(layer.weight, bits[i])
                params[f'{name}_bias'] = torch.round(
                    layer.bias * ascale).cpu().numpy().astype(np.int16)

        # Output layer (dual bias: fc{N}_bias == rest, fc{N}_bias_start == start)
        out_idx = len(self.layers) + 1
        with torch.no_grad():
            params[f'fc{out_idx}_weight'] = quant_w(self.output_layer.weight, bits[-1])
            params[f'fc{out_idx}_bias'] = torch.round(
                self.bias_rest * ascale).cpu().numpy().astype(np.int16)
            params[f'fc{out_idx}_bias_start'] = torch.round(
                self.bias_start * ascale).cpu().numpy().astype(np.int16)

        return params


# ============================================================
# Generation (for --chat mode)
# ============================================================

def generate_response(model, query, query_encoder, context_encoder,
                      max_len=50, use_int=True, device=None):
    """Generate a response character by character with dual bias."""
    model.eval()
    query_vec = query_encoder.encode(query)
    output = ""

    with torch.no_grad():
        for pos in range(max_len):
            context_vec = context_encoder.encode(output)
            full_input = np.concatenate([query_vec, context_vec])
            x = torch.tensor(full_input, dtype=torch.float32).unsqueeze(0)
            if device:
                x = x.to(device)

            use_start = pos < DUAL_BIAS_THRESHOLD
            logits = model.forward_with_bias(x, use_start_bias=use_start, use_int=use_int)
            next_idx = logits.argmax(dim=1).item()

            if next_idx == EOS_IDX:
                break
            char = idx_to_char(next_idx)
            if char == '\x00':
                break
            output += char

    return output.strip()


# ============================================================
# Training loop
# ============================================================

def get_device():
    """Select best available device: XPU > CUDA > CPU."""
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        dev = torch.device('xpu')
        print(f"Using XPU: {torch.xpu.get_device_name(0)}")
        return dev
    if torch.cuda.is_available():
        dev = torch.device('cuda')
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        return dev
    print("Using CPU")
    return torch.device('cpu')


def train(epochs=300, lr=0.002, save_best=False, batch_size=8192, quant_target_epoch=300):
    """Train NEOCHAT model with mini-batch SGD."""
    print("=" * 60)
    print("NEOCHAT Training — Dual Bias, 24-bit eZ80 Mode")
    print("=" * 60)
    device = get_device()

    # Load all pairs
    all_pairs = load_chunk(sys.stdin, 0)
    total_pairs = len(all_pairs)

    if total_pairs == 0:
        print("No training data!")
        return None

    # Validate charset compliance
    valid_chars = set(CHARSET) - {'\x00'}
    valid_pairs = []
    skipped = 0
    for query, response in all_pairs:
        clean_resp = response.upper()
        if all(c in valid_chars for c in clean_resp):
            valid_pairs.append((query, clean_resp))
        else:
            skipped += 1
    all_pairs = valid_pairs
    if skipped:
        print(f"Skipped {skipped} pairs with invalid characters")

    print(f"Charset ({NUM_CHARS} chars): {repr(CHARSET[:-1])} + EOS")
    print(f"Loaded {len(all_pairs)} valid pairs")
    print(f"Architecture: {INPUT_SIZE} → {' → '.join(map(str, HIDDEN_SIZES))} → {NUM_CHARS}")
    print(f"Dual bias threshold: first {DUAL_BIAS_THRESHOLD} chars use bias_start")

    query_encoder = TrigramEncoder(num_buckets=SPEC['query_buckets'])
    context_encoder = ContextEncoder(num_buckets=SPEC['context_buckets'],
                                     context_len=SPEC['context_len'])

    # Generate all character-level examples upfront (CPU)
    print("Generating character examples...")
    all_examples = []
    for i, (query, response) in enumerate(all_pairs):
        examples = create_training_examples_with_pos(
            query, response, query_encoder, context_encoder)
        all_examples.extend(examples)
        if (i + 1) % 20000 == 0:
            print(f"  {i+1}/{len(all_pairs)} pairs → {len(all_examples)} examples so far")

    n_examples = len(all_examples)
    print(f"Generated {n_examples:,} character examples")

    # Build tensors and move entirely to GPU (1.6M x 256 x 4B = ~1.6GB, fits in VRAM)
    print("Moving data to GPU...")
    X_all = torch.tensor(np.stack([ex[0] for ex in all_examples]), dtype=torch.float32).to(device)
    y_all = torch.tensor(np.array([ex[1] for ex in all_examples]), dtype=torch.long).to(device)
    pos_all = torch.tensor(np.array([ex[2] for ex in all_examples]), dtype=torch.long).to(device)
    del all_examples  # Free CPU memory

    n_batches = (n_examples + batch_size - 1) // batch_size
    print(f"Mini-batch training: {n_batches} batches of {batch_size}")

    model = None
    total_epochs = 0
    best_int_acc = 0.0
    best_epoch = 0
    best_state = None

    # Try resume
    try:
        checkpoint = torch.load(CHECKPOINT_FILE, weights_only=False)
        arch = checkpoint.get('architecture', {})
        if arch.get('num_classes') == NUM_CHARS and arch.get('hidden_sizes') == HIDDEN_SIZES:
            model = NeochatModel()
            model.load_state_dict(filter_legacy_state(checkpoint['model_state']))
            total_epochs = checkpoint.get('total_epochs', 0)
            best_int_acc = checkpoint.get('best_int_acc', 0.0)
            best_epoch = checkpoint.get('best_epoch', 0)
            print(f"Resumed: {total_epochs} epochs, best IntAcc: {best_int_acc:.1%}")
        else:
            print("Architecture changed, starting fresh")
    except FileNotFoundError:
        print("No checkpoint, starting fresh")
    except Exception as e:
        print(f"Can't load checkpoint: {e}, starting fresh")

    if model is None:
        model = NeochatModel()
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {total_params:,}")

    model = model.to(device)
    # weight_decay is LOAD-BEARING FOR QUANTIZATION (do not remove): although the
    # 0.85-quantile quantizer is scale-relative, decay is not a uniform rescale --
    # it pulls larger weights down proportionally, keeping the weight distribution
    # compact so the fixed-scale rounding stays clean. Empirically (ez80research
    # loop, ~34 experiments): wd=0 -> 0.444 IntAcc (catastrophe), wd=1e-4 -> 0.604,
    # wd=2e-4 -> 0.575. Keep at 1e-4.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=SPEC['weight_decay'])
    # Cosine schedule spans the GLOBAL training horizon and is fast-forwarded by
    # the epochs already trained, so re-running on an existing checkpoint continues
    # the curve instead of restarting at full LR each invocation.
    horizon = max(quant_target_epoch, total_epochs + epochs)
    scheduler = torch.optim.lr_scheduler.PolynomialLR(
        optimizer, total_iters=horizon, power=2.0)  # polynomial decay (experiment) vs cosine
    # Compute budget (P6): in 'grad_steps' mode, stop after a fixed number of
    # optimizer steps regardless of wall-clock, so a slower-but-better arch gets
    # the SAME amount of training as a fast one (fair comparison). 'wall_s' keeps
    # the legacy behavior (run all epochs; the harness timeout is the cap).
    _budget = SPEC.get('compute_budget', {'mode': 'wall_s', 'limit': 0})
    max_steps = _budget['limit'] if _budget.get('mode') == 'grad_steps' else None
    global_step = 0
    budget_hit = False
    if total_epochs > 0:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # benign "step before optimizer.step" notices
            for _ in range(total_epochs):
                scheduler.step()
    criterion = nn.CrossEntropyLoss()

    interrupted = False
    t_start = time.time()
    for epoch in range(epochs):
        try:
            model.train()
            t_epoch = time.time()

            # QT ramp based on global epoch count so resume doesn't reset
            global_epoch = total_epochs + epoch
            _qt0 = SPEC['qt_start']
            quant_temp = _qt0 + (1 - _qt0) * min(
                1.0, global_epoch / (quant_target_epoch * SPEC['qt_ramp_factor']))

            # Shuffle indices each epoch (on GPU to avoid CPU-GPU sync)
            perm = torch.randperm(n_examples, device=device)
            epoch_loss = 0.0
            epoch_correct = 0

            for batch_idx in range(n_batches):
                start = batch_idx * batch_size
                end = min(start + batch_size, n_examples)
                idx = perm[start:end]

                X_batch = X_all[idx]
                y_batch = y_all[idx]
                pos_batch = pos_all[idx]

                optimizer.zero_grad()

                outputs = model(X_batch, positions=pos_batch, quant_temp=quant_temp)
                ce_loss = criterion(outputs, y_batch)
                quant_loss = model.compute_quantization_loss() * SPEC['quant_loss_weight']

                loss = ce_loss + quant_loss
                loss.backward()
                optimizer.step()
                global_step += 1

                epoch_loss += ce_loss.item() * (end - start)
                epoch_correct += (outputs.argmax(dim=1) == y_batch).sum().item()
                if max_steps and global_step >= max_steps:
                    budget_hit = True
                    break

            current_epoch = total_epochs + epoch + 1
            avg_loss = epoch_loss / n_examples
            avg_acc = epoch_correct / n_examples
            elapsed = time.time() - t_epoch

            # Evaluate IntAcc every 10 epochs (expensive), print CE every epoch
            if (epoch + 1) % 10 == 0:
                with torch.no_grad():
                    eval_idx = torch.randperm(n_examples, device=device)[:min(50000, n_examples)]
                    X_eval = X_all[eval_idx]
                    y_eval = y_all[eval_idx]
                    pos_eval = pos_all[eval_idx]

                    int_outputs = model(X_eval, positions=pos_eval, use_int=True)
                    int_preds = int_outputs.argmax(dim=1)
                    int_acc = (int_preds == y_eval).float().mean().item()

                    if int_acc > best_int_acc:
                        best_int_acc = int_acc
                        best_epoch = current_epoch
                        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                        marker = " *BEST*"
                    else:
                        marker = ""

                    print(f"  Epoch {current_epoch}: CE={avg_loss:.4f} "
                          f"Acc={avg_acc:.1%} IntAcc={int_acc:.1%} "
                          f"QT={quant_temp:.2f} [{elapsed:.1f}s]{marker}")
            else:
                print(f"  Epoch {current_epoch}: CE={avg_loss:.4f} "
                      f"Acc={avg_acc:.1%} QT={quant_temp:.2f} [{elapsed:.1f}s]")

            scheduler.step()

            if budget_hit:
                print(f"  [compute] grad-step budget reached "
                      f"({global_step}/{max_steps}) — stopping")
                break

        except KeyboardInterrupt:
            print("\nInterrupted!")
            interrupted = True
            break

    # One grep-able compute line for the harness to log (mode/steps/limit/hit).
    print(f"[compute] mode={_budget.get('mode')} steps={global_step} "
          f"limit={_budget.get('limit')} hit={1 if budget_hit else 0}")

    total_epochs += epoch + 1

    # save-best fallback: IntAcc is only evaluated every 10 epochs, so a short run
    # (or one interrupted before epoch 10) would have best_state=None and silently
    # save "latest" with best_int_acc=0. Do one final IntAcc eval so --save-best
    # always persists a ranked model.
    if save_best and best_state is None and n_examples > 0:
        model.eval()
        with torch.no_grad():
            eval_idx = torch.randperm(n_examples, device=device)[:min(50000, n_examples)]
            int_out = model(X_all[eval_idx], positions=pos_all[eval_idx], use_int=True)
            best_int_acc = (int_out.argmax(dim=1) == y_all[eval_idx]).float().mean().item()
        best_epoch = total_epochs
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        print(f"  (save-best fallback: final IntAcc={best_int_acc:.1%} @ epoch {best_epoch})")

    # Save checkpoint (CPU for portability)
    if save_best and best_state:
        save_state = {k: v.cpu() for k, v in best_state.items()}
        save_note = "best"
    else:
        save_state = {k: v.cpu() for k, v in model.state_dict().items()}
        save_note = "latest"

    torch.save({
        'model_state': save_state,
        'architecture': {
            'input_size': INPUT_SIZE,
            'hidden_sizes': HIDDEN_SIZES,
            'num_classes': NUM_CHARS,
        },
        # Frozen resolved spec: every downstream stage reads THIS, never the live
        # modelspec.py, so the model is always built/eval'd with the spec it was
        # trained with (the anti-skew invariant).
        'modelspec': modelspec.to_json(SPEC),
        'charset': CHARSET,
        'total_epochs': total_epochs,
        'best_int_acc': best_int_acc,
        'best_epoch': best_epoch,
        'dual_bias_threshold': DUAL_BIAS_THRESHOLD,
        'grad_steps_run': global_step,
        'compute_mode': _budget.get('mode'),
        'budget_limit': _budget.get('limit'),
        'budget_hit': budget_hit,
    }, CHECKPOINT_FILE)
    print(f"Saved {save_note} → {CHECKPOINT_FILE} "
          f"(epochs: {total_epochs}, best: {best_int_acc:.1%} @ {best_epoch})")

    print(f"\n{'='*60}")
    print(f"Done: {total_epochs} epochs, Best IntAcc: {best_int_acc:.1%} @ epoch {best_epoch}")
    print("=" * 60)

    return model


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train NEOCHAT (dual bias, 24-bit eZ80)')
    parser.add_argument('--epochs', '-e', type=int, default=300, help='Training epochs')
    parser.add_argument('--file', '-f', type=str, help='Training data file (default: stdin)')
    parser.add_argument('--batch-size', '-b', type=int, default=8192, help='Mini-batch size')
    parser.add_argument('--lr', type=float, default=0.002, help='Learning rate')
    parser.add_argument('--save-best', action='store_true', help='Save best model')
    parser.add_argument('--quant-target', type=int, default=300,
                        help='Global epoch at which QT reaches 1.0 (set high when resuming)')
    parser.add_argument('--chat', action='store_true', help='Interactive chat after training')
    args = parser.parse_args()

    if args.file:
        import io
        with open(args.file) as f:
            sys.stdin = io.StringIO(f.read())

    model = train(epochs=args.epochs, lr=args.lr, save_best=args.save_best,
                  batch_size=args.batch_size, quant_target_epoch=args.quant_target)

    if args.chat and model is not None:
        device = get_device()
        model = model.to(device)

        print("\n" + "=" * 60)
        print("NEOCHAT Chat (type '!' to exit)")
        print("=" * 60)

        query_encoder = TrigramEncoder()
        context_encoder = ContextEncoder()

        while True:
            try:
                query = input("> ").strip()
                if not query:
                    continue
                if query == '!':
                    break
                response = generate_response(model, query, query_encoder,
                                             context_encoder, max_len=50,
                                             device=device)
                print(response)
            except (EOFError, KeyboardInterrupt):
                break
        print("\nBye!")
