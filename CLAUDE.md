# CLAUDE.md

## Project Overview

eZ80-AI: a micro language model that runs on the Ti-84 Plus CE graphing calculator. Trains character-level neural networks in Python with 2-bit quantization-aware training (QAT), then compiles them into native eZ80 ADL-mode binaries. All inference is pure 24-bit signed integer math -- no floats on hardware.

## Build Commands

Build Ti-84 Plus CE .8xp binary:
```bash
python3 buildchat84.py --model model.npz
```

Train a model:
```bash
python3 train.py -f labeled_data.txt --epochs 300 --save-best --chat
```

Export PyTorch model to .npz:
```bash
python3 exportmodel.py
```

Prepare training data (downloads nq_open from HuggingFace):
```bash
python3 prepare_data.py
```

Interactive chat (GPU-accelerated testing):
```bash
python3 chat.py
python3 chat.py -q "hello"
```

Run automated tests:
```bash
python3 test_model.py --samples 100
```

No formal test suite exists. Validation is done via training metrics (IntAcc) and interactive chat.

## Architecture

### Pipeline

Training data -> `train.py` -> model (.pt) -> `exportmodel.py` -> model (.npz) -> `buildchat84.py` -> .8xp + .8xv files (in bin/)

### Key modules

- **`train.py`** -- Training loop. NeochatModel with dual output biases, 24-bit integer simulation, progressive quantization. Architecture: 256->512->512->256->43.
- **`encoding.py`** -- Text encoding: TrigramEncoder (query->128 hash buckets), ContextEncoder (recent chars->128 hash buckets), training example generation, data loading.
- **`libqat.py`** -- Quantization-aware training primitives. OverflowAwareLinear with 2-bit weights {-2,-1,0,+1}, straight-through estimator, 24-bit overflow regularization.
- **`libez80.py`** -- eZ80 ADL-mode machine code builder. Emits raw instructions with label/fixup system for 24-bit address resolution. Includes .LIS/.SIS prefixed instructions for 16-bit math within 24-bit addressing mode.
- **`loadmodel.py`** -- Loads models from .pt (PyTorch) or .npz (NumPy) formats.
- **`buildchat84.py`** -- Converts model to Ti-84 CE binary. Emits eZ80 code for trigram tokenization, neural net inference (multiply-accumulate loops), argmax, D_J Context Attention, and TI-OS I/O. Outputs .8xp program + .8xv weight AppVars.
- **`exportmodel.py`** -- Exports PyTorch checkpoint to .npz with 2-bit quantized weights.
- **`prepare_data.py`** -- Downloads nq_open dataset, combines with personality data, outputs shuffled training file.

### Neural network design

- **Input encoding**: 256 dimensions -- first 128 from trigram hashing of input text (fuzzy, order-invariant), second 128 from context encoding of recently generated characters.
- **Hidden layers**: 512->512->256, ReLU activation, division-by-4 scaling between layers (arithmetic right-shift on eZ80).
- **Output**: 43 neurons (space + digits + letters + punctuation + EOS), dual bias sets (first 3 chars vs rest). Argmax selects next character.
- **Weight packing**: 4 weights per byte (2-bit, LSB first).
- **Integer math**: All accumulation uses 24-bit signed integers (eZ80 native register width).

### eZ80 ADL mode caveats

See libez80.py header for 5 documented hardware caveats affecting .LIS prefixed instructions, IY register usage, and 8-bit register loads.

### Dependencies

- **Core**: Python 3.8+, NumPy
- **Training**: PyTorch (with XPU support for Intel GPUs)
- **Data generation**: HuggingFace `datasets` library
