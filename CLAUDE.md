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

Faithfulness gates (run after any codegen/spec change): `python3 test_intkernel.py` and `python3 test_faithfulness.py`; `ez80research/evaluate.py --samples 8000` runs the full gate set (CONTRACT/EXPORT/BUILD/FAITH/SIZE) and measures IntAcc independently.

## Architecture

### Pipeline

Training data -> `train.py` -> model (.pt) -> `exportmodel.py` -> model (.npz) -> `buildchat84.py` -> .8xp + .8xv files (in bin/)

### Key modules

- **`train.py`** -- Training loop. NeochatModel with dual output biases, 24-bit integer simulation, progressive quantization. Architecture comes from `modelspec.DEFAULT_SPEC` (currently 1024 -> 1600 -> 1408 -> 896 -> 43).
- **`encoding.py`** -- Text encoding: TrigramEncoder (query) and ContextEncoder (recent chars) hash into spec-driven bucket counts (currently 512 + 512).
- **`libqat.py`** -- Quantization-aware training primitives. OverflowAwareLinear with 2-bit weights {-2,-1,0,+1}, straight-through estimator, 24-bit overflow regularization.
- **`libez80.py`** -- eZ80 ADL-mode machine code builder. Emits raw instructions with label/fixup system for 24-bit address resolution. Includes .LIS/.SIS prefixed instructions for 16-bit math within 24-bit addressing mode.
- **`loadmodel.py`** -- Loads models from .pt (PyTorch) or .npz (NumPy) formats.
- **`buildchat84.py`** -- Converts model to Ti-84 CE binary. Emits eZ80 code for trigram tokenization, neural net inference (multiply-accumulate loops), argmax, and TI-OS I/O. On-device generation is a faithful mirror of train.py's integer path (`_forward_int`/`generate_response`): pure argmax + dual bias, stop at EOS or 50 chars, no device-only heuristics. Outputs .8xp program + .8xv weight AppVars. Weight AppVars are flash-resident: emitted with the archived flag, read in place from archive at runtime (loader computes ChkFindSym ptr + 9 + 1 + name_len + 2 and verifies an 8-byte per-AppVar magic header; a RAM-resident var is archived one-way at startup). Only the program image counts against RAM; weights gate against `sizes.FLASH_BUDGET_BYTES`.
- **`exportmodel.py`** -- Exports PyTorch checkpoint to .npz with 2-bit quantized weights.
- **`prepare_data.py`** -- Downloads nq_open dataset, combines with personality data, outputs shuffled training file.

### Neural network design

- **Input encoding**: spec-driven (currently 1024 dims) -- first half from trigram hashing of input text (fuzzy, order-invariant), second half from context encoding of recently generated characters.
- **Hidden layers**: spec-driven (currently 1600->1408->896), ReLU activation, per-layer arithmetic right-shift between layers.
- **Output**: 43 neurons (space + digits + letters + punctuation + EOS), dual bias sets (first 3 chars vs rest). Argmax selects next character. Output shift is larger (currently 4) so logits fit the device's int16 stores.
- **Weight packing**: 4 weights per byte (2-bit). On device, 2-bit layers use sparse-input column-major packing (4 consecutive neurons per byte) with a RAM-resident 24-bit accumulator array and jump-table byte dispatch -- ~8.7x faster than the old row-major loop and bit-exact (see CAVEAT_AUDIT.md + commits 7f27af8/5ecca03).
- **Integer math**: All accumulation uses 24-bit signed integers (eZ80 native register width); inter-layer activations and logits are stored int16.

### eZ80 ADL mode caveats

See libez80.py header for the documented hardware caveats (suffix-prefixed instructions, IY register usage, 8-bit register loads) and **CAVEAT_AUDIT.md** for the 2026-06 audit of each against the CEmu CPU core. Two of five were misdiagnosed: the real failure of suffixed memory/stack ops is `{MBASE, addr16}` address translation (they execute in Z80 mode; ALL such forms are broken on TI-OS and are no longer emittable), and `.LIS SBC`'s S flag is mode-width-correct -- the observed failures came from S-only signed compares being overflow-blind. The workarounds all remain valid. Loop counters next to pointers in RAM still use full 24-bit loads/stores: 8-bit register loads do not clear the pair's upper byte (confirmed), and a 16-bit store into a 3-byte slot would corrupt the neighbor. (Per CEmu, suffixed 16-bit *pair* writes like `.SIS LD HL,nn` actually DO zero bits 16-23 -- the old claim here was backwards -- but Zilog documents the upper byte as undefined in Z80 mode, so the codegen never relies on it.)

### Source of truth: the device must mirror the Python sim

The Python integer path (`train.py._forward_int` and `generate_response`) is the source of truth; `buildchat84.py` must reproduce it exactly. Numerics ARE gated: `faithgate.py` executes the emitted bytes in `ez80interp.py` (CEmu-aligned) and compares against `intkernel` bit-for-bit, and `test_faithfulness.py` checks device-contract generation against the sim end-to-end (including int16 activation/logit range pins). Run both after any codegen or spec change. Do NOT add on-device-only behavior (context attention, logit/EOS heuristics, repeat fallbacks, etc.); generation is a plain argmax + dual bias that stops at EOS or 50 chars, identical to `generate_response`. A released build once shipped device-only mechanisms + a 16-bit-counter quirk and produced on-calc gibberish; both were fixed by making the calc faithful to the sim.

### Dependencies

- **Core**: Python 3.8+, NumPy
- **Training**: PyTorch (with XPU support for Intel GPUs)
- **Data generation**: HuggingFace `datasets` library
