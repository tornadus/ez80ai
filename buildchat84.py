#!/usr/bin/env python3
"""
Build NEOCHAT -- eZ80 native 24-bit accumulator neural net for Ti-84 Plus CE (.8xp)

Features:
  - 43-character charset (digits, punctuation)
  - Dual bias sets for output layer (fc4_bias / fc4_bias_start)
  - Faithful mirror of the Python integer path (train.py: _forward_int /
    generate_response): pure argmax + dual bias, stop at EOS or max_len=50.
    NO device-only heuristics -- on-device output must match the Python sim
    character-for-character. (The build's numerics are NOT auto-verified against
    the sim, so any divergence is silent; keep them in lockstep. A prior release
    shipped device-only attention/heuristics + a 16-bit-counter quirk and produced
    on-calc gibberish; those were removed and the loop counters made 24-bit.)

Usage:
    python3 buildchat84.py --model model.npz --output NEOCHAT.8xp
    On calculator: Asm(prgmNEOCHAT)

Keys: Alpha labels for letters, DEL=backspace, ENTER=submit, CLEAR=clear, MODE=quit
"""

import os
import sys
import numpy as np
import struct

from libez80 import eZ80Builder
from loadmodel import load_model_params, load_dual_bias_threshold, load_spec_from_model
import sizes

# Ti-84 Plus CE TI-OS routine addresses
TI_ClrScrn = 0x020814       # Clear home screen
TI_HomeUp = 0x020828         # Cursor to top-left
TI_PutC = 0x0207B8           # Print character in A
TI_NewLine = 0x0207F0        # Advance cursor to next line
TI_GetCSC = 0x02014C         # Non-blocking key scan -> scan code in A
TI_RunIndicOff = 0x020848    # Hide run indicator
TI_RunIndicOn = 0x020844     # Show run indicator (restore before exit)
TI_Mov9ToOP1 = 0x020320      # Copy 9 bytes to OP1 (variable name buffer)
TI_ChkFindSym = 0x02050C     # Find symbol by name in OP1 -> DE=data, A=flash page
TI_Arc_Unarc = 0x021448      # Toggle archive status of variable in OP1

# AppVar type ID
APPVAR_TYPE = 0x15            # TI AppVar variable type

# AppVar names for weight data (8 chars max, padded with 0)
APPVAR_NAMES = ['NEOA', 'NEOB', 'NEOC', 'NEOD']

# Ti-84 Plus CE memory
USERMEM = 0xD1A881           # Where programs load after preamble strip
TI_CURSOR_COL = 0xD00596    # Homescreen cursor column (uint8_t)
TI_CURSOR_ROW = 0xD00595    # Homescreen cursor row (uint8_t)
TI_DRAW_FGCOLOR = 0xD026AC  # Small font foreground color (16-bit BGR565)
TI_DRAW_BGCOLOR = 0xD026AA  # Small font background color (16-bit BGR565)

MAX_OUTPUT_LEN = 50          # Maximum characters to generate

# TI-84 CE keyboard scan codes (from CEdev getcsc.h / ti84pce.inc)
SK_ENTER = 0x09
SK_ADD = 0x0A
SK_SUB = 0x0B
SK_MUL = 0x0C
SK_DIV = 0x0D
SK_POWER = 0x0E
SK_CLEAR = 0x0F
SK_CHS = 0x11      # (-) negate key
SK_3 = 0x12
SK_6 = 0x13
SK_9 = 0x14
SK_RPAREN = 0x15
SK_TAN = 0x16
SK_VARS = 0x17
SK_DECPNT = 0x19
SK_2 = 0x1A
SK_5 = 0x1B
SK_8 = 0x1C
SK_LPAREN = 0x1D
SK_COS = 0x1E
SK_PRGM = 0x1F
SK_STAT = 0x20
SK_0 = 0x21
SK_1 = 0x22
SK_4 = 0x23
SK_7 = 0x24
SK_COMMA = 0x25
SK_SIN = 0x26
SK_APPS = 0x27
SK_GRAPHVAR = 0x28
SK_STO = 0x29
SK_LN = 0x2A
SK_LOG = 0x2B
SK_SQUARE = 0x2C
SK_RECIP = 0x2D
SK_MATH = 0x2E
SK_ALPHA = 0x2F
SK_GRAPH = 0x31
SK_TRACE = 0x32
SK_ZOOM = 0x33
SK_WINDOW = 0x34
SK_YEQU = 0x35
SK_2ND = 0x36
SK_MODE = 0x37
SK_DEL = 0x38

# Scan code -> ASCII mapping table (letters + space + a little punctuation).
#
# NOTE on the charset asymmetry (deliberate): the model's OUTPUT charset includes
# digits 0-9, comma and dash so it can answer things like "WHEN WAS ..." with a
# number — and it does in practice. Those keys are intentionally NOT mapped for
# INPUT here: a query is only ever hashed into trigram buckets, so what matters is
# letters/space, and keeping the input keypad simple avoids accidental mode keys.
# Relatedly, query text is hashed as-is (lower-cased) and is NOT filtered to the
# output charset — any character a query contains just contributes to a trigram
# bucket, which is fine and intended.
_ALPHA_MAP = "\0\0\0\0\0\0\0\0\0\0\"WRMH\0\0?\0VQLG\0\0:ZUPKFC\0 YTOJEB\0\0XSNIDA\0\0\0\0\0\0\0\0"
SCAN_TO_ASCII = {}
for i, ch in enumerate(_ALPHA_MAP):
    if ch != '\0' and ch.isalpha():
        SCAN_TO_ASCII[i] = ord(ch)
# Add space (0 key) and useful punctuation
SCAN_TO_ASCII[SK_0] = ord(' ')     # [0] -> Space
SCAN_TO_ASCII[SK_CHS] = ord('?')   # [(-)] -> ?
SCAN_TO_ASCII[SK_DECPNT] = ord('.') # [.] -> .
SCAN_TO_ASCII[SK_ADD] = ord('!')    # [+] -> !


def pack_weights(weights: np.ndarray, bits: int = 2) -> bytes:
    """Pack `bits`-bit signed weights, 8//bits codes per byte, LSB first.

    Each weight w in [-2^(bits-1), 2^(bits-1)-1] is stored as the unsigned code
    w + 2^(bits-1) (so 0 -> the mid code). The final partial byte is padded with
    the zero-code. bits=2 -> 4/byte (codes pad with 2 -> 0xAA all-zero sentinel);
    bits=4 -> 2/byte (pad with 8 -> 0x88 sentinel)."""
    off = 1 << (bits - 1)
    cpb = 8 // bits
    lo, hi = -off, off - 1
    mapped = (np.clip(weights.flatten(), lo, hi) + off).astype(np.uint8)
    pad = (-len(mapped)) % cpb
    if pad:
        mapped = np.concatenate([mapped, np.full(pad, off, dtype=np.uint8)])
    mapped = mapped.reshape(-1, cpb)
    shifts = np.arange(cpb, dtype=np.uint16) * bits
    return (mapped.astype(np.uint16) << shifts).sum(axis=1).astype(np.uint8).tobytes()


def zero_byte(bits: int) -> int:
    """The packed byte whose every code is the zero-weight code (the zero-skip
    sentinel). bits=2 -> 0xAA, bits=4 -> 0x88."""
    off = 1 << (bits - 1)
    cpb = 8 // bits
    return sum(off << (k * bits) for k in range(cpb))


# Back-compat alias (2-bit).
def pack_2bit_weights(weights: np.ndarray) -> bytes:
    return pack_weights(weights, 2)


def build_8xp(code: bytes, name: str) -> bytes:
    """Package machine code into a .8xp file for Ti-84 Plus CE"""
    name = name.upper()[:8]

    # Program body = preamble + machine code
    preamble = bytes([0xEF, 0x7B])  # Asm84CEPrgm token
    program_body = preamble + code
    body_size = len(program_body)

    # Variable data = size field (2 bytes) + program body
    var_data = struct.pack('<H', body_size) + program_body
    var_data_len = len(var_data)

    # Build variable entry (data section)
    data_section = bytearray()
    data_section.extend([0x0D, 0x00])               # Flag: 0x0D (with version/flag fields)
    data_section.extend(struct.pack('<H', var_data_len))  # Variable data length
    data_section.append(0x06)                         # Type: protected program
    data_section.extend(name.encode('ascii').ljust(8, b'\x00'))  # Name (8 bytes)
    data_section.append(0x00)                         # Version
    data_section.append(0x00)                         # Flag: in RAM
    data_section.extend(struct.pack('<H', var_data_len))  # Variable data length (repeat)
    data_section.extend(var_data)                     # Variable data

    # Checksum: sum of all data section bytes
    checksum = sum(data_section) & 0xFFFF

    # Build file header (55 bytes)
    header = bytearray()
    header.extend(b'**TI83F*')                        # Signature (8 bytes)
    header.extend([0x1A, 0x0A, 0x00])                 # Further signature (3 bytes)
    comment = b'Built by z80ai eZ80 builder'
    header.extend(comment.ljust(42, b'\x00'))          # Comment (42 bytes)
    header.extend(struct.pack('<H', len(data_section)))  # Data section length

    # Assemble final file
    result = bytes(header) + bytes(data_section) + struct.pack('<H', checksum)
    return result


def build_8xv(data: bytes, name: str) -> bytes:
    """Package raw data into a .8xv AppVar file for Ti-84 Plus CE."""
    name = name.upper()[:8]

    # Variable data = size field (2 bytes) + payload
    var_data = struct.pack('<H', len(data)) + data
    var_data_len = len(var_data)

    # Build variable entry (data section)
    data_section = bytearray()
    data_section.extend([0x0D, 0x00])
    data_section.extend(struct.pack('<H', var_data_len))
    data_section.append(APPVAR_TYPE)                      # Type: AppVar
    data_section.extend(name.encode('ascii').ljust(8, b'\x00'))
    data_section.append(0x00)                              # Version
    data_section.append(0x00)                              # Flag: in RAM
    data_section.extend(struct.pack('<H', var_data_len))
    data_section.extend(var_data)

    checksum = sum(data_section) & 0xFFFF

    header = bytearray()
    header.extend(b'**TI83F*')
    header.extend([0x1A, 0x0A, 0x00])
    comment = b'NEOCHAT weight data'
    header.extend(comment.ljust(42, b'\x00'))
    header.extend(struct.pack('<H', len(data_section)))

    return bytes(header) + bytes(data_section) + struct.pack('<H', checksum)


def build_scan_table(b):
    """Emit scan code -> ASCII lookup table as a 256-byte array"""
    table = [0] * 256
    for sc, ascii_val in SCAN_TO_ASCII.items():
        if sc < 256:
            table[sc] = ascii_val
    b.label('SCANTBL')
    b.db(*table)


def build_autoreg(model_path: str = 'model.npz', debug: bool = False):
    """Build the autoregressive inference for Ti-84 Plus CE.

    When debug=True, the production inference codegen is left 100% intact and
    REUSED; the only change is that after the AppVars are loaded the program is
    routed to a DBG_RUN instrumentation routine (instead of CHAT). DBG_RUN reads
    one query, runs ONE forward pass (genpos=0, empty/space context) through the
    exact same LAYER/RELU/ARGMAX subroutines, and dumps 24-bit checksums of each
    intermediate buffer plus the argmax to the home screen. This lets us
    binary-search where the on-device math diverges from the Python sim.
    """

    # Load model
    print(f"Loading model from {model_path}...")
    params, arch, charset = load_model_params(model_path)

    eos_idx = len(charset) - 1
    num_chars = len(charset)
    print(f"Charset ({num_chars} chars): {repr(charset[:-1])} + EOS")
    assert num_chars == 43, f"Expected 43-char charset, got {num_chars}"
    assert eos_idx == 42, f"Expected EOS at index 42, got {eos_idx}"

    # Discover layers (exclude fcN_bias_start which is a secondary bias, not a layer).
    # Sort NUMERICALLY by the fc index so fc10 follows fc9 (not alphabetical).
    layer_keys = [k for k in params.keys() if k.endswith('_weight')]
    layer_names = sorted((k.replace('_weight', '') for k in layer_keys),
                         key=lambda nm: int(''.join(ch for ch in nm if ch.isdigit())))
    num_layers = len(layer_names)

    # Get layer dimensions
    layer_sizes = []
    for i, name in enumerate(layer_names):
        w = params[f'{name}_weight']
        if i == 0:
            layer_sizes.append(w.shape[1])
        layer_sizes.append(w.shape[0])

    input_size = layer_sizes[0]   # 256 (128 query + 128 context)
    output_size = layer_sizes[-1]

    _spec_pre = load_spec_from_model(model_path)
    print(f"Architecture: {' -> '.join(map(str, layer_sizes))}")
    print(f"Input: {input_size} ({_spec_pre['query_buckets']} query + "
          f"{_spec_pre['context_buckets']} context)")
    print(f"Output: {output_size} characters")

    # Check for dual bias on the OUTPUT layer (its name is fcN, not hardcoded fc4).
    has_dual_bias = f'{layer_names[-1]}_bias_start' in params
    # Position threshold the model was trained with (sim's DUAL_BIAS_THRESHOLD).
    # Driven from the model file so the device branch can never silently disagree
    # with the sim if the threshold is ever changed.
    dual_bias_threshold = load_dual_bias_threshold(model_path)
    if has_dual_bias:
        print(f"Dual bias: {layer_names[-1]}_bias (rest) + {layer_names[-1]}_bias_start "
              f"(first {dual_bias_threshold} chars)")
    else:
        print("WARNING: No dual bias found in model")

    # === Spec-driven, fully generic AppVar layout (the "sharder") ===
    # Replaces the old hardcoded NEOA-D + manual layer-2 split. Each layer is
    # split into output-neuron ranges ("shards") small enough to fit an AppVar,
    # then shards are greedily packed into AppVars (<= MAX_APPVAR_BYTES each).
    # For the baseline [512,512,256] arch this reproduces the exact NEOA-D blobs
    # byte-for-byte (greedy + the same layer-2 split); for any other arch it just
    # works. Output-neuron split points are byte-aligned because every input dim
    # is a multiple of 4 under 2-bit packing (asserted below).
    spec = load_spec_from_model(model_path)
    shifts = spec['inter_layer_shift']
    wbits = spec['weight_bits']               # per-layer weight bit-width
    ascale = spec['activation_scale']
    qb = spec['query_buckets']                # encoding is a DOF (P5)
    cb = spec['context_buckets']
    clen = spec['context_len']
    ctx_orders = spec['context_ngram_orders']
    ctx_max_order = max(ctx_orders)
    assert (qb & (qb - 1)) == 0 and (cb & (cb - 1)) == 0, "bucket counts must be pow2"
    assert qb <= 256 and cb <= 256, \
        "emitted tokenizer masks buckets with an 8-bit AND; bucket counts must be <= 256"
    assert ctx_orders == list(range(1, ctx_max_order + 1)), \
        "emitted tokenizer supports contiguous context n-gram orders 1..N only"
    assert spec['query_ngram_orders'] == [3], "emitted tokenizer is trigram-query only"
    assert len(shifts) == num_layers, (len(shifts), num_layers)
    assert len(wbits) == num_layers, (len(wbits), num_layers)
    MAX_AV = sizes.MAX_APPVAR_BYTES
    out_name = layer_names[-1]

    def pack_bias_bytes(bias_arr):
        """Pack bias array as little-endian uint16 bytes (two's complement)."""
        data = bytearray()
        for v in bias_arr:
            data.extend(struct.pack('<H', int(v) & 0xFFFF))
        return bytes(data)

    def av_name(i):
        """NEOA..NEOZ, then NEOAA.. (<=8 chars). i=0..3 -> NEOA..NEOD (baseline)."""
        return 'NEO' + (chr(65 + i) if i < 26
                        else chr(65 + i // 26 - 1) + chr(65 + i % 26))

    # Per-layer shard ranges; split any layer whose packed weights+bias exceed
    # one AppVar. The output layer carries TWO bias sets (rest + start).
    layer_plan = []
    flat_shards = []
    for li, name in enumerate(layer_names):
        w = params[f'{name}_weight']
        n_out, n_in = int(w.shape[0]), int(w.shape[1])
        assert n_in % 4 == 0, f"layer {li} input dim {n_in} must be a multiple of 4"
        is_out = (li == num_layers - 1)
        bias_sets = 2 if (is_out and has_dual_bias) else 1
        layer_bytes = (n_out * n_in * wbits[li]) // 8 + n_out * 2 * bias_sets
        nsplit = max(1, -(-layer_bytes // MAX_AV))
        base, rem, lo, ranges = n_out // nsplit, n_out % nsplit, 0, []
        for k in range(nsplit):
            sz = base + (1 if k < rem else 0)
            ranges.append((lo, lo + sz)); lo += sz
        layer_plan.append({'li': li, 'n_in': n_in, 'n_out': n_out,
                           'shift': shifts[li], 'bits': wbits[li],
                           'is_out': is_out, 'shards': []})
        for (a, c) in ranges:
            flat_shards.append((li, a, c))

    def shard_blobs(li, lo, hi):
        name = layer_names[li]
        wb = pack_weights(params[f'{name}_weight'][lo:hi, :], wbits[li])
        if li == num_layers - 1:                       # output: bias pre-scaled
            sh = shifts[li]
            brest = pack_bias_bytes(params[f'{name}_bias'][lo:hi] * (1 << sh))
            bstart = (pack_bias_bytes(params[f'{out_name}_bias_start'][lo:hi] * (1 << sh))
                      if has_dual_bias else brest)
            return wb, brest, bstart
        return wb, pack_bias_bytes(params[f'{name}_bias'][lo:hi]), None

    appvar_blobs = {}
    av_idx, cur = 0, bytearray()
    for (li, lo, hi) in flat_shards:
        wb, b1, b2 = shard_blobs(li, lo, hi)
        sz = len(wb) + len(b1) + (len(b2) if b2 is not None else 0)
        if len(cur) > 0 and len(cur) + sz > MAX_AV:
            appvar_blobs[av_name(av_idx)] = bytes(cur); av_idx += 1; cur = bytearray()
        w_off = len(cur); cur += wb
        b_off = len(cur); cur += b1
        bs_off = None
        if b2 is not None:
            bs_off = len(cur); cur += b2
        layer_plan[li]['shards'].append(
            {'av': av_idx, 'w_off': w_off, 'b_off': b_off, 'bs_off': bs_off,
             'lo': lo, 'hi': hi})
    appvar_blobs[av_name(av_idx)] = bytes(cur)
    av_names = [av_name(i) for i in range(av_idx + 1)]

    # Ping-pong buffer assignment: layer 0 reads TOKBUF, the output layer writes
    # OUTBUF, hidden layers alternate PING/PONG (in != out every layer).
    hidden_out_sizes = layer_sizes[1:-1]
    max_hidden = max(hidden_out_sizes) if hidden_out_sizes else output_size
    pingpong = ['PING', 'PONG']
    for li, plan in enumerate(layer_plan):
        plan['in_buf'] = 'TOKBUF' if li == 0 else pingpong[(li - 1) % 2]
        plan['out_buf'] = 'OUTBUF' if plan['is_out'] else pingpong[li % 2]

    for nm in av_names:
        blob = appvar_blobs[nm]
        print(f"  AppVar {nm}: {len(blob):,} bytes ({len(blob)/1024:.1f} KB)")

    # === Now build the program (code only, no weights) ===

    # The program has 5 "virtual layers" for dispatch:
    # LAYER1:  256->512  (from NEOA)
    # LAYER2A: 512->256  (from NEOB, first half of Layer 2)
    # LAYER2B: 512->256  (from NEOC, second half of Layer 2)
    # LAYER3:  512->256  (from NEOD offset 0)
    # LAYER4_REST:  256->43   (from NEOD offset l4_woffset, bias at l4_boffset)
    # LAYER4_START: 256->43   (from NEOD offset l4_woffset, bias at l4_bs_offset)

    b = eZ80Builder(org=USERMEM)

    # === MAIN ===
    b.label('START')

    # Save IY (TI-OS flags pointer)
    b.ld_mem_label_iy('SAVED_IY')

    # Turn off run indicator
    b.call_addr(TI_RunIndicOff)

    # === Load AppVars ===
    for i, _nm in enumerate(av_names):
        b.ld_hl_label(f'AVNAME{i}')
        b.call_addr(TI_Mov9ToOP1)
        b.call_addr(TI_ChkFindSym)
        b.jp_c('AV_ERR')              # carry set = not found
        # Decide RAM vs archived by WHERE ChkFindSym's data pointer (DE) points,
        # not by the archive-flag register (A/B), which proved unreliable on real
        # hardware. RAM is >= 0xD00000; archived vars live in flash (< 0xD00000).
        # Testing the pointer also means we never Arc_Unarc a var that's already
        # in RAM, so the calc's archive layout is left exactly as we found it.
        b.ex_de_hl()                  # HL = data pointer
        b.ld_mem_label_hl('AVTMP')    # stash all 3 bytes
        b.ex_de_hl()                  # restore DE = data pointer
        b.ld_a_mem_label('AVTMP_HI')  # A = high byte of the pointer
        b.cp_n(0xD0)
        b.jr_nc(f'AV_RAM{i}')         # high byte >= 0xD0 -> already in RAM, use as-is

        # Archived (pointer in flash) -- unarchive it, then re-find
        b.ld_hl_label(f'AVNAME{i}')
        b.call_addr(TI_Mov9ToOP1)
        b.call_addr(TI_Arc_Unarc)
        b.ld_a_n(1)
        b.ld_mem_label_a(f'AVARCED{i}')
        b.ld_hl_label(f'AVNAME{i}')
        b.call_addr(TI_Mov9ToOP1)
        b.call_addr(TI_ChkFindSym)
        b.jp_c('AV_ERR')

        b.label(f'AV_RAM{i}')
        b.inc_de()
        b.inc_de()                    # Skip 2-byte size header
        # Store the AppVar weight-data pointer. ED-53 (LD (nn),DE) can corrupt
        # adjacent memory on real hardware (libez80 caveat #2), so move the
        # pointer into HL and use the safe LD (nn),HL (0x22) instead. DE/HL are
        # not needed afterwards (the loop re-finds each AppVar).
        b.ex_de_hl()
        b.ld_mem_label_hl(f'AVPTR{i}')

    b.jr('AV_OK')

    b.label('AV_ERR')
    b.call_addr(TI_ClrScrn)
    b.call_addr(TI_HomeUp)
    b.ld_a_n(ord('E'))
    b.call_addr(TI_PutC)
    b.ld_a_n(ord('R'))
    b.call_addr(TI_PutC)
    b.ld_a_n(ord('R'))
    b.call_addr(TI_PutC)
    b.call_addr(TI_RunIndicOn)
    b.ret()

    b.label('AV_OK')

    # Clear screen
    b.call_addr(TI_ClrScrn)
    b.call_addr(TI_HomeUp)

    # Enter chat mode (or the debug dump in --debug builds)
    if debug:
        b.jp('DBG_RUN')
    else:
        b.jp('CHAT')

    # === CHAT MODE ===
    b.label('CHAT')
    b.call_addr(TI_ClrScrn)
    b.call_addr(TI_HomeUp)

    b.label('CHAT_LOOP')

    # Check cursor row -- if on line 7+ (of 0-9), clear screen for space
    b.ld_hl_nn(TI_CURSOR_ROW)
    b.ld_a_hl()
    b.cp_n(7)
    b.jr_c('CHAT_NOCLS')
    b.label('WAIT_KEY')
    b.call_addr(TI_GetCSC)
    b.or_a()
    b.jr_z('WAIT_KEY')
    b.call_addr(TI_ClrScrn)
    b.call_addr(TI_HomeUp)
    b.jr('CHAT_PROMPT')

    b.label('CHAT_NOCLS')
    b.or_a()
    b.jr_z('CHAT_PROMPT')
    b.call_addr(TI_NewLine)

    b.label('CHAT_PROMPT')

    # Print prompt "> "
    b.ld_a_n(ord('>'))
    b.call_addr(TI_PutC)
    b.ld_a_n(ord(' '))
    b.call_addr(TI_PutC)

    # Read input
    b.call('READ_INPUT')

    # Check if MODE was pressed (quit)
    b.ld_a_mem_label('RI_FLAG')
    b.or_a()
    b.jp_nz('CHAT_EXIT')

    # Check if CLEAR was pressed
    b.ld_a_mem_label('RI_CLEAR_FLAG')
    b.or_a()
    b.jr_nz('CHAT_LOOP')

    # Check if empty
    b.ld_a_mem_label('INPLEN')
    b.or_a()
    b.jr_z('CHAT_LOOP')

    # Process and generate
    b.call('TOKENIZE')
    b.call('CLEAR_CTX')
    b.call('GENERATE')
    # Restore IY and clear inverse mode after response
    b.ld_iy_mem_label('SAVED_IY')
    b.ld_hl_nn(0xD00085)
    b.ld_a_hl()
    b.and_n(0xF7)             # Clear bit 3 (textInverse)
    b.ld_hl_a()

    b.jp('CHAT_LOOP')

    b.label('CHAT_EXIT')
    # Re-archive any AppVars we unarchived at startup
    b.ld_iy_mem_label('SAVED_IY')
    for i, _nm in enumerate(av_names):
        b.ld_a_mem_label(f'AVARCED{i}')
        b.or_a()
        b.jr_z(f'AV_NOARC{i}')
        b.ld_hl_label(f'AVNAME{i}')
        b.call_addr(TI_Mov9ToOP1)
        b.call_addr(TI_Arc_Unarc)
        b.label(f'AV_NOARC{i}')
    # Clean up
    b.call_addr(TI_ClrScrn)
    b.call_addr(TI_HomeUp)
    b.call_addr(TI_RunIndicOn)
    b.ret()

    # === READ_INPUT: Read a line using GetCSC + scan table ===
    b.label('READ_INPUT')
    b.xor_a()
    b.ld_mem_label_a('INPLEN')
    b.ld_mem_label_a('RI_FLAG')
    b.ld_mem_label_a('RI_CLEAR_FLAG')

    b.label('RI_LOOP')
    b.call_addr(TI_GetCSC)
    b.or_a()
    b.jr_z('RI_LOOP')

    # Check for MODE (quit)
    b.cp_n(SK_MODE)
    b.jp_z('RI_MODE')

    # Check for ENTER (submit)
    b.cp_n(SK_ENTER)
    b.jp_z('RI_DONE')

    # Check for CLEAR
    b.cp_n(SK_CLEAR)
    b.jp_z('RI_CLEAR')

    # Check for DEL (backspace)
    b.cp_n(SK_DEL)
    b.jp_z('RI_DELETE')

    # Look up scan code in table
    b.ld_hl_label('SCANTBL')
    b.ld_bc_nn(0)
    b.ld_c_a()
    b.add_hl_bc()
    b.ld_a_hl()
    b.or_a()
    b.jr_z('RI_LOOP')

    # Check if buffer full. Cap at 60 to match the Python side, which truncates
    # queries to 60 chars (encoding.parse_pair / prepare_data.normalize_query);
    # accepting more on-calc would tokenize long queries differently than trained.
    b.ld_b_a()
    b.ld_a_mem_label('INPLEN')
    b.cp_n(60)
    b.jr_nc('RI_LOOP')

    # Store character
    b.push_bc()
    b.ld_hl_label('INPBUF')
    b.ld_bc_nn(0)
    b.ld_c_a()
    b.add_hl_bc()
    b.pop_bc()
    b.ld_a_b()
    b.ld_hl_a()

    # Increment length
    b.ld_a_mem_label('INPLEN')
    b.inc_a()
    b.ld_mem_label_a('INPLEN')

    # Echo character
    b.ld_a_b()
    b.call_addr(TI_PutC)

    b.jr('RI_LOOP')

    b.label('RI_DELETE')
    b.ld_a_mem_label('INPLEN')
    b.or_a()
    b.jr_z('RI_LOOP')

    b.dec_a()
    b.ld_mem_label_a('INPLEN')

    # Visual backspace
    b.ld_hl_nn(TI_CURSOR_COL)
    b.ld_a_hl()
    b.or_a()
    b.jr_z('RI_LOOP')
    b.dec_a()
    b.ld_hl_a()
    b.ld_a_n(ord(' '))
    b.call_addr(TI_PutC)
    b.ld_hl_nn(TI_CURSOR_COL)
    b.ld_a_hl()
    b.dec_a()
    b.ld_hl_a()
    b.jr('RI_LOOP')

    b.label('RI_CLEAR')
    # CLEAR key: reset input, clear screen
    b.xor_a()
    b.ld_mem_label_a('INPLEN')
    b.ld_a_n(1)
    b.ld_mem_label_a('RI_CLEAR_FLAG')
    b.call_addr(TI_ClrScrn)
    b.call_addr(TI_HomeUp)
    b.jp('RI_DONE')

    b.label('RI_MODE')
    b.ld_a_n(1)
    b.ld_mem_label_a('RI_FLAG')
    b.xor_a()
    b.ld_mem_label_a('INPLEN')

    b.label('RI_DONE')
    # Strip TRAILING spaces so the device sees query.strip() exactly like the sim
    # (train.py/chat.py/test_model.py all .strip() the query before encoding, and
    # TOKENIZE already skips LEADING spaces). Without this, a typed trailing space
    # produces a different trigram vector and a different on-calc answer (~16% of
    # queries). An all-spaces query collapses to INPLEN=0 here and is skipped by
    # CHAT, matching Python's empty-after-strip behaviour.
    b.label('RI_STRIP')
    b.ld_a_mem_label('INPLEN')
    b.or_a()
    b.jr_z('RI_STRIP_DONE')        # empty -> nothing to trim
    b.dec_a()                       # A = index of last char
    b.ld_hl_label('INPBUF')
    b.ld_bc_nn(0)
    b.ld_c_a()
    b.add_hl_bc()                   # HL -> INPBUF[last]
    b.ld_a_hl()
    b.cp_n(ord(' '))
    b.jr_nz('RI_STRIP_DONE')       # last char not a space -> done
    b.ld_a_mem_label('INPLEN')
    b.dec_a()
    b.ld_mem_label_a('INPLEN')
    b.jr('RI_STRIP')
    b.label('RI_STRIP_DONE')
    b.call_addr(TI_NewLine)
    b.ret()

    # === GENERATE: Main generation loop ===
    # Faithful to train.generate_response: pure argmax + dual-bias-by-position,
    # stop at EOS or after MAX_OUTPUT_LEN (=50) characters. No attention, no
    # logit boosting, no confidence gating, no repeat detection.
    b.label('GENERATE')
    b.ld_a_n(MAX_OUTPUT_LEN)
    b.ld_mem_label_a('GENCNT')     # Remaining-character / max-length counter
    b.xor_a()
    b.ld_mem_label_a('GENPOS')     # Generation position counter (0-based)

    b.label('GENLOOP')
    # One generic forward pass (all layers + genpos-selected output) -> OUTBUF.
    b.call('FORWARD')

    # === ARGMAX -> RESULT (plain argmax over logits) ===
    b.call('ARGMAX')

    # === EOS check: stop when argmax == EOS index ===
    b.ld_a_mem_label('RESULT')
    b.cp_n(eos_idx)
    b.jr_z('GEN_STOP')       # argmax == EOS: stop generation

    # === Print character -- restore IY for TI-OS, set inverse mode
    b.ld_iy_mem_label('SAVED_IY')
    b.ld_hl_nn(0xD00085)     # textFlags
    b.ld_a_hl()
    b.or_n(0x08)              # OR 0x08 -- set bit 3 (textInverse)
    b.ld_hl_a()
    b.call('PRINTCH')

    # Update context
    b.call('UPDATE_CTX')

    # Increment GENPOS
    b.ld_a_mem_label('GENPOS')
    b.inc_a()
    b.ld_mem_label_a('GENPOS')

    # MAX-LENGTH stop: loop while GENCNT (remaining of MAX_OUTPUT_LEN) nonzero
    b.ld_a_mem_label('GENCNT')
    b.dec_a()
    b.ld_mem_label_a('GENCNT')
    b.jp_nz('GENLOOP')

    b.label('GEN_STOP')
    b.ret()

    # === PRINTCH: Print character from RESULT via TI-OS ===
    b.label('PRINTCH')
    b.ld_a_mem_label('RESULT')
    b.ld_hl_label('CHARTBL')
    b.ld_bc_nn(0)
    b.ld_c_a()
    b.add_hl_bc()
    b.ld_a_hl()
    b.call_addr(TI_PutC)
    b.ret()

    # === UPDATE_CTX: Update context encoding with new character ===
    b.label('UPDATE_CTX')
    b.ld_hl_label('CTXCHARS')
    b.inc_hl()
    b.ld_de_label('CTXCHARS')
    b.ld_bc_nn(clen - 1)
    b.ldir()

    # Store new character at end
    b.ld_a_mem_label('RESULT')
    b.ld_hl_label('CHARTBL')
    b.ld_bc_nn(0)
    b.ld_c_a()
    b.add_hl_bc()
    b.ld_a_hl()
    # Convert to lowercase
    b.cp_n(ord('A'))
    b.jr_c('UPD_STORE')
    b.cp_n(ord('Z') + 1)
    b.jr_nc('UPD_STORE')
    b.add_a_n(0x20)
    b.label('UPD_STORE')
    b.ld_hl_label('CTXCHARS')
    b.ld_de_nn(clen - 1)
    b.add_hl_de()
    b.ld_hl_a()

    b.call('ENCODE_CTX')
    b.ret()

    # === ENCODE_CTX: Encode CTXCHARS into context buckets ===
    b.label('ENCODE_CTX')
    # Clear context buckets (the cb buckets after the qb query buckets in TOKBUF)
    b.ld_hl_label('TOKBUF')
    b.ld_de_nn(qb * 2)  # offset to context region = query_buckets * 2 bytes
    b.add_hl_de()
    b.push_hl()
    b.pop_de()
    b.inc_de()
    b.xor_a()
    b.ld_hl_a()
    b.ld_bc_nn(cb * 2 - 1)  # context_buckets * 2 - 1
    b.ldir()

    # Hash n-grams
    b.ld_a_n(0)
    b.ld_mem_label_a('CTXPOS')

    b.ld_a_n(1)
    b.ld_mem_label_a('CTXN')

    b.label('CTX_NLOOP')
    b.xor_a()
    b.ld_mem_label_a('CTXPOS')

    b.label('CTX_PLOOP')
    b.ld_a_n(clen + 1)             # positions for length-n: 0..clen-n  (< clen+1-n)
    b.ld_hl_label('CTXN')
    b.sub_hl_ind()
    b.ld_b_a()
    b.ld_a_mem_label('CTXPOS')
    b.cp_b()
    b.jr_nc('CTX_NEXT_N')

    b.call('CTX_HASH')

    b.ld_a_mem_label('CTXPOS')
    b.inc_a()
    b.ld_mem_label_a('CTXPOS')
    b.jr('CTX_PLOOP')

    b.label('CTX_NEXT_N')
    b.ld_a_mem_label('CTXN')
    b.inc_a()
    b.ld_mem_label_a('CTXN')
    b.cp_n(ctx_max_order + 1)      # context n-gram orders 1..ctx_max_order
    b.jr_c('CTX_NLOOP')
    b.ret()

    # === CTX_HASH: Hash n-gram at position CTXPOS with length CTXN ===
    b.label('CTX_HASH')
    # hash = pos * 7
    b.ld_a_mem_label('CTXPOS')
    b.ld_l_a()
    b.ld_h_n(0)
    b.add_hl_hl_16()  # *2
    b.add_hl_hl_16()  # *4
    b.add_hl_hl_16()  # *8
    b.push_hl()
    b.pop_de()
    b.ld_a_mem_label('CTXPOS')
    b.ld_l_a()
    b.ld_h_n(0)
    b.ex_de_hl()
    b.or_a()
    b.sbc_hl_de_16()  # *7
    b.push_hl()

    # Get pointer to chars
    b.ld_hl_label('CTXCHARS')
    b.ld_a_mem_label('CTXPOS')
    b.ld_bc_nn(0)
    b.ld_c_a()
    b.add_hl_bc()
    b.ex_de_hl()

    b.pop_hl()

    # For each char in n-gram
    b.ld_a_mem_label('CTXN')
    b.ld_b_a()

    b.label('CTX_HLOOP')
    b.push_bc()
    # hash = hash * 31 + char
    b.push_hl()
    b.add_hl_hl_16()  # *2
    b.add_hl_hl_16()  # *4
    b.add_hl_hl_16()  # *8
    b.add_hl_hl_16()  # *16
    b.add_hl_hl_16()  # *32
    b.pop_bc()
    b.or_a()
    b.sbc_hl_bc_16()  # *31
    b.ld_a_de()
    b.ld_c_a()
    b.ld_b_n(0)
    b.add_hl_bc_16()  # + char
    b.inc_de()
    b.pop_bc()
    b.djnz('CTX_HLOOP')

    # bucket = (hash & (cb-1)); context region starts at TOKBUF + qb buckets
    b.ld_a_l()
    b.and_n(cb - 1)

    # Add to bucket (context is at TOKBUF + query_buckets*2 bytes)
    b.ld_hl_nn(0)
    b.ld_l_a()
    b.add_hl_hl()
    b.ld_de_label('TOKBUF')
    b.ld_bc_nn(qb * 2)
    b.add_hl_bc()
    b.add_hl_de()

    # Increment bucket value by activation_scale (device folds the *scale here)
    b.ld_e_hl()
    b.inc_hl()
    b.ld_d_hl()
    b.push_hl()
    b.ld_hl_nn_16(ascale)
    b.add_hl_de_16()
    b.ex_de_hl()
    b.pop_hl()
    b.ld_hl_d()
    b.dec_hl()
    b.ld_hl_e()
    b.ret()

    # === CLEAR_CTX: Initialize context with spaces ===
    b.label('CLEAR_CTX')
    b.ld_hl_label('CTXCHARS')
    b.ld_a_n(ord(' '))
    for _ in range(clen):
        b.ld_hl_a()
        b.inc_hl()

    b.jp('ENCODE_CTX')


    # === Layer dispatch stubs (generic; one per shard) ===
    # Each stub loads its weight/bias pointers (AVPTRn + offset), input/output
    # buffers (output offset for split layers), dimensions, and the per-layer
    # right-shift (LSHIFT, applied at runtime in LAYER), then jumps to LAYER.
    def emit_stub(label, av_idx, w_off, b_off, in_buf, out_buf, out_off,
                  in_size, out_size, shift, bits):
        b.label(label)
        b.ld_hl_mem_label(f'AVPTR{av_idx}')        # HL = weight pointer
        if w_off > 0:
            b.ld_de_nn(w_off); b.add_hl_de()
        b.ld_ix_label(in_buf)                       # IX = input buffer
        if out_off > 0:                            # IY = output buffer + offset
            b.push_hl()
            b.ld_hl_label(out_buf); b.ld_de_nn(out_off); b.add_hl_de()
            b.push_hl(); b.pop_iy()
            b.pop_hl()
        else:
            b.ld_iy_label(out_buf)
        b.push_hl()                                # DE = bias pointer
        b.ld_hl_mem_label(f'AVPTR{av_idx}')
        if b_off > 0:
            b.ld_de_nn(b_off); b.add_hl_de()
        b.push_hl(); b.pop_de()
        b.pop_hl()
        b.push_hl()                                # dims (24-bit, see NEURCNT note)
        b.ld_hl_nn(out_size); b.ld_mem_label_hl('NEURCNT')
        b.ld_hl_nn(in_size); b.ld_mem_label_hl('INCNT')
        b.pop_hl()
        b.ld_a_n(shift); b.ld_mem_label_a('LSHIFT')  # per-layer right-shift
        b.jp(f'LAYER_B{bits}')                        # per-bit-width LAYER variant

    for plan in layer_plan:
        li, in_buf, out_buf = plan['li'], plan['in_buf'], plan['out_buf']
        shift, bits = plan['shift'], plan['bits']
        for k, sh in enumerate(plan['shards']):
            in_size, out_size, out_off = plan['n_in'], sh['hi'] - sh['lo'], sh['lo'] * 2
            if plan['is_out']:
                emit_stub(f'L{li}_R{k}', sh['av'], sh['w_off'], sh['b_off'],
                          in_buf, out_buf, out_off, in_size, out_size, shift, bits)
                emit_stub(f'L{li}_S{k}', sh['av'], sh['w_off'], sh['bs_off'],
                          in_buf, out_buf, out_off, in_size, out_size, shift, bits)
            else:
                emit_stub(f'L{li}_{k}', sh['av'], sh['w_off'], sh['b_off'],
                          in_buf, out_buf, out_off, in_size, out_size, shift, bits)

    # === FORWARD: full forward pass (all layers + genpos-selected output) ===
    b.label('FORWARD')
    for plan in layer_plan[:-1]:                    # hidden layers
        for k in range(len(plan['shards'])):
            b.call(f"L{plan['li']}_{k}")
        b.ld_bc_nn(plan['n_out'])                   # ReLU over the full layer output
        b.ld_hl_label(plan['out_buf'])
        b.call('RELU')
    out_plan = layer_plan[-1]
    b.ld_a_mem_label('GENPOS')
    b.cp_n(dual_bias_threshold)
    b.jr_nc('FWD_REST')
    for k in range(len(out_plan['shards'])):        # first chars: start bias
        b.call(f"L{out_plan['li']}_S{k}")
    b.ret()
    b.label('FWD_REST')
    for k in range(len(out_plan['shards'])):        # rest: rest bias
        b.call(f"L{out_plan['li']}_R{k}")
    b.ret()

    # === LAYER: neural-net layer computation (one variant per weight bit-width) ===
    # 24-bit native accumulator. The decode (LWT/LSAME) is bit-width-specific:
    # 8//bits codes per packed byte, code = w + 2^(bits-1), zero-skip sentinel =
    # zero_byte(bits). 2-bit uses the fast special-cased MULADD; wider widths use
    # the general multiply MULADD_GEN. Internal labels are suffixed per variant.
    def emit_layer_routine(bits):
        sfx = f'_B{bits}'
        cpb = 8 // bits                 # codes per packed byte
        mask = (1 << bits) - 1
        off = 1 << (bits - 1)           # zero-code offset
        zb = zero_byte(bits)            # all-zero-weights sentinel byte
        muladd = 'MULADD' if bits == 2 else 'MULADD_GEN'

        b.label(f'LAYER_B{bits}')
        b.ld_mem_label_hl('SAVW')       # weight pointer -> SAVW (safe LD (nn),HL)
        b.ex_de_hl()                    # bias pointer (DE) -> SAVB via HL (caveat #2)
        b.ld_mem_label_hl('SAVB')

        b.label(f'LNEUR{sfx}')
        b.ld_hl_nn(0)
        b.ld_mem_label_hl('ACC')
        b.push_ix()
        b.pop_hl()
        b.ld_mem_label_hl('CURIN')
        b.ld_de_nn(0)
        b.ld_hl_mem_label('SAVW')
        b.push_hl()
        b.ld_hl_mem_label('INCNT')
        b.ld_mem_label_hl('WTCNT')
        b.pop_hl()
        b.ld_c_n(0)

        b.label(f'LWT{sfx}')
        b.ld_a_c()
        b.and_n(cpb - 1)                # load a new packed byte every cpb weights
        b.jr_nz(f'LSAME{sfx}')
        b.ld_hl_mem_label('SAVW')
        b.ld_a_hl()
        b.ld_mem_label_a('PACKED')
        b.inc_hl()
        b.ld_mem_label_hl('SAVW')

        # ZERO-SKIP: packed byte == sentinel -> all cpb weights are zero
        b.cp_n(zb)
        b.jr_nz(f'LSAME{sfx}')

        b.ld_hl_mem_label('CURIN')
        b.ld_de_nn(cpb * 2)            # advance input past cpb skipped weights
        b.add_hl_de()
        b.ld_mem_label_hl('CURIN')
        b.ld_a_c()
        b.add_a_n(cpb)
        b.ld_c_a()
        b.push_hl()
        b.ld_hl_mem_label('WTCNT')
        for _ in range(cpb):
            b.dec_hl()
        b.ld_mem_label_hl('WTCNT')
        b.ld_a_h()
        b.or_l()
        b.pop_hl()
        b.jp_nz(f'LWT{sfx}')
        b.jp(f'LWT_DONE{sfx}')

        b.label(f'LSAME{sfx}')
        b.ld_a_mem_label('PACKED')
        b.and_n(mask)
        b.sub_n(off)                   # code -> signed weight
        b.ld_mem_label_a('WEIGHT')
        b.ld_a_mem_label('PACKED')
        for _ in range(bits):          # shift to the next code
            b.rrca()
        b.ld_mem_label_a('PACKED')
        b.ld_hl_mem_label('CURIN')
        b.ld_e_hl()
        b.inc_hl()
        b.ld_d_hl()
        b.inc_hl()
        b.ld_mem_label_hl('CURIN')
        b.ld_a_mem_label('WEIGHT')
        b.call(muladd)
        b.inc_c()
        b.push_hl()
        b.ld_hl_mem_label('WTCNT')
        b.dec_hl()
        b.ld_mem_label_hl('WTCNT')
        b.ld_a_h()
        b.or_l()
        b.pop_hl()
        b.jp_nz(f'LWT{sfx}')

        b.label(f'LWT_DONE{sfx}')

        # Post inner loop: add bias THEN shift (output bias is pre-scaled, so this
        # matches (matmul >> shift) + bias for the output layer too).
        b.ld_hl_label('ACC')
        b.ld_e_hl()
        b.inc_hl()
        b.ld_d_hl()
        b.inc_hl()
        b.ld_a_hl()

        b.ld_hl_mem_label('SAVB')
        b.ld_c_hl()                       # LD C, (HL)
        b.inc_hl()
        b.ld_b_hl()
        b.inc_hl()
        b.ld_mem_label_hl('SAVB')

        b.ld_h_a()
        b.ld_a_e()
        b.add_a_c()                      # ADD A, C
        b.ld_e_a()
        b.ld_a_d()
        b.adc_a_b()                      # ADC A, B
        b.ld_d_a()
        b.push_af()
        b.ld_a_b()
        b.add_a_a()
        b.sbc_a_a()                      # SBC A, A
        b.ld_b_a()
        b.pop_af()
        b.ld_a_h()
        b.adc_a_b()                      # ADC A, B

        # Arithmetic right shift by LSHIFT (per-layer divide; SRA;RR;RR = floor).
        # Value is in A:D:E (high:mid:low); preserved across the counter load via
        # push_af/pop_af. shift==0 -> no shift.
        b.push_af()
        b.ld_a_mem_label('LSHIFT')
        b.ld_mem_label_a('LSHIFT_CNT')
        b.pop_af()
        b.label(f'LSH_LOOP{sfx}')
        b.push_af()
        b.ld_a_mem_label('LSHIFT_CNT')
        b.or_a()
        b.jr_z(f'LSH_DONE{sfx}')
        b.dec_a()
        b.ld_mem_label_a('LSHIFT_CNT')
        b.pop_af()
        b.sra_a()                         # SRA A
        b.rr_d()                          # RR D
        b.rr_e()                          # RR E
        b.jr(f'LSH_LOOP{sfx}')
        b.label(f'LSH_DONE{sfx}')
        b.pop_af()

        # Store result to output buffer
        b.ld_iyd_e(0x00)                 # LD (IY+0), E
        b.ld_iyd_d(0x01)                 # LD (IY+1), D
        b.inc_iy()
        b.inc_iy()

        # Outer loop: decrement neuron counter
        b.push_hl()
        b.ld_hl_mem_label('NEURCNT')
        b.dec_hl()
        b.ld_mem_label_hl('NEURCNT')
        b.ld_a_h()
        b.or_l()
        b.pop_hl()
        b.jp_nz(f'LNEUR{sfx}')
        b.ret()

    for bits in sorted(set(wbits)):
        emit_layer_routine(bits)

    # === MULADD: Multiply-accumulate (24-bit native) ===
    b.label('MULADD')
    b.or_a()
    b.jr_z('MA_RET')
    b.jp_m('MA_NEG')
    # weight == +1: ACC += DE
    b.ld_hl_mem_label('ACC')
    b.add_hl_de()
    b.ld_mem_label_hl('ACC')
    b.ret()

    b.label('MA_NEG')
    b.cp_n(0xFF)
    b.jr_z('MA_N1')
    # weight == -2: ACC -= 2*DE. Each SBC HL,DE subtracts DE *and the carry*, so
    # the carry MUST be cleared before BOTH subtractions. Without the second
    # `or a` the first SBC's borrow leaks into the second, computing
    # ACC - 2*DE - borrow and silently diverging from the sim (train._forward_int
    # does an exact -2*DE). This is an assembly-level bug, so the host
    # test_faithfulness.py (which models the device CONTRACT, not the literal
    # bytes) does NOT catch it -- verify on hardware via the --debug per-layer
    # checksum build if you touch MA_NEG.
    b.ld_hl_mem_label('ACC')
    b.or_a()
    b.sbc_hl_de()
    b.or_a()
    b.sbc_hl_de()
    b.ld_mem_label_hl('ACC')
    b.ret()

    b.label('MA_N1')
    # weight == -1: ACC -= DE
    b.ld_hl_mem_label('ACC')
    b.or_a()
    b.sbc_hl_de()
    b.ld_mem_label_hl('ACC')

    b.label('MA_RET')
    b.ret()

    # === MULADD_GEN: general signed multiply-accumulate (for >2-bit weights) ===
    # A = signed weight, DE = input value, ACC += A*DE via |A| add/subtract steps.
    # Only emitted (and only called) when some layer uses a bit-width other than 2.
    if any(bt != 2 for bt in wbits):
        b.label('MULADD_GEN')
        b.or_a()
        b.ret_z()                        # weight 0 -> nothing
        b.jp_m('MAG_NEG')
        b.ld_b_a()                        # B = weight (positive); loop B times
        b.label('MAG_PLOOP')
        b.ld_hl_mem_label('ACC')
        b.add_hl_de()
        b.ld_mem_label_hl('ACC')
        b.djnz('MAG_PLOOP')
        b.ret()
        b.label('MAG_NEG')
        b.cpl()                           # A = -weight = |weight|
        b.inc_a()
        b.ld_b_a()
        b.label('MAG_NLOOP')
        b.ld_hl_mem_label('ACC')
        b.or_a()                          # clear carry before each SBC
        b.sbc_hl_de()
        b.ld_mem_label_hl('ACC')
        b.djnz('MAG_NLOOP')
        b.ret()

    # === ReLU (HL = buffer pointer, BC = element count; set by FORWARD) ===
    b.label('RELU')
    b.ld_e_hl()
    b.inc_hl()
    b.ld_d_hl()
    b.bit_7_d()
    b.jr_z('RPOS')
    b.dec_hl()
    b.xor_a()
    b.ld_hl_a()
    b.inc_hl()
    b.ld_hl_a()
    b.label('RPOS')
    b.inc_hl()
    b.dec_bc_16()
    b.ld_a_b()
    b.or_c()
    b.jp_nz('RELU')
    b.ret()

    # === ARGMAX (plain argmax over logits, first-max tie-break) ===
    # Mirrors torch.argmax: scans low->high index, only a STRICTLY greater logit
    # replaces the best, so ties keep the lowest index. (Previously also tracked a
    # second-best in SECL/SECH/SECI for a heuristic that was removed; that dead
    # bookkeeping is gone — the best-comparison logic below is unchanged.)
    b.label('ARGMAX')
    b.ld_hl_label('OUTBUF')
    # Load first value as initial best
    b.ld_a_hl()
    b.ld_mem_label_a('MAXL')
    b.inc_hl()
    b.ld_a_hl()
    b.ld_mem_label_a('MAXH')
    b.inc_hl()
    b.xor_a()
    b.ld_mem_label_a('MAXI')
    b.ld_b_n(num_chars - 1)    # 42 remaining candidates
    b.ld_c_n(1)                 # current index

    b.label('AMLP')
    b.ld_e_hl()
    b.inc_hl()
    b.ld_d_hl()
    b.inc_hl()
    b.push_hl()
    b.push_bc()

    # Signed 16-bit compare: is D:E > MAXH:MAXL? (strictly greater -> new best)
    b.ld_a_d()
    b.ld_hl_label('MAXH')
    b.xor_hl()                # XOR (HL)
    b.jp_m('AM_DSIGN')

    # Same sign: unsigned compare
    b.ld_a_d()
    b.cp_hl()
    b.jr_c('AM_SKIP')        # candidate < best -> skip
    b.jr_nz('AM_NEW_BEST')    # candidate > best -> new best
    b.ld_hl_label('MAXL')
    b.ld_a_e()
    b.cp_hl()
    b.jr_c('AM_SKIP')
    b.jr_z('AM_SKIP')        # equal -> keep first (not strictly greater)
    b.jr('AM_NEW_BEST')

    b.label('AM_DSIGN')
    b.bit_7_d()
    b.jr_nz('AM_SKIP')       # candidate negative, best positive -> skip

    b.label('AM_NEW_BEST')
    b.ld_a_e()
    b.ld_mem_label_a('MAXL')
    b.ld_a_d()
    b.ld_mem_label_a('MAXH')
    b.pop_bc()
    b.ld_a_c()
    b.ld_mem_label_a('MAXI')
    b.pop_hl()
    b.inc_c()
    b.dec_b()
    b.jp_nz('AMLP')
    b.jp('AM_DONE')

    b.label('AM_SKIP')
    b.pop_bc()
    b.pop_hl()
    b.inc_c()
    b.dec_b()
    b.jp_nz('AMLP')

    b.label('AM_DONE')
    b.ld_a_mem_label('MAXI')
    b.ld_mem_label_a('RESULT')
    b.ret()

    # === TOKENIZE (query into the first qb buckets of TOKBUF) ===
    b.label('TOKENIZE')
    b.ld_hl_label('TOKBUF')
    b.ld_de_label('TOKBUF')
    b.inc_de()
    b.ld_bc_nn(qb * 2 - 1)        # clear query region (query_buckets * 2 bytes)
    b.ld_a_n(0)
    b.ld_hl_a()
    b.ldir()

    b.ld_a_mem_label('INPLEN')
    b.or_a()
    b.jp_z('TOK_DONE')
    b.ld_mem_label_a('TOKLEN')

    b.ld_de_label('INPBUF')

    b.label('TOK_SKIP_SPACE')
    b.ld_a_mem_label('TOKLEN')
    b.or_a()
    b.jp_z('TOK_DONE')
    b.ld_a_de()
    b.cp_n(ord(' '))
    b.jr_nz('TOK_START')
    b.inc_de()
    b.ld_a_mem_label('TOKLEN')
    b.dec_a()
    b.ld_mem_label_a('TOKLEN')
    b.jr('TOK_SKIP_SPACE')

    b.label('TOK_START')
    b.ld_a_n(ord(' '))
    b.ld_mem_label_a('TOKC1')
    b.ld_a_de()
    b.cp_n(ord('A'))
    b.jr_c('TOK_FIRST_LOW')
    b.cp_n(ord('Z') + 1)
    b.jr_nc('TOK_FIRST_LOW')
    b.add_a_n(0x20)
    b.label('TOK_FIRST_LOW')
    b.ld_mem_label_a('TOKC2')
    b.inc_de()
    b.ld_a_mem_label('TOKLEN')
    b.dec_a()
    b.ld_mem_label_a('TOKLEN')

    b.label('TOK_LOOP')
    b.ld_a_mem_label('TOKLEN')
    b.or_a()
    b.jr_z('TOK_TRAIL')
    b.ld_a_de()
    b.cp_n(ord('A'))
    b.jr_c('TOK_LOW1')
    b.cp_n(ord('Z') + 1)
    b.jr_nc('TOK_LOW1')
    b.add_a_n(0x20)
    b.label('TOK_LOW1')
    b.ld_mem_label_a('TOKC3')
    b.call('TOK_HASH')
    b.ld_a_mem_label('TOKC2')
    b.ld_mem_label_a('TOKC1')
    b.ld_a_mem_label('TOKC3')
    b.ld_mem_label_a('TOKC2')
    b.inc_de()
    b.ld_a_mem_label('TOKLEN')
    b.dec_a()
    b.ld_mem_label_a('TOKLEN')
    b.jr('TOK_LOOP')

    b.label('TOK_TRAIL')
    b.ld_a_n(ord(' '))
    b.ld_mem_label_a('TOKC3')
    b.call('TOK_HASH')
    b.jr('TOK_DONE')

    # === TOK_HASH ===
    b.label('TOK_HASH')
    b.push_de()
    b.ld_a_mem_label('TOKC1')
    b.ld_l_a()
    b.ld_h_n(0)
    b.push_hl()
    b.add_hl_hl_16()
    b.add_hl_hl_16()
    b.add_hl_hl_16()
    b.add_hl_hl_16()
    b.add_hl_hl_16()
    b.pop_de()
    b.or_a()
    b.sbc_hl_de_16()
    b.ld_a_mem_label('TOKC2')
    b.ld_c_a()
    b.ld_b_n(0)
    b.add_hl_bc_16()
    b.push_hl()
    b.add_hl_hl_16()
    b.add_hl_hl_16()
    b.add_hl_hl_16()
    b.add_hl_hl_16()
    b.add_hl_hl_16()
    b.pop_de()
    b.or_a()
    b.sbc_hl_de_16()
    b.ld_a_mem_label('TOKC3')
    b.ld_c_a()
    b.ld_b_n(0)
    b.add_hl_bc_16()

    b.ld_a_l()
    b.and_n(qb - 1)               # query bucket = hash & (query_buckets - 1)

    b.ld_hl_nn(0)
    b.ld_l_a()
    b.add_hl_hl()
    b.push_de()
    b.ld_de_label('TOKBUF')
    b.add_hl_de()
    b.ld_e_hl()
    b.inc_hl()
    b.ld_d_hl()
    b.ld_bc_nn_16(ascale)
    b.ex_de_hl()
    b.add_hl_bc_16()
    b.ex_de_hl()
    b.ld_a_d()
    b.ld_hl_a()
    b.dec_hl()
    b.ld_a_e()
    b.ld_hl_a()
    b.pop_de()
    b.pop_de()
    b.ret()

    b.label('TOK_DONE')
    b.ret()

    # ================================================================
    # DEBUG instrumentation (only emitted when debug=True). Reuses the
    # production TOKENIZE / CLEAR_CTX / LAYER* / RELU* / ARGMAX routines
    # verbatim, capturing a 24-bit byte-checksum of each intermediate
    # buffer and dumping them to the home screen as 6 hex digits.
    #
    # CHECKSUM DEFINITION: sum every raw byte of the buffer into a 24-bit
    # accumulator (wrap mod 2^24), over the buffer's on-device byte layout.
    # All activation/logit/token buffers store each value as a little-endian
    # 16-bit (two's-complement) word, so we just sum count*2 raw bytes.
    # ================================================================
    if debug:
        # Byte lengths of each buffer on the device (value count * 2 bytes/LE16).
        TOKBUF_BYTES = input_size * 2        # 256 buckets  -> 512 bytes
        L1_BYTES     = 512 * 2               # BUF_A (RELU1) -> 1024 bytes
        L2_BYTES     = 512 * 2               # BUF_B+BUF_B_MID contiguous -> 1024 bytes
        L3_BYTES     = 256 * 2               # BUF_A (RELU3) -> 512 bytes
        OUT_BYTES    = output_size * 2       # OUTBUF (43)   -> 86 bytes

        b.label('DBG_RUN')
        # --- Prompt + read one query line (reuse READ_INPUT) -------------
        b.call_addr(TI_ClrScrn)
        b.call_addr(TI_HomeUp)
        b.ld_a_n(ord('>'))
        b.call_addr(TI_PutC)
        b.ld_a_n(ord(' '))
        b.call_addr(TI_PutC)
        b.call('READ_INPUT')
        # If MODE pressed -> quit cleanly.
        b.ld_a_mem_label('RI_FLAG')
        b.or_a()
        b.jp_nz('DBG_EXIT')

        # --- Build the input vector exactly like production -------------
        b.call('TOKENIZE')      # query -> TOKBUF[0:256]
        b.call('CLEAR_CTX')     # ctx = 8 spaces, encode -> TOKBUF[256:512]

        # genpos = 0 so LAYER4_START (start bias) is used, matching first char.
        b.xor_a()
        b.ld_mem_label_a('GENPOS')

        # P0 sanity: copy AVPTR0 into DBG_P0 for display.
        b.ld_hl_mem_label('AVPTR0')
        b.ld_mem_label_hl('DBG_P0')

        # --- TOK checksum (TOKBUF) --------------------------------------
        b.ld_hl_label('TOKBUF')
        b.ld_bc_nn(TOKBUF_BYTES)
        b.call('CHECKSUM')
        b.ld_hl_mem_label('DBGACC')
        b.ld_mem_label_hl('DBG_TOK')

        # --- Full forward pass -> OUTBUF, then checksum it --------------
        # (Per-layer checksums are gone with the generalized codegen; the host
        # faithfulness gate now verifies every layer at the byte level.)
        b.call('FORWARD')
        b.ld_hl_label('OUTBUF')
        b.ld_bc_nn(OUT_BYTES)
        b.call('CHECKSUM')
        b.ld_hl_mem_label('DBGACC')
        b.ld_mem_label_hl('DBG_OUT')

        # --- ARGMAX -> RESULT (index); fetch winning logit (LE16) -------
        b.call('ARGMAX')
        # DBG_ARGV = OUTBUF[RESULT] as signed 16-bit, sign-extended to 24 bits.
        # OUTBUF stores each logit as a little-endian 16-bit word (2 bytes/elem).
        b.ld_a_mem_label('RESULT')
        b.ld_hl_nn(0)
        b.ld_l_a()
        b.add_hl_hl()                 # HL = index*2 (16-bit element stride)
        b.ld_de_label('OUTBUF')
        b.add_hl_de()                 # HL -> &OUTBUF[RESULT]
        b.ld_e_hl()                   # E = low byte
        b.inc_hl()
        b.ld_d_hl()                   # D = high byte -> DE = logit (LE16)
        # Store the two value bytes, then a third sign byte, into DBG_ARGV.
        b.ld_a_e()
        b.ld_mem_label_a('DBG_ARGV')      # DBG_ARGV+0 = low
        b.ld_a_d()
        b.ld_mem_label_a('DBG_ARGV1')     # DBG_ARGV+1 = high
        b.ld_a_n(0)                       # sign byte = 0x00, or 0xFF if D bit7 set
        b.bit_7_d()
        b.jr_z('DBG_ARGV_HI0')
        b.ld_a_n(0xFF)
        b.label('DBG_ARGV_HI0')
        b.ld_mem_label_a('DBG_ARGV2')     # DBG_ARGV+2 = sign

        # --- Display everything -----------------------------------------
        b.ld_iy_mem_label('SAVED_IY')   # restore IY before any TI-OS call
        b.call_addr(TI_ClrScrn)
        b.call_addr(TI_HomeUp)

        # P0 (AVPTR0 runtime RAM address)
        b.ld_a_n(ord('P'))
        b.call_addr(TI_PutC)
        b.ld_a_n(ord('0'))
        b.call_addr(TI_PutC)
        b.call('DBG_SP')
        b.ld_hl_mem_label('DBG_P0')
        b.call('DBG_PUT_HL6')
        b.call_addr(TI_NewLine)

        # TOK
        b.call('DBG_LBL_TOK')
        b.ld_hl_mem_label('DBG_TOK')
        b.call('DBG_PUT_HL6')
        b.call_addr(TI_NewLine)

        # OUT
        b.call('DBG_LBL_OUT')
        b.ld_hl_mem_label('DBG_OUT')
        b.call('DBG_PUT_HL6')
        b.call_addr(TI_NewLine)

        # ARG: index (2 hex) + space + winning logit (6 hex)
        b.ld_a_n(ord('A'))
        b.call_addr(TI_PutC)
        b.ld_a_n(ord('R'))
        b.call_addr(TI_PutC)
        b.ld_a_n(ord('G'))
        b.call_addr(TI_PutC)
        b.call('DBG_SP')
        b.ld_a_mem_label('RESULT')
        b.call('DBG_PUT_A2')           # argmax index, 2 hex digits
        b.call('DBG_SP')
        b.ld_hl_mem_label('DBG_ARGV')
        b.call('DBG_PUT_HL6')          # winning logit, 6 hex digits
        b.call_addr(TI_NewLine)

        # Wait for a key, then loop back for another query.
        b.label('DBG_WAITK')
        b.call_addr(TI_GetCSC)
        b.or_a()
        b.jr_z('DBG_WAITK')
        b.cp_n(SK_MODE)
        b.jr_z('DBG_EXIT')
        b.jp('DBG_RUN')

        b.label('DBG_EXIT')
        b.call_addr(TI_ClrScrn)
        b.call_addr(TI_HomeUp)
        b.call_addr(TI_RunIndicOn)
        b.ret()

        # --- CHECKSUM: HL=ptr, BC=byte count -> DBGACC (24-bit sum) ------
        # Sums every raw byte in [HL, HL+BC) into the 3-byte accumulator
        # DBGACC/DBGACC1/DBGACC2 (low/mid/high), wrapping mod 2^24.
        b.label('CHECKSUM')
        b.xor_a()
        b.ld_mem_label_a('DBGACC')
        b.ld_mem_label_a('DBGACC1')
        b.ld_mem_label_a('DBGACC2')
        b.label('CKS_LOOP')
        b.ld_a_b()
        b.or_c()
        b.ret_z()                     # count == 0 -> done
        b.ld_a_hl()                   # A = next byte
        b.push_hl()
        b.push_bc()
        b.ld_b_a()                    # B = incoming byte (HL/BC now free)
        b.ld_a_mem_label('DBGACC')
        b.add_a_b()
        b.ld_mem_label_a('DBGACC')
        b.ld_a_mem_label('DBGACC1')
        b.adc_a_n(0)
        b.ld_mem_label_a('DBGACC1')
        b.ld_a_mem_label('DBGACC2')
        b.adc_a_n(0)
        b.ld_mem_label_a('DBGACC2')
        b.pop_bc()
        b.pop_hl()
        b.inc_hl()
        b.dec_bc()
        b.jr('CKS_LOOP')

        # --- DBG_PUT_HL6: print HL (24-bit) as 6 hex digits -------------
        # DBGHEX/DBGHEX1/DBGHEX2 are 3 contiguous bytes (low/mid/high), so the
        # 24-bit store writes all three; print high, mid, low (big-endian).
        b.label('DBG_PUT_HL6')
        b.ld_mem_label_hl('DBGHEX')
        b.ld_a_mem_label('DBGHEX2')   # high byte
        b.call('DBG_PUT_A2')
        b.ld_a_mem_label('DBGHEX1')   # mid byte
        b.call('DBG_PUT_A2')
        b.ld_a_mem_label('DBGHEX')    # low byte
        b.call('DBG_PUT_A2')
        b.ret()

        # --- DBG_PUT_A2: print A as 2 hex digits ------------------------
        b.label('DBG_PUT_A2')
        b.push_af()
        b.rrca()
        b.rrca()
        b.rrca()
        b.rrca()
        b.call('DBG_NIB')
        b.pop_af()
        b.call('DBG_NIB')
        b.ret()

        # --- DBG_NIB: print low nibble of A as hex char -----------------
        b.label('DBG_NIB')
        b.and_n(0x0F)
        b.cp_n(10)
        b.jr_c('DBG_NIB_DIG')
        b.add_a_n(ord('A') - 10)
        b.jr('DBG_NIB_OUT')
        b.label('DBG_NIB_DIG')
        b.add_a_n(ord('0'))
        b.label('DBG_NIB_OUT')
        b.call_addr(TI_PutC)
        b.ret()

        # --- DBG_SP: print a space --------------------------------------
        b.label('DBG_SP')
        b.ld_a_n(ord(' '))
        b.call_addr(TI_PutC)
        b.ret()

        # --- Label printers (avoid string-table machinery) --------------
        def emit_label_printer(name, text):
            b.label(name)
            for ch in text:
                b.ld_a_n(ord(ch))
                b.call_addr(TI_PutC)
            b.call('DBG_SP')
            b.ret()
        emit_label_printer('DBG_LBL_TOK', 'TOK')
        emit_label_printer('DBG_LBL_OUT', 'OUT')

    # === DATA ===

    # Character table (43 chars)
    b.label('CHARTBL')
    for c in charset:
        if c == '\x00':
            b.db(0)
        else:
            b.db(ord(c))

    # Scan code -> ASCII lookup table (256 bytes)
    build_scan_table(b)

    # Variables -- 24-bit accumulator, 24-bit counters, 24-bit pointers
    #
    # NEURCNT/INCNT/WTCNT are 3-byte (24-bit) so their loads/stores are full
    # 24-bit ops (LD HL,nn / LD HL,(nn) / LD (nn),HL).  A 24-bit LD HL,nn clears
    # all 24 bits of HL (the .SIS 16-bit LD HL,nn did NOT -- caveat 4 -- leaving a
    # stale upper byte), and a 24-bit store writes exactly 3 bytes into a 3-byte
    # slot, so it can never under/overrun into the adjacent SAVW weight pointer.
    # This is the uninitialized-state / pointer-corruption fix.
    b.label('NEURCNT'); b.d3(0)
    b.label('INCNT');   b.d3(0)
    b.label('WTCNT');   b.d3(0)
    b.label('SAVW');    b.d3(0)
    b.label('SAVB');    b.d3(0)
    b.label('CURIN');   b.d3(0)
    b.label('PACKED');  b.db(0)
    b.label('WEIGHT');  b.db(0)
    b.label('LSHIFT');     b.db(0)   # per-layer right-shift, set by each stub
    b.label('LSHIFT_CNT'); b.db(0)   # working counter for the runtime shift loop
    b.label('ACC');     b.d3(0)
    b.label('MAXL');    b.db(0)
    b.label('MAXH');    b.db(0)
    b.label('MAXI');    b.db(0)
    b.label('RESULT');  b.db(0)
    b.label('GENCNT');  b.db(0)
    b.label('GENPOS');  b.db(0)       # Generation position (0-based, for dual bias)
    b.label('TOKLEN');  b.db(0)
    b.label('TOKC1');   b.db(0)
    b.label('TOKC2');   b.db(0)
    b.label('TOKC3');   b.db(0)
    b.label('CTXPOS');  b.db(0)
    b.label('CTXN');    b.db(0)
    b.label('CTXCHARS'); b.ds(clen)

    # Input buffer and flags
    b.label('INPLEN');  b.db(0)
    b.label('RI_FLAG'); b.db(0)
    b.label('RI_CLEAR_FLAG'); b.db(0)
    b.label('SAVED_IY'); b.d3(0)
    b.label('INPBUF'); b.ds(62)

    # Computation buffers (ping-pong; each holds the widest layer's output).
    # Split layers write contiguous slices of their output buffer.
    b.label('TOKBUF'); b.ds(input_size * 2)
    b.label('PING'); b.ds(max_hidden * 2)
    b.label('PONG'); b.ds(max_hidden * 2)
    b.label('OUTBUF'); b.ds(output_size * 2)

    # AppVar data pointers
    for i in range(len(av_names)):
        b.label(f'AVPTR{i}'); b.d3(0)

    # Scratch for the RAM-vs-archived pointer test: a 3-byte slot whose high
    # byte (AVTMP_HI, offset +2) is read to classify ChkFindSym's data pointer.
    b.label('AVTMP'); b.db(0); b.db(0)
    b.label('AVTMP_HI'); b.db(0)

    # AppVar name data
    for i, name in enumerate(av_names):
        b.label(f'AVNAME{i}')
        b.db(APPVAR_TYPE)
        for c in name.ljust(8, '\x00'):
            b.db(ord(c))

    # Archive tracking flags
    for i in range(len(av_names)):
        b.label(f'AVARCED{i}'); b.db(0)

    # Debug-only scratch (emitted only when debug=True so production output is
    # byte-for-byte unchanged). 24-bit checksum slots use d3 (3-byte) so the
    # 24-bit LD (nn),HL store fills them exactly. DBGACC/DBGHEX/DBG_ARGV are
    # three CONTIGUOUS single bytes (low, mid, high) so a 24-bit load/store sees
    # them as one little-endian 24-bit value.
    if debug:
        b.label('DBG_P0');  b.d3(0)
        b.label('DBG_TOK'); b.d3(0)
        b.label('DBG_L1');  b.d3(0)
        b.label('DBG_L2');  b.d3(0)
        b.label('DBG_L3');  b.d3(0)
        b.label('DBG_OUT'); b.d3(0)
        b.label('DBGACC');  b.db(0)
        b.label('DBGACC1'); b.db(0)
        b.label('DBGACC2'); b.db(0)
        b.label('DBGHEX');  b.db(0)
        b.label('DBGHEX1'); b.db(0)
        b.label('DBGHEX2'); b.db(0)
        b.label('DBG_ARGV');  b.db(0)
        b.label('DBG_ARGV1'); b.db(0)
        b.label('DBG_ARGV2'); b.db(0)

    # Metadata for the host-side faithfulness gate (faithgate.py). Pure Python
    # attached to the builder — it emits NO bytes, so the build stays byte-for-byte
    # identical. It tells the eZ80 interpreter how to drive one forward pass:
    # which routines to CALL in order, the genpos-selected output routine, and the
    # buffer/flag labels to read/write. When the codegen is generalized (more/
    # fewer layers, different shards), update this list to match.
    b.forward_meta = {
        'input_size': input_size,
        'output_size': output_size,
        'num_chars': num_chars,
        'dual_bias_threshold': dual_bias_threshold,
        'appvar_names': av_names,
        'forward': 'FORWARD',     # one entry point: reads GENPOS, writes OUTBUF
        'argmax': 'ARGMAX',
    }

    return b, appvar_blobs


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Build NEOCHAT .8xp for Ti-84 Plus CE (24-bit native)')
    parser.add_argument('--model', '-m', type=str, default='model.npz',
                        help='Model file to load')
    parser.add_argument('--output', '-o', type=str,
                        default=os.path.join(os.path.dirname(__file__), 'bin', 'NEOCHAT.8xp'),
                        help='Output .8xp file')
    parser.add_argument('--name', '-n', type=str, default=None,
                        help='Program name on calculator (max 8 chars, default: from output filename)')
    parser.add_argument('--debug', action='store_true',
                        help='Build an instrumented DEBUG program (NEOCHAT_DBG.8xp) that '
                             'dumps per-layer buffer checksums. Reuses the SAME weight '
                             'AppVars (NEOA-D); does not touch the production build.')
    args = parser.parse_args()

    # In --debug mode, default the output to NEOCHAT_DBG.8xp (unless the user
    # overrode -o) so the production NEOCHAT.8xp is never touched.
    default_output = os.path.join(os.path.dirname(__file__), 'bin', 'NEOCHAT.8xp')
    if args.debug and args.output == default_output:
        args.output = os.path.join(os.path.dirname(__file__), 'bin', 'NEOCHAT_DBG.8xp')

    # Derive program name from output filename if not specified. The on-calc
    # name is capped at 8 chars, so the debug build is launched as prgmNEOCDBG.
    if args.name:
        prog_name = args.name
    elif args.debug:
        prog_name = 'NEOCDBG'
    else:
        prog_name = os.path.splitext(os.path.basename(args.output))[0]

    print(f"Building NEOCHAT{' [DEBUG]' if args.debug else ''} (24-bit native) -> {args.output}...\n")

    b, appvar_blobs = build_autoreg(args.model, debug=args.debug)

    # Show key addresses
    print("\nKey addresses:")
    for name in ['START', 'GENERATE', 'FORWARD', 'LAYER', 'MULADD', 'ARGMAX',
                 'TOKENIZE', 'UPDATE_CTX', 'ENCODE_CTX', 'CLEAR_CTX',
                 'CHARTBL', 'TOKBUF', 'OUTBUF', 'PING', 'PONG']:
        if name in b.labels:
            print(f"  {name}: {b.labels[name]:06X}h")

    # Resolve labels and get raw code
    b.resolve()

    print(f"\nProgram: {len(b.code)} bytes ({len(b.code)/1024:.1f} KB)")

    # Report AppVar sizes.
    total_av = sum(len(d) for d in appvar_blobs.values())
    for av_name, av_data in appvar_blobs.items():
        print(f"AppVar {av_name}: {len(av_data):,} bytes")
    print(f"\nTotal weight data: {total_av:,} bytes ({total_av/1024:.1f} KB)")

    # Hard gate: refuse to ship a model that won't fit the calculator's RAM.
    # Runs BEFORE writing ANY output file so an over-budget model fails loudly and
    # cleanly (BudgetError) without leaving a half-written .8xp/.8xv behind. The
    # working buffers (TOKBUF/BUF_A/...) are emitted into the program image via
    # ds(), so len(b.code) already includes them — total_ram() must not add them
    # again.
    import sizes as _sizes
    _ram = _sizes.total_ram(len(b.code), total_av)
    print(f"Runtime RAM: ~{_ram/1024:.1f} KB "
          f"(program incl. buffers {len(b.code)/1024:.1f} + weights {total_av/1024:.1f}); "
          f"budget {_sizes.RAM_BUDGET_BYTES/1024:.0f} KB")
    _sizes.validate_or_raise({
        'program': len(b.code),
        'appvars': {n: len(d) for n, d in appvar_blobs.items()},
        'total_weight': total_av,
    })

    # Within budget: create the output dir and write the program .8xp.
    output_dir = os.path.dirname(args.output) or '.'
    os.makedirs(output_dir, exist_ok=True)
    xp_data = build_8xp(b.code, prog_name)
    with open(args.output, 'wb') as f:
        f.write(xp_data)

    # Write the AppVar files. The DEBUG build reuses the SAME weight AppVars as
    # production (NEOA-D), so we skip re-writing the .8xv files to avoid touching
    # production artifacts.
    if not args.debug:
        for av_name, av_data in appvar_blobs.items():
            xv_data = build_8xv(av_data, av_name)
            av_path = os.path.join(output_dir, f'{av_name}.8xv')
            with open(av_path, 'wb') as f:
                f.write(xv_data)
            print(f"  wrote {av_path}")

    print(f"Program name: {prog_name.upper()[:8]}")
    print(f"Saved to {args.output}")
    if args.debug:
        print(f"\n[DEBUG] Transfer to calculator:")
        print(f"  {args.output}   (the instrumented program)")
        print(f"  plus the existing weight AppVars: " + ', '.join(f'{n}.8xv' for n in appvar_blobs))
        print(f"Run: Asm(prgm{prog_name.upper()[:8]})")
    else:
        print(f"\nTransfer ALL files to calculator:")
        print(f"  {args.output}")
        for av_name in appvar_blobs:
            print(f"  {av_name}.8xv")
        print(f"Run: Asm(prgm{prog_name.upper()[:8]})")
