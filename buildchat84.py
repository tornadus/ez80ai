#!/usr/bin/env python3
"""
Build NEOCHAT -- eZ80 native 24-bit accumulator neural net for Ti-84 Plus CE (.8xp)

Features:
  - 43-character charset (digits, punctuation)
  - Dual bias sets for output layer (fc4_bias / fc4_bias_start)
  - D_J Context Attention (from JAM XL): 32-slot key/value memory
  - Confidence gating: "JUST ASK" on low first-char margin
  - EOS suppression when margin is low and generation is short
  - Repeat detection: 3 consecutive identical chars -> fallback
  - Progressive EOL bias after position 17

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
from loadmodel import load_model_params

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

# Scan code -> ASCII mapping table (same as snarkchat: letters + space + punctuation)
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


def pack_2bit_weights(weights: np.ndarray) -> bytes:
    """Pack 2-bit weights: 4 per byte, LSB first (same as ZX Spectrum version)"""
    flat = weights.flatten()
    mapped = np.clip(flat + 2, 0, 3).astype(np.uint8)

    packed = []
    for i in range(0, len(mapped), 4):
        chunk = mapped[i:i+4]
        if len(chunk) < 4:
            chunk = np.pad(chunk, (0, 4 - len(chunk)), constant_values=2)
        byte = int(chunk[0]) | (int(chunk[1]) << 2) | (int(chunk[2]) << 4) | (int(chunk[3]) << 6)
        packed.append(byte)

    return bytes(packed)


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


def build_autoreg(model_path: str = 'model.npz'):
    """Build the autoregressive inference for Ti-84 Plus CE"""

    # Load model
    print(f"Loading model from {model_path}...")
    params, arch, charset = load_model_params(model_path)

    eos_idx = len(charset) - 1
    num_chars = len(charset)
    print(f"Charset ({num_chars} chars): {repr(charset[:-1])} + EOS")
    assert num_chars == 43, f"Expected 43-char charset, got {num_chars}"
    assert eos_idx == 42, f"Expected EOS at index 42, got {eos_idx}"

    # Discover layers (exclude fc4_bias_start which is a secondary bias, not a layer)
    layer_keys = [k for k in params.keys() if k.endswith('_weight')]
    layer_names = sorted(k.replace('_weight', '') for k in layer_keys)
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

    print(f"Architecture: {' -> '.join(map(str, layer_sizes))}")
    print(f"Input: {input_size} (128 query + 128 context)")
    print(f"Output: {output_size} characters")

    # Check for dual bias
    has_dual_bias = 'fc4_bias_start' in params
    if has_dual_bias:
        print(f"Dual bias: fc4_bias (rest) + fc4_bias_start (first 3 chars)")
    else:
        print("WARNING: No dual bias found in model")

    # Pack weights and biases per original layer
    packed_weights = []
    biases = []
    for name in layer_names:
        packed_weights.append(pack_2bit_weights(params[f'{name}_weight']))
        biases.append(params[f'{name}_bias'])

    # === Build AppVar data blobs ===
    # Split Layer 2 (512x512, 66KB) into two 256-output halves

    def pack_bias_bytes(bias_arr):
        """Pack bias array as little-endian uint16 bytes (two's complement)."""
        data = bytearray()
        for v in bias_arr:
            val = int(v) & 0xFFFF
            data.extend(struct.pack('<H', val))
        return bytes(data)

    # Layer 2 split: first 256 outputs and last 256 outputs
    w2 = params[f'{layer_names[1]}_weight']  # Shape: (512, 512)
    b2 = params[f'{layer_names[1]}_bias']    # Shape: (512,)
    w2a = w2[:256, :]   # First 256 output neurons
    w2b = w2[256:, :]   # Last 256 output neurons
    b2a = b2[:256]
    b2b = b2[256:]

    # NEOD: L3 weights + L3 biases + L4 weights + L4 bias (rest) + L4 bias_start
    # NOTE: L4 biases are multiplied by 4 to compensate for the LAYER routine's
    # divide-by-4 step.  In Python _forward_int, the output layer adds bias AFTER
    # the divide, but the eZ80 LAYER routine adds bias BEFORE dividing.
    # Pre-scaling by 4 ensures: (matmul + 4*bias) >> 2 == (matmul >> 2) + bias
    l3_w_packed = pack_2bit_weights(params[f'{layer_names[2]}_weight'])
    l3_b_packed = pack_bias_bytes(params[f'{layer_names[2]}_bias'])
    l4_w_packed = pack_2bit_weights(params[f'{layer_names[3]}_weight'])
    l4_b_packed = pack_bias_bytes(params[f'{layer_names[3]}_bias'] * 4)
    l4_bs_packed = pack_bias_bytes(params['fc4_bias_start'] * 4) if has_dual_bias else l4_b_packed

    appvar_blobs = {
        'NEOA': pack_2bit_weights(params[f'{layer_names[0]}_weight']) +
                 pack_bias_bytes(params[f'{layer_names[0]}_bias']),
        'NEOB': pack_2bit_weights(w2a) + pack_bias_bytes(b2a),
        'NEOC': pack_2bit_weights(w2b) + pack_bias_bytes(b2b),
        'NEOD': l3_w_packed + l3_b_packed + l4_w_packed + l4_b_packed + l4_bs_packed,
    }

    # Compute weight/bias offsets within each AppVar
    l1_wsize = len(pack_2bit_weights(params[f'{layer_names[0]}_weight']))
    l2a_wsize = len(pack_2bit_weights(w2a))
    l2b_wsize = len(pack_2bit_weights(w2b))
    l3_wsize = len(l3_w_packed)
    l3_bsize = len(l3_b_packed)
    l4_woffset = l3_wsize + l3_bsize  # L4 weights start in NEOD
    l4_wsize = len(l4_w_packed)
    l4_boffset = l4_woffset + l4_wsize  # L4 rest bias start in NEOD
    l4_bsize = len(l4_b_packed)
    l4_bs_offset = l4_boffset + l4_bsize  # L4 start bias in NEOD

    for name, blob in appvar_blobs.items():
        print(f"  AppVar {name}: {len(blob):,} bytes ({len(blob)/1024:.1f} KB)")

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
    for i, av_name in enumerate(APPVAR_NAMES):
        b.ld_hl_label(f'AVNAME{i}')
        b.call_addr(TI_Mov9ToOP1)
        b.call_addr(TI_ChkFindSym)
        b.jp_c('AV_ERR')              # carry set = not found
        b.or_a()
        b.jr_z(f'AV_RAM{i}')

        # Archived -- unarchive it, then re-find
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
        b.ld_mem_label_de(f'AVPTR{i}')

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

    # Clear attention state on startup
    b.call('CTX_CLEAR')

    # Enter chat mode
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
    b.call('CTX_CLEAR')       # Clear attention on screen clear
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
    for i, av_name in enumerate(APPVAR_NAMES):
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

    # Check if buffer full
    b.ld_b_a()
    b.ld_a_mem_label('INPLEN')
    b.cp_n(62)
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
    # CLEAR key: reset input, clear screen, clear attention state
    b.xor_a()
    b.ld_mem_label_a('INPLEN')
    b.ld_a_n(1)
    b.ld_mem_label_a('RI_CLEAR_FLAG')
    b.call_addr(TI_ClrScrn)
    b.call_addr(TI_HomeUp)
    b.call('CTX_CLEAR')       # Clear attention on CLEAR key
    b.jp('RI_DONE')

    b.label('RI_MODE')
    b.ld_a_n(1)
    b.ld_mem_label_a('RI_FLAG')
    b.xor_a()
    b.ld_mem_label_a('INPLEN')

    b.label('RI_DONE')
    b.call_addr(TI_NewLine)
    b.ret()

    # === GENERATE: Main generation loop ===
    b.label('GENERATE')
    b.ld_a_n(MAX_OUTPUT_LEN)
    b.ld_mem_label_a('GENCNT')
    b.xor_a()
    b.ld_mem_label_a('GENPOS')     # Generation position counter (0-based)
    b.ld_mem_label_a('LASTCH')     # Last character for repeat detection
    b.ld_mem_label_a('REPCNT')     # Repeat count

    # Write to D_J attention before generation
    b.call('CTX_WRITE')

    b.label('GENLOOP')
    # 5 virtual layers: L1 -> RELU1 -> L2A+L2B -> RELU2 -> L3 -> RELU3 -> L4
    b.call('LAYER1')     # 256->512, output to BUF_A
    b.call('RELU1')      # ReLU BUF_A (512 values)

    # D_J Context Attention: mix context after RELU1, before LAYER2A
    b.call('CTX_ATTEND')

    b.call('LAYER2A')    # 512->256, first half to BUF_B[0..255]
    b.call('LAYER2B')    # 512->256, second half to BUF_B[256..511]
    b.call('RELU2')      # ReLU BUF_B (512 values)
    b.call('LAYER3')     # 512->256, output to BUF_A
    b.call('RELU3')      # ReLU BUF_A (256 values)

    # Select LAYER4 variant based on GENPOS
    b.ld_a_mem_label('GENPOS')
    b.cp_n(3)
    b.jr_nc('GEN_L4_REST')
    b.call('LAYER4_START')   # first 3 chars: use start bias
    b.jr('GEN_L4_DONE')
    b.label('GEN_L4_REST')
    b.call('LAYER4_REST')    # subsequent: use rest bias
    b.label('GEN_L4_DONE')

    # === Progressive EOL bias ===
    # If GENPOS > 17: add 16 + (GENPOS - 17) * 16 to EOS logit
    b.ld_a_mem_label('GENPOS')
    b.cp_n(18)
    b.jr_c('GEN_NO_EOL_BIAS')
    # A = GENPOS >= 18
    b.sub_n(17)           # A = GENPOS - 17 (1, 2, 3, ...)
    # Multiply by 16: shift left 4
    b.sla_a()             # SLA A
    b.sla_a()             # SLA A
    b.sla_a()             # SLA A
    b.sla_a()             # SLA A
    b.add_a_n(16)         # A = 16 + (GENPOS-17)*16
    # Add to OUTBUF[EOS_IDX * 2] (16-bit, little-endian)
    b.ld_hl_label('OUTBUF')
    b.ld_de_nn(eos_idx * 2)
    b.add_hl_de()
    # Load current EOS logit
    b.ld_e_hl()           # low byte
    b.inc_hl()
    b.ld_d_hl()           # high byte
    # Add A (unsigned) to DE (16-bit): E += A, D += carry
    b.push_hl()
    b.ld_c_a()
    b.ld_b_n(0)
    b.ex_de_hl()
    b.add_hl_bc_16()
    b.ex_de_hl()
    # Store back
    b.pop_hl()
    b.ld_a_d()
    b.ld_hl_a()           # high byte
    b.dec_hl()
    b.ld_a_e()
    b.ld_hl_a()           # low byte
    b.label('GEN_NO_EOL_BIAS')

    # === ARGMAX with second-best tracking ===
    b.call('ARGMAX')

    # === Confidence gating ===
    # If GENPOS == 0 and margin < 3: UNSURE
    b.ld_a_mem_label('GENPOS')
    b.or_a()
    b.jr_nz('GEN_CONF_OK')
    # First char: check margin (best - second)
    # 16-bit margin: MAXL:MAXH - SECL:SECH
    b.ld_a_mem_label('MAXL')
    b.ld_hl_label('SECL')
    b.sub_hl_ind()                       # A = MAXL - SECL (low byte of difference)
    b.ld_mem_label_a('MARGIN_L')
    b.ld_a_mem_label('MAXH')
    b.ld_hl_label('SECH')
    b.sbc_a_hl()                         # SBC A, (HL) -- high byte with borrow
    b.ld_mem_label_a('MARGIN_H')
    # If high byte != 0, margin is large -> OK
    b.or_a()
    b.jr_nz('GEN_CONF_OK')
    b.ld_a_mem_label('MARGIN_L')
    b.cp_n(3)
    b.jr_nc('GEN_CONF_OK')
    # Margin < 3: unsure
    b.jp('GEN_UNSURE')

    b.label('GEN_CONF_OK')

    # === EOS check with margin guard ===
    b.ld_a_mem_label('RESULT')
    b.cp_n(eos_idx)
    b.jr_nz('GEN_NOT_EOS')

    # EOS is best. If GENPOS < 15 and margin < 5: use second best
    b.ld_a_mem_label('GENPOS')
    b.cp_n(15)
    b.jr_nc('GEN_EOS_ACCEPT')
    # Check margin
    b.ld_a_mem_label('MAXL')
    b.ld_hl_label('SECL')
    b.sub_hl_ind()
    b.ld_mem_label_a('MARGIN_L')
    b.ld_a_mem_label('MAXH')
    b.ld_hl_label('SECH')
    b.sbc_a_hl()                         # SBC A, (HL)
    b.ld_mem_label_a('MARGIN_H')
    b.or_a()
    b.jr_nz('GEN_EOS_ACCEPT')    # high byte nonzero = large margin
    b.ld_a_mem_label('MARGIN_L')
    b.cp_n(5)
    b.jr_nc('GEN_EOS_ACCEPT')    # margin >= 5: accept
    # Suppress EOS: use second best
    b.ld_a_mem_label('SECI')
    b.ld_mem_label_a('RESULT')
    # Check if second best is also EOS (unlikely but safe)
    b.cp_n(eos_idx)
    b.jr_z('GEN_EOS_ACCEPT')
    b.jr('GEN_NOT_EOS')

    b.label('GEN_EOS_ACCEPT')
    b.ret()     # EOS accepted: return from GENERATE

    b.label('GEN_NOT_EOS')

    # === Repeat detection ===
    b.ld_a_mem_label('RESULT')
    b.ld_hl_label('LASTCH')
    b.cp_hl()
    b.jr_nz('GEN_NEWCH')
    # Same character: increment repeat count
    b.ld_a_mem_label('REPCNT')
    b.inc_a()
    b.ld_mem_label_a('REPCNT')
    b.cp_n(3)
    b.jr_nc('GEN_FALLBACK')     # 3+ repeats -> fallback
    b.jr('GEN_PRINT')

    b.label('GEN_NEWCH')
    b.ld_a_mem_label('RESULT')
    b.ld_mem_label_a('LASTCH')
    b.ld_a_n(1)
    b.ld_mem_label_a('REPCNT')

    b.label('GEN_PRINT')
    # Print character -- restore IY for TI-OS, set inverse mode
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

    # Loop
    b.ld_a_mem_label('GENCNT')
    b.dec_a()
    b.ld_mem_label_a('GENCNT')
    b.jp_nz('GENLOOP')
    b.ret()

    # === GEN_UNSURE: Model unsure about first char ===
    b.label('GEN_UNSURE')
    b.ld_iy_mem_label('SAVED_IY')
    b.ld_hl_nn(0xD00085)
    b.ld_a_hl()
    b.or_n(0x08)              # OR 0x08
    b.ld_hl_a()
    b.ld_hl_label('UNSURE_MSG')
    b.label('GU_LOOP')
    b.ld_a_hl()
    b.or_a()
    b.jr_z('GU_DONE')
    b.push_hl()
    b.call_addr(TI_PutC)
    b.pop_hl()
    b.inc_hl()
    b.jr('GU_LOOP')
    b.label('GU_DONE')
    b.ret()

    # === GEN_FALLBACK: Repeat detected, print fallback message ===
    b.label('GEN_FALLBACK')
    b.ld_iy_mem_label('SAVED_IY')
    b.ld_hl_nn(0xD00085)
    b.ld_a_hl()
    b.or_n(0x08)              # OR 0x08
    b.ld_hl_a()
    # Pick one of 3 fallback messages based on GENPOS as pseudo-random seed
    b.ld_a_mem_label('GENPOS')
    b.and_n(0x03)
    b.cp_n(3)
    b.jr_c('GF_PICK')
    b.xor_a()
    b.label('GF_PICK')
    # A = 0, 1, or 2: select message
    b.or_a()
    b.jr_nz('GF_NOT0')
    b.ld_hl_label('FB_MSG0')
    b.jr('GF_PRINT')
    b.label('GF_NOT0')
    b.cp_n(1)
    b.jr_nz('GF_NOT1')
    b.ld_hl_label('FB_MSG1')
    b.jr('GF_PRINT')
    b.label('GF_NOT1')
    b.ld_hl_label('FB_MSG2')
    b.label('GF_PRINT')
    b.ld_a_hl()
    b.or_a()
    b.jr_z('GF_DONE')
    b.push_hl()
    b.call_addr(TI_PutC)
    b.pop_hl()
    b.inc_hl()
    b.jr('GF_PRINT')
    b.label('GF_DONE')
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
    b.ld_bc_nn(7)
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
    b.ld_de_nn(7)
    b.add_hl_de()
    b.ld_hl_a()

    b.call('ENCODE_CTX')
    b.ret()

    # === ENCODE_CTX: Encode CTXCHARS into context buckets ===
    b.label('ENCODE_CTX')
    # Clear context buckets (last 128 of TOKBUF)
    b.ld_hl_label('TOKBUF')
    b.ld_de_nn(256)  # 128 buckets * 2 bytes
    b.add_hl_de()
    b.push_hl()
    b.pop_de()
    b.inc_de()
    b.xor_a()
    b.ld_hl_a()
    b.ld_bc_nn(255)  # 128*2 - 1
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
    b.ld_a_n(9)
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
    b.cp_n(4)
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

    # bucket = (hash & 127) + 128
    b.ld_a_l()
    b.and_n(127)

    # Add to bucket (context is at TOKBUF + 256)
    b.ld_hl_nn(0)
    b.ld_l_a()
    b.add_hl_hl()
    b.ld_de_label('TOKBUF')
    b.ld_bc_nn(256)
    b.add_hl_bc()
    b.add_hl_de()

    # Increment bucket value by 32
    b.ld_e_hl()
    b.inc_hl()
    b.ld_d_hl()
    b.push_hl()
    b.ld_hl_nn_16(32)
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
    for _ in range(8):
        b.ld_hl_a()
        b.inc_hl()

    b.jp('ENCODE_CTX')

    # === D_J Context Attention Routines ===

    # --- CTX_CLEAR: Zero all attention state ---
    b.label('CTX_CLEAR')
    b.ld_hl_label('CTX_KEY')
    b.push_hl()
    b.pop_de()
    b.inc_de()
    b.xor_a()
    b.ld_hl_a()
    # Total bytes: 128 (key) + 128 (val) + 32 (age) + 4 (query) + 1 + 1 + 1 = 295
    b.ld_bc_nn(294)    # 295 - 1
    b.ldir()
    b.ret()

    # --- CTX_ATTEND: After RELU1, before LAYER2A ---
    # Build 4-dim query from TOKBUF[0:3] via D_J rotation
    # q[0]=thash[3]-thash[2], q[1]=-thash[2], q[2]=thash[0]-thash[2], q[3]=thash[1]-thash[2]
    # Then dot-product scan over 32 slots, add best value to BUF_A[0:3]
    b.label('CTX_ATTEND')
    # Load TOKBUF[0..3] as 16-bit values but use low byte only for query construction
    # thash values are in TOKBUF (first 128 buckets, 16-bit each)
    # TOKBUF[i] is at TOKBUF + i*2 (16-bit little-endian)
    # We use the low byte of TOKBUF[0..3] as the hash values

    # Load thash[2] low byte into B (used by all query components)
    b.ld_hl_label('TOKBUF')
    b.ld_de_nn(4)           # offset to TOKBUF[2]
    b.add_hl_de()
    b.ld_a_hl()             # A = thash[2] low byte
    b.ld_b_a()              # B = thash[2] (saved for reuse)

    # q[0] = thash[3] - thash[2]
    b.ld_hl_label('TOKBUF')
    b.ld_de_nn(6)           # offset to TOKBUF[3]
    b.add_hl_de()
    b.ld_a_hl()             # A = thash[3]
    b.sub_b()               # SUB B  (A = thash[3] - thash[2])
    b.ld_hl_label('CTX_QUERY')
    b.ld_hl_a()             # q[0]

    # q[1] = -thash[2]
    b.xor_a()
    b.sub_b()               # SUB B  (A = 0 - thash[2])
    b.ld_hl_label('CTX_QUERY')
    b.inc_hl()
    b.ld_hl_a()             # q[1]

    # q[2] = thash[0] - thash[2]
    b.ld_hl_label('TOKBUF')
    b.ld_a_hl()             # A = thash[0]
    b.sub_b()               # SUB B
    b.ld_hl_label('CTX_QUERY')
    b.ld_de_nn(2)
    b.add_hl_de()
    b.ld_hl_a()             # q[2]

    # q[3] = thash[1] - thash[2]
    b.ld_hl_label('TOKBUF')
    b.ld_de_nn(2)           # offset to TOKBUF[1]
    b.add_hl_de()
    b.ld_a_hl()             # A = thash[1]
    b.sub_b()               # SUB B
    b.ld_hl_label('CTX_QUERY')
    b.ld_de_nn(3)
    b.add_hl_de()
    b.ld_hl_a()             # q[3]

    # --- Dot product scan over 32 slots ---
    b.xor_a()
    b.ld_mem_label_a('CTX_BEST')    # best slot = 0
    b.ld_a_n(0x80)                   # worst possible score (-128)
    b.ld_mem_label_a('CTX_SCORE')

    b.xor_a()
    b.ld_mem_label_a('CA_SLOT')     # slot counter

    b.label('CA_SLOT_LOOP')
    # Compute dot product for current slot
    # key base = CTX_KEY + slot*4
    b.ld_a_mem_label('CA_SLOT')
    b.sla_a()             # SLA A (*2)
    b.sla_a()             # SLA A (*4)
    b.ld_hl_label('CTX_KEY')
    b.ld_de_nn(0)
    b.ld_e_a()
    b.add_hl_de()         # HL = CTX_KEY + slot*4
    b.ld_mem_label_hl('CA_KPTR')    # save key pointer

    # acc = 0 (dot product accumulator, 8-bit signed)
    b.xor_a()
    b.ld_mem_label_a('CA_ACC')

    # For each of 4 dimensions: acc += query[i] * key[i]
    # Using signed 8-bit multiply approximation (repeated addition)
    b.ld_a_n(0)
    b.ld_mem_label_a('CA_DIM')

    b.label('CA_DIM_LOOP')
    # Load query[dim]
    b.ld_hl_label('CTX_QUERY')
    b.ld_a_mem_label('CA_DIM')
    b.ld_de_nn(0)
    b.ld_e_a()
    b.add_hl_de()
    b.ld_a_hl()           # A = query[dim] (signed)
    b.ld_mem_label_a('CA_Q')

    # Load key[dim]
    b.ld_hl_mem_label('CA_KPTR')
    b.ld_a_mem_label('CA_DIM')
    b.ld_de_nn(0)
    b.ld_e_a()
    b.add_hl_de()
    b.ld_a_hl()           # A = key[dim] (signed, [-3..3])
    b.ld_mem_label_a('CA_K')

    # Signed 8-bit multiply: product = query * key
    # Both values are small, so we use repeated addition
    b.call('DOT_MUL')

    b.ld_a_mem_label('CA_DIM')
    b.inc_a()
    b.ld_mem_label_a('CA_DIM')
    b.cp_n(4)
    b.jr_c('CA_DIM_LOOP')

    # Compare accumulated score with best (signed 8-bit)
    b.ld_a_mem_label('CA_ACC')
    b.ld_hl_label('CTX_SCORE')
    # Signed compare: A > (HL)?
    b.ld_b_a()            # save score in B
    b.sub_hl_ind()        # A = score - best_score (may overflow)
    # Handle signed overflow
    b.jp_m('CA_NOT_BEST')
    # score >= best_score (or overflow handled): update
    b.ld_a_b()
    b.ld_mem_label_a('CTX_SCORE')
    b.ld_a_mem_label('CA_SLOT')
    b.ld_mem_label_a('CTX_BEST')

    b.label('CA_NOT_BEST')
    b.ld_a_mem_label('CA_SLOT')
    b.inc_a()
    b.ld_mem_label_a('CA_SLOT')
    b.cp_n(32)            # 32 slots
    b.jp_nz('CA_SLOT_LOOP')

    # --- Add best slot's value to BUF_A[0:3] if score > 0 ---
    b.ld_a_mem_label('CTX_SCORE')
    b.or_a()
    b.ret_z()             # score == 0: no match
    b.jp_m('CA_RET')      # score < 0: no match

    # value base = CTX_VAL + best*4
    b.ld_a_mem_label('CTX_BEST')
    b.sla_a()             # SLA A (*2)
    b.sla_a()             # SLA A (*4)
    b.ld_hl_label('CTX_VAL')
    b.ld_de_nn(0)
    b.ld_e_a()
    b.add_hl_de()         # HL = CTX_VAL + best*4
    b.ld_mem_label_hl('CA_VPTR')

    # Add 4 values to BUF_A[0:3] (8-bit values, sign-extended to 16-bit, add to 16-bit BUF_A)
    b.ld_a_n(0)
    b.ld_mem_label_a('CA_DIM')

    b.label('CA_ADD_LOOP')
    # Load 8-bit value from CTX_VAL
    b.ld_hl_mem_label('CA_VPTR')
    b.ld_a_mem_label('CA_DIM')
    b.ld_de_nn(0)
    b.ld_e_a()
    b.add_hl_de()
    b.ld_a_hl()           # A = value (8-bit, treated as unsigned 0-3 from quant)
    # Sign extend to 16-bit (these are unsigned 0-3 values, so just zero-extend)
    b.ld_e_a()
    b.ld_d_n(0)           # DE = 16-bit value

    # BUF_A[dim] is 16-bit at BUF_A + dim*2
    b.ld_hl_label('BUF_A')
    b.ld_a_mem_label('CA_DIM')
    b.sla_a()             # SLA A (*2 for word offset)
    b.ld_bc_nn(0)
    b.ld_c_a()
    b.add_hl_bc()         # HL = BUF_A + dim*2

    # Load current BUF_A value (16-bit)
    b.push_hl()           # save pointer
    b.ld_c_hl()           # LD C, (HL) -- low byte
    b.inc_hl()
    b.ld_b_hl()           # B = high byte

    # Add DE to BC with saturation at 0x7FFF
    b.push_hl()
    b.ld_hl_nn_16(0)
    b.ld_h_b()            # LD H, B
    b.ld_l_c()            # LD L, C
    b.add_hl_de_16()      # HL = old + value (16-bit)
    # Check for overflow (positive saturation)
    b.bit_7_d()           # Was the addend negative? No (always 0-3), skip neg check
    # Check if result went negative (overflow)
    b.ld_a_h()
    b.and_n(0x80)
    b.jr_z('CA_NO_SAT')
    # Saturated: check if both operands were positive
    b.ld_a_b()            # old high byte
    b.and_n(0x80)
    b.jr_nz('CA_NO_SAT')  # old was negative, no saturation needed
    b.ld_hl_nn_16(0x7FFF)
    b.label('CA_NO_SAT')
    b.ld_b_h()            # LD B, H
    b.ld_c_l()            # LD C, L
    b.pop_hl()            # HL = BUF_A + dim*2 + 1
    b.ld_a_b()
    b.ld_hl_a()           # store high byte
    b.pop_hl()            # HL = BUF_A + dim*2
    b.ld_a_c()
    b.ld_hl_a()           # store low byte

    b.ld_a_mem_label('CA_DIM')
    b.inc_a()
    b.ld_mem_label_a('CA_DIM')
    b.cp_n(4)
    b.jr_c('CA_ADD_LOOP')

    b.label('CA_RET')
    b.ret()

    # --- DOT_MUL: CA_ACC += CA_Q * CA_K (signed 8-bit multiply) ---
    b.label('DOT_MUL')
    b.ld_a_mem_label('CA_Q')
    b.or_a()
    b.ret_z()             # query == 0: skip
    b.ld_a_mem_label('CA_K')
    b.or_a()
    b.ret_z()             # key == 0: skip

    # Determine sign of product
    b.ld_a_mem_label('CA_Q')
    b.ld_hl_label('CA_K')
    b.xor_hl()            # XOR (HL)
    b.and_n(0x80)
    b.ld_mem_label_a('CA_SIGN')   # bit 7 = sign of product

    # abs(query)
    b.ld_a_mem_label('CA_Q')
    b.or_a()
    b.jp_m('DM_NEGQ')
    b.jr('DM_ABSQ_OK')
    b.label('DM_NEGQ')
    b.cpl()               # CPL (complement A)
    b.inc_a()
    b.label('DM_ABSQ_OK')
    b.ld_mem_label_a('CA_ABSQ')

    # abs(key)
    b.ld_a_mem_label('CA_K')
    b.or_a()
    b.jp_m('DM_NEGK')
    b.jr('DM_ABSK_OK')
    b.label('DM_NEGK')
    b.cpl()               # CPL
    b.inc_a()
    b.label('DM_ABSK_OK')

    # Multiply by repeated addition: product = abs(key) * abs(query)
    # abs(key) is small (0-3), use as loop count
    b.ld_b_a()            # B = abs(key) (loop count)
    b.ld_hl_label('CA_ABSQ')
    b.xor_a()             # A = 0 (accumulator)
    b.label('DM_MUL_LP')
    b.add_a_hl()          # ADD A, (HL) -- repeated addition
    b.djnz('DM_MUL_LP')

    # A = |product|. Apply sign.
    b.ld_c_a()            # save |product|
    b.ld_a_mem_label('CA_SIGN')
    b.or_a()
    b.jr_z('DM_POS')
    # Negative product: CA_ACC -= |product|
    b.ld_a_mem_label('CA_ACC')
    b.sub_c()             # SUB C
    b.ld_mem_label_a('CA_ACC')
    b.ret()
    b.label('DM_POS')
    # Positive product: CA_ACC += |product|
    b.ld_a_mem_label('CA_ACC')
    b.add_a_c()           # ADD A, C
    b.ld_mem_label_a('CA_ACC')
    b.ret()

    # --- CTX_WRITE: Store current context in oldest slot ---
    b.label('CTX_WRITE')
    # Find oldest slot (highest age)
    b.xor_a()
    b.ld_mem_label_a('CTX_WSLOT')
    b.ld_hl_label('CTX_AGE')
    b.ld_a_hl()
    b.ld_mem_label_a('CTX_SCORE')  # reuse as max_age

    b.ld_a_n(1)
    b.ld_mem_label_a('CW_IDX')

    b.label('CW_FIND')
    b.ld_hl_label('CTX_AGE')
    b.ld_a_mem_label('CW_IDX')
    b.ld_de_nn(0)
    b.ld_e_a()
    b.add_hl_de()
    b.ld_a_hl()                    # A = age[idx]
    b.ld_hl_label('CTX_SCORE')
    b.cp_hl()                      # compare with max age
    b.jr_c('CW_NOT_OLD')           # if age < max, skip
    b.ld_mem_label_a('CTX_SCORE')
    b.ld_a_mem_label('CW_IDX')
    b.ld_mem_label_a('CTX_WSLOT')
    b.label('CW_NOT_OLD')
    b.ld_a_mem_label('CW_IDX')
    b.inc_a()
    b.ld_mem_label_a('CW_IDX')
    b.cp_n(32)
    b.jr_c('CW_FIND')

    # Write key: query values clamped to [-3, 3]
    b.ld_a_mem_label('CTX_WSLOT')
    b.sla_a()             # SLA A (*2)
    b.sla_a()             # SLA A (*4)
    b.ld_hl_label('CTX_KEY')
    b.ld_de_nn(0)
    b.ld_e_a()
    b.add_hl_de()         # HL = CTX_KEY + slot*4
    b.ld_mem_label_hl('CW_KPTR')

    # Write 4 query values clamped to [-3, 3]
    b.ld_a_n(0)
    b.ld_mem_label_a('CW_DIM')

    b.label('CW_KEY_LOOP')
    b.ld_hl_label('CTX_QUERY')
    b.ld_a_mem_label('CW_DIM')
    b.ld_de_nn(0)
    b.ld_e_a()
    b.add_hl_de()
    b.ld_a_hl()           # A = query[dim] (signed)
    # Clamp to [-3, 3]
    b.call('CLAMP_S3')
    # Store to key
    b.push_af()
    b.ld_hl_mem_label('CW_KPTR')
    b.ld_a_mem_label('CW_DIM')
    b.ld_de_nn(0)
    b.ld_e_a()
    b.add_hl_de()
    b.pop_af()
    b.ld_hl_a()

    b.ld_a_mem_label('CW_DIM')
    b.inc_a()
    b.ld_mem_label_a('CW_DIM')
    b.cp_n(4)
    b.jr_c('CW_KEY_LOOP')

    # Write value: quantized BUF_A[0:3] >> 6 (giving 0-3 range)
    b.ld_a_mem_label('CTX_WSLOT')
    b.sla_a()             # SLA A (*2)
    b.sla_a()             # SLA A (*4)
    b.ld_hl_label('CTX_VAL')
    b.ld_de_nn(0)
    b.ld_e_a()
    b.add_hl_de()         # HL = CTX_VAL + slot*4
    b.ld_mem_label_hl('CW_VPTR')

    b.ld_a_n(0)
    b.ld_mem_label_a('CW_DIM')

    b.label('CW_VAL_LOOP')
    # Load BUF_A[dim] high byte (16-bit value, take high byte >> 6 for 0-3 range)
    b.ld_hl_label('BUF_A')
    b.ld_a_mem_label('CW_DIM')
    b.sla_a()             # SLA A (*2 for word)
    b.inc_a()             # +1 for high byte
    b.ld_de_nn(0)
    b.ld_e_a()
    b.add_hl_de()
    b.ld_a_hl()           # A = high byte of BUF_A[dim]
    # >> 6: shift right 6 times to get 0-3
    b.rrca()
    b.rrca()
    b.rrca()
    b.rrca()
    b.rrca()
    b.rrca()
    b.and_n(0x03)         # mask to 2 bits

    # Store to value slot
    b.push_af()
    b.ld_hl_mem_label('CW_VPTR')
    b.ld_a_mem_label('CW_DIM')
    b.ld_de_nn(0)
    b.ld_e_a()
    b.add_hl_de()
    b.pop_af()
    b.ld_hl_a()

    b.ld_a_mem_label('CW_DIM')
    b.inc_a()
    b.ld_mem_label_a('CW_DIM')
    b.cp_n(4)
    b.jr_c('CW_VAL_LOOP')

    # Reset age of written slot
    b.ld_hl_label('CTX_AGE')
    b.ld_a_mem_label('CTX_WSLOT')
    b.ld_de_nn(0)
    b.ld_e_a()
    b.add_hl_de()
    b.xor_a()
    b.ld_hl_a()

    # Increment all other ages (saturate at 255)
    b.xor_a()
    b.ld_mem_label_a('CW_IDX')
    b.label('CW_AGE_LOOP')
    b.ld_a_mem_label('CW_IDX')
    b.ld_hl_label('CTX_WSLOT')
    b.cp_hl()
    b.jr_z('CW_AGE_SKIP')
    # Load age
    b.ld_hl_label('CTX_AGE')
    b.ld_de_nn(0)
    b.ld_e_a()
    b.add_hl_de()
    b.ld_a_hl()
    b.cp_n(255)
    b.jr_z('CW_AGE_SKIP')
    b.inc_a()
    b.ld_hl_a()
    b.label('CW_AGE_SKIP')
    b.ld_a_mem_label('CW_IDX')
    b.inc_a()
    b.ld_mem_label_a('CW_IDX')
    b.cp_n(32)
    b.jr_c('CW_AGE_LOOP')
    b.ret()

    # --- CLAMP_S3: Clamp signed A to [-3, 3] ---
    b.label('CLAMP_S3')
    b.or_a()
    b.jp_m('CS3_NEG')
    # Positive: clamp to 3
    b.cp_n(4)
    b.jr_c('CS3_DONE')
    b.ld_a_n(3)
    b.jr('CS3_DONE')
    b.label('CS3_NEG')
    # Negative: clamp to -3 (0xFD)
    b.cp_n(0xFD)
    b.jr_nc('CS3_DONE')   # A >= 0xFD (-3): already in range
    b.ld_a_n(0xFD)        # clamp to -3
    b.label('CS3_DONE')
    b.ret()

    # === Layer dispatch stubs ===
    def emit_layer_stub(label, av_idx, w_offset, b_offset, in_buf, out_buf,
                        in_size, out_size, fall_through=False):
        """Emit a layer dispatch stub that loads pointers from an AppVar."""
        b.label(label)
        # HL = weight pointer = AVPTR[av_idx] + w_offset
        b.ld_hl_mem_label(f'AVPTR{av_idx}')
        if w_offset > 0:
            b.ld_de_nn(w_offset)
            b.add_hl_de()
        # Save weight pointer, load bias pointer
        b.push_hl()
        b.ld_hl_mem_label(f'AVPTR{av_idx}')
        b.ld_de_nn(b_offset)
        b.add_hl_de()
        b.push_hl()
        b.pop_de()         # DE = bias pointer
        b.pop_hl()         # HL = weight pointer
        # Set buffers
        b.ld_ix_label(in_buf)
        b.ld_iy_label(out_buf)
        # Set dimensions (16-bit)
        b.push_hl()
        b.ld_hl_nn_16(out_size)
        b.ld_mem_label_hl_16('NEURCNT')
        b.ld_hl_nn_16(in_size)
        b.ld_mem_label_hl_16('INCNT')
        b.pop_hl()
        if not fall_through:
            b.jp('LAYER')

    # L1: 256->512 from NEOA (AVPTR0), weights at +0, biases at +l1_wsize
    emit_layer_stub('LAYER1', 0, 0, l1_wsize, 'TOKBUF', 'BUF_A', 256, 512)

    # L2A: 512->256 from NEOB (AVPTR1)
    emit_layer_stub('LAYER2A', 1, 0, l2a_wsize, 'BUF_A', 'BUF_B', 512, 256)

    # L2B: 512->256 from NEOC (AVPTR2)
    emit_layer_stub('LAYER2B', 2, 0, l2b_wsize, 'BUF_A', 'BUF_B_MID', 512, 256)

    # L3: 512->256 from NEOD (AVPTR3), weights at +0, biases at +l3_wsize
    emit_layer_stub('LAYER3', 3, 0, l3_wsize, 'BUF_B', 'BUF_A', 512, 256)

    # L4_REST: 256->43 from NEOD (AVPTR3), weights at l4_woffset, bias at l4_boffset
    emit_layer_stub('LAYER4_REST', 3, l4_woffset, l4_boffset,
                    'BUF_A', 'OUTBUF', 256, output_size)

    # L4_START: 256->43 from NEOD (AVPTR3), weights at l4_woffset, bias at l4_bs_offset
    emit_layer_stub('LAYER4_START', 3, l4_woffset, l4_bs_offset,
                    'BUF_A', 'OUTBUF', 256, output_size, fall_through=True)

    # === LAYER: Neural network layer computation ===
    # 24-bit native accumulator
    b.label('LAYER')
    b.ld_mem_label_hl('SAVW')
    b.ld_mem_label_de('SAVB')

    b.label('LNEUR')
    b.ld_hl_nn(0)
    b.ld_mem_label_hl('ACC')
    b.push_ix()
    b.pop_hl()
    b.ld_mem_label_hl('CURIN')
    b.ld_de_nn(0)
    b.ld_hl_mem_label('SAVW')
    b.push_hl()
    b.ld_hl_mem_label_16('INCNT')
    b.ld_mem_label_hl_16('WTCNT')
    b.pop_hl()
    b.ld_c_n(0)

    b.label('LWT')
    b.ld_a_c()
    b.and_n(0x03)
    b.jr_nz('LSAME')
    b.ld_hl_mem_label('SAVW')
    b.ld_a_hl()
    b.ld_mem_label_a('PACKED')
    b.inc_hl()
    b.ld_mem_label_hl('SAVW')

    # ZERO-SKIP: if packed byte is 0xAA, all 4 weights are zero
    b.cp_n(0xAA)
    b.jr_nz('LSAME')

    b.ld_hl_mem_label('CURIN')
    b.ld_de_nn(8)
    b.add_hl_de()
    b.ld_mem_label_hl('CURIN')
    b.ld_a_c()
    b.add_a_n(4)
    b.ld_c_a()
    b.push_hl()
    b.ld_hl_mem_label_16('WTCNT')
    b.dec_hl_16()
    b.dec_hl_16()
    b.dec_hl_16()
    b.dec_hl_16()
    b.ld_mem_label_hl_16('WTCNT')
    b.ld_a_h()
    b.or_l()
    b.pop_hl()
    b.jp_nz('LWT')
    b.jp('LWT_DONE')

    b.label('LSAME')
    b.ld_a_mem_label('PACKED')
    b.and_n(0x03)
    b.sub_n(2)
    b.ld_mem_label_a('WEIGHT')
    b.ld_a_mem_label('PACKED')
    b.rrca()
    b.rrca()
    b.ld_mem_label_a('PACKED')
    b.ld_hl_mem_label('CURIN')
    b.ld_e_hl()
    b.inc_hl()
    b.ld_d_hl()
    b.inc_hl()
    b.ld_mem_label_hl('CURIN')
    b.ld_a_mem_label('WEIGHT')
    b.call('MULADD')
    b.inc_c()
    b.push_hl()
    b.ld_hl_mem_label_16('WTCNT')
    b.dec_hl_16()
    b.ld_mem_label_hl_16('WTCNT')
    b.ld_a_h()
    b.or_l()
    b.pop_hl()
    b.jp_nz('LWT')

    b.label('LWT_DONE')

    # Post inner loop: add bias THEN divide by 4
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

    # Arithmetic right shift by 2 (divide by 4)
    for _ in range(2):
        b.sra_a()                     # SRA A
        b.rr_d()                      # RR D
        b.rr_e()                      # RR E

    # Store result to output buffer
    b.ld_iyd_e(0x00)                 # LD (IY+0), E
    b.ld_iyd_d(0x01)                 # LD (IY+1), D
    b.inc_iy()
    b.inc_iy()

    # Outer loop: decrement neuron counter
    b.push_hl()
    b.ld_hl_mem_label_16('NEURCNT')
    b.dec_hl_16()
    b.ld_mem_label_hl_16('NEURCNT')
    b.ld_a_h()
    b.or_l()
    b.pop_hl()
    b.jp_nz('LNEUR')
    b.ret()

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
    # weight == -2: ACC -= 2*DE
    b.ld_hl_mem_label('ACC')
    b.or_a()
    b.sbc_hl_de()
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

    # === ReLU stubs ===
    relu_configs = [
        ('RELU1', 'BUF_A', 512),
        ('RELU2', 'BUF_B', 512),
        ('RELU3', 'BUF_A', 256),
    ]
    for idx, (label, buf_name, count) in enumerate(relu_configs):
        b.label(label)
        b.ld_bc_nn(0)
        b.ld_c_n(count & 0xFF)
        b.ld_b_n((count >> 8) & 0xFF)
        b.ld_hl_label(buf_name)
        if idx == len(relu_configs) - 1:
            pass  # Fall through to RELU
        else:
            b.jr('RELU')

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

    # === ARGMAX with second-best tracking ===
    b.label('ARGMAX')
    b.ld_hl_label('OUTBUF')
    # Load first value as initial best AND second
    b.ld_a_hl()
    b.ld_mem_label_a('MAXL')
    b.ld_mem_label_a('SECL')
    b.inc_hl()
    b.ld_a_hl()
    b.ld_mem_label_a('MAXH')
    b.ld_mem_label_a('SECH')
    b.inc_hl()
    b.xor_a()
    b.ld_mem_label_a('MAXI')
    b.ld_mem_label_a('SECI')
    b.ld_b_n(num_chars - 1)    # 42 remaining candidates
    b.ld_c_n(1)                 # current index

    b.label('AMLP')
    b.ld_e_hl()
    b.inc_hl()
    b.ld_d_hl()
    b.inc_hl()
    b.push_hl()
    b.push_bc()

    # Signed 16-bit compare: is D:E > MAXH:MAXL?
    b.ld_a_d()
    b.ld_hl_label('MAXH')
    b.xor_hl()                # XOR (HL)
    b.jp_m('AM_DSIGN')

    # Same sign: unsigned compare
    b.ld_a_d()
    b.cp_hl()
    b.jr_c('AM_CHK_SEC')      # candidate < best -> check second
    b.jr_nz('AM_NEW_BEST')    # candidate > best -> new best
    b.ld_hl_label('MAXL')
    b.ld_a_e()
    b.cp_hl()
    b.jr_c('AM_CHK_SEC')
    b.jr_z('AM_CHK_SEC')
    b.jr('AM_NEW_BEST')

    b.label('AM_DSIGN')
    b.bit_7_d()
    b.jr_nz('AM_CHK_SEC')     # Candidate negative -> check second

    b.label('AM_NEW_BEST')
    # Old best becomes second best
    b.ld_a_mem_label('MAXL')
    b.ld_mem_label_a('SECL')
    b.ld_a_mem_label('MAXH')
    b.ld_mem_label_a('SECH')
    b.ld_a_mem_label('MAXI')
    b.ld_mem_label_a('SECI')
    # New best
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

    # Check if candidate > second best
    b.label('AM_CHK_SEC')
    # Signed 16-bit compare: is D:E > SECH:SECL?
    b.ld_a_d()
    b.ld_hl_label('SECH')
    b.xor_hl()                # XOR (HL)
    b.jp_m('AM_DSIGN2')

    b.ld_a_d()
    b.cp_hl()
    b.jr_c('AM_SKIP')
    b.jr_nz('AM_NEW_SEC')
    b.ld_hl_label('SECL')
    b.ld_a_e()
    b.cp_hl()
    b.jr_c('AM_SKIP')
    b.jr_z('AM_SKIP')
    b.jr('AM_NEW_SEC')

    b.label('AM_DSIGN2')
    b.bit_7_d()
    b.jr_nz('AM_SKIP')

    b.label('AM_NEW_SEC')
    b.ld_a_e()
    b.ld_mem_label_a('SECL')
    b.ld_a_d()
    b.ld_mem_label_a('SECH')
    b.pop_bc()
    b.ld_a_c()
    b.ld_mem_label_a('SECI')
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

    # === TOKENIZE (query into first 128 buckets) ===
    b.label('TOKENIZE')
    b.ld_hl_label('TOKBUF')
    b.ld_de_label('TOKBUF')
    b.inc_de()
    b.ld_bc_nn(255)
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
    b.and_n(127)

    b.ld_hl_nn(0)
    b.ld_l_a()
    b.add_hl_hl()
    b.push_de()
    b.ld_de_label('TOKBUF')
    b.add_hl_de()
    b.ld_e_hl()
    b.inc_hl()
    b.ld_d_hl()
    b.ld_bc_nn_16(32)
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

    # Unsure message
    b.label('UNSURE_MSG')
    for c in 'JUST ASK':
        b.db(ord(c))
    b.db(0)

    # Fallback messages
    b.label('FB_MSG0')
    for c in 'TRY AGAIN':
        b.db(ord(c))
    b.db(0)

    b.label('FB_MSG1')
    for c in 'BEATS ME':
        b.db(ord(c))
    b.db(0)

    b.label('FB_MSG2')
    for c in 'NOT SURE':
        b.db(ord(c))
    b.db(0)

    # Variables -- 24-bit accumulator, 16-bit counters, 24-bit pointers
    b.label('NEURCNT'); b.dw(0)
    b.label('INCNT');   b.dw(0)
    b.label('WTCNT');   b.dw(0)
    b.label('SAVW');    b.d3(0)
    b.label('SAVB');    b.d3(0)
    b.label('CURIN');   b.d3(0)
    b.label('PACKED');  b.db(0)
    b.label('WEIGHT');  b.db(0)
    b.label('ACC');     b.d3(0)
    b.label('MAXL');    b.db(0)
    b.label('MAXH');    b.db(0)
    b.label('MAXI');    b.db(0)
    b.label('SECL');    b.db(0)       # Second-best low byte
    b.label('SECH');    b.db(0)       # Second-best high byte
    b.label('SECI');    b.db(0)       # Second-best index
    b.label('RESULT');  b.db(0)
    b.label('GENCNT');  b.db(0)
    b.label('GENPOS');  b.db(0)       # Generation position (0-based, for dual bias)
    b.label('LASTCH');  b.db(0)       # Last char for repeat detection
    b.label('REPCNT');  b.db(0)       # Repeat count
    b.label('MARGIN_L'); b.db(0)      # Confidence margin low byte
    b.label('MARGIN_H'); b.db(0)      # Confidence margin high byte
    b.label('TOKLEN');  b.db(0)
    b.label('TOKC1');   b.db(0)
    b.label('TOKC2');   b.db(0)
    b.label('TOKC3');   b.db(0)
    b.label('CTXPOS');  b.db(0)
    b.label('CTXN');    b.db(0)
    b.label('CTXCHARS'); b.ds(8)

    # Input buffer and flags
    b.label('INPLEN');  b.db(0)
    b.label('RI_FLAG'); b.db(0)
    b.label('RI_CLEAR_FLAG'); b.db(0)
    b.label('SAVED_IY'); b.d3(0)
    b.label('INPBUF'); b.ds(62)

    # D_J Context Attention state (295 bytes)
    b.label('CTX_KEY');   b.ds(128)   # 32 slots x 4 bytes
    b.label('CTX_VAL');   b.ds(128)   # 32 slots x 4 bytes
    b.label('CTX_AGE');   b.ds(32)    # 32 slots x 1 byte
    b.label('CTX_QUERY'); b.ds(4)     # Current query vector
    b.label('CTX_SCORE'); b.db(0)
    b.label('CTX_BEST');  b.db(0)
    b.label('CTX_WSLOT'); b.db(0)

    # Attention temp variables
    b.label('CA_SLOT');   b.db(0)
    b.label('CA_KPTR');   b.d3(0)
    b.label('CA_VPTR');   b.d3(0)
    b.label('CA_ACC');    b.db(0)
    b.label('CA_DIM');    b.db(0)
    b.label('CA_Q');      b.db(0)
    b.label('CA_K');      b.db(0)
    b.label('CA_SIGN');   b.db(0)
    b.label('CA_ABSQ');   b.db(0)
    b.label('CW_IDX');    b.db(0)
    b.label('CW_DIM');    b.db(0)
    b.label('CW_KPTR');   b.d3(0)
    b.label('CW_VPTR');   b.d3(0)

    # Computation buffers
    b.label('TOKBUF'); b.ds(input_size * 2)
    max_hidden = 512
    b.label('BUF_A'); b.ds(max_hidden * 2)
    b.label('BUF_B'); b.ds(256 * 2)
    b.label('BUF_B_MID'); b.ds(256 * 2)
    b.label('OUTBUF'); b.ds(output_size * 2)

    # AppVar data pointers
    for i in range(len(APPVAR_NAMES)):
        b.label(f'AVPTR{i}'); b.d3(0)

    # AppVar name data
    for i, name in enumerate(APPVAR_NAMES):
        b.label(f'AVNAME{i}')
        b.db(APPVAR_TYPE)
        for c in name.ljust(8, '\x00'):
            b.db(ord(c))

    # Archive tracking flags
    for i in range(len(APPVAR_NAMES)):
        b.label(f'AVARCED{i}'); b.db(0)

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
    args = parser.parse_args()

    # Derive program name from output filename if not specified
    if args.name:
        prog_name = args.name
    else:
        prog_name = os.path.splitext(os.path.basename(args.output))[0]

    print(f"Building NEOCHAT (24-bit native) -> {args.output}...\n")

    b, appvar_blobs = build_autoreg(args.model)

    # Show key addresses
    print("\nKey addresses:")
    for name in ['START', 'GENERATE', 'LAYER', 'MULADD', 'ARGMAX', 'TOKENIZE',
                 'UPDATE_CTX', 'CTX_ATTEND', 'CTX_WRITE', 'CTX_CLEAR',
                 'CHARTBL', 'TOKBUF', 'OUTBUF', 'BUF_A', 'BUF_B']:
        if name in b.labels:
            print(f"  {name}: {b.labels[name]:06X}h")

    # Resolve labels and get raw code
    b.resolve()

    # Package program as .8xp
    xp_data = build_8xp(b.code, prog_name)
    with open(args.output, 'wb') as f:
        f.write(xp_data)

    print(f"\nProgram: {len(b.code)} bytes ({len(b.code)/1024:.1f} KB)")

    # Report AppVar sizes (no files written yet).
    total_av = sum(len(d) for d in appvar_blobs.values())
    for av_name, av_data in appvar_blobs.items():
        print(f"AppVar {av_name}: {len(av_data):,} bytes")
    print(f"\nTotal weight data: {total_av:,} bytes ({total_av/1024:.1f} KB)")

    # Hard gate: refuse to ship a model that won't fit the calculator's RAM.
    # Runs BEFORE writing any .8xv so an over-budget model fails loudly and
    # cleanly (BudgetError) instead of crashing inside build_8xv on the uint16
    # AppVar-size limit.
    import sizes as _sizes
    _ram = _sizes.total_ram(len(b.code), total_av)
    print(f"Runtime RAM: ~{_ram/1024:.1f} KB "
          f"(program {len(b.code)/1024:.1f} + weights {total_av/1024:.1f} "
          f"+ buffers {_sizes.RUNTIME_BUFFER_BYTES/1024:.1f}); "
          f"budget {_sizes.RAM_BUDGET_BYTES/1024:.0f} KB")
    _sizes.validate_or_raise({
        'program': len(b.code),
        'appvars': {n: len(d) for n, d in appvar_blobs.items()},
        'total_weight': total_av,
    })

    # Within budget: write the AppVar files.
    output_dir = os.path.dirname(args.output) or '.'
    for av_name, av_data in appvar_blobs.items():
        xv_data = build_8xv(av_data, av_name)
        av_path = os.path.join(output_dir, f'{av_name}.8xv')
        with open(av_path, 'wb') as f:
            f.write(xv_data)
        print(f"  wrote {av_path}")

    print(f"Program name: {prog_name.upper()[:8]}")
    print(f"Saved to {args.output}")
    print(f"\nTransfer ALL files to calculator:")
    print(f"  {args.output}")
    for av_name in appvar_blobs:
        print(f"  {av_name}.8xv")
    print(f"Run: Asm(prgm{prog_name.upper()[:8]})")
