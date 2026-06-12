#!/usr/bin/env python3
"""
Calculator memory budget for NEOCHAT models on the Ti-84 Plus CE.

Single source of truth for "does a model fit on the calculator". Imported by
buildchat84.py (so the build fails loudly when over budget) and by the
experiment harness ez80research/evaluate.py (so the agentic loop rejects
infeasible models). Keep this file pure and dependency-free.

What actually constrains a model (flash-resident weights, 2026-06):

  * WEIGHTS LIVE IN ARCHIVED FLASH and are read in place by the inference code
    (measured cost <= ~2% of inference time on all hardware revisions). The
    binding constraint on model size is therefore the FLASH budget — the free
    user archive we can reasonably expect on a calculator that has a few other
    things installed.

  * RAM holds only the program image (code + the ds() working buffers). That is
    ~10-15 KB and sits far below the 141 KB RAM budget; the RAM gate is kept as
    a backstop against a pathological program-size regression.

  * Per-AppVar size is a .8xv FORMAT limit, not a model limit. Each AppVar
    payload is length-prefixed with a uint16, so <= 65535 bytes. A layer's
    weights can be split across AS MANY AppVars as needed to respect this — the
    number of AppVars is a build-layout detail, not a hardware cap.
"""

# RAM constraint: only the program image (code + ds() buffers) is RAM-resident
# now that weights are read in place from flash. The 141 KB figure was
# calibrated ON-DEVICE in the unarchive-to-RAM era (filler-AppVar binary
# search: the OS minimum free-to-run is ~5.0-5.5 KB; 141 KB of accounting
# leaves ~7.2 KB free). It predates flash-resident weights but remains a valid
# ceiling for the program image, which sits an order of magnitude below it.
RAM_BUDGET_BYTES = 141 * 1024     # program image only; OS free-to-run cliff ~5.0-5.5 KB

# Flash constraint: the archived weight AppVars. A CE with the pre-installed
# apps and nothing else has roughly 1.8 MB of free archive; gate at 1.5 MB so
# users can run the program even with a few other things on the calculator.
FLASH_BUDGET_BYTES = 1_500_000

# .8xv payload hard limit. The on-disk var entry length-prefixes the payload with
# a uint16 TWICE (build_8xv: var_data = u16(len) + payload, then the entry stores
# u16(len(var_data)) = len+2). So the largest payload that fits both fields is
# 65535 - 2 = 65533, not 65535. A layer is split across as many AppVars as needed
# to respect this; it is NOT a per-layer feasibility limit.
MAX_APPVAR_BYTES = 65533

# Fixed RAM consumers (the program image's working buffers).
RUNTIME_BUFFER_BYTES = 3 * 1024   # TOKBUF/BUF_A/BUF_B/OUTBUF working buffers
# Program CODE estimate (logic only — the pregate adds the ds() working buffers
# separately). The real emitted code (sans buffers) is ~2.0-2.5 KB; the old 8 KB
# was a ~5 KB over-estimate that was harmless under the old 150 KB budget but
# rejects feasible models under the corrected tight budget. 2500 keeps the pregate
# only a few hundred bytes conservative vs the real build size.
PROGRAM_ESTIMATE_BYTES = 2500      # program code (logic), arch-only pre-gate only


class BudgetError(Exception):
    """Raised when a built model exceeds a calculator memory budget."""


def weight_data_bytes(arch, weight_bits=2):
    """Total packed bytes of weights+bias+dual-bias for an architecture dict
    {'input_size', 'hidden_sizes', 'num_classes'}. Mirrors the build's packing:
    `weight_bits`-bit weights (8//bits per byte) + int16 bias per neuron, plus a
    second (dual) bias set on the output layer. `weight_bits` is a scalar (every
    layer) or a per-layer list of length len(hidden_sizes)+1. Independent of AppVar
    sharding."""
    dims = [arch['input_size']] + list(arch['hidden_sizes']) + [arch['num_classes']]
    nlayers = len(dims) - 1
    bits = weight_bits if isinstance(weight_bits, (list, tuple)) else [weight_bits] * nlayers
    weights = bias = 0
    for i in range(nlayers):
        n_in, n_out = dims[i], dims[i + 1]
        weights += (n_in * n_out * bits[i]) // 8   # 8//bits weights per byte
        bias += n_out * 2                          # int16 bias / neuron
    dual_bias = arch['num_classes'] * 2
    return {'weights': weights, 'bias': bias,
            'dual_bias': dual_bias, 'total': weights + bias + dual_bias}


def total_ram(program_bytes, weight_total_bytes=0):
    """Bytes that must be resident in RAM at runtime: the program image only.
    Weights live in archived flash and are read in place, so they no longer
    count against RAM (`weight_total_bytes` is accepted for caller
    compatibility and ignored — gate it against FLASH_BUDGET_BYTES instead).
    The working buffers (TOKBUF/PING/PONG/OUTBUF) are emitted INTO the program
    image via ds(), so when `program_bytes` is the real build size
    (len(b.code)) it already includes them — do NOT add them again."""
    return program_bytes


def estimate_memory(arch):
    """Fast arch-only pre-gate (no build needed). Returns components plus
    'total_ram', the minimum number of AppVars the weights would shard into, and
    a 'fits' flag against the RAM (program) and flash (weights) budgets. Uses a
    program-size ESTIMATE; the real gate in check_budget() uses the build's
    actual program size."""
    w = weight_data_bytes(arch)
    # The program ESTIMATE doesn't include the ds() working buffers, so fold a
    # buffer allowance into it here (the real gate in check_budget() doesn't,
    # because the real program size already contains them).
    ram = total_ram(PROGRAM_ESTIMATE_BYTES + RUNTIME_BUFFER_BYTES)
    return {
        **w,
        'weight_total': w['total'],
        'total_ram': ram,
        'min_appvars': -(-w['total'] // MAX_APPVAR_BYTES),  # ceil
        'fits': ram <= RAM_BUDGET_BYTES and w['total'] <= FLASH_BUDGET_BYTES,
    }


def pregate(spec):
    """Fast spec-only feasibility PRE-gate (no train, no build). Reject an
    infeasible model in <1s instead of after a full train+build. Bit-width-aware,
    and sizes the real ds() working buffers (TOKBUF + PING + PONG + OUTBUF) from
    the spec, plus a conservative program-code allowance. Deliberately slightly
    OVER-estimates so it never green-lights a model the real build gate rejects.
    RAM gates the program image; flash gates the packed weights.
    Returns (fits, est_ram_bytes, reasons)."""
    arch = {'input_size': spec['input_size'],
            'hidden_sizes': spec['hidden_sizes'],
            'num_classes': spec['num_classes']}
    w = weight_data_bytes(arch, spec['weight_bits'])
    hidden = spec['hidden_sizes']
    max_hidden = max(hidden) if hidden else spec['num_classes']
    # ds() working buffers emitted into the program image (see buildchat84 DATA).
    buffers = (spec['input_size'] * 2          # TOKBUF
               + 2 * max_hidden * 2             # PING + PONG
               + spec['num_classes'] * 2        # OUTBUF
               + 512)                           # INPBUF/CTX/scratch slack
    est_ram = PROGRAM_ESTIMATE_BYTES + buffers
    reasons = []
    if est_ram > RAM_BUDGET_BYTES:
        reasons.append(f'PREGATE_RAM_OVER({est_ram}>{RAM_BUDGET_BYTES})')
    if w['total'] > FLASH_BUDGET_BYTES:
        reasons.append(f'PREGATE_FLASH_OVER({w["total"]}>{FLASH_BUDGET_BYTES})')
    return (len(reasons) == 0, est_ram, reasons)


def check_budget(sizes):
    """Apply the HARD GATE against REAL build numbers.

    `sizes` keys: 'program' (code bytes), 'total_weight' (sum of all AppVar
    payload bytes), 'appvars' ({name: bytes}). Returns (ok, reasons); empty
    reasons == OK.

    Three independent failure modes:
      * RAM_OVER       — the program image (code + buffers) exceeds the RAM
                         budget (backstop; weights don't count, they're flash).
      * FLASH_OVER     — total packed weight data exceeds the flash budget
                         (the real feasibility limit on model size).
      * APPVAR_TOO_BIG — the build emitted an AppVar above the .8xv 65535-byte
                         format limit. This means the build's SPLITTER is
                         incomplete for this model, not that the model is
                         inherently infeasible — but it still can't ship as-is,
                         so it's a hard fail.
    """
    reasons = []

    ram = total_ram(sizes.get('program', 0))
    if ram > RAM_BUDGET_BYTES:
        reasons.append(f'RAM_OVER({ram}>{RAM_BUDGET_BYTES})')

    flash = sizes.get('total_weight', 0)
    if flash > FLASH_BUDGET_BYTES:
        reasons.append(f'FLASH_OVER({flash}>{FLASH_BUDGET_BYTES})')

    for name, nbytes in sizes.get('appvars', {}).items():
        if nbytes > MAX_APPVAR_BYTES:
            reasons.append(f'APPVAR_TOO_BIG({name}:{nbytes}>{MAX_APPVAR_BYTES})')

    return (len(reasons) == 0, reasons)


def validate_or_raise(sizes):
    """Loud failure helper for the build pipeline. Raises BudgetError listing
    every violation when the model is over budget."""
    ok, reasons = check_budget(sizes)
    if not ok:
        raise BudgetError(
            'model exceeds Ti-84 Plus CE memory budget: ' + ', '.join(reasons)
        )
