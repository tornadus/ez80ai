#!/usr/bin/env python3
"""
Calculator memory budget for NEOCHAT models on the Ti-84 Plus CE.

Single source of truth for "does a model fit on the calculator". Imported by
buildchat84.py (so the build fails loudly when over budget) and by the
experiment harness ez80research/evaluate.py (so the agentic loop rejects
infeasible models). Keep this file pure and dependency-free.

What actually constrains a model (corrected — do NOT assume 1 layer = 1 AppVar):

  * RAM is the binding constraint. The README has you free >=150 KB of RAM, and
    archived AppVars are UNARCHIVED INTO RAM when the program starts. So at
    runtime the program code + ALL packed weight data + the working buffers must
    coexist in RAM. That total, not flash, is the ceiling. (Flash only needs to
    hold the AppVars while archived, and the calc has a few MB of it.)

  * Per-AppVar size is a .8xv FORMAT limit, not a model limit. Each AppVar
    payload is length-prefixed with a uint16, so <= 65535 bytes. A layer's
    weights can be split across AS MANY AppVars as needed to respect this — the
    number of AppVars is a build-layout detail, not a hardware cap. (The current
    buildchat84.py happens to emit a fixed NEOA-D layout, but that is the build's
    choice; it is not what bounds feasibility.)

  * AppVar COUNT is therefore not separately gated: with a <=150 KB RAM ceiling
    and <=65535 B per AppVar, a feasible model has at most ~2-3 AppVars anyway,
    so RAM binds first.

NOTE on a possible future lever: if the inference code is changed to read
weights directly from ARCHIVED AppVars in flash (rather than relying on the OS
unarchiving them to RAM), the binding budget becomes flash (megabytes) instead
of RAM, dramatically enlarging the feasible model space. That is not how the
build works today, so we gate on RAM.
"""

# Binding constraint: program + weights + buffers, all resident in RAM at run.
RAM_BUDGET_BYTES = 150 * 1024     # >=150 KB free RAM (README)

# .8xv payload hard limit (uint16 length field). A layer is split across as many
# AppVars as needed to respect this; it is NOT a per-layer feasibility limit.
MAX_APPVAR_BYTES = 65535

# Fixed RAM consumers besides the packed weights.
RUNTIME_BUFFER_BYTES = 3 * 1024   # TOKBUF/BUF_A/BUF_B/OUTBUF working buffers
PROGRAM_ESTIMATE_BYTES = 8 * 1024  # program code, for the arch-only pre-gate only


class BudgetError(Exception):
    """Raised when a built model exceeds the calculator RAM budget."""


def weight_data_bytes(arch):
    """Total packed bytes of weights+bias+shift+dual-bias for an architecture
    dict {'input_size', 'hidden_sizes', 'num_classes'}. Mirrors the build's
    packing: 2-bit weights (4 per byte), int16 bias + uint8 shift per neuron,
    plus a second (dual) bias set on the output layer. This total is independent
    of how the bytes are sharded into AppVars."""
    dims = [arch['input_size']] + list(arch['hidden_sizes']) + [arch['num_classes']]
    weights = bias = shift = 0
    for i in range(len(dims) - 1):
        n_in, n_out = dims[i], dims[i + 1]
        weights += (n_in * n_out) // 4   # 4 weights / byte (2-bit packing)
        bias += n_out * 2                # int16 bias / neuron
        shift += n_out * 1               # uint8 shift / neuron
    dual_bias = arch['num_classes'] * 2
    return {'weights': weights, 'bias': bias, 'shift': shift,
            'dual_bias': dual_bias, 'total': weights + bias + shift + dual_bias}


def total_ram(program_bytes, weight_total_bytes):
    """Bytes that must be resident in RAM at runtime."""
    return program_bytes + weight_total_bytes + RUNTIME_BUFFER_BYTES


def estimate_memory(arch):
    """Fast arch-only pre-gate (no build needed). Returns components plus
    'total_ram', the minimum number of AppVars the weights would shard into, and
    a 'fits' flag against the RAM budget. Uses a program-size ESTIMATE; the real
    gate in check_budget() uses the build's actual program size."""
    w = weight_data_bytes(arch)
    ram = total_ram(PROGRAM_ESTIMATE_BYTES, w['total'])
    return {
        **w,
        'weight_total': w['total'],
        'total_ram': ram,
        'min_appvars': -(-w['total'] // MAX_APPVAR_BYTES),  # ceil
        'fits': ram <= RAM_BUDGET_BYTES,
    }


def check_budget(sizes):
    """Apply the HARD GATE against REAL build numbers.

    `sizes` keys: 'program' (code bytes), 'total_weight' (sum of all AppVar
    payload bytes), 'appvars' ({name: bytes}). Returns (ok, reasons); empty
    reasons == OK.

    Two independent failure modes:
      * RAM_OVER       — program + weights + buffers exceed the RAM budget
                         (the real feasibility limit).
      * APPVAR_TOO_BIG — the build emitted an AppVar above the .8xv 65535-byte
                         format limit. This means the build's SPLITTER is
                         incomplete for this model, not that the model is
                         inherently infeasible — but it still can't ship as-is,
                         so it's a hard fail.
    """
    reasons = []

    ram = total_ram(sizes.get('program', 0), sizes.get('total_weight', 0))
    if ram > RAM_BUDGET_BYTES:
        reasons.append(f'RAM_OVER({ram}>{RAM_BUDGET_BYTES})')

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
            'model exceeds Ti-84 Plus CE RAM budget: ' + ', '.join(reasons)
        )
