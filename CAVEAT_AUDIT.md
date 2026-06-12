# eZ80 ADL-mode caveat audit vs CEmu (2026-06-12)

Reference: CEmu eZ80 core at `/home/tornadus/Documents/CEmu/src/core/`
(`cpu.c`, `cpu.h`, `registers.h`). CEmu is the community's hardware-validated
TI-84 Plus CE emulator; all verdicts below treat it as ground truth for CPU
semantics. Line numbers refer to those files as of this audit.

## How CEmu models suffixes (read this first)

A suffix byte (x=1, z==y, z<4 in the opcode octal decomposition) sets two
mode bits for the *next* instruction (`cpu.c:1155-1168`):

```c
cpu.L  = context.s;   // bit 0 of the suffix opcode
cpu.IL = context.r;   // bit 1 of the suffix opcode
```

- `IL` controls **immediate/address fetch width**: `cpu_fetch_word()` reads 2
  bytes, plus a third iff `IL` (`cpu.c:97-104`).
- `L` controls **everything else**: register-pair width and masking
  (`cpu_read_rp`/`cpu_write_rp`, `cpu.c:311-331`), data read/write width
  (`cpu_read_word`/`cpu_write_word` read/write a third byte iff `L`,
  `cpu.c:125-139`), which stack pointer is used (`cpu.registers.stack[mode]`,
  i.e. SPS when `L=0`, `cpu.c:141-164`), and — critically — **data address
  translation**: every data byte access goes through
  `cpu_address_mode(address, cpu.L)` (`cpu.c:116-123`), which with `L=0`
  rewrites the address to `(MBASE<<16) | (address & 0xFFFF)` (`cpu.c:71-76`).

So in an ADL program, a `.SIS`/`.LIS` instruction is a *Z80-mode* instruction:
16-bit registers, MBASE-relative data addresses, SPS stack. The 24-bit
immediate of a `.LIS` memory op is fetched (3 bytes, so instruction length is
as libez80 emits it) but its top byte is **ignored** for the data access.
On TI-OS, MBASE = 0xD0, and userMem programs live at 0xD1A881+ — so a suffixed
absolute load/store aimed at a program label accesses 0xD0xxxx instead:
wrong data on loads, corruption of unrelated OS RAM on stores.

---

## Caveat 1 — suffix prefix byte values (0x40/.SIS, 0x49/.SIL, 0x52/.LIS, 0x5B/.LIL)

**Claim:** many online sources have 0x49 and 0x52 swapped; the values in
libez80 (`SIS=0x40`, `SIL=0x49`, `LIS=0x52`) are correct.

**CEmu:** `cpu.h:45-76` defines `s` = opcode bit 0, `r` = opcode bit 1;
`cpu.c:1165-1167` sets `L=s`, `IL=r`. Decoding:

| byte | s (L: reg/data) | r (IL: imm) | meaning |
|------|------|------|---------|
| 0x40 | 0 | 0 | .SIS — 16-bit imm, 16-bit reg |
| 0x49 | 1 | 0 | .SIL — 16-bit imm, 24-bit reg |
| 0x52 | 0 | 1 | .LIS — 24-bit imm, 16-bit reg |
| 0x5B | 1 | 1 | .LIL — 24-bit imm, 24-bit reg |

**Verdict: CONFIRMED.** libez80's constants and its derivation comment are
exactly CEmu's decode.

---

## Caveat 2 — ".LIS ED-prefixed STOREs (ED 43, ED 53) may corrupt adjacent memory; non-ED stores (LIS 22) and loads are safe"

**Claim + workaround:** `ld_mem_label_de_16`/`ld_mem_label_bc_16` are marked
unreliable-on-hardware; the codegen routes every pointer store through
`LD (nn),HL` (0x22, 24-bit) or 8-bit `LD (nn),A` stores (workaround sites:
`buildchat84.py` LAYER prologues "bias pointer (DE) -> SAVB via HL (caveat
#2)", AV loader "Store via HL: ED-53 ... can corrupt adjacent memory").

**CEmu:** `.LIS LD (Mmn),rr` is `cpu.c:1468-1470` →
`cpu_write_word(cpu_fetch_word(), cpu_read_rp(p))`. With `L=0`,
`cpu_write_word` writes **exactly 2 bytes** (`cpu.c:133-139`) — there is no
3-byte over-write, ED-prefixed or not. But `cpu_write_byte` translates the
address with `cpu_address_mode(addr, L=0)` = `{MBASE, addr&0xFFFF}`
(`cpu.c:71-76,120-123`). The same applies to the "safe" non-ED forms
(`.LIS LD (Mmn),HL` / `LD HL,(Mmn)`, `cpu.c:1058-1059,1074-1075`) and to
**loads**.

**Verdict: WRONG as diagnosed — but the workaround must stay.** The true
semantics: *every* suffixed absolute memory access (ED or not, load or store)
targets `{MBASE, addr16}`, not the 24-bit label address. On TI-OS this is
~0xD0xxxx instead of 0xD1xxxx — stores corrupt unrelated OS RAM (what was
observed and called "adjacent memory"), and loads read garbage ("loads appear
to be safe" was luck/never stressed). The ED-vs-non-ED and store-vs-load
distinctions in the original caveat are spurious.

**Unlock:** none for memory ops — they are *more* broken than documented and
have been removed from libez80 (the interpreter now refuses them loudly).
What this audit does establish is that **register-only** suffixed ops
(`ADD/SBC HL,rr`, `INC/DEC rr`, `EX DE,HL`, immediate loads) touch no memory
and are fully safe — which the codegen already relies on.

---

## Caveat 3 — ".LIS SBC flags unreliable for signed compare; S may reflect bit 23 instead of bit 15"

**Claim + workaround:** signed 16-bit comparisons avoid `.LIS SBC` flags;
ARGMAX uses an 8-bit XOR-sign-bits compare instead.

**CEmu:** `SBC HL,rp` is `cpu.c:1449-1458`. Operands are masked to the mode
width, the result is masked, and the flags are computed *at the mode width*:
`cpuflag_sign_w(r->HL, cpu.L)` tests `0x8000 << (mode<<3)` (`registers.h:202`)
— i.e. **bit 15** when `L=0`. Z is computed on the masked result (the write
`REG_WRITE_EX(HL, ..., cpu_mask_mode(new_word, cpu.L))` zeroes bits 16-31, so
`cpuflag_zero(r->HL)` sees only the 16-bit result). C is the 16-bit borrow
(`cpuflag_carry_w`, `registers.h:199`). PV is a correct 16-bit signed
overflow (`cpuflag_overflow_w_sub`, `registers.h:212`).

**Verdict: WRONG (per CEmu).** S after `.LIS SBC` is the sign of the 16-bit
result. The likely source of the empirical failures: an S-only signed compare
is **overflow-blind** on any CPU — when the subtraction overflows (e.g.
0x7FFF - 0x8000), S is the "wrong" sign even though the flag itself is
correct. A correct signed compare needs S xor PV (`JP PE/PO`), which libez80
never had. The 8-bit XOR-sign workaround in ARGMAX is itself a correct
overflow-safe compare, so it stays (ARGMAX is 43 elements/char — not hot).

**Unlock:** `.LIS SBC HL,DE` + S-xor-PV would be a valid 16-bit signed
compare if a hot loop ever needs one. None of the current hot loops do.

---

## Caveat 4 — "8-bit register loads do NOT clear the upper byte of the 24-bit pair"

**Claim + workaround:** `LD L,A`-style loads leave bits 16-23 stale; always
`LD rr,0` (24-bit) before assembling a pointer from 8-bit pieces.

**CEmu:** registers are unions where B/C/D/E/H/L are single `uint8_t` fields
of the 32-bit pair (`registers.h:114-129`); 8-bit writes go through
`cpu_write_reg` (`cpu.c:260-272`) which assigns only that byte. The upper
byte (`BCU`/`DEU`/`HLU`) is untouched.

**Verdict: CONFIRMED.** The `LD rr,0`-first discipline is correct and stays.

**Nuance found while auditing (affects the CLAUDE.md note):** 16-bit
*register-pair* writes in short mode behave the opposite way in CEmu —
`cpu_write_rp` (`cpu.c:322-331`) masks the value to 16 bits and assigns the
full 32-bit field, so `.SIS LD HL,nn`, `.LIS ADD HL,rr`, `.LIS INC rr` etc.
**zero** bits 16-23. The project note "a `.SIS LD HL,nn` does not clear
register bits 16-23" is contradicted by CEmu. (Zilog documents the upper
byte as effectively undefined in Z80 mode, so *relying* on the zeroing would
still be unwise; the 24-bit counters introduced by that fix are also the
right fix for the 2-byte-store-into-3-byte-slot problem and stay.) Two
hardware-verified exceptions where CEmu *preserves* the upper byte: the BC
decrement inside block instructions (`cpu_dec_bc_partial_mode`,
`cpu.c:410-419`, "Do not mask BC") and the destination of `EX (SP),rr`
(`cpu_write_index_partial_mode`, `cpu.c:218-228`).

---

## Caveat 5 — "TI-OS uses IY as flags base pointer (0xD00080); save/restore around syscalls"

**Claim + workaround:** IY is saved at START (`SAVED_IY`) and restored before
every TI-OS call.

**CEmu:** not a CPU property — `cpu.c` treats IY as an ordinary index
register. The constraint is the TI-OS ABI (system flags are addressed as
`(IY+offset)` with IY = flags base 0xD00080 inside every OS routine).

**Verdict: CONFIRMED (as an OS contract, not a CPU caveat).** Workaround
stays. Inference code is free to use IY between syscalls, which it does.

---

## ez80interp.py divergences from CEmu (found + fixed in this audit)

The gate interpreter must agree with CEmu for every instruction the builder
can emit. Three divergences found:

1. **Suffixed register-pair writes preserved the upper byte; CEmu zeroes it**
   (`cpu_write_rp`, `cpu.c:322-331`). Fixed: every `.LIS`/`.SIS` register op
   in `_lis`/`_sis` (ADD/SBC/INC/DEC/LD-immediate/EX/POP) now writes
   `value & 0xFFFF` into the pair, clearing bits 16-23. The emitted codegen
   never *reads* bits 16-23 after a suffixed op (it defensively rebuilds
   pointers — those defenses are kept), so faithgate stays green.
2. **Suffixed memory/stack ops were modeled as flat 24-bit accesses; CEmu
   uses `{MBASE, addr16}` and the SPS stack** (see caveat 2). These forms are
   unusable in this program and are no longer emittable: removed from
   libez80, and `ez80interp` now raises `UnsupportedOpcode` for them (loud
   gate failure instead of silent mis-execution, per the interpreter's
   scope contract).
3. **`BIT 7,r` left S stale; CEmu sets S = the tested bit (for bit 7) and
   PV = Z** (`cpu.c:1296-1301`). Fixed. (Emitted code only branches on Z
   after BIT, so this was latent, not live.)

---

## Unexploited eZ80 instructions confirmed in CEmu (relevant to the hot loops)

- **`LD rr,(IX/IY+d)` / `LD (IX/IY+d),rr`** — 24-bit register-pair load/store
  through an index register with displacement, one instruction
  (`cpu.c:1142-1148`). This is the enabler for a memory-resident 24-bit
  accumulator array: `LD HL,(IY+d)` / `ADD HL,DE` / `LD (IY+d),HL`.
  Added to libez80/ez80interp as `ld_hl_iyd` / `ld_iyd_hl` (FD 27 d /
  FD 2F d; x=0,z=7,q,p=2 with the FD prefix).
- **`LEA IY, IY+d`** (`ED 33 d`, `cpu.c:1397-1404`) — pointer bump by a
  constant without touching flags or other registers (IX form already used).
- **`LD A,(IX+d)`** (`DD 7E d`, standard r=A indexed load, `cpu.c:245-258`)
  — lets the packed-weight pointer live in IX so BC is freed for…
- **`DJNZ`** (`cpu.c:996-1002`) — single-instruction decrement-and-branch on
  B for the inner byte counter (the old loop burned 4 memory ops per packed
  byte on a RAM counter).
- **`LD rr,(HL)` / `LD (HL),rr`** (ED page, `cpu.c:1421-1428`), notably
  `LD HL,(HL)` = ED 27: one-instruction pointer-table indirection.
- **`JP (HL)`** (`cpu.c:1239-1253`): one-byte computed jump — with ED 27,
  the whole Option-2 jump-table dispatch is `LD HL,(HL)` + `JP (HL)`.
- **`MLT rp`** (`cpu.c:1504-1510`): 8x8→16 unsigned multiply. Not useful for
  {-2,-1,0,1} weights; noted for completeness (e.g. future index math).
- Block I/O & compare instructions (`cpu_execute_bli`): nothing applicable
  beyond the already-used `LDIR`.

## Summary table

| # | Caveat | Verdict |
|---|--------|---------|
| 1 | Suffix bytes 0x40/0x49/0x52/0x5B | CONFIRMED |
| 2 | .LIS ED stores corrupt adjacent memory; non-ED + loads safe | WRONG diagnosis (real failure is `{MBASE,addr16}` translation of ALL suffixed memory ops; workaround kept, forms removed) |
| 3 | .LIS SBC sign flag reflects bit 23 | WRONG per CEmu (S is bit 15; real pitfall is overflow-blind S-only compares; workaround kept) |
| 4 | 8-bit loads don't clear upper byte | CONFIRMED (but suffixed 16-bit pair writes DO zero it in CEmu, contra the CLAUDE.md note) |
| 5 | TI-OS IY flags pointer | CONFIRMED (OS contract) |
