"""
eZ80 machine code builder for ADL (24-bit) mode.

Self-contained builder for generating Ti-84 Plus CE machine code with:
- 24-bit address fixups and immediates (ADL default .LIL mode)
- .LIS prefixed instructions for 16-bit register pair operations at 24-bit addresses
- .SIS prefixed instructions for fully 16-bit operations (2-byte immediates)
- IY save/restore for TI-OS compatibility
- Label/fixup system for forward references

IMPORTANT eZ80 ADL mode caveats. Audited 2026-06 against the CEmu CPU core
(~/Documents/CEmu/src/core/{cpu.c,cpu.h,registers.h}; line refs are to that
checkout). Two of the five original caveats were misdiagnosed; the corrected
semantics below are the reference. ez80interp.py was aligned with CEmu in the
same audit (suffixed pair writes zero bits 16-23; BIT 7,r sets S; suffixed
memory/stack opcodes raise UnsupportedOpcode instead of mis-executing).

How CEmu models suffixes (the mechanism behind caveats 1-3): a suffix byte
sets two mode bits for the NEXT instruction (cpu.c:1155-1168): IL = opcode
bit 1 controls immediate/address FETCH width; L = opcode bit 0 controls
everything else — register-pair width/masking (cpu_read_rp/cpu_write_rp,
cpu.c:311-331), data access width (cpu.c:125-139), WHICH STACK is used (SPS
when L=0, cpu.c:141-164), and data address translation: every data access
goes through cpu_address_mode(addr, L) (cpu.c:116-123), which with L=0
rewrites the address to {MBASE, addr & 0xFFFF} (cpu.c:71-76).

  1. Suffix prefix bytes: 0x40=.SIS, 0x49=.SIL, 0x52=.LIS, 0x5B=.LIL
     Many online sources have 0x49 and 0x52 SWAPPED. CONFIRMED via the CEmu
     decode above (L = bit 0, IL = bit 1 of the suffix opcode).
  2. Suffixed (.SIS/.LIS) instructions that touch MEMORY or the STACK are
     unusable in an ADL program: the CPU executes them in Z80 mode, so data
     accesses go to {MBASE, addr16} — NOT the 24-bit label address. On TI-OS
     MBASE=0xD0 while userMem programs live at 0xD1A881+, so a suffixed
     absolute access aimed at a program label hits 0xD0xxxx instead: garbage
     reads and OS-RAM-corrupting writes ~64KB away. Pushes/pops additionally
     use the Z80 SPS stack. The historical caveat here ("ED 43/ED 53 stores
     corrupt adjacent memory; non-ED stores and loads are safe") was this,
     misdiagnosed: there is no 3-byte over-write (a short store writes exactly
     2 bytes, cpu.c:133-139) and the ED-vs-non-ED / store-vs-load distinctions
     were spurious — ALL suffixed memory forms are equally broken ("loads are
     safe" was luck, never stressed). These emitters have been REMOVED;
     ez80interp rejects the opcodes loudly. Register-only suffixed ops
     (ADD/SBC HL,rr / INC/DEC rr / EX DE,HL / immediate loads) touch no
     memory and are fully safe.
  3. Flags from .LIS SBC are CORRECT for the 16-bit result per CEmu: operands,
     result and flags are computed at mode width — S = bit 15 of the result
     (registers.h:202), Z on the masked 16-bit result, C = 16-bit borrow
     (registers.h:199), PV = true 16-bit signed overflow (registers.h:212).
     The old "S may reflect bit 23" claim is WRONG; the empirical failures it
     explained came from S-only signed compares being overflow-blind on any
     Z80-family CPU (e.g. 0x7FFF - 0x8000 overflows and flips S). A correct
     signed compare needs S xor PV (JP PE/PO). The 8-bit XOR-sign-bits compare
     used by ARGMAX is a correct overflow-safe alternative and stays.
  4. 8-bit register loads (LD H,n / LD C,A / etc.) do NOT clear the upper byte
     of the 24-bit register pair — registers are byte-unions and 8-bit writes
     touch only their byte (registers.h:114-129, cpu.c:260-272). Always use
     LD rr,0 (24-bit) before assembling a pointer from 8-bit pieces. Nuance:
     suffixed 16-bit PAIR writes (.SIS LD HL,nn / .LIS ADD HL,rr / INC rr...)
     DO zero bits 16-23 in CEmu (cpu_write_rp masks then assigns the full
     field, cpu.c:322-331) — the opposite of what this project once claimed —
     with two hardware-verified exceptions that PRESERVE the upper byte: BC
     decrements inside block instructions (cpu.c:410-419) and the destination
     of EX (SP),rr (cpu.c:218-228). Zilog documents the upper byte as
     undefined in Z80 mode, so the codegen relies on NEITHER behavior.
  5. TI-OS uses IY as flags base pointer (0xD00080); save IY before
     computation and restore before any TI-OS syscall (_PutC, _NewLine, ...).
     This is an OS ABI contract, not a CPU caveat (cpu.c treats IY as an
     ordinary index register). Inference code may use IY freely between
     syscalls (the fast layer loops do).

Instructions verified safe in the audit and now load-bearing in the hot
loops: LD rr,(IX/IY+d) / LD (IX/IY+d),rr (cpu.c:1142-1148), LEA IY,IY+d
(ED 33), LD A,(IX+d), DJNZ, LD HL,(HL) (ED 27) + JP (HL) for jump-table
dispatch. MLT rp exists (8x8->16 unsigned) but is useless for {-2,-1,0,1}
weights.
"""

from typing import List, Dict, Tuple


class eZ80Builder:
    """eZ80 code builder for ADL mode (24-bit addressing).

    Default origin is userMem (0xD1A881) where TI-OS loads CE assembly programs
    after stripping the 2-byte Asm84CEPrgm preamble.
    """

    # Suffix prefix bytes (in ADL mode)
    # Derived from CEmu cpu.c: cpu.L = bit 0 (register size), cpu.IL = bit 1 (immediate size)
    # Prefix byte z-field: 0x40=0, 0x49=1, 0x52=2, 0x5B=3
    SIS = 0x40  # z=0: IL=0 S-imm, L=0 S-reg -> .SIS (16-bit addr, 16-bit reg)
    SIL = 0x49  # z=1: IL=0 S-imm, L=1 L-reg -> .SIL (16-bit addr, 24-bit reg)
    LIS = 0x52  # z=2: IL=1 L-imm, L=0 S-reg -> .LIS (24-bit addr, 16-bit reg)
    # LIL = 0x5B  z=3: IL=1 L-imm, L=1 L-reg -> .LIL (default in ADL mode)

    def __init__(self, org: int = 0xD1A881):
        self.org = org
        self.code = bytearray()
        self.labels: Dict[str, int] = {}
        self.fixups: List[Tuple[int, str, str]] = []  # (offset, label, type)

    # === Core emission ===

    def addr(self) -> int:
        return self.org + len(self.code)

    def label(self, name: str):
        self.labels[name] = self.addr()

    def emit(self, *bytes):
        for b in bytes:
            self.code.append(b & 0xFF)

    def emit_word(self, val: int):
        """Emit a 16-bit value (little-endian)."""
        self.emit(val & 0xFF, (val >> 8) & 0xFF)

    def emit_addr(self, val: int):
        """Emit a 24-bit value (little-endian)."""
        self.emit(val & 0xFF, (val >> 8) & 0xFF, (val >> 16) & 0xFF)

    # === Fixups ===

    def fixup_word(self, label: str):
        """Emit 3-byte placeholder for 24-bit absolute address."""
        self.fixups.append((len(self.code), label, 'abs24'))
        self.emit(0, 0, 0)

    def fixup_word_16(self, label: str):
        """Emit 2-byte placeholder for 16-bit fixup (used with .SIS prefix)."""
        self.fixups.append((len(self.code), label, 'abs'))
        self.emit(0, 0)

    def fixup_rel(self, label: str):
        """Emit placeholder byte for relative jump."""
        self.fixups.append((len(self.code), label, 'rel'))
        self.emit(0)

    def resolve(self):
        """Apply all fixups: abs24 (3-byte), abs (2-byte), rel (1-byte signed)."""
        for offset, label, ftype in self.fixups:
            if label not in self.labels:
                raise ValueError(f"Unknown label: {label}")
            target = self.labels[label]

            if ftype == 'abs24':
                self.code[offset] = target & 0xFF
                self.code[offset + 1] = (target >> 8) & 0xFF
                self.code[offset + 2] = (target >> 16) & 0xFF
            elif ftype == 'abs':
                self.code[offset] = target & 0xFF
                self.code[offset + 1] = (target >> 8) & 0xFF
            elif ftype == 'rel':
                from_addr = self.org + offset + 1
                rel = target - from_addr
                if rel < -128 or rel > 127:
                    raise ValueError(f"Relative jump out of range: {label} = {rel}")
                self.code[offset] = rel & 0xFF

    def save(self, filename: str):
        self.resolve()
        with open(filename, 'wb') as f:
            f.write(self.code)
        print(f"Wrote {len(self.code)} bytes to {filename}")

    # === Control flow ===

    def nop(self): self.emit(0x00)
    def ret(self): self.emit(0xC9)
    def ret_z(self): self.emit(0xC8)
    def ret_nz(self): self.emit(0xC0)
    def rst(self, n): self.emit(0xC7 | n)
    def halt(self): self.emit(0x76)
    def di(self): self.emit(0xF3)
    def ei(self): self.emit(0xFB)

    def call(self, label: str):
        self.emit(0xCD)
        self.fixup_word(label)

    def call_addr(self, addr: int):
        self.emit(0xCD)
        self.emit_addr(addr)

    def jp(self, label: str):
        self.emit(0xC3)
        self.fixup_word(label)

    def jp_nz(self, label: str):
        self.emit(0xC2)
        self.fixup_word(label)

    def jp_z(self, label: str):
        self.emit(0xCA)
        self.fixup_word(label)

    def jp_m(self, label: str):
        self.emit(0xFA)
        self.fixup_word(label)

    def jp_c(self, label: str):
        self.emit(0xDA)
        self.fixup_word(label)

    def jp_hl(self):
        """JP (HL) (E9) -- computed jump to the 24-bit address in HL
        (CEmu cpu.c:1239-1253, 'JP (rr)')."""
        self.emit(0xE9)

    def jr(self, label: str):
        self.emit(0x18)
        self.fixup_rel(label)

    def jr_nz(self, label: str):
        self.emit(0x20)
        self.fixup_rel(label)

    def jr_z(self, label: str):
        self.emit(0x28)
        self.fixup_rel(label)

    def jr_nc(self, label: str):
        self.emit(0x30)
        self.fixup_rel(label)

    def jr_c(self, label: str):
        self.emit(0x38)
        self.fixup_rel(label)

    def djnz(self, label: str):
        self.emit(0x10)
        self.fixup_rel(label)

    # === 24-bit immediate loads (ADL default .LIL) ===

    def ld_hl_nn(self, val): self.emit(0x21); self.emit_addr(val)
    def ld_de_nn(self, val): self.emit(0x11); self.emit_addr(val)
    def ld_bc_nn(self, val): self.emit(0x01); self.emit_addr(val)
    def ld_ix_nn(self, val): self.emit(0xDD, 0x21); self.emit_addr(val)
    def ld_iy_nn(self, val): self.emit(0xFD, 0x21); self.emit_addr(val)

    # === 24-bit label loads ===

    def ld_hl_label(self, label): self.emit(0x21); self.fixup_word(label)
    def ld_de_label(self, label): self.emit(0x11); self.fixup_word(label)
    def ld_bc_label(self, label): self.emit(0x01); self.fixup_word(label)
    def ld_ix_label(self, label): self.emit(0xDD, 0x21); self.fixup_word(label)
    def ld_iy_label(self, label): self.emit(0xFD, 0x21); self.fixup_word(label)

    # === 8-bit immediate loads ===

    def ld_a_n(self, val): self.emit(0x3E, val & 0xFF)
    def ld_b_n(self, val): self.emit(0x06, val & 0xFF)
    def ld_c_n(self, val): self.emit(0x0E, val & 0xFF)
    def ld_d_n(self, val): self.emit(0x16, val & 0xFF)
    def ld_e_n(self, val): self.emit(0x1E, val & 0xFF)
    def ld_h_n(self, val): self.emit(0x26, val & 0xFF)
    def ld_l_n(self, val): self.emit(0x2E, val & 0xFF)
    def ld_hl_n(self, val): self.emit(0x36, val & 0xFF)  # LD (HL),n

    # === 24-bit memory load/store ===

    def ld_hl_hl_ind(self):
        """LD HL,(HL) (ED 27) -- 24-bit pointer-table indirection in one
        instruction (CEmu cpu.c:1421-1428, ED page x=0,z=7,q=0,p=2)."""
        self.emit(0xED, 0x27)

    def ld_hl_mem_label(self, label): self.emit(0x2A); self.fixup_word(label)
    def ld_mem_label_hl(self, label): self.emit(0x22); self.fixup_word(label)
    # NOTE: the unsuffixed ED 53 / ED 43 stores below are plain ADL-mode
    # 24-bit stores (3 bytes at the 24-bit address) — correct per CEmu. The
    # historical "may corrupt adjacent memory" worry was a misdiagnosis of
    # the SUFFIXED (.LIS) forms (see caveat #2). The codegen still routes
    # pointer stores through ld_mem_label_hl by convention.
    def ld_mem_label_de(self, label): self.emit(0xED, 0x53); self.fixup_word(label)
    def ld_mem_label_bc(self, label): self.emit(0xED, 0x43); self.fixup_word(label)
    def ld_bc_mem_label(self, label): self.emit(0xED, 0x4B); self.fixup_word(label)
    def ld_de_mem_label(self, label): self.emit(0xED, 0x5B); self.fixup_word(label)
    def ld_mem_label_sp(self, label): self.emit(0xED, 0x73); self.fixup_word(label)
    def ld_sp_mem_label(self, label): self.emit(0xED, 0x7B); self.fixup_word(label)
    def ld_a_mem_label(self, label): self.emit(0x3A); self.fixup_word(label)
    def ld_mem_label_a(self, label): self.emit(0x32); self.fixup_word(label)
    # IX absolute load/store (DD 2A / DD 22), 24-bit.
    def ld_ix_mem_label(self, label): self.emit(0xDD, 0x2A); self.fixup_word(label)
    def ld_mem_label_ix(self, label): self.emit(0xDD, 0x22); self.fixup_word(label)

    # === 8-bit register-register loads ===

    def ld_a_bc(self): self.emit(0x0A)
    def ld_a_hl(self): self.emit(0x7E)
    def ld_hl_a(self): self.emit(0x77)   # LD (HL),A
    def ld_e_hl(self): self.emit(0x5E)   # LD E,(HL)
    def ld_d_hl(self): self.emit(0x56)   # LD D,(HL)
    def ld_b_hl(self): self.emit(0x46)   # LD B,(HL)
    def ld_c_hl(self): self.emit(0x4E)   # LD C,(HL)
    def ld_a_b(self): self.emit(0x78)
    def ld_a_c(self): self.emit(0x79)
    def ld_a_d(self): self.emit(0x7A)
    def ld_a_e(self): self.emit(0x7B)
    def ld_a_h(self): self.emit(0x7C)
    def ld_a_l(self): self.emit(0x7D)
    def ld_b_a(self): self.emit(0x47)
    def ld_b_c(self): self.emit(0x41)
    def ld_b_e(self): self.emit(0x43)
    def ld_b_h(self): self.emit(0x44)
    def ld_c_a(self): self.emit(0x4F)
    def ld_c_l(self): self.emit(0x4D)
    def ld_d_a(self): self.emit(0x57)
    def ld_d_h(self): self.emit(0x54)
    def ld_e_a(self): self.emit(0x5F)
    def ld_e_c(self): self.emit(0x59)
    def ld_e_l(self): self.emit(0x5D)
    def ld_h_a(self): self.emit(0x67)
    def ld_h_b(self): self.emit(0x60)
    def ld_l_a(self): self.emit(0x6F)
    def ld_l_c(self): self.emit(0x69)
    def ld_l_d(self): self.emit(0x6A)
    def ld_l_e(self): self.emit(0x6B)
    def ld_a_de(self): self.emit(0x1A)   # LD A,(DE)
    def ld_de_a(self): self.emit(0x12)   # LD (DE),A
    def ld_hl_d(self): self.emit(0x72)   # LD (HL),D
    def ld_hl_e(self): self.emit(0x73)   # LD (HL),E

    # === IX/IY indexed loads ===

    def ld_a_ixd(self, d): self.emit(0xDD, 0x7E, d & 0xFF)  # LD A,(IX+d)
    def ld_l_ixd(self, d): self.emit(0xDD, 0x6E, d & 0xFF)
    def ld_h_ixd(self, d): self.emit(0xDD, 0x66, d & 0xFF)
    def ld_e_ixd(self, d): self.emit(0xDD, 0x5E, d & 0xFF)  # LD E,(IX+d)
    def ld_d_ixd(self, d): self.emit(0xDD, 0x56, d & 0xFF)  # LD D,(IX+d)
    def ld_ixd_l(self, d): self.emit(0xDD, 0x75, d & 0xFF)
    def ld_ixd_h(self, d): self.emit(0xDD, 0x74, d & 0xFF)
    def ld_iyd_l(self, d): self.emit(0xFD, 0x75, d & 0xFF)
    def ld_iyd_h(self, d): self.emit(0xFD, 0x74, d & 0xFF)
    def ld_iyd_d(self, d): self.emit(0xFD, 0x72, d & 0xFF)  # LD (IY+d),D
    def ld_iyd_e(self, d): self.emit(0xFD, 0x73, d & 0xFF)  # LD (IY+d),E

    # 24-bit register-pair load/store through IY+d — core eZ80 ADL forms
    # (CEmu cpu.c:1142-1148: x=0,z=7 with the FD prefix; q selects load vs
    # store, p=2 selects HL; cpu_read_word/cpu_write_word move 3 bytes in
    # ADL mode). The enabler for the RAM-resident 24-bit accumulator array.
    def ld_hl_iyd(self, d): self.emit(0xFD, 0x27, d & 0xFF)  # LD HL,(IY+d)
    def ld_iyd_hl(self, d): self.emit(0xFD, 0x2F, d & 0xFF)  # LD (IY+d),HL

    # === Stack pointer loads ===

    def ld_sp_hl(self): self.emit(0xF9)
    def ld_sp_ix(self): self.emit(0xDD, 0xF9)
    def ld_sp_iy(self): self.emit(0xFD, 0xF9)

    # === 8-bit arithmetic ===

    def add_a_n(self, val): self.emit(0xC6, val & 0xFF)
    def add_a_a(self): self.emit(0x87)
    def add_a_b(self): self.emit(0x80)
    def add_a_c(self): self.emit(0x81)
    def add_a_e(self): self.emit(0x83)
    def add_a_h(self): self.emit(0x84)
    def add_a_l(self): self.emit(0x85)
    def add_a_hl(self): self.emit(0x86)   # ADD A,(HL)
    def adc_a_n(self, val): self.emit(0xCE, val & 0xFF)  # ADC A,n
    def adc_a_a(self): self.emit(0x8F)
    def adc_a_b(self): self.emit(0x88)
    def adc_a_d(self): self.emit(0x8A)
    def adc_a_hl(self): self.emit(0x8E)   # ADC A,(HL)
    def sub_n(self, val): self.emit(0xD6, val & 0xFF)
    def sub_b(self): self.emit(0x90)
    def sub_c(self): self.emit(0x91)
    def sub_h(self): self.emit(0x94)
    def sub_l(self): self.emit(0x95)
    def sub_a_hl(self): self.emit(0x96)   # SUB (HL)
    sub_hl_ind = sub_a_hl                 # alias
    def sbc_a_a(self): self.emit(0x9F)
    def sbc_a_hl(self): self.emit(0x9E)   # SBC A,(HL)
    def cp_n(self, val): self.emit(0xFE, val & 0xFF)
    def cp_a(self): self.emit(0xBF)
    def cp_b(self): self.emit(0xB8)
    def cp_hl(self): self.emit(0xBE)      # CP (HL)
    def inc_a(self): self.emit(0x3C)
    def dec_a(self): self.emit(0x3D)

    # === Logic ===

    def and_n(self, val): self.emit(0xE6, val & 0xFF)
    def and_a(self): self.emit(0xA7)
    def or_a(self): self.emit(0xB7)
    def or_c(self): self.emit(0xB1)
    def or_d(self): self.emit(0xB2)
    def or_l(self): self.emit(0xB5)
    def or_n(self, val): self.emit(0xF6, val & 0xFF)
    def xor_a(self): self.emit(0xAF)
    def xor_hl(self): self.emit(0xAE)     # XOR (HL)
    def cpl(self): self.emit(0x2F)

    # === 24-bit register pair arithmetic ===

    def add_hl_bc(self): self.emit(0x09)
    def add_hl_de(self): self.emit(0x19)
    def add_hl_hl(self): self.emit(0x29)
    def add_hl_sp(self): self.emit(0x39)
    def sbc_hl_de(self): self.emit(0xED, 0x52)
    def sbc_hl_bc(self): self.emit(0xED, 0x42)
    def inc_bc(self): self.emit(0x03)
    def dec_bc(self): self.emit(0x0B)
    def inc_de(self): self.emit(0x13)
    def inc_hl(self): self.emit(0x23)
    def dec_hl(self): self.emit(0x2B)
    def dec_sp(self): self.emit(0x3B)
    def inc_b(self): self.emit(0x04)
    def dec_b(self): self.emit(0x05)
    def inc_c(self): self.emit(0x0C)
    def dec_c(self): self.emit(0x0D)
    def inc_d(self): self.emit(0x14)
    def dec_d(self): self.emit(0x15)
    def inc_e(self): self.emit(0x1C)
    def dec_e(self): self.emit(0x1D)
    def inc_h(self): self.emit(0x24)
    def dec_h(self): self.emit(0x25)
    def inc_l(self): self.emit(0x2C)
    def inc_ix(self): self.emit(0xDD, 0x23)
    def inc_iy(self): self.emit(0xFD, 0x23)

    def lea_ix_d(self, d):
        """LEA IX, IX+d (ED 32 d) -- core eZ80 instruction (emitted pervasively
        by the CE C toolchain), signed 8-bit displacement."""
        self.emit(0xED, 0x32, d & 0xFF)

    def lea_iy_d(self, d):
        """LEA IY, IY+d (ED 33 d) -- CEmu cpu.c:1397-1404 (z=3 selects IY)."""
        self.emit(0xED, 0x33, d & 0xFF)

    # === Shifts and rotates ===

    def rlca(self): self.emit(0x07)
    def rrca(self): self.emit(0x0F)
    def rra(self): self.emit(0x1F)
    def sla_a(self): self.emit(0xCB, 0x27)
    def sla_l(self): self.emit(0xCB, 0x25)
    def sra_a(self): self.emit(0xCB, 0x2F)
    def sra_c(self): self.emit(0xCB, 0x29)
    def sra_h(self): self.emit(0xCB, 0x2C)
    def srl_e(self): self.emit(0xCB, 0x3B)
    def rl_c(self): self.emit(0xCB, 0x11)
    def rl_h(self): self.emit(0xCB, 0x14)
    def rr_d(self): self.emit(0xCB, 0x1A)
    def rr_e(self): self.emit(0xCB, 0x1B)
    def rr_l(self): self.emit(0xCB, 0x1D)

    # === Bit test ===

    def bit_7_a(self): self.emit(0xCB, 0x7F)
    def bit_7_c(self): self.emit(0xCB, 0x79)
    def bit_7_d(self): self.emit(0xCB, 0x7A)
    def bit_7_h(self): self.emit(0xCB, 0x7C)

    # === Stack ===

    def push_af(self): self.emit(0xF5)
    def push_bc(self): self.emit(0xC5)
    def push_de(self): self.emit(0xD5)
    def push_hl(self): self.emit(0xE5)
    def push_ix(self): self.emit(0xDD, 0xE5)
    def push_iy(self): self.emit(0xFD, 0xE5)
    def pop_af(self): self.emit(0xF1)
    def pop_bc(self): self.emit(0xC1)
    def pop_de(self): self.emit(0xD1)
    def pop_hl(self): self.emit(0xE1)
    def pop_ix(self): self.emit(0xDD, 0xE1)
    def pop_iy(self): self.emit(0xFD, 0xE1)

    # === Block transfer ===

    def ldir(self): self.emit(0xED, 0xB0)

    # === Exchange ===

    def ex_de_hl(self): self.emit(0xEB)
    def ex_sp_hl(self): self.emit(0xE3)
    def ex_sp_ix(self): self.emit(0xDD, 0xE3)
    def ex_sp_iy(self): self.emit(0xFD, 0xE3)
    def ex_af_af(self): self.emit(0x08)
    def exx(self): self.emit(0xD9)

    # === Data ===

    def db(self, *vals):
        for v in vals:
            self.emit(v)

    def dw(self, *vals):
        for v in vals:
            self.emit_word(v & 0xFFFF)

    def d3(self, *vals):
        """Emit 24-bit values (for 3-byte pointer variables in data section)."""
        for v in vals:
            self.emit_addr(v & 0xFFFFFF)

    def ds(self, n):
        for _ in range(n):
            self.emit(0)

    def ascii(self, s):
        for c in s:
            self.emit(ord(c))

    # ================================================================
    # .LIS / .SIS prefixed instructions — REGISTER-ONLY forms.
    # Used for 16-bit math in the inference core.
    #
    # Per CEmu (see caveat #2 in the module header), every
    # suffixed instruction that touches memory or the stack executes in
    # Z80 mode: data addresses become {MBASE, addr16} and pushes/pops use
    # the SPS stack. Those forms can never address program data in an ADL
    # program and were REMOVED from this builder; ez80interp rejects their
    # opcodes. Only register-register/immediate suffixed ops remain.
    #
    # Values written by these ops are masked to 16 bits and ZERO the
    # pair's bits 16-23 (CEmu cpu_write_rp); do not rely on either the old
    # "preserved upper byte" model or the zeroing (Zilog: undefined).
    # Flags: C/Z/S/PV all reflect the 16-bit result (S-only signed
    # compares are overflow-blind on any CPU; see caveat #3).
    # ================================================================

    # 16-bit register pair arithmetic
    def add_hl_de_16(self): self.emit(self.LIS, 0x19)
    def add_hl_bc_16(self): self.emit(self.LIS, 0x09)
    def add_hl_hl_16(self): self.emit(self.LIS, 0x29)
    def add_hl_sp_16(self): self.emit(self.LIS, 0x39)
    def sbc_hl_de_16(self): self.emit(self.LIS, 0xED, 0x52)
    def sbc_hl_bc_16(self): self.emit(self.LIS, 0xED, 0x42)

    # 16-bit immediate loads (.SIS: 2-byte immediate, register-only)
    def ld_hl_nn_16(self, val): self.emit(self.SIS, 0x21); self.emit_word(val)
    def ld_de_nn_16(self, val): self.emit(self.SIS, 0x11); self.emit_word(val)
    def ld_bc_nn_16(self, val): self.emit(self.SIS, 0x01); self.emit_word(val)

    # 16-bit inc/dec for register pairs
    def inc_hl_16(self): self.emit(self.LIS, 0x23)
    def inc_de_16(self): self.emit(self.LIS, 0x13)
    def inc_bc_16(self): self.emit(self.LIS, 0x03)
    def dec_hl_16(self): self.emit(self.LIS, 0x2B)
    def dec_bc_16(self): self.emit(self.LIS, 0x0B)

    # 16-bit exchange (register-only)
    def ex_de_hl_16(self): self.emit(self.LIS, 0xEB)

    # ================================================================
    # IY save/restore (for TI-OS compatibility)
    # TI-OS uses IY as flags base pointer (0xD00080). Save before computation
    # that uses IY, restore before TI-OS syscalls.
    # ================================================================

    def ld_mem_label_iy(self, label):
        """Store IY to memory (24-bit). Use to save TI-OS IY value."""
        self.emit(0xFD, 0x22); self.fixup_word(label)

    def ld_iy_mem_label(self, label):
        """Load IY from memory (24-bit). Use to restore TI-OS IY value."""
        self.emit(0xFD, 0x2A); self.fixup_word(label)
