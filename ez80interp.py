#!/usr/bin/env python3
"""
ez80interp.py — a minimal eZ80 ADL-mode interpreter that EXECUTES the real
machine code emitted by libez80/buildchat84, so the faithfulness gate can run the
actual emitted bytes (not a re-modeled contract) and compare against intkernel.

GRADER-OWNED. Closes the gap test_faithfulness.py explicitly cannot: assembly-
level bugs that live in the literal bytes (e.g. the MA_NEG carry bug), invisible
to a Python re-implementation of the contract.

SCOPE: it implements exactly the opcode subset eZ80Builder can emit (a closed
set). Any byte it does not recognize raises UnsupportedOpcode — so if the codegen
ever starts emitting a new instruction, the gate fails loudly instead of silently
mis-executing. Only the compute routines are ever run (TOKENIZE, LAYER*, RELU*,
ARGMAX, …); TI-OS I/O is never entered, so no syscalls are emulated.

ADL mode model: BC/DE/HL/IX/IY/SP/PC are 24-bit; A/flags are 8-bit. 8-bit loads
preserve the pair's upper byte (caveat #4). .LIS/.SIS ops compute on the low 16
bits; the low bits — all the codegen ever reads back from them — are exact.
"""


class UnsupportedOpcode(Exception):
    pass


class CPUError(Exception):
    pass


MASK24 = 0xFFFFFF
RET_SENTINEL = 0xBADBAD          # return address that ends run()


def _s8(v):
    v &= 0xFF
    return v - 256 if v & 0x80 else v


class CPU:
    def __init__(self, mem, base, org):
        self.mem = mem            # bytearray
        self.base = base          # address of mem[0]
        self.a = 0
        self.bc = self.de = self.hl = 0
        self.ix = self.iy = 0
        self.sp = base + len(mem) - 16   # stack near top of region
        self.pc = 0
        self.fz = self.fc = self.fs = False
        self.steps = 0

    # --- memory ---
    def r8(self, addr):
        return self.mem[addr - self.base]

    def w8(self, addr, v):
        self.mem[addr - self.base] = v & 0xFF

    def r16(self, addr):
        return self.r8(addr) | (self.r8(addr + 1) << 8)

    def w16(self, addr, v):
        self.w8(addr, v); self.w8(addr + 1, v >> 8)

    def r24(self, addr):
        return self.r8(addr) | (self.r8(addr + 1) << 8) | (self.r8(addr + 2) << 16)

    def w24(self, addr, v):
        self.w8(addr, v); self.w8(addr + 1, v >> 8); self.w8(addr + 2, v >> 16)

    # --- 8-bit register views over the 24-bit pairs (upper byte preserved) ---
    @property
    def b(self): return (self.bc >> 8) & 0xFF
    @b.setter
    def b(self, v): self.bc = (self.bc & 0xFF00FF) | ((v & 0xFF) << 8)
    @property
    def c(self): return self.bc & 0xFF
    @c.setter
    def c(self, v): self.bc = (self.bc & 0xFFFF00) | (v & 0xFF)
    @property
    def d(self): return (self.de >> 8) & 0xFF
    @d.setter
    def d(self, v): self.de = (self.de & 0xFF00FF) | ((v & 0xFF) << 8)
    @property
    def e(self): return self.de & 0xFF
    @e.setter
    def e(self, v): self.de = (self.de & 0xFFFF00) | (v & 0xFF)
    @property
    def h(self): return (self.hl >> 8) & 0xFF
    @h.setter
    def h(self, v): self.hl = (self.hl & 0xFF00FF) | ((v & 0xFF) << 8)
    @property
    def l(self): return self.hl & 0xFF
    @l.setter
    def l(self, v): self.hl = (self.hl & 0xFFFF00) | (v & 0xFF)

    # --- fetch ---
    def fetch8(self):
        v = self.r8(self.pc); self.pc = (self.pc + 1) & MASK24; return v

    def fetch16(self):
        v = self.r16(self.pc); self.pc = (self.pc + 2) & MASK24; return v

    def fetch24(self):
        v = self.r24(self.pc); self.pc = (self.pc + 3) & MASK24; return v

    # --- flag helpers ---
    def _setzs8(self, v):
        v &= 0xFF
        self.fz = (v == 0)
        self.fs = bool(v & 0x80)

    def _add8(self, val, carry=0):
        r = self.a + val + carry
        self.fc = r > 0xFF
        self.a = r & 0xFF
        self._setzs8(self.a)

    def _sub8(self, val, carry=0, store=True):
        r = self.a - val - carry
        self.fc = r < 0                  # borrow
        res = r & 0xFF
        self.fz = (res == 0)
        self.fs = bool(res & 0x80)
        if store:
            self.a = res

    def _add24(self, reg, val):
        r = reg + val
        self.fc = r > MASK24
        return r & MASK24

    def _sbc24(self, reg, val):
        r = reg - val - (1 if self.fc else 0)
        self.fc = r < 0
        res = r & MASK24
        self.fz = (res == 0)
        self.fs = bool(res & 0x800000)
        return res

    # --- run loop ---
    def call(self, target):
        self.sp = (self.sp - 3) & MASK24
        self.w24(self.sp, self.pc)
        self.pc = target

    def run(self, entry, max_steps=20_000_000):
        self.pc = entry
        self.sp = (self.sp - 3) & MASK24
        self.w24(self.sp, RET_SENTINEL)
        self.steps = 0
        while True:
            if self.pc == RET_SENTINEL:
                return
            self.step()
            self.steps += 1
            if self.steps > max_steps:
                raise CPUError(f"step limit exceeded at pc={self.pc:06X}")

    def step(self):
        op = self.fetch8()
        if op == 0xCB:
            return self._cb()
        if op == 0xDD:
            return self._idx('ix')
        if op == 0xFD:
            return self._idx('iy')
        if op == 0xED:
            return self._ed()
        if op == 0x52:               # .LIS prefix (16-bit register ops)
            return self._lis()
        if op == 0x40:               # .SIS prefix (16-bit imm + reg)
            return self._sis()
        return self._main(op)

    # --- main page ---
    def _main(self, op):
        a = self
        if op == 0x00: return                          # nop
        # immediate 24-bit loads
        if op == 0x21: a.hl = a.fetch24(); return
        if op == 0x11: a.de = a.fetch24(); return
        if op == 0x01: a.bc = a.fetch24(); return
        # 8-bit immediate loads
        if op == 0x3E: a.a = a.fetch8(); return
        if op == 0x06: a.b = a.fetch8(); return
        if op == 0x0E: a.c = a.fetch8(); return
        if op == 0x16: a.d = a.fetch8(); return
        if op == 0x1E: a.e = a.fetch8(); return
        if op == 0x26: a.h = a.fetch8(); return
        if op == 0x2E: a.l = a.fetch8(); return
        if op == 0x36: a.w8(a.hl, a.fetch8()); return  # LD (HL),n
        # absolute mem load/store (24-bit address operand)
        if op == 0x2A: a.hl = a.r24(a.fetch24()); return
        if op == 0x22: a.w24(a.fetch24(), a.hl); return
        if op == 0x3A: a.a = a.r8(a.fetch24()); return
        if op == 0x32: a.w8(a.fetch24(), a.a); return
        # (HL)/(DE)/(BC) loads
        if op == 0x7E: a.a = a.r8(a.hl); return
        if op == 0x77: a.w8(a.hl, a.a); return
        if op == 0x5E: a.e = a.r8(a.hl); return
        if op == 0x56: a.d = a.r8(a.hl); return
        if op == 0x46: a.b = a.r8(a.hl); return
        if op == 0x4E: a.c = a.r8(a.hl); return
        if op == 0x72: a.w8(a.hl, a.d); return
        if op == 0x73: a.w8(a.hl, a.e); return
        if op == 0x0A: a.a = a.r8(a.bc); return
        if op == 0x1A: a.a = a.r8(a.de); return
        if op == 0x12: a.w8(a.de, a.a); return
        # reg-reg loads
        if op in _LD_RR:
            dst, src = _LD_RR[op]
            setattr(a, dst, getattr(a, src)); return
        # 8-bit ALU
        if op == 0xC6: a._add8(a.fetch8()); return
        if op == 0xCE: a._add8(a.fetch8(), 1 if a.fc else 0); return
        if op == 0xD6: a._sub8(a.fetch8()); return
        if op == 0xFE: a._sub8(a.fetch8(), store=False); return
        if op == 0xE6: a.a &= a.fetch8(); a.fc = False; a._setzs8(a.a); return
        if op == 0xF6: a.a |= a.fetch8(); a.fc = False; a._setzs8(a.a); return
        if op == 0x87: a._add8(a.a); return
        if op == 0x80: a._add8(a.b); return
        if op == 0x81: a._add8(a.c); return
        if op == 0x83: a._add8(a.e); return
        if op == 0x84: a._add8(a.h); return
        if op == 0x85: a._add8(a.l); return
        if op == 0x86: a._add8(a.r8(a.hl)); return
        if op == 0x8F: a._add8(a.a, 1 if a.fc else 0); return
        if op == 0x88: a._add8(a.b, 1 if a.fc else 0); return
        if op == 0x8A: a._add8(a.d, 1 if a.fc else 0); return
        if op == 0x8E: a._add8(a.r8(a.hl), 1 if a.fc else 0); return
        if op == 0x90: a._sub8(a.b); return
        if op == 0x91: a._sub8(a.c); return
        if op == 0x94: a._sub8(a.h); return
        if op == 0x95: a._sub8(a.l); return
        if op == 0x96: a._sub8(a.r8(a.hl)); return
        if op == 0x9F:                                  # SBC A,A
            v = 0xFF if a.fc else 0x00
            borrow = 1 if a.fc else 0
            a._sub8(a.a, borrow); a.a = v; a._setzs8(v); a.fc = bool(borrow); return
        if op == 0x9E: a._sub8(a.r8(a.hl), 1 if a.fc else 0); return
        if op == 0xB8: a._sub8(a.b, store=False); return  # CP B
        if op == 0xBF: a._sub8(a.a, store=False); return  # CP A
        if op == 0xBE: a._sub8(a.r8(a.hl), store=False); return  # CP (HL)
        if op == 0xA7: a.fc = False; a._setzs8(a.a); return  # AND A
        if op == 0xB7: a.fc = False; a._setzs8(a.a); return  # OR A
        if op == 0xB1: a.a |= a.c; a.fc = False; a._setzs8(a.a); return
        if op == 0xB5: a.a |= a.l; a.fc = False; a._setzs8(a.a); return
        if op == 0xAF: a.a = 0; a.fc = False; a._setzs8(0); return  # XOR A
        if op == 0xAE: a.a ^= a.r8(a.hl); a.fc = False; a._setzs8(a.a); return
        if op == 0x2F: a.a ^= 0xFF; return              # CPL
        if op == 0x3C: a.a = (a.a + 1) & 0xFF; a._setzs8(a.a); return  # INC A
        if op == 0x3D: a.a = (a.a - 1) & 0xFF; a._setzs8(a.a); return  # DEC A
        # 24-bit pair add
        if op == 0x09: a.hl = a._add24(a.hl, a.bc); return
        if op == 0x19: a.hl = a._add24(a.hl, a.de); return
        if op == 0x29: a.hl = a._add24(a.hl, a.hl); return
        if op == 0x39: a.hl = a._add24(a.hl, a.sp); return
        # 24-bit inc/dec
        if op == 0x03: a.bc = (a.bc + 1) & MASK24; return
        if op == 0x0B: a.bc = (a.bc - 1) & MASK24; return
        if op == 0x13: a.de = (a.de + 1) & MASK24; return
        if op == 0x23: a.hl = (a.hl + 1) & MASK24; return
        if op == 0x2B: a.hl = (a.hl - 1) & MASK24; return
        # 8-bit inc/dec on pair-halves (Z/S only)
        if op == 0x04: a.b = (a.b + 1) & 0xFF; a._setzs8(a.b); return
        if op == 0x05: a.b = (a.b - 1) & 0xFF; a._setzs8(a.b); return
        if op == 0x0C: a.c = (a.c + 1) & 0xFF; a._setzs8(a.c); return
        if op == 0x0D: a.c = (a.c - 1) & 0xFF; a._setzs8(a.c); return
        if op == 0x14: a.d = (a.d + 1) & 0xFF; a._setzs8(a.d); return
        if op == 0x15: a.d = (a.d - 1) & 0xFF; a._setzs8(a.d); return
        if op == 0x1C: a.e = (a.e + 1) & 0xFF; a._setzs8(a.e); return
        if op == 0x1D: a.e = (a.e - 1) & 0xFF; a._setzs8(a.e); return
        if op == 0x24: a.h = (a.h + 1) & 0xFF; a._setzs8(a.h); return
        if op == 0x25: a.h = (a.h - 1) & 0xFF; a._setzs8(a.h); return
        if op == 0x2C: a.l = (a.l + 1) & 0xFF; a._setzs8(a.l); return
        # rotates on A
        if op == 0x07:                                  # RLCA
            a.fc = bool(a.a & 0x80); a.a = ((a.a << 1) | (1 if a.fc else 0)) & 0xFF; return
        if op == 0x0F:                                  # RRCA
            a.fc = bool(a.a & 0x01); a.a = ((a.a >> 1) | ((1 if a.fc else 0) << 7)) & 0xFF; return
        if op == 0x1F:                                  # RRA
            nc = bool(a.a & 0x01); a.a = ((a.a >> 1) | ((1 if a.fc else 0) << 7)) & 0xFF; a.fc = nc; return
        # exchange / stack / block
        if op == 0xEB: a.hl, a.de = a.de, a.hl; return
        if op == 0xE3:
            t = a.r24(a.sp); a.w24(a.sp, a.hl); a.hl = t; return
        if op == 0xF5: a._push((a.a << 8) | a._flags_byte()); return
        if op == 0xC5: a._push(a.bc); return
        if op == 0xD5: a._push(a.de); return
        if op == 0xE5: a._push(a.hl); return
        if op == 0xF1:
            v = a._pop(); a.a = (v >> 8) & 0xFF; a._set_flags_byte(v & 0xFF); return
        if op == 0xC1: a.bc = a._pop(); return
        if op == 0xD1: a.de = a._pop(); return
        if op == 0xE1: a.hl = a._pop(); return
        # control flow
        if op == 0xC9: a.pc = a._pop(); return          # RET
        if op == 0xC8:                                   # RET Z
            if a.fz: a.pc = a._pop()
            return
        if op == 0xC0:                                   # RET NZ
            if not a.fz: a.pc = a._pop()
            return
        if op == 0xCD: a.call(a.fetch24()); return       # CALL
        if op == 0xC3: a.pc = a.fetch24(); return        # JP
        if op == 0xC2:
            t = a.fetch24();  a.pc = t if not a.fz else a.pc; return
        if op == 0xCA:
            t = a.fetch24();  a.pc = t if a.fz else a.pc; return
        if op == 0xFA:
            t = a.fetch24();  a.pc = t if a.fs else a.pc; return
        if op == 0xDA:
            t = a.fetch24();  a.pc = t if a.fc else a.pc; return
        if op == 0x18: d = _s8(a.fetch8()); a.pc = (a.pc + d) & MASK24; return  # JR
        if op == 0x20:
            d = _s8(a.fetch8());  a.pc = (a.pc + d) & MASK24 if not a.fz else a.pc; return
        if op == 0x28:
            d = _s8(a.fetch8());  a.pc = (a.pc + d) & MASK24 if a.fz else a.pc; return
        if op == 0x30:
            d = _s8(a.fetch8());  a.pc = (a.pc + d) & MASK24 if not a.fc else a.pc; return
        if op == 0x38:
            d = _s8(a.fetch8());  a.pc = (a.pc + d) & MASK24 if a.fc else a.pc; return
        if op == 0x10:                                   # DJNZ
            d = _s8(a.fetch8()); a.b = (a.b - 1) & 0xFF
            if a.b != 0: a.pc = (a.pc + d) & MASK24
            return
        raise UnsupportedOpcode(f"main opcode {op:02X} at pc={(a.pc-1)&MASK24:06X}")

    def _push(self, v):
        self.sp = (self.sp - 3) & MASK24
        self.w24(self.sp, v)

    def _pop(self):
        v = self.r24(self.sp)
        self.sp = (self.sp + 3) & MASK24
        return v

    def _flags_byte(self):
        return (0x80 if self.fs else 0) | (0x40 if self.fz else 0) | (0x01 if self.fc else 0)

    def _set_flags_byte(self, v):
        self.fs = bool(v & 0x80); self.fz = bool(v & 0x40); self.fc = bool(v & 0x01)

    # --- CB page (shifts/rotates/bit) ---
    def _cb(self):
        op = self.fetch8()
        a = self
        if op == 0x27:                                   # SLA A
            a.fc = bool(a.a & 0x80); a.a = (a.a << 1) & 0xFF; a._setzs8(a.a); return
        if op == 0x25:                                   # SLA L
            v = a.l; a.fc = bool(v & 0x80); v = (v << 1) & 0xFF; a.l = v; a._setzs8(v); return
        if op == 0x2F:                                   # SRA A
            a.fc = bool(a.a & 0x01); a.a = ((a.a >> 1) | (a.a & 0x80)) & 0xFF; a._setzs8(a.a); return
        if op == 0x29:                                   # SRA C
            v = a.c; a.fc = bool(v & 0x01); v = ((v >> 1) | (v & 0x80)) & 0xFF; a.c = v; a._setzs8(v); return
        if op == 0x2C:                                   # SRA H
            v = a.h; a.fc = bool(v & 0x01); v = ((v >> 1) | (v & 0x80)) & 0xFF; a.h = v; a._setzs8(v); return
        if op == 0x3B:                                   # SRL E
            v = a.e; a.fc = bool(v & 0x01); v = (v >> 1) & 0xFF; a.e = v; a._setzs8(v); return
        if op in (0x11, 0x14, 0x1A, 0x1B, 0x1D):         # RR/RL on D/E/L/C/H
            return self._cb_rot(op)
        if op in (0x7F, 0x79, 0x7A, 0x7C):               # BIT 7,r
            reg = {0x7F: 'a', 0x79: 'c', 0x7A: 'd', 0x7C: 'h'}[op]
            a.fz = not (getattr(a, reg) & 0x80); return
        raise UnsupportedOpcode(f"CB {op:02X}")

    def _cb_rot(self, op):
        a = self
        if op == 0x1A:                                   # RR D
            v = a.d; nc = v & 1; v = (v >> 1) | ((1 if a.fc else 0) << 7); a.d = v & 0xFF; a.fc = bool(nc); a._setzs8(a.d); return
        if op == 0x1B:                                   # RR E
            v = a.e; nc = v & 1; v = (v >> 1) | ((1 if a.fc else 0) << 7); a.e = v & 0xFF; a.fc = bool(nc); a._setzs8(a.e); return
        if op == 0x1D:                                   # RR L
            v = a.l; nc = v & 1; v = (v >> 1) | ((1 if a.fc else 0) << 7); a.l = v & 0xFF; a.fc = bool(nc); a._setzs8(a.l); return
        if op == 0x11:                                   # RL C
            v = a.c; nc = v & 0x80; v = ((v << 1) | (1 if a.fc else 0)) & 0xFF; a.c = v; a.fc = bool(nc); a._setzs8(v); return
        if op == 0x14:                                   # RL H
            v = a.h; nc = v & 0x80; v = ((v << 1) | (1 if a.fc else 0)) & 0xFF; a.h = v; a.fc = bool(nc); a._setzs8(v); return
        raise UnsupportedOpcode(f"CB rot {op:02X}")

    # --- DD/FD index-register page ---
    def _idx(self, which):
        op = self.fetch8()
        a = self
        cur = a.ix if which == 'ix' else a.iy

        def setidx(v):
            if which == 'ix': a.ix = v & MASK24
            else: a.iy = v & MASK24

        if op == 0x21: setidx(a.fetch24()); return       # LD IX/IY,nn  (also labels)
        if op == 0x22: a.w24(a.fetch24(), cur); return    # LD (nn),IX/IY
        if op == 0x2A: setidx(a.r24(a.fetch24())); return  # LD IX/IY,(nn)
        if op == 0x23: setidx((cur + 1) & MASK24); return  # INC IX/IY
        if op == 0xE5: a._push(cur); return               # PUSH
        if op == 0xE1: setidx(a._pop()); return           # POP
        if op == 0xF9: a.sp = cur; return                 # LD SP,IX/IY
        if op == 0xE3:                                     # EX (SP),IX/IY
            t = a.r24(a.sp); a.w24(a.sp, cur); setidx(t); return
        if op == 0x6E: d = _s8(a.fetch8()); a.l = a.r8((cur + d) & MASK24); return  # LD L,(idx+d)
        if op == 0x66: d = _s8(a.fetch8()); a.h = a.r8((cur + d) & MASK24); return  # LD H,(idx+d)
        if op == 0x75: d = _s8(a.fetch8()); a.w8((cur + d) & MASK24, a.l); return   # LD (idx+d),L
        if op == 0x74: d = _s8(a.fetch8()); a.w8((cur + d) & MASK24, a.h); return   # LD (idx+d),H
        if op == 0x72: d = _s8(a.fetch8()); a.w8((cur + d) & MASK24, a.d); return   # LD (idx+d),D
        if op == 0x73: d = _s8(a.fetch8()); a.w8((cur + d) & MASK24, a.e); return   # LD (idx+d),E
        raise UnsupportedOpcode(f"{which.upper()} {op:02X}")

    # --- ED page ---
    def _ed(self):
        op = self.fetch8()
        a = self
        if op == 0x52: a.hl = a._sbc24(a.hl, a.de); return  # SBC HL,DE
        if op == 0x42: a.hl = a._sbc24(a.hl, a.bc); return  # SBC HL,BC
        if op == 0x53: a.w24(a.fetch24(), a.de); return     # LD (nn),DE
        if op == 0x43: a.w24(a.fetch24(), a.bc); return     # LD (nn),BC
        if op == 0x4B: a.bc = a.r24(a.fetch24()); return    # LD BC,(nn)
        if op == 0x5B: a.de = a.r24(a.fetch24()); return    # LD DE,(nn)
        if op == 0x73: a.w24(a.fetch24(), a.sp); return     # LD (nn),SP
        if op == 0x7B: a.sp = a.r24(a.fetch24()); return    # LD SP,(nn)
        if op == 0xB0:                                       # LDIR (24-bit count)
            while a.bc != 0:
                a.w8(a.de, a.r8(a.hl))
                a.hl = (a.hl + 1) & MASK24
                a.de = (a.de + 1) & MASK24
                a.bc = (a.bc - 1) & MASK24
            return
        raise UnsupportedOpcode(f"ED {op:02X}")

    # --- .LIS prefix (16-bit register arithmetic; low 16 bits are exact) ---
    def _lis(self):
        op = self.fetch8()
        a = self
        if op == 0xED:
            op2 = self.fetch8()
            if op2 == 0x52:                                 # SBC HL,DE (16)
                a.hl = a._lo16_sbc(a.hl, a.de); return
            if op2 == 0x42:                                 # SBC HL,BC (16)
                a.hl = a._lo16_sbc(a.hl, a.bc); return
            if op2 == 0x53: a.w16(a.fetch24(), a.de); return
            if op2 == 0x43: a.w16(a.fetch24(), a.bc); return
            if op2 == 0x4B: a.bc = (a.bc & 0xFF0000) | a.r16(a.fetch24()); return
            if op2 == 0x5B: a.de = (a.de & 0xFF0000) | a.r16(a.fetch24()); return
            if op2 == 0xB0:                                 # LDIR (16-bit count)
                cnt = a.bc & 0xFFFF
                while cnt != 0:
                    a.w8(a.de, a.r8(a.hl))
                    a.hl = (a.hl + 1) & MASK24
                    a.de = (a.de + 1) & MASK24
                    cnt -= 1
                a.bc = a.bc & 0xFF0000
                return
            raise UnsupportedOpcode(f"LIS ED {op2:02X}")
        if op == 0x19: a.hl = a._lo16_add(a.hl, a.de); return
        if op == 0x09: a.hl = a._lo16_add(a.hl, a.bc); return
        if op == 0x29: a.hl = a._lo16_add(a.hl, a.hl); return
        if op == 0x39: a.hl = a._lo16_add(a.hl, a.sp); return
        if op == 0x2A: a.hl = (a.hl & 0xFF0000) | a.r16(a.fetch24()); return
        if op == 0x22: a.w16(a.fetch24(), a.hl); return
        if op == 0x23: a.hl = (a.hl & 0xFF0000) | ((a.hl + 1) & 0xFFFF); return
        if op == 0x13: a.de = (a.de & 0xFF0000) | ((a.de + 1) & 0xFFFF); return
        if op == 0x03: a.bc = (a.bc & 0xFF0000) | ((a.bc + 1) & 0xFFFF); return
        if op == 0x2B: a.hl = (a.hl & 0xFF0000) | ((a.hl - 1) & 0xFFFF); return
        if op == 0x0B: a.bc = (a.bc & 0xFF0000) | ((a.bc - 1) & 0xFFFF); return
        if op == 0xE5: a._push16(a.hl); return
        if op == 0xD5: a._push16(a.de); return
        if op == 0xC5: a._push16(a.bc); return
        if op == 0xE1: a.hl = (a.hl & 0xFF0000) | a._pop16(); return
        if op == 0xD1: a.de = (a.de & 0xFF0000) | a._pop16(); return
        if op == 0xC1: a.bc = (a.bc & 0xFF0000) | a._pop16(); return
        if op == 0xEB: a.hl, a.de = a.de, a.hl; return
        raise UnsupportedOpcode(f"LIS {op:02X}")

    def _lo16_add(self, reg, val):
        r = (reg & 0xFFFF) + (val & 0xFFFF)
        self.fc = r > 0xFFFF
        return (reg & 0xFF0000) | (r & 0xFFFF)

    def _lo16_sbc(self, reg, val):
        r = (reg & 0xFFFF) - (val & 0xFFFF) - (1 if self.fc else 0)
        self.fc = r < 0
        res = r & 0xFFFF
        self.fz = (res == 0)
        return (reg & 0xFF0000) | res

    def _push16(self, v):
        self.sp = (self.sp - 2) & MASK24
        self.w16(self.sp, v)

    def _pop16(self):
        v = self.r16(self.sp)
        self.sp = (self.sp + 2) & MASK24
        return v

    # --- .SIS prefix (16-bit immediate loads) ---
    def _sis(self):
        op = self.fetch8()
        a = self
        if op == 0x21: a.hl = (a.hl & 0xFF0000) | a.fetch16(); return
        if op == 0x11: a.de = (a.de & 0xFF0000) | a.fetch16(); return
        if op == 0x01: a.bc = (a.bc & 0xFF0000) | a.fetch16(); return
        raise UnsupportedOpcode(f"SIS {op:02X}")


# Register-register load opcode table: opcode -> (dst, src)
_LD_RR = {
    0x78: ('a', 'b'), 0x79: ('a', 'c'), 0x7A: ('a', 'd'), 0x7B: ('a', 'e'),
    0x7C: ('a', 'h'), 0x7D: ('a', 'l'),
    0x47: ('b', 'a'), 0x41: ('b', 'c'), 0x43: ('b', 'e'), 0x44: ('b', 'h'),
    0x4F: ('c', 'a'), 0x4D: ('c', 'l'),
    0x57: ('d', 'a'), 0x54: ('d', 'h'),
    0x5F: ('e', 'a'), 0x59: ('e', 'c'), 0x5D: ('e', 'l'),
    0x67: ('h', 'a'), 0x60: ('h', 'b'),
    0x6F: ('l', 'a'), 0x69: ('l', 'c'), 0x6A: ('l', 'd'), 0x6B: ('l', 'e'),
}
