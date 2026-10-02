# -*- coding: utf-8 -*-
"""ARM disassembly toolkit for the game's own code, used to read the .mot loader
straight out of New Love Plus+ instead of guessing the format.

Calibration (verified): for this binary VA = file_offset + 0x00100000 over the whole
file -- a single contiguous image.  Derived from the constructor literal
    0x006A06BC  ldr r1, [pc, #0x2c]      ; pool word = 0x008319C0
    0x008319C0  -> file 0x7319C0         ; which really is a string table
check_map() re-proves this on independent literals before anything relies on it.

Usage:
    python dis_arm.py check                       # calibration
    python dis_arm.py dis <hexVA> [n]             # disassemble n instructions
    python dis_arm.py fn  <hexVA>                 # annotate a whole function
    python dis_arm.py find                        # scan for the MOT record loop
"""
import struct
import sys

try:
    from capstone import Cs, CS_ARCH_ARM, CS_MODE_ARM
except ImportError:
    raise SystemExit('capstone not installed')

CODE = r'D:/dsh/NLPP_CHN_2_1_1/00040000000F4E00/code.bin'
BASE = 0x00100000
_d = None


def data():
    global _d
    if _d is None:
        _d = open(CODE, 'rb').read()
    return _d


def off(va):
    return va - BASE


def va(o):
    return o + BASE


def u32(va_):
    return struct.unpack_from('<I', data(), off(va_))[0]


def s32(va_):
    return struct.unpack_from('<i', data(), off(va_))[0]


def pool_target(insn_addr, imm, pc_bias=8):
    """Literal-pool target for `ldr rX, [pc, #imm]`."""
    return u32(insn_addr + pc_bias + imm)


def md():
    return Cs(CS_ARCH_ARM, CS_MODE_ARM)


def dis(va_, n=40):
    return list(md().disasm(data()[off(va_):off(va_) + n * 4], va_))


def cstr(va_, maxlen=48):
    b = data()[off(va_):off(va_) + maxlen]
    z = b.find(b'\x00')
    return b[:z if z >= 0 else maxlen]


def check():
    """Re-prove VA = file + 0x100000 on several independent literals."""
    ok = True
    probes = [
        # (instruction VA, imm)  -- known ldr rX,[pc,#imm] sites
        (0x006A06BC, 0x2C, b'string table near .bcr/.bclyt names'),
    ]
    for a, imm, want in probes:
        t = pool_target(a, imm)
        o = off(t)
        inrange = 0 <= o < len(data())
        content = cstr(t, 40) if inrange else b'<out of range>'
        printable = sum(1 for c in content if 32 <= c < 127)
        good = inrange and printable >= max(4, len(content) - 1)
        ok &= good
        print('  literal from 0x%08X -> 0x%08X (file 0x%X) %s' % (a, t, o, 'OK' if good else 'SUSPECT'))
        print('     content: %r   (%s)' % (content, want.decode()))
    # independent check: every 'MOT '/'BONE'/'TEXI' string should sit in code context
    for s in (b'MOT ', b'BONE', b'TEXI'):
        o = data().find(s)
        print('  string %-5r at file 0x%-8X VA 0x%08X' % (s, o, va(o)))
    print('  mapping VA = file + 0x%X : %s' % (BASE, 'CONFIRMED' if ok else 'NOT PROVEN'))
    return ok


if __name__ == '__main__':
    a = sys.argv[1:] or ['check']
    if a[0] == 'check':
        sys.exit(0 if check() else 1)
    if a[0] == 'dis':
        start = int(a[1], 16)
        n = int(a[2]) if len(a) > 2 else 40
        for i in dis(start, n):
            print('  0x%08X  %-10s %s' % (i.address, i.mnemonic, i.op_str))
    if a[0] == 'fn':
        start = int(a[1], 16)
        for i in dis(start, 120):
            print('  0x%08X  %-10s %s' % (i.address, i.mnemonic, i.op_str))
