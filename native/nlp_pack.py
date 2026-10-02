# -*- coding: utf-8 -*-
"""
NLP native asset pipeline -- part 1: the `img.bin` PACK container + SERI descriptors.

Format reference (authoritative):
  gdkchan/NLPUnpacker  NLPUnpacker/Program.cs    (PACK + SERI readers)
  gdkchan/Ohana3DS-Rebirth  Models/NewLovePlus   (SMES/SMAT/BONE consumers)
"""
import struct
import zlib
import os

BLOCK_ALIGN = 0x800
BLOCK_MASK = BLOCK_ALIGN - 1


class PackFile(object):
    __slots__ = ('index', 'pack', 'signature', 'name', 'raw_len', 'raw_off',
                 'flags', 'comp_len', 'comp_off', 'unknown0', 'unknown1')

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))

    @property
    def compressed(self):
        return bool(self.flags & 1) and self.comp_len > 0

    @property
    def signature_str(self):
        return self.signature.decode('latin1')

    def __repr__(self):
        return '<%s #%d %r %d%s>' % (os.path.basename(self.pack.path), self.index,
                                     self.name, self.raw_len,
                                     ' zlib' if self.compressed else '')


class Pack(object):
    def __init__(self, path, base, header):
        self.path = path
        self.base = base
        (self.signature, self.file_count, self.strptrs_rel, self.strtab_rel,
         self.data_rel, self.decomp_len, self.comp_len, self.padding) = header
        self.files = []

    @property
    def strptrs_off(self):
        return self.base + self.strptrs_rel

    @property
    def strtab_off(self):
        return self.base + self.strtab_rel

    @property
    def end(self):
        return self.base + self.comp_len

    def __repr__(self):
        return '<Pack @0x%X %d files>' % (self.base, self.file_count)


def read_pack(f, base):
    f.seek(base)
    head = f.read(0x20)
    if len(head) < 0x20 or head[:4] != b'PACK':
        return None
    fc = struct.unpack_from('<I', head, 4)[0] >> 16
    hdr = (head[:4], fc,
           struct.unpack_from('<I', head, 8)[0],
           struct.unpack_from('<I', head, 0xC)[0],
           struct.unpack_from('<I', head, 0x10)[0],
           struct.unpack_from('<I', head, 0x14)[0],
           struct.unpack_from('<I', head, 0x18)[0],
           struct.unpack_from('<I', head, 0x1C)[0])
    p = Pack(f.name, base, hdr)

    f.seek(base + 0x20)
    entries = f.read(fc * 0x20)
    f.seek(p.strptrs_off)
    strptrs = struct.unpack_from('<%dI' % fc, f.read(fc * 4), 0) if fc else ()
    # read a generous slice of the string table
    tab_len = max(0x1000, min(0x20000, p.end - p.strtab_off)) if p.strtab_off < p.end else 0x1000
    f.seek(p.strtab_off)
    strtab = f.read(tab_len)

    for i in range(fc):
        e = entries[i * 0x20:(i + 1) * 0x20]
        if len(e) < 0x20:
            break
        u0, rl, ro, u1, fl, cpl, cpo = struct.unpack_from('<IIIIIII', e, 4)
        off = strptrs[i] if i < len(strptrs) else 0
        nm = b''
        if off < len(strtab):
            end = strtab.find(b'\x00', off)
            nm = strtab[off:end if end >= 0 else off + 64]
        p.files.append(PackFile(index=i, pack=p, signature=e[:4],
                                name=nm.decode('latin1'), raw_len=rl,
                                raw_off=ro + base, flags=fl, comp_len=cpl,
                                comp_off=cpo + base, unknown0=u0, unknown1=u1))
    return p


def iter_packs(path):
    size = os.path.getsize(path)
    with open(path, 'rb') as f:
        pos = 0
        while pos + 0x20 <= size:
            p = read_pack(f, pos)
            if p is None:
                pos = (pos & ~BLOCK_MASK) + BLOCK_ALIGN
                continue
            yield p
            nxt = p.end
            if nxt <= pos:
                break
            pos = nxt
            if pos & BLOCK_MASK:
                pos = (pos & ~BLOCK_MASK) + BLOCK_ALIGN


def pack_bytes(f, pf):
    """Raw (decompressed) payload of one pack entry."""
    if pf.compressed:
        f.seek(pf.comp_off)
        return zlib.decompress(f.read(pf.comp_len))
    f.seek(pf.raw_off)
    return f.read(pf.raw_len)


def read_seri_file(path, pf):
    """Return (seri_bytes, string_table_bytes) for a pack entry, reading the
    containing pack header so the SERI pointer/string tables can be resolved."""
    with open(path, 'rb') as f:
        p = pf.pack
        f.seek(p.strptrs_off)
        strptrs = struct.unpack_from('<%dI' % p.file_count, f.read(p.file_count * 4), 0)
        f.seek(p.strtab_off)
        strtab = f.read(max(0x1000, min(0x100000, max(0, p.end - p.strtab_off))))
        payload = pack_bytes(f, pf)
    return payload, strptrs, strtab


# --------------------------------------------------------------------------- SERI
class Seri(object):
    def __init__(self):
        self.params = {}
        self.types = {}

    def get(self, name, default=None):
        return self.params.get(name, default)

    def __repr__(self):
        return '<Seri %s>' % ', '.join('%s=%r' % (k, v)
                                       for k, v in list(self.params.items())[:10])


def parse_seri_block(block, strptrs, strtab):
    """Parse a SERI structure that lives at block[0] ('SERI').
    `strptrs`/`strtab` are pack-relative tables; offsets inside the SERI resolve
    against them (NLPUnpacker semantics)."""
    def cstr(tab, off, limit=256):
        if off < 0 or off >= len(tab):
            return ''
        end = tab.find(b'\x00', off)
        return tab[off:end if end >= 0 else min(len(tab), off + limit)].decode('latin1')

    def name_at(o):
        return cstr(strtab, o)

    if block[:4] != b'SERI':
        return None
    out = Seri()
    a = struct.unpack_from('<I', block, 4)[0]
    values_off = a + 4
    count = struct.unpack_from('<H', block, 8)[0]
    types_off = values_off - count
    if types_off < 0 or types_off + count > len(block):
        return None

    def strptr(v):
        """NLPUnpacker: value is a 1-based index into the pack's string-pointer table."""
        k = v - 1
        if k < 0 or k >= len(strptrs):
            return ''
        return cstr(strtab, strptrs[k])

    def value(vo, t, name):
        if t == 's':
            return cstr(strtab, vo)
        if t == 'i':
            v = struct.unpack_from('<i', block, values_off + vo)[0]
            return strptr(v) if name in ('bone', 'smes', 'smat', 'tex', 'hair_length') else v
        if t == 'b':
            return block[values_off + vo] == 1
        if t == 'f':
            return struct.unpack_from('<f', block, values_off + vo)[0]
        if t == 'a':
            o = values_off + vo
            if o + 4 > len(block):
                return []
            dt = chr(block[o])
            n = struct.unpack_from('<H', block, o + 2)[0]
            ptrs = struct.unpack_from('<%dH' % n, block, o + 4)
            if dt == 's':
                return [cstr(strtab, q) for q in ptrs]
            if dt == 'i':
                if name in ('texi', 'model', 'cloth', 'list', 'meshname', 'matname'):
                    res = []
                    for q in ptrs:
                        sp = struct.unpack_from('<I', block, values_off + q)[0]
                        res.append((sp, strptr(sp)))
                    return res
                return [struct.unpack_from('<i', block, values_off + q)[0] for q in ptrs]
            if dt == 'b':
                return [block[values_off + q] == 1 for q in ptrs]
            if dt == 'f':
                return [struct.unpack_from('<f', block, values_off + q)[0] for q in ptrs]
            if dt == 'h':
                res = []
                for q in ptrs:
                    o2 = values_off + q
                    n2 = struct.unpack_from('<H', block, o2)[0]
                    t2 = o2 + 2 + n2 * 4
                    d = {}
                    for k in range(n2):
                        no, xo = struct.unpack_from('<HH', block, o2 + 2 + k * 4)
                        nm2 = name_at(no)
                        d[nm2] = value(xo, chr(block[t2 + k]), nm2)
                    res.append(d)
                return res
            return []
        return None

    for i in range(count):
        no, vo = struct.unpack_from('<HH', block, 0xA + i * 4)
        t = chr(block[types_off + i])
        nm = name_at(no)
        out.types[nm] = t
        out.params[nm] = value(vo, t, nm)
    return out
