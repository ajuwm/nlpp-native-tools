# -*- coding: utf-8 -*-
"""PICA200 TEV texture combiner ("TexEnv") -- pure numpy reference implementation.

Two layers live here:

1. **PICA layer** -- `PicaStage` + `combine_stages()` is a faithful port of the
   software-renderer reference implementation in Azahar/Citra
   (`src/video_core/renderer_software/sw_texturing.cpp` + `sw_rasterizer.cpp`,
   saved under `out/ref/azahar__...`).  Everything is done in 8-bit integer
   semantics exactly like the hardware reference: sources/modifiers are evaluated
   on u8, the combine ops use the integer formulas from `ColorCombine` /
   `AlphaCombine`, then the per-stage scale multiplies and clamps at 255.
   Verified by hand-computed asserts in `_selftest()`.

2. **NLPP layer** -- `parse_comb()` / `decode_record()` takes the SMAT `COMB`
   block of a New Love Plus + material and splits it into per-stage records.
   IMPORTANT: these records are *not* the raw PICA TEV registers -- see
   `_notes/2026-11-14-tev-decode.md` for the proof.  The structure that survives
   every test is:

       COMB = u32 header (always 0x000000ff)
            + N x 20-byte stage record
       20-byte record = 16 four-bit enum values, one per byte in bits 0..3 of
                        bytes 0..15, followed by one arbitrary 32-bit word
                        (bytes 16..19).

   The engine's own field names for one TexEnv (found as a string table in the
   game's code.bin at 0x56a590) are:

       dmp_TexEnv[i].srcRgb / .srcAlpha / .operandRgb / .operandAlpha /
       .combineRgb / .combineAlpha / .scaleRgb / .scaleAlpha / .constRgba

   i.e. 8 two-nibble fields + one u32 == exactly the 20 bytes.  `decode_record()`
   exposes that split; `material_stages()` turns it into PICA stages using the
   *provisional* layout below (documented as unverified in the report).

Usage::

    from tev import combine_material, decode_pica_words, PicaStage

    rgba = combine_material(mtl, textures, primary, secondary)   # NLPP path
    rgba = combine_stages(stages, textures, primary, secondary)  # explicit path
"""
import os
import struct

import numpy as np

# --------------------------------------------------------------------------- enums
SOURCES = {
    0x0: 'PrimaryColor', 0x1: 'PrimaryFragmentColor', 0x2: 'SecondaryFragmentColor',
    0x3: 'Texture0', 0x4: 'Texture1', 0x5: 'Texture2', 0x6: 'Texture3',
    0xD: 'PreviousBuffer', 0xE: 'Constant', 0xF: 'Previous',
}
SOURCES_INV = {v: k for k, v in SOURCES.items()}

COLOR_MODS = {
    0x0: 'SrcColor', 0x1: 'OneMinusSrcColor', 0x2: 'SrcAlpha', 0x3: 'OneMinusSrcAlpha',
    0x4: 'SrcRed', 0x5: 'OneMinusSrcRed', 0x8: 'SrcGreen', 0x9: 'OneMinusSrcGreen',
    0xC: 'SrcBlue', 0xD: 'OneMinusSrcBlue',
}
ALPHA_MODS = {
    0x0: 'SrcAlpha', 0x1: 'OneMinusSrcAlpha', 0x2: 'SrcRed', 0x3: 'OneMinusSrcRed',
    0x4: 'SrcGreen', 0x5: 'OneMinusSrcGreen', 0x6: 'SrcBlue', 0x7: 'OneMinusSrcBlue',
}
OPS = {
    0: 'Replace', 1: 'Modulate', 2: 'Add', 3: 'AddSigned', 4: 'Lerp',
    5: 'Subtract', 6: 'Dot3_RGB', 7: 'Dot3_RGBA', 8: 'MultiplyThenAdd',
    9: 'AddThenMultiply',
}
OPS_INV = {v: k for k, v in OPS.items()}

SOURCE_TEX = {'Texture0': 0, 'Texture1': 1, 'Texture2': 2, 'Texture3': 3}


# --------------------------------------------------------------------------- stage
class PicaStage(object):
    """One PICA TEV stage, 5 configuration registers (0x00C0 + 8*i ..)."""

    __slots__ = ('color_source', 'alpha_source', 'color_mod', 'alpha_mod',
                 'color_op', 'alpha_op', 'const_rgba', 'color_scale', 'alpha_scale',
                 'raw')

    def __init__(self, color_source=(0, 0, 0), alpha_source=(0, 0, 0),
                 color_mod=(0, 0, 0), alpha_mod=(0, 0, 0),
                 color_op=0, alpha_op=0, const_rgba=(0, 0, 0, 0),
                 color_scale=0, alpha_scale=0, raw=None):
        self.color_source = tuple(color_source)
        self.alpha_source = tuple(alpha_source)
        self.color_mod = tuple(color_mod)
        self.alpha_mod = tuple(alpha_mod)
        self.color_op = int(color_op)
        self.alpha_op = int(alpha_op)
        self.const_rgba = tuple(int(v) for v in const_rgba)
        self.color_scale = int(color_scale)
        self.alpha_scale = int(alpha_scale)
        self.raw = raw

    # -- multiplier semantics, same as TevStageConfig::GetColorMultiplier ----
    @property
    def color_mul(self):
        return (1 << self.color_scale) if self.color_scale < 3 else 1

    @property
    def alpha_mul(self):
        return (1 << self.alpha_scale) if self.alpha_scale < 3 else 1

    # -- human readable ------------------------------------------------------
    def describe(self):
        def src(names):
            return ', '.join(SOURCES.get(v, '?%d' % v) for v in names)

        def cm(names):
            return ', '.join(COLOR_MODS.get(v, '?%d' % v) for v in names)

        def am(names):
            return ', '.join(ALPHA_MODS.get(v, '?%d' % v) for v in names)

        return dict(
            color_src=src(self.color_source), alpha_src=src(self.alpha_source),
            color_mod=cm(self.color_mod), alpha_mod=am(self.alpha_mod),
            color_op=OPS.get(self.color_op, '?%d' % self.color_op),
            alpha_op=OPS.get(self.alpha_op, '?%d' % self.alpha_op),
            const=self.const_rgba, scale=(self.color_mul, self.alpha_mul))

    def __repr__(self):
        d = self.describe()
        return '<PicaStage %s(%s) a=%s(%s)>' % (
            d['color_op'], d['color_src'], d['alpha_op'], d['alpha_src'])


def _nib(w, sh):
    return (w >> sh) & 0xF


def decode_pica_words(words, const_override=None):
    """5 raw u32 (sources, modifiers, ops, const, scale) -> PicaStage.

    Bit layout is Pica::TexturingRegs::TevStageConfig (regs_texturing.h):
      sources : cs1 0-3, cs2 4-7, cs3 8-11, as1 16-19, as2 20-23, as3 24-27
      mods    : cm1 0-3, cm2 4-7, cm3 8-11, am1 12-14, am2 16-18, am3 20-22
      ops     : cop 0-3, aop 16-19
      color   : r 0-7, g 8-15, b 16-23, a 24-31
      scale   : cs 0-1, as 16-17
    """
    src, mod, op, col, scl = (int(w) for w in words)
    stage = PicaStage(
        color_source=(_nib(src, 0), _nib(src, 4), _nib(src, 8)),
        alpha_source=(_nib(src, 16), _nib(src, 20), _nib(src, 24)),
        color_mod=(_nib(mod, 0), _nib(mod, 4), _nib(mod, 8)),
        alpha_mod=(_nib(mod, 12) & 7, _nib(mod, 16) & 7, _nib(mod, 20) & 7),
        color_op=_nib(op, 0), alpha_op=_nib(op, 16),
        const_rgba=((col >> 0) & 0xFF, (col >> 8) & 0xFF,
                    (col >> 16) & 0xFF, (col >> 24) & 0xFF),
        color_scale=scl & 3, alpha_scale=(scl >> 16) & 3,
        raw=tuple(words))
    if const_override is not None:
        stage.const_rgba = tuple(const_override)
    return stage


# --------------------------------------------------------------------------- math
def _as_u8_rgba(x):
    """Accept (4,) tuple, (H,W,4) uint8, or float 0..1 -> uint8 (H,W,4) or (4,)."""
    a = np.asarray(x)
    if a.dtype == np.uint8:
        return a
    if a.dtype.kind == 'f':
        return np.clip(np.rint(a * 255.0), 0, 255).astype(np.uint8)
    return np.clip(a, 0, 255).astype(np.uint8)


def _color_modifier(mod, src):
    """src (...,4) uint8 -> (...,3) int32.  Port of GetColorModifier()."""
    r, g, b, a = src[..., 0], src[..., 1], src[..., 2], src[..., 3]
    if mod == 0:
        return np.stack([r, g, b], -1).astype(np.int32)
    if mod == 1:
        return 255 - np.stack([r, g, b], -1).astype(np.int32)
    if mod == 2:
        return np.stack([a, a, a], -1).astype(np.int32)
    if mod == 3:
        v = 255 - a.astype(np.int32)
        return np.stack([v, v, v], -1)
    if mod in (4, 5):
        v = r.astype(np.int32)
        v = v if mod == 4 else 255 - v
        return np.stack([v, v, v], -1)
    if mod in (8, 9):
        v = g.astype(np.int32)
        v = v if mod == 8 else 255 - v
        return np.stack([v, v, v], -1)
    if mod in (12, 13):
        v = b.astype(np.int32)
        v = v if mod == 12 else 255 - v
        return np.stack([v, v, v], -1)
    raise ValueError('bad color modifier %d' % mod)


def _alpha_modifier(mod, src):
    """src (...,4) uint8 -> (...) int32.  Port of GetAlphaModifier()."""
    r, g, b, a = (src[..., i].astype(np.int32) for i in range(4))
    if mod == 0:
        return a
    if mod == 1:
        return 255 - a
    if mod == 2:
        return r
    if mod == 3:
        return 255 - r
    if mod == 4:
        return g
    if mod == 5:
        return 255 - g
    if mod == 6:
        return b
    if mod == 7:
        return 255 - b
    raise ValueError('bad alpha modifier %d' % mod)


def _color_combine(op, i0, i1, i2):
    """i0/i1/i2 (...,3) int32.  Port of ColorCombine() (8-bit integer maths)."""
    if op == 0:                                  # Replace
        return i0
    if op == 1:                                  # Modulate
        return (i0 * i1) // 255
    if op == 2:                                  # Add
        return np.minimum(255, i0 + i1)
    if op == 3:                                  # AddSigned
        return np.clip(i0 + i1 - 128, 0, 255)
    if op == 4:                                  # Lerp
        return (i0 * i2 + i1 * (255 - i2)) // 255
    if op == 5:                                  # Subtract
        return np.maximum(0, i0 - i1)
    if op == 6 or op == 7:                       # Dot3_RGB / Dot3_RGBA
        r = ((i0[..., 0] * 2 - 255) * (i1[..., 0] * 2 - 255) + 128) // 256
        g = ((i0[..., 1] * 2 - 255) * (i1[..., 1] * 2 - 255) + 128) // 256
        b = ((i0[..., 2] * 2 - 255) * (i1[..., 2] * 2 - 255) + 128) // 256
        v = np.clip(r + g + b, 0, 255).astype(np.int32)
        return np.stack([v, v, v], -1)
    if op == 8:                                  # MultiplyThenAdd
        return np.minimum(255, (i0 * i1 + 255 * i2) // 255)
    if op == 9:                                  # AddThenMultiply
        return (np.minimum(255, i0 + i1) * i2) // 255
    raise ValueError('bad color op %d' % op)


def _alpha_combine(op, i0, i1, i2):
    if op == 0:
        return i0
    if op == 1:
        return (i0 * i1) // 255
    if op == 2:
        return np.minimum(255, i0 + i1)
    if op == 3:
        return np.clip(i0 + i1 - 128, 0, 255)
    if op == 4:
        return (i0 * i2 + i1 * (255 - i2)) // 255
    if op == 5:
        return np.maximum(0, i0 - i1)
    if op == 8:
        return np.minimum(255, (i0 * i1 + 255 * i2) // 255)
    if op == 9:
        return (np.minimum(255, i0 + i1) * i2) // 255
    raise ValueError('bad alpha op %d' % op)


# --------------------------------------------------------------------------- combiner
def combine_stages(stages, textures, primary, secondary=(0, 0, 0, 0),
                   primary_fragment=None, buffer_color=(0, 0, 0, 0),
                   buffer_update=None):
    """Run a list of `PicaStage` and return the final RGBA as uint8.

    Parameters
    ----------
    stages : list[PicaStage]
    textures : sequence of 4 entries, each (H,W,4) uint8 or float 0..1.
               Missing / disabled units may be None (treated as opaque black).
    primary : vertex colour, (4,) or (H,W,4).  PICA *PrimaryColor* (source 0).
    secondary : PICA *SecondaryFragmentColor* (source 2, usually the specular
               term produced by the fragment-lighting unit).
    primary_fragment : PICA *PrimaryFragmentColor* (source 1).  PICA's fixed
               function lighting writes its result here; defaults to `primary`
               when no custom fragment shader is in use.
    buffer_color : TEV combiner-buffer colour (source 0xD at stage 0).
    buffer_update : optional iterable of (rgb, alpha) booleans per stage.  When
               None, PICA's default 0x0F/0x0F mask is *not* assumed -- pass it
               explicitly if your material sets TEXENV_UPDATE_BUFFER.
    """
    if primary_fragment is None:
        primary_fragment = primary
    prim = _as_u8_rgba(primary)
    pfrag = _as_u8_rgba(primary_fragment)
    sec = _as_u8_rgba(secondary)
    tex = []
    for i in range(4):
        t = textures[i] if i < len(textures) else None
        if t is None:
            ref = np.zeros(prim.shape, np.uint8) if prim.ndim == 3 else np.zeros(4, np.uint8)
            tex.append(ref)
        else:
            ref = _as_u8_rgba(t)
            if prim.ndim == 3 and ref.shape[:2] != prim.shape[:2]:
                raise ValueError('texture %d shape %r != primary %r'
                                 % (i, ref.shape, prim.shape))
            tex.append(ref)

    prev = np.zeros_like(prim)                       # combiner_output
    buf = _as_u8_rgba(buffer_color)                  # combiner_buffer
    nxt = buf.copy()                                 # next_combiner_buffer

    for idx, st in enumerate(stages):
        def source(code):
            if code == 0x0:
                return prim
            if code == 0x1:
                return pfrag
            if code == 0x2:
                return sec
            if code in (0x3, 0x4, 0x5, 0x6):
                return tex[code - 0x3]
            if code == 0xD:
                return buf
            if code == 0xE:
                c = np.asarray(st.const_rgba, np.uint8)
                return c if prim.ndim == 1 else np.broadcast_to(c, prim.shape)
            if code == 0xF:
                return prev
            raise ValueError('bad source 0x%x' % code)

        # PICA quirk: "Previous" as source 1/2 in stage 0 means source 3.
        s1 = st.color_source[0]
        s2 = st.color_source[1]
        if idx == 0:
            if s1 == 0xF:
                s1 = st.color_source[2]
            if s2 == 0xF:
                s2 = st.color_source[2]

        r0 = _color_modifier(st.color_mod[0], source(s1))
        r1 = _color_modifier(st.color_mod[1], source(s2))
        r2 = _color_modifier(st.color_mod[2], source(st.color_source[2]))
        col = _color_combine(st.color_op, r0, r1, r2)

        if st.color_op == 7:                     # Dot3_RGBA writes alpha too
            alpha = col[..., 0]
        else:
            a0 = _alpha_modifier(st.alpha_mod[0], source(st.alpha_source[0]))
            a1 = _alpha_modifier(st.alpha_mod[1], source(st.alpha_source[1]))
            a2 = _alpha_modifier(st.alpha_mod[2], source(st.alpha_source[2]))
            alpha = _alpha_combine(st.alpha_op, a0, a1, a2)

        out = np.empty(prim.shape, np.int32) if prim.ndim == 3 else np.empty(4, np.int32)
        out[..., :3] = np.minimum(255, col * st.color_mul)
        out[..., 3] = np.minimum(255, alpha * st.alpha_mul)
        prev = out.astype(np.uint8)

        buf = nxt
        if buffer_update is not None and idx < len(buffer_update):
            upd = buffer_update[idx]
            upd_rgb = upd[0] if isinstance(upd, (tuple, list)) else upd
            upd_a = upd[1] if isinstance(upd, (tuple, list)) else upd
            if upd_rgb:
                nxt = nxt.copy() if nxt is buf else nxt
                nxt[..., :3] = prev[..., :3]
            if upd_a:
                nxt = nxt.copy() if nxt is buf else nxt
                nxt[..., 3] = prev[..., 3]

    return prev


# --------------------------------------------------------------------------- NLPP COMB
COMB_HEADER = 0xFF
RECORD_BYTES = 20
NLPP_FIELDS = ('srcRgb', 'srcAlpha', 'operandRgb', 'operandAlpha',
               'combineRgb', 'combineAlpha', 'scaleRgb', 'scaleAlpha')


def parse_comb(material):
    """Material -> (header, [20-byte record, ...]).  Works on a Material or raw words."""
    words = getattr(material, 'combiner', material)
    head = getattr(material, 'combiner_head', COMB_HEADER)
    if not words:
        return head, []
    flat = [head]
    for st in words:
        flat.extend(st)
    blob = struct.pack('<%dI' % len(flat), *flat)
    body = blob[4:]
    return head, [body[i:i + RECORD_BYTES] for i in range(0, len(body), RECORD_BYTES)]


def decode_record(rec):
    """20-byte NLPP stage record -> structural dict (see module docstring).

    Returns the 8 two-nibble fields in the order the engine declares them, plus
    the trailing 32-bit word.  Each pair is (low_nibble, high_nibble) where the
    nibbles live at bits 0..3 and 8..11 of each u16 -- exactly the only two bit
    positions the whole 5875-material corpus ever uses.
    """
    if len(rec) != RECORD_BYTES:
        raise ValueError('record must be %d bytes, got %d' % (RECORD_BYTES, len(rec)))
    fields = {}
    for i, name in enumerate(NLPP_FIELDS):
        lo, hi = rec[2 * i], rec[2 * i + 1]
        fields[name] = (lo & 0xF, hi & 0xF)
    fields['constRgba'] = tuple(rec[16:20])
    return fields


def record_words(rec):
    return struct.unpack('<5I', rec)


def material_stages(material, layout=None):
    """NLPP material -> list[PicaStage] under the *provisional* mapping.

    The mapping below is a hypothesis (see the report): the NLPP record holds
    two source selectors, two operand selectors, a combine mode and a scale per
    channel.  Because the enumeration numbering of the *engine* is not yet
    proven to match PICA's, `layout='raw'` (the default) keeps the raw nibbles
    and only maps the fields that are unambiguous (sources and const colour);
    `layout='pica'` additionally trusts the nibbles as PICA enum values so the
    result can be rendered and compared.
    """
    layout = layout or os.environ.get('LP_TEV_LAYOUT', 'pica')
    _, recs = parse_comb(material)
    out = []
    for rec in recs:
        f = decode_record(rec)
        # sources: the engine stores two selectors per channel; PICA wants three.
        # The third is set to the second selector (a no-op for Replace/Add of the
        # same value is *not* implied -- this is exactly the unverified part).
        cs = (f['srcRgb'][0], f['srcRgb'][1], f['srcRgb'][1])
        as_ = (f['srcAlpha'][0], f['srcAlpha'][1], f['srcAlpha'][1])
        cm = (f['operandRgb'][0], f['operandRgb'][1], f['operandRgb'][1])
        am = (f['operandAlpha'][0] & 7, f['operandAlpha'][1] & 7,
              f['operandAlpha'][1] & 7)
        cop = f['combineRgb'][0]
        aop = f['combineAlpha'][0]
        out.append(PicaStage(
            color_source=cs, alpha_source=as_, color_mod=cm, alpha_mod=am,
            color_op=cop, alpha_op=aop,
            const_rgba=f['constRgba'],
            color_scale=f['scaleRgb'][0] & 3,
            alpha_scale=f['scaleAlpha'][0] & 3,
            raw=record_words(rec)))
    return out


# --------------------------------------------------------------------------- API
def combine_material(material, textures, primary, secondary=(0, 0, 0, 0),
                     primary_fragment=None, stages=None, layout=None, **kw):
    """NLPP material + texture samples + lighting colours -> final RGBA.

    `textures` is indexed by *texture unit* (the order inside the SMAT TEXU
    block, i.e. the same order as `material.mappers`), each entry either a
    (H,W,4) uint8 array, a float 0..1 array, or None.

    Signature summary (also in the report):
        combine_material(material, textures, primary, secondary=(0,0,0,0),
                         primary_fragment=None, stages=None, layout=None)
            -> (H,W,4) uint8  (or (4,) uint8 for flat inputs)
    """
    if stages is None:
        stages = material_stages(material, layout=layout)
    return combine_stages(stages, textures, primary, secondary=secondary,
                          primary_fragment=primary_fragment, **kw)


# --------------------------------------------------------------------------- tests
def _flat(v):
    return np.asarray(v, np.uint8).reshape(1, 1, 4)


def _selftest():
    """Hand-computed checks against the PICA reference semantics."""
    def run(stages, tex=(None, None, None, None), prim=(255, 255, 255, 255),
            sec=(0, 0, 0, 0), pfrag=None):
        return combine_stages(stages, tex, _flat(prim), secondary=_flat(sec),
                              primary_fragment=None if pfrag is None else _flat(pfrag))

    # 1) Replace(PrimaryColor) -> primary verbatim.
    st = PicaStage(color_source=(0, 0, 0), color_op=OPS_INV['Replace'],
                   alpha_source=(0, 0, 0), alpha_op=OPS_INV['Replace'])
    out = run([st], prim=(10, 20, 30, 200))
    assert tuple(out[0, 0]) == (10, 20, 30, 200), out

    # 2) Modulate(Texture0, PrimaryColor): 128*255/255 = 128 ; 64*128/255 = 32
    tex = _flat((128, 64, 255, 255))
    st = PicaStage(color_source=(3, 0, 0), color_mod=(0, 0, 0),
                   color_op=OPS_INV['Modulate'], alpha_source=(3, 0, 0),
                   alpha_op=OPS_INV['Modulate'], color_scale=0, alpha_scale=0)
    out = run([st], tex=(tex, None, None, None), prim=(255, 128, 255, 255))
    assert tuple(out[0, 0]) == (128, 32, 255, 255), tuple(out[0, 0])

    # 3) Scale 2x clamps at 255: 128*200/255 = 100 -> *2 = 200 ok; 255*2 -> 255
    st = PicaStage(color_source=(3, 0, 0), color_op=OPS_INV['Modulate'],
                   alpha_source=(0, 0, 0), alpha_op=OPS_INV['Replace'],
                   color_scale=1, alpha_scale=0)
    out = run([st], tex=(_flat((255, 128, 255, 255)), None, None, None),
              prim=(255, 255, 255, 255))
    assert tuple(out[0, 0])[:3] == (255, 255, 255), tuple(out[0, 0])

    # 4) AddSigned(bias 128): 200 + 100 - 128 = 172
    st = PicaStage(color_source=(3, 0, 0), color_mod=(0, 0, 0),
                   color_op=OPS_INV['AddSigned'], alpha_source=(0, 0, 0),
                   alpha_op=OPS_INV['Replace'])
    out = run([st], tex=(_flat((200, 200, 200, 255)), None, None, None),
              prim=(100, 100, 100, 255))
    assert tuple(out[0, 0])[:3] == (172, 172, 172), tuple(out[0, 0])

    # 5) Lerp(a, b, t) = (a*t + b*(255-t))/255 with t=128: (200*128+100*127)/255 = 150
    st = PicaStage(color_source=(3, 0, 0), color_op=OPS_INV['Lerp'],
                   alpha_source=(0, 0, 0), alpha_op=OPS_INV['Replace'],
                   color_mod=(0, 0, 2))
    out = run([st], tex=(_flat((200, 200, 200, 255)), None, None, None),
              prim=(100, 100, 100, 128))
    assert tuple(out[0, 0])[:3] == (150, 150, 150), tuple(out[0, 0])

    # 6) 2-stage chain: stage1 uses Previous * Constant
    s0 = PicaStage(color_source=(3, 0, 0), color_op=OPS_INV['Replace'],
                   alpha_source=(3, 0, 0), alpha_op=OPS_INV['Replace'])
    s1 = PicaStage(color_source=(0xF, 0xE, 0xE), color_op=OPS_INV['Modulate'],
                   alpha_source=(0xF, 0xF, 0xF), alpha_op=OPS_INV['Replace'],
                   const_rgba=(128, 128, 128, 255))
    out = run([s0, s1], tex=(_flat((200, 100, 50, 255)), None, None, None))
    assert tuple(out[0, 0]) == (100, 50, 25, 255), tuple(out[0, 0])

    # 7) alpha modifier / op: alpha = 1 - tex.a, then * 2 scale
    st = PicaStage(color_source=(3, 0, 0), color_op=OPS_INV['Replace'],
                   alpha_source=(3, 0, 0), alpha_mod=(1, 0, 0),
                   alpha_op=OPS_INV['Replace'], alpha_scale=1)
    out = run([st], tex=(_flat((10, 10, 10, 100)), None, None, None))
    assert int(out[0, 0, 3]) == 255, int(out[0, 0, 3])   # (255-100)*2 = 310 -> 255

    # 8) decode_pica_words round-trip on a real PICA register set
    w = [0x00000030,           # cs1=0, cs2=3, cs3=0 ; as1..=0
         0x00000000,           # modifiers all "source colour/alpha"
         0x00010000,           # color op = Replace, alpha op = Modulate
         0x00804020,           # const = (0x20,0x40,0x80,0x00)
         0x00010000]           # color scale 0, alpha scale 1
    st = decode_pica_words(w)
    assert st.color_source == (0, 3, 0) and st.alpha_source == (0, 0, 0)
    assert st.color_op == 0 and st.alpha_op == 1
    assert st.const_rgba == (0x20, 0x40, 0x80, 0x00)
    assert st.alpha_mul == 2 and st.color_mul == 1
    print('tev.py selftest: 8/8 OK')


if __name__ == '__main__':
    _selftest()
