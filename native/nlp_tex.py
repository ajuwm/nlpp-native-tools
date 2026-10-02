# -*- coding: utf-8 -*-
"""Native PICA200 texture decoder for New Love Plus .texi payloads.

Byte/bit layouts and the tile order follow gdkchan/Ohana3DS-Rebirth
`Ohana/TextureCodec.cs` (decode()).

format ids (from the TEXI descriptor):
    0 l4  1 l8  2 a4  3 la4  4 la8  5 hilo8  6 l8b  7 rgb565
    8 rgba5551  9 rgba4  0xa rgba8  0xb rgb8  0xc etc1  0xd etc1a4
"""
import struct
import os

FMT = {
    0: 'l4', 1: 'l8', 2: 'a4', 3: 'la4', 4: 'la8', 5: 'hilo8',
    6: 'l8b', 7: 'rgb565', 8: 'rgba5551', 9: 'rgba4',
    0xa: 'rgba8', 0xb: 'rgb8', 0xc: 'etc1', 0xd: 'etc1a4',
}

_BPP = {'rgba8': 4, 'rgb8': 3, 'rgba5551': 2, 'rgb565': 2, 'rgba4': 2,
        'la8': 2, 'la4': 1, 'l8': 1, 'l8b': 1, 'l4': 1, 'a4': 1}

# Ohana TextureCodec.tileOrder -- position inside one 8x8 tile
_TILE_ORDER = (0, 1, 8, 9, 2, 3, 10, 11, 16, 17, 24, 25, 18, 19, 26, 27,
               4, 5, 12, 13, 6, 7, 14, 15, 20, 21, 28, 29, 22, 23, 30, 31,
               32, 33, 40, 41, 34, 35, 42, 43, 48, 49, 56, 57, 50, 51, 58, 59,
               36, 37, 44, 45, 38, 39, 46, 47, 52, 53, 60, 61, 54, 55, 62, 63)


def _tile_xy():
    out = []
    for v in _TILE_ORDER:
        x = v % 8
        y = (v - x) // 8
        out.append((x, y))
    return out


_TILE_XY = _tile_xy()


def _e3(v):
    return (v << 3) | (v >> 2)


def _e4(v):
    return (v << 4) | v


def _e6(v):
    return (v << 2) | (v >> 4)


# ------------------------------------------------------------------ ETC1
# Port of Ohana3DS TextureCodec ETC1 path.  NOTE the two non-standard details:
#   * the 8 bytes of a block are read as [0..3] = index bits (LE) and
#     [4..7] = colour header (LE)  -- i.e. the reverse of the usual ETC1 layout
#   * after decoding, blocks are re-ordered by etc1Scramble()
_ETC1_LUT = ((2, 8, -2, -8), (5, 17, -5, -17), (9, 29, -9, -29), (13, 42, -13, -42),
             (18, 60, -18, -60), (24, 80, -24, -80), (33, 106, -33, -106),
             (47, 183, -47, -183))


def _sat(v):
    return 0 if v < 0 else (255 if v > 255 else v)


def _s8(v):
    v &= 0xFF
    return v - 256 if v >= 128 else v


def _etc1_decoded_block(b):
    """b: 8 input bytes -> 64 bytes RGB (4x4, row-major).

    Byte order verified empirically (native/etc1_variants.py): BOTH halves are read
    big-endian.  Measured adjacent-pixel |diff| was 0.75 for big/big versus 13.7 for
    little/little, and the resulting image is the navy sailor uniform.
    """
    block_top = int.from_bytes(b[4:8], 'big')
    block_bottom = int.from_bytes(b[0:4], 'big')

    flip = (block_top & 0x1000000) > 0
    difference = (block_top & 0x2000000) > 0

    if difference:
        r1 = block_top & 0xF8
        g1 = (block_top & 0xF800) >> 8
        b1 = (block_top & 0xF80000) >> 16
        r2 = ((r1 >> 3) + (_s8((block_top & 7) << 5) >> 5)) & 0xFFFFFFFF
        g2 = ((g1 >> 3) + (_s8((block_top & 0x700) >> 3) >> 5)) & 0xFFFFFFFF
        b2 = ((b1 >> 3) + (_s8((block_top & 0x70000) >> 11) >> 5)) & 0xFFFFFFFF
        r1 |= r1 >> 5
        g1 |= g1 >> 5
        b1 |= b1 >> 5
        r2 = ((r2 << 3) | (r2 >> 2)) & 0xFFFFFFFF
        g2 = ((g2 << 3) | (g2 >> 2)) & 0xFFFFFFFF
        b2 = ((b2 << 3) | (b2 >> 2)) & 0xFFFFFFFF
    else:
        r1 = block_top & 0xF0
        g1 = (block_top & 0xF000) >> 8
        b1 = (block_top & 0xF00000) >> 16
        r2 = (block_top & 0xF) << 4
        g2 = (block_top & 0xF00) >> 4
        b2 = (block_top & 0xF0000) >> 12
        r1 |= r1 >> 4
        g1 |= g1 >> 4
        b1 |= b1 >> 4
        r2 |= r2 >> 4
        g2 |= g2 >> 4
        b2 |= b2 >> 4

    table1 = (block_top >> 29) & 7
    table2 = (block_top >> 26) & 7
    out = bytearray(64)

    def put(r, g, b, x, y, block, table, out):
        index = x * 4 + y
        msb = (block << 1) & 0xFFFFFFFF
        if index < 8:
            sel = ((block >> (index + 24)) & 1) + ((msb >> (index + 8)) & 2)
        else:
            sel = ((block >> (index + 8)) & 1) + ((msb >> (index - 8)) & 2)
        d = _ETC1_LUT[table][sel]
        o = (y * 4 + x) * 4
        # RGB order, NOT the B,G,R that Ohana writes (it targets a BGRA bitmap).
        out[o] = _sat(r + d)
        out[o + 1] = _sat(g + d)
        out[o + 2] = _sat(b + d)
        out[o + 3] = 255

    if not flip:
        for y in range(4):
            for x in range(2):
                put(r1, g1, b1, x, y, block_bottom, table1, out)
                put(r2, g2, b2, x + 2, y, block_bottom, table2, out)
    else:
        for y in range(2):
            for x in range(4):
                put(r1, g1, b1, x, y, block_bottom, table1, out)
                put(r2, g2, b2, x, y + 2, block_bottom, table2, out)
    return bytes(out)


def _etc1_scramble(width, height):
    n = (width // 4) * (height // 4)
    order = [0] * n
    base_acc = 0
    row_acc = 0
    base_num = 0
    row_num = 0
    for tile in range(n):
        if (tile % (width // 4) == 0) and tile > 0:
            if row_acc < 1:
                row_acc += 1
                row_num += 2
                base_num = row_num
            else:
                row_acc = 0
                base_num -= 2
                row_num = base_num
        order[tile] = base_num
        if base_acc < 1:
            base_acc += 1
            base_num += 1
        else:
            base_acc = 0
            base_num += 3
    return order


def decode_etc1(data, width, height, alpha=False):
    """PICA200 ETC1 / ETC1A4 -> linear RGBA."""
    # 1. decode every block, in file order
    dec = bytearray(width * height * 4)
    off = 0
    for by in range(height // 4):
        for bx in range(width // 4):
            if alpha:
                blk = bytes(data[off + 8:off + 16]) + bytes(data[off:off + 8])
                raw = _etc1_decoded_block(blk)
                alpha_block = data[off:off + 8]
                off += 16
            else:
                raw = _etc1_decoded_block(data[off:off + 8])
                alpha_block = b'\xff' * 8
                off += 8
            toggle = False
            ao = 0
            for tX in range(4):
                for tY in range(4):
                    dst = ((bx * 4 + tX) + (by * 4 + tY) * width) * 4
                    src = (tX + tY * 4) * 4
                    dec[dst:dst + 3] = raw[src:src + 3]
                    if alpha:
                        a = (alpha_block[ao] & 0xF0) >> 4 if toggle else (alpha_block[ao] & 0xF)
                        if toggle:
                            ao += 1
                        dec[dst + 3] = (a << 4) | a
                    else:
                        dec[dst + 3] = 255
                    toggle = not toggle

    # 2. un-scramble the block order
    order = _etc1_scramble(width, height)
    out = bytearray(width * height * 4)
    i = 0
    for tY in range(height // 4):
        for tX in range(width // 4):
            TX = order[i] % (width // 4)
            TY = (order[i] - TX) // (width // 4)
            for y in range(4):
                for x in range(4):
                    s = ((TX * 4 + x) + (TY * 4 + y) * width) * 4
                    d = ((tX * 4 + x) + (tY * 4 + y) * width) * 4
                    out[d:d + 4] = dec[s:s + 4]
            i += 1
    return bytes(out)


def bpp_of(fmt):
    name = FMT.get(fmt, fmt) if isinstance(fmt, int) else fmt
    return _BPP.get(name, 0)


def decode(data, width, height, fmt):
    """Decode a tiled PICA200 texture into linear RGBA (w*h*4 bytes)."""
    name = FMT.get(fmt, fmt) if isinstance(fmt, int) else fmt
    if name in ('etc1', 'etc1a4'):
        return decode_etc1(data, width, height, name == 'etc1a4')
    out = bytearray(width * height * 4)
    off = 0
    tiles_x = width // 8
    tiles_y = height // 8

    if name == 'rgba8':
        # STORED ORDER IS A,B,G,R (not A,R,G,B).
        # Ohana copies file bytes [1,2,3] into a GDI+ Format32bppArgb buffer, whose
        # in-memory byte order is B,G,R -- so file[1]=B, file[2]=G, file[3]=R, file[0]=A.
        for ty in range(tiles_y):
            for tx in range(tiles_x):
                for px in range(64):
                    x, y = _TILE_XY[px]
                    o = ((tx * 8 + x) + (ty * 8 + y) * width) * 4
                    a = data[off]
                    b = data[off + 1]
                    g = data[off + 2]
                    r = data[off + 3]
                    out[o:o + 4] = bytes((r, g, b, a))
                    off += 4
    elif name == 'rgb8':
        # STORED ORDER IS B,G,R (Ohana copies the 3 bytes straight into a BGRA buffer).
        for ty in range(tiles_y):
            for tx in range(tiles_x):
                for px in range(64):
                    x, y = _TILE_XY[px]
                    o = ((tx * 8 + x) + (ty * 8 + y) * width) * 4
                    b = data[off]
                    g = data[off + 1]
                    r = data[off + 2]
                    out[o:o + 4] = bytes((r, g, b, 255))
                    off += 3
    elif name in ('rgba5551', 'rgb565', 'rgba4'):
        for ty in range(tiles_y):
            for tx in range(tiles_x):
                for px in range(64):
                    x, y = _TILE_XY[px]
                    o = ((tx * 8 + x) + (ty * 8 + y) * width) * 4
                    v = data[off] | (data[off + 1] << 8)
                    off += 2
                    if name == 'rgba5551':
                        r = (v >> 11) & 0x1F
                        g = (v >> 6) & 0x1F
                        b = (v >> 1) & 0x1F
                        a = 255 if (v & 1) else 0
                        out[o:o + 4] = bytes((_e3(r), _e3(g), _e3(b), a))
                    elif name == 'rgb565':
                        r = (v >> 11) & 0x1F
                        g = (v >> 5) & 0x3F
                        b = v & 0x1F
                        out[o:o + 4] = bytes((_e3(r), _e6(g), _e3(b), 255))
                    else:
                        out[o:o + 4] = bytes((_e4((v >> 12) & 0xF), _e4((v >> 8) & 0xF),
                                              _e4((v >> 4) & 0xF), _e4(v & 0xF)))
    elif name == 'la8':
        for ty in range(tiles_y):
            for tx in range(tiles_x):
                for px in range(64):
                    x, y = _TILE_XY[px]
                    o = ((tx * 8 + x) + (ty * 8 + y) * width) * 4
                    l, a = data[off], data[off + 1]
                    off += 2
                    out[o:o + 4] = bytes((l, l, l, a))
    elif name == 'la4':
        for ty in range(tiles_y):
            for tx in range(tiles_x):
                for px in range(64):
                    x, y = _TILE_XY[px]
                    o = ((tx * 8 + x) + (ty * 8 + y) * width) * 4
                    v = data[off]; off += 1
                    l = _e4((v >> 4) & 0xF)
                    a = _e4(v & 0xF)
                    out[o:o + 4] = bytes((l, l, l, a))
    elif name in ('l8', 'l8b'):
        for ty in range(tiles_y):
            for tx in range(tiles_x):
                for px in range(64):
                    x, y = _TILE_XY[px]
                    o = ((tx * 8 + x) + (ty * 8 + y) * width) * 4
                    l = data[off]; off += 1
                    out[o:o + 4] = bytes((l, l, l, 255 if name == 'l8' else l))
    elif name == 'l4':
        for ty in range(tiles_y):
            for tx in range(tiles_x):
                for px in range(64):
                    x, y = _TILE_XY[px]
                    o = ((tx * 8 + x) + (ty * 8 + y) * width) * 4
                    v = data[off + (px >> 1)]
                    l = _e4((v >> 4) if (px & 1) == 0 else (v & 0xF))
                    out[o:o + 4] = bytes((l, l, l, 255))
                off += 32
    elif name == 'a4':
        for ty in range(tiles_y):
            for tx in range(tiles_x):
                for px in range(64):
                    x, y = _TILE_XY[px]
                    o = ((tx * 8 + x) + (ty * 8 + y) * width) * 4
                    v = data[off + (px >> 1)]
                    a = _e4((v >> 4) if (px & 1) == 0 else (v & 0xF))
                    out[o:o + 4] = bytes((255, 255, 255, a))
                off += 32
    else:
        raise ValueError('unsupported texture format %r' % (fmt,))
    return bytes(out)


def write_png(path, rgba, width, height):
    from PIL import Image
    Image.frombytes('RGBA', (width, height), rgba).save(path)
    return path
