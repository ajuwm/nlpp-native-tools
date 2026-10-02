# -*- coding: utf-8 -*-
"""Validate the record->bone table against the bind pose, and work out the >253 indices.

m_lower011 (nk==1, so its values are ABSOLUTE and must equal the bind pose) has the table
    [303,304,305,307,308,309,311, 84, 312,314, 85, 315,317, 86, 318,320, 87, 321]
Only 84..87 are inside the 254-bone rig, so those four records are a clean test: their
values must reproduce the bind translation of bones 84..87.

If that holds, the table is confirmed and the values above 253 index a LARGER (merged)
rig, which we then have to account for.
"""
import sys
import struct

sys.path.insert(0, r'D:/dsh/nlpp/native')
sys.stdout.reconfigure(encoding='utf-8')
import nlp_pack as N          # noqa: E402
import nlp_model as M         # noqa: E402
import nlp_mot as MOT         # noqa: E402
from compose import character_parts   # noqa: E402

IMG = r'D:/dsh/nlpp/img.bin'
NAME = 'm_lower011.mot'

bones = None
for nm in character_parts('m', 'm_00_200', 'm_01_012'):
    try:
        m = M.load_model(IMG, nm)
    except Exception:
        continue
    if bones is None:
        bones = m.bones
by_idx = {b.index: b for b in bones}
print('rig: %d bones (0..%d)' % (len(bones), max(by_idx)))

raw = None
with open(IMG, 'rb') as f:
    for p in N.iter_packs(IMG):
        for pf in p.files:
            if pf.name == NAME:
                raw = N.pack_bytes(f, pf)
                break
        if raw:
            break

count = struct.unpack_from('<I', raw, 8)[0]
bt = struct.unpack_from('<%dI' % count, raw, struct.unpack_from('<I', raw, 0x0C)[0])
print('%s  count=%d' % (NAME, count))
print('   table = %s' % list(bt))
print()

mm = MOT.load_mot_final(raw, NAME)
print('records whose bone index is INSIDE the 254-bone rig:')
for ri, r in enumerate(mm.records):
    if r.bone >= len(bones):
        continue
    b = by_idx[r.bone]
    print('   rec%-3d bone=%-4d channels=%s' % (ri, r.bone, ['%s%d' % (c, a) for c, a in r.channels]))
    print('        values k0 = %s' % [round(v, 4) for v in r.values[0]])
    print('        bind  T=%s R=%s' % (tuple(round(x, 4) for x in b.translation),
                                       tuple(round(x, 4) for x in b.rotation)))
print()
print('records whose bone index is OUTSIDE the rig (first 8):')
n = 0
for ri, r in enumerate(mm.records):
    if r.bone < len(bones):
        continue
    print('   rec%-3d bone=%-4d  channels=%s  k0=%s'
          % (ri, r.bone, ['%s%d' % (c, a) for c, a in r.channels],
             [round(v, 4) for v in r.values[0]]))
    n += 1
    if n >= 8:
        break
print()
above = [v for v in bt if v >= len(bones)]
print('indices >= %d : %s' % (len(bones), above))
if above:
    print('   minus %d -> %s' % (len(bones), [v - len(bones) for v in above]))
