# -*- coding: utf-8 -*-
"""Which bones do the ARM vertices bind to?

Same method that identified 78..96 as the hair: ask the mesh, not the filenames.  Take the
vertices that lie in the arm region geometrically (far from the body centre line, at
shoulder height) and list the bones they are weighted to.  Those bone indices are what an
m_upperNNN layer has to drive.
"""
import sys
import collections
import numpy as np

sys.path.insert(0, r'D:/dsh/nlpp/native')
sys.stdout.reconfigure(encoding='utf-8')
import nlp_model as M         # noqa: E402
from compose import character_parts   # noqa: E402

IMG = r'D:/dsh/nlpp/img.bin'

rows = []
for nm in character_parts('m', 'm_00_200', 'm_01_012'):
    try:
        m = M.load_model(IMG, nm)
    except Exception:
        continue
    pos, bone_of = [], []
    for me in m.meshes:
        for pr in me.prims:
            for v in pr.vertices.values():
                pos.append(v['pos'])
                bone_of.append((v.get('bones') or [-1])[0])
    if not pos:
        continue
    P = np.array(pos)
    rows.append((nm, P, np.array(bone_of)))

allP = np.vstack([r[1] for r in rows])
ymin, ymax = allP[:, 1].min(), allP[:, 1].max()
H = ymax - ymin
print('character bbox  Y=[%.1f..%.1f]  height=%.1f' % (ymin, ymax, H))
print()

# arm region: |x| beyond 45% of the half-width, and y in the upper 45% of the body
half_x = max(abs(allP[:, 0]).max(), 1.0)
GATE_X = 0.45 * half_x
print('arm gate: |x| > %.1f' % GATE_X)
print()

arm_bones = collections.Counter()
for nm, P, B in rows:
    m = np.abs(P[:, 0]) > GATE_X
    if not m.any():
        print('%-16s (no arm vertices)' % nm)
        continue
    c = collections.Counter(B[m].tolist())
    print('%-16s %5d arm verts   bones: %s' % (nm, int(m.sum()), sorted(c)[:18]))
    for b, n in c.items():
        if b >= 0:
            arm_bones[b] += n

print()
print('bones weighted by arm vertices, most used first:')
for b, n in arm_bones.most_common(24):
    print('   bone %-4d %5d verts' % (b, n))
print()
print('=> an m_upperNNN layer must drive these indices:')
print('   %s' % sorted(arm_bones))
