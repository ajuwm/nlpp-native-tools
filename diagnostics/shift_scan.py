# -*- coding: utf-8 -*-
"""Is the record -> bone pairing shifted?

The rig's bind pose is a T-pose: bone 30 -> 33 points along +X (the arm sticks out
sideways), and the leg already points straight down along -Y.  So a standing pose needs a
BIG rotation on the arm (~90 deg) and a SMALL one on the leg.

Every calm motion has the opposite: ~81 on the legs (12..18) and only ~18 on the arms.
That is the shape of a shifted pairing.  Test it: keep the table but offset the pairing,
bone[i] = table[(i + k) % count], and score

    armDrop : how far the arm vertices fall (want LARGE -- the arms come down)
    legHold : how close the leg length stays to the bind pose (want SMALL change)

over k = 0..count-1.
"""
import sys
import numpy as np

sys.path.insert(0, r'D:/dsh/nlpp/native')
sys.stdout.reconfigure(encoding='utf-8')
import nlp_pack as N          # noqa: E402
import nlp_model as M         # noqa: E402
import nlp_mot as MOT         # noqa: E402
import nlp_render as R        # noqa: E402
from compose import character_parts   # noqa: E402

IMG = r'D:/dsh/nlpp/img.bin'
NAME = 'm_00010_40.mot'
T = 0

objs, bones = [], None
for nm in character_parts('m', 'm_00_200', 'm_01_012'):
    try:
        m = M.load_model(IMG, nm)
    except Exception:
        continue
    if bones is None:
        bones = m.bones
    for me in m.meshes:
        for pr in me.prims:
            if not pr.vertices or not pr.indices:
                continue
            order = sorted(pr.vertices.keys())
            remap = {vi: k for k, vi in enumerate(order)}
            verts = []
            for vi in order:
                v = dict(pr.vertices[vi])
                w = list(v.get('weights') or [])
                s = sum(w)
                if s > 0:
                    v['weights'] = [x / s for x in w]
                verts.append(v)
            objs.append(dict(verts=verts, indices=[remap[i] for i in pr.indices if i in remap]))
MOT.BIND_ROT = {b.index: tuple(b.rotation) for b in bones}
wb = R.world_matrices(bones, {})
binv = [np.linalg.inv(m) if m is not None else None for m in wb]
BIND = np.vstack([R.skin_positions(o['verts'], bones, wb)[0] for o in objs])
own = np.concatenate([np.array([(v.get('bones') or [0])[0] for v in o['verts']]) for o in objs])
ARM = (own >= 30) & (own <= 63)
LEG = (own >= 12) & (own <= 18)
print('arm verts %d   leg verts %d' % (int(ARM.sum()), int(LEG.sum())))
print('bind: arm meanY=%.1f  leg spanY=%.1f' %
      (BIND[ARM][:, 1].mean(), BIND[LEG][:, 1].max() - BIND[LEG][:, 1].min()))

raw = None
with open(IMG, 'rb') as f:
    for p in N.iter_packs(IMG):
        for pf in p.files:
            if pf.name == NAME:
                raw = N.pack_bytes(f, pf)
                break
        if raw:
            break

import struct
count = struct.unpack_from('<I', raw, 8)[0]
bt = list(struct.unpack_from('<%dI' % count, raw, struct.unpack_from('<I', raw, 0x0C)[0]))
print('count=%d  table[:8]=%s' % (count, bt[:8]))


def evaluate(k):
    mm = MOT.load_mot_final(raw, NAME)
    # re-map the bones with the shift
    for i, r in enumerate(mm.records):
        v = bt[(i + k) % count]
        r.bone = (v - 225) if v >= 225 else v
        if r.bone >= len(bones):
            r.bone = 9999
    p = MOT.sample_present(mm, T)
    pose = {}
    for bone, ch in p.items():
        if bone >= len(bones):
            continue
        d = [0.0] * 9
        for ax, v in ch.get('T', {}).items():
            d[ax] = v
        for ax, v in ch.get('R', {}).items():
            d[3 + ax] = v
        d[6] = d[7] = d[8] = 1.0
        pose[bone] = tuple(d)
    wp = R.world_matrices(bones, pose)
    world = [(wp[i] @ binv[i]) if (wp[i] is not None and binv[i] is not None) else wp[i]
             for i in range(len(wp))]
    pts = np.vstack([R.skin_positions(o['verts'], bones, world)[0] for o in objs])
    arm_drop = BIND[ARM][:, 1].mean() - pts[ARM][:, 1].mean()
    leg_span = pts[LEG][:, 1].max() - pts[LEG][:, 1].min()
    bind_leg = BIND[LEG][:, 1].max() - BIND[LEG][:, 1].min()
    return arm_drop, abs(leg_span - bind_leg)


rows = []
for k in range(count):
    ad, ld = evaluate(k)
    rows.append((ad, ld, k))
rows.sort(key=lambda r: -r[0])
print()
print('%-6s %-12s %-12s' % ('shift k', 'armDrop', 'legChange'))
for ad, ld, k in rows[:10]:
    print('%-6d %-12.1f %-12.1f' % (k, ad, ld))
print()
best = rows[0]
print('best: k=%d armDrop=%.1f legChange=%.1f  (k=0 is the current pairing: armDrop=%.1f)'
      % (best[2], best[0], best[1], [r for r in rows if r[2] == 0][0][0]))
print('want armDrop LARGE (arms come down) and legChange small (legs stay straight)')
