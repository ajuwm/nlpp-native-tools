# -*- coding: utf-8 -*-
"""Which body region actually moves?

Legs are visibly animating now; the arms still look frozen.  Rather than guess, split the
skinned vertices by the bone they are weighted to and measure the frame-to-frame
displacement per region.  That says exactly which chains the motion drives.
"""
import sys
import collections
import numpy as np

sys.path.insert(0, r'D:/dsh/nlpp/native')
sys.stdout.reconfigure(encoding='utf-8')
import nlp_pack as N          # noqa: E402
import nlp_model as M         # noqa: E402
import nlp_mot as MOT         # noqa: E402
import nlp_render as R        # noqa: E402
from compose import character_parts   # noqa: E402

IMG = r'D:/dsh/nlpp/img.bin'
MOTIONS = ['m_00000_40.mot', 'm_09070_21.mot', 'm_31783_31.mot']

REGIONS = [('hip/spine', 0, 11), ('lower leg', 12, 18), ('legs', 19, 32),
           ('arms', 33, 63), ('upper', 64, 122), ('rest', 123, 253)]

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
wb = R.world_matrices(bones, {})
binv = [np.linalg.inv(m) if m is not None else None for m in wb]

# vertex -> its dominant bone
vb = []
for o in objs:
    for v in o['verts']:
        bs = v.get('bones') or [0]
        vb.append(bs[0])
vb = np.array(vb, np.int32)

raws = {}
with open(IMG, 'rb') as f:
    for p in N.iter_packs(IMG):
        for pf in p.files:
            if pf.name in MOTIONS and pf.name not in raws:
                raws[pf.name] = N.pack_bytes(f, pf)
        if len(raws) == len(MOTIONS):
            break


def sample_pts(mm, t):
    p = MOT.sample_present(mm, t)
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
    pts = []
    for o in objs:
        pos, _ = R.skin_positions(o['verts'], bones, world)
        pts.append(pos)
    return np.vstack(pts)


for name in MOTIONS:
    if name not in raws:
        continue
    mm = MOT.load_mot_final(raws[name], name)
    dur = max(4, int(mm.duration) or 60)
    frames = [sample_pts(mm, int(dur * k / 8.0)) for k in range(8)]
    base = frames[0]
    disp = np.max(np.stack([np.abs(f - base) for f in frames[1:]]), axis=0).max(axis=1)
    # which records drive arm bones at all?
    driven = collections.Counter()
    for r in mm.records:
        if r.channels:
            for lo, hi in ((0, 11), (12, 18), (19, 32), (33, 63), (64, 122), (123, 253)):
                if lo <= r.bone <= hi:
                    driven[(lo, hi)] += 1
    print('=== %s  (dur=%d) ===' % (name, dur))
    print('   records per region: %s' % {('%d-%d' % k): v for k, v in sorted(driven.items())})
    for label, lo, hi in REGIONS:
        m = (vb >= lo) & (vb <= hi)
        if not m.any():
            print('   %-10s (no vertices)' % label)
            continue
        d = disp[m]
        print('   %-10s verts=%-6d  maxDisp=%7.2f  meanDisp=%6.3f'
              % (label, int(m.sum()), d.max(), d.mean()))
    print()
