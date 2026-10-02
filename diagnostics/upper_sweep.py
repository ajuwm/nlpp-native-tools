# -*- coding: utf-8 -*-
"""Score all 15 m_upperNNN layers: arm movement vs arm distortion.

m_upper001 folds the right arm into a flat plate.  Rather than reverse-engineer how the
game picks the layer, measure every candidate:

    movement   : how far the arm vertices travel across the motion  -> want LARGE
    flatness   : min/max bbox side of the arm region                 -> want NOT tiny
                 (a folded plate collapses one dimension toward 0)
    height     : bbox height spread                                  -> want small

Rank and keep the best.
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
BODY = 'm_00000_40.mot'
HAND = 'm_hand000.mot'
FRAMES = [0, 8, 16, 24, 32, 40, 48]

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

# arm vertices = weighted to bones 33..63
vb = np.concatenate([np.array([(v.get('bones') or [0])[0] for v in o['verts']]) for o in objs])
ARM = (vb >= 33) & (vb <= 63)
print('arm vertices: %d of %d' % (int(ARM.sum()), len(vb)))

cands = []
allraw = {}
with open(IMG, 'rb') as f:
    for p in N.iter_packs(IMG):
        for pf in p.files:
            if pf.name in (BODY, HAND) or (pf.name.startswith('m_upper') and pf.name.endswith('.mot')):
                if pf.name not in allraw:
                    allraw[pf.name] = N.pack_bytes(f, pf)
        if len(allraw) > 20:
            break
cands = sorted(k for k in allraw if k.startswith('m_upper'))
print('candidates: %d -> %s' % (len(cands), cands))


def measure(names):
    pts_all, hs = [], []
    for t in FRAMES:
        flat = []
        for name in names:
            mm = MOT.load_mot_final(allraw[name], name)
            p = MOT.sample_present(mm, t)
            d = {}
            for bone, ch in p.items():
                if bone >= len(bones):
                    continue
                dd = {}
                if ch.get('T'):
                    dd['T'] = dict(ch['T'])
                if ch.get('R'):
                    dd['R'] = dict(ch['R'])
                if dd:
                    d[bone] = dd
            flat.append(d)
        st = MOT.stack_poses(flat)
        pose = {}
        for bone, ch in st.items():
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
        pts_all.append(pts)
        hs.append(pts[:, 1].max() - pts[:, 1].min())
    base = pts_all[0]
    move = float(np.max([np.abs(p - base).max(axis=1)[ARM].max() for p in pts_all[1:]]))
    arm = np.vstack([p[ARM] for p in pts_all])
    sides = arm.max(axis=0) - arm.min(axis=0)
    flat = float(sides.min() / max(sides.max(), 1e-6))
    return move, flat, max(hs) - min(hs)


print()
print('%-16s %-10s %-10s %-10s' % ('upper layer', 'armMove', 'flatness', 'heightDrift'))
rows = []
base_move, base_flat, base_h = measure([BODY, HAND])
print('%-16s %-10.1f %-10.3f %-10.1f   (no upper layer)' % ('(none)', base_move, base_flat, base_h))
for name in cands:
    mv, fl, hd = measure([BODY, HAND, name])
    rows.append((name, mv, fl, hd))
    print('%-16s %-10.1f %-10.3f %-10.1f' % (name.replace('.mot', ''), mv, fl, hd))
print()
good = [r for r in rows if r[2] > base_flat * 0.9]
good.sort(key=lambda r: -r[1])
print('candidates that do NOT flatten the arm (flatness >= %.3f), best movement first:' % (base_flat * 0.9))
for name, mv, fl, hd in good[:6]:
    print('   %-16s move=%.1f flatness=%.3f heightDrift=%.1f' % (name, mv, fl, hd))
if not good:
    print('   none -- every m_upperNNN folds the arm')
