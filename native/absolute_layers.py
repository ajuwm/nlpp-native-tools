# -*- coding: utf-8 -*-
"""Layered motions (nk==1) probably store ABSOLUTE rotations, not deltas.

They are single-keyframe records; their translation is already treated as absolute
(`m.absolute_translation`).  The rotations are still added to the bind pose, which leaves
the arms in a T-pose and gives the hands a wrong orientation.

Test both semantics side by side:
    delta    : local_R = bind_R + value                 (current)
    absolute : local_R = value  ->  pose delta = value - bind_R
and render the same frames so the difference is visible.
"""
import sys
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, r'D:/dsh/nlpp/native')
sys.stdout.reconfigure(encoding='utf-8')
import nlp_pack as N          # noqa: E402
import nlp_model as M         # noqa: E402
import nlp_mot as MOT         # noqa: E402
import nlp_render as R        # noqa: E402
from compose import character_parts, LAYERS   # noqa: E402

IMG = r'D:/dsh/nlpp/img.bin'
OUT = r'D:/dsh/nlpp/out/absolute_layers.png'
FRAMES = [0, 10, 20, 30, 40]
SPEC = [('m_00000_40.mot', 0, 1)] + list(LAYERS['default'][1:])

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
            objs.append(dict(verts=verts, uv=np.array([v['uv'] for v in verts], np.float64),
                             texture=None,
                             indices=[remap[i] for i in pr.indices if i in remap]))
bind_r = {b.index: list(b.rotation) for b in bones}
wb = R.world_matrices(bones, {})
binv = [np.linalg.inv(m) if m is not None else None for m in wb]
allpos = np.vstack([np.array([v['pos'] for v in o['verts']]) for o in objs])
BLO, BHI = allpos.min(axis=0), allpos.max(axis=0)
CTR = (BLO + BHI) / 2.0
HGT = float(max(BHI[1] - BLO[1], 20.0))

mots = []
with open(IMG, 'rb') as f:
    got = set()
    for p in N.iter_packs(IMG):
        for pf in p.files:
            for name, base, per in SPEC:
                if pf.name == name and name not in got:
                    got.add(name)
                    mots.append(MOT.load_mot_final(N.pack_bytes(f, pf), name, base=base, per=per))
        if len(got) == len(SPEC):
            break
print('layers: %s' % [(m.name, 'abs' if m.absolute_translation else 'delta') for m in mots])


def pose_at(t, absolute_layers):
    flat = []
    for mm in mots:
        p = MOT.sample_present(mm, t)
        conv = absolute_layers and mm.absolute_translation
        d = {}
        for bone, ch in p.items():
            if bone >= len(bones):
                continue
            dd = {}
            if ch.get('T'):
                dd['T'] = dict(ch['T'])
            if ch.get('R'):
                if conv:
                    br = bind_r.get(bone, [0.0, 0.0, 0.0])
                    dd['R'] = {ax: v - br[ax] for ax, v in ch['R'].items()}
                else:
                    dd['R'] = dict(ch['R'])
            if dd:
                d[bone] = dd
        flat.append(d)
    st = MOT.stack_poses(flat)
    out = {}
    for bone, ch in st.items():
        if bone >= len(bones):
            continue
        d = [0.0] * 9
        for ax, v in ch.get('T', {}).items():
            d[ax] = v
        for ax, v in ch.get('R', {}).items():
            d[3 + ax] = v
        d[6] = d[7] = d[8] = 1.0
        out[bone] = tuple(d)
    return out


tiles = []
for absolute_layers in (False, True):
    for t in FRAMES:
        wp = R.world_matrices(bones, pose_at(t, absolute_layers))
        world = [(wp[i] @ binv[i]) if (wp[i] is not None and binv[i] is not None) else wp[i]
                 for i in range(len(wp))]
        fobj = []
        for o in objs:
            pos, nrm = R.skin_positions(o['verts'], bones, world)
            g = dict(o); g['pos'] = pos; g['nrm'] = nrm
            fobj.append(g)
        fb = R.Framebuffer(190, 280, bg=(0.12, 0.13, 0.16))
        R.render_scene(fb, fobj, CTR + np.array([0.0, 0.0, 1.0]) * (2.6 * HGT),
                       CTR, up=(0, 1, 0), fovy=30.0)
        tiles.append(('%s t=%d' % ('ABS' if absolute_layers else 'delta', t),
                      Image.fromarray((np.clip(fb.color, 0, 1) * 255).astype(np.uint8))))

cols = len(FRAMES)
cw = max(i.width for _l, i in tiles)
chh = max(i.height for _l, i in tiles) + 14
sheet = Image.new('RGB', (cw * cols, chh * 2), (22, 22, 26))
dr = ImageDraw.Draw(sheet)
for i, (lbl, im) in enumerate(tiles):
    x = (i % cols) * cw
    y = (i // cols) * chh
    sheet.paste(im, (x, y + 14))
    dr.text((x + 3, y + 2), lbl, fill=(230, 230, 230))
sheet.save(OUT)
print('sheet -> %s' % OUT)
