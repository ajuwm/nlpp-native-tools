# -*- coding: utf-8 -*-
"""Render SEVERAL motions side by side in one sheet, so one bad motion cannot mislead.

The previous rounds kept judging the whole decoder on m_00011_40 -- the single file I
happened to pick early on.  Spread the sample instead: walk the .mot catalogue, take
motions from different id ranges, and render 6 frames of each into one contact sheet.
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
from compose import character_parts   # noqa: E402

IMG = r'D:/dsh/nlpp/img.bin'
OUT = r'D:/dsh/nlpp/out/motion_sheet.png'
N_MOT = 6
N_FR = 6

# ---- body parts -------------------------------------------------------------
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
            objs.append(dict(verts=verts,
                             uv=np.array([v['uv'] for v in verts], np.float64),
                             texture=None,
                             indices=[remap[i] for i in pr.indices if i in remap]))
bind_t = {b.index: list(b.translation) for b in bones}
wb = R.world_matrices(bones, {})
binv = [np.linalg.inv(m) if m is not None else None for m in wb]
BIND_LO = np.vstack([o['verts'][0]['pos'] for o in objs])  # placeholder, replaced below
allpos = np.vstack([np.array([v['pos'] for v in o['verts']]) for o in objs])
BLO, BHI = allpos.min(axis=0), allpos.max(axis=0)
CTR = (BLO + BHI) / 2.0
HGT = float(max(BHI[1] - BLO[1], 20.0))

# ---- pick motions spread over the catalogue --------------------------------
cands = []
with open(IMG, 'rb') as f:
    for p in N.iter_packs(IMG):
        for pf in p.files:
            nm = pf.name
            if nm.startswith('m_') and nm.endswith('.mot') and '__' not in nm:
                cands.append((nm, pf))
        if len(cands) > 4000:
            break
cands.sort(key=lambda x: x[0])
step = max(1, len(cands) // N_MOT)
picked = [cands[i * step] for i in range(N_MOT)]
print('catalogue: %d body motions; sampling %s' % (len(cands), [p[0] for p in picked]))

raws = {}
with open(IMG, 'rb') as f:
    for nm, pf in picked:
        try:
            raws[nm] = N.pack_bytes(f, pf)
        except Exception:
            pass


def pose_of(mm, t):
    p = MOT.sample_present(mm, t)
    out = {}
    for bone, ch in p.items():
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
for nm, pf in picked:
    raw = raws.get(nm)
    if not raw:
        continue
    mm = MOT.load_mot_final(raw, nm)
    dur = max(4, int(mm.duration) or 60)
    for k in range(N_FR):
        t = int(dur * k / float(N_FR))
        wp = R.world_matrices(bones, pose_of(mm, t))
        world = [(wp[i] @ binv[i]) if (wp[i] is not None and binv[i] is not None) else wp[i]
                 for i in range(len(wp))]
        fobj = []
        for o in objs:
            pos, nrm = R.skin_positions(o['verts'], bones, world)
            g = dict(o)
            g['pos'] = pos
            g['nrm'] = nrm
            fobj.append(g)
        fb = R.Framebuffer(150, 220, bg=(0.12, 0.13, 0.16))
        R.render_scene(fb, fobj, CTR + np.array([0.0, 0.0, 1.0]) * (2.6 * HGT),
                       CTR, up=(0, 1, 0), fovy=30.0)
        tiles.append(('%s t=%d' % (nm.replace('.mot', ''), t),
                      Image.fromarray((np.clip(fb.color, 0, 1) * 255).astype(np.uint8))))

cols = N_FR
rows = (len(tiles) + cols - 1) // cols
cw = max(i.width for _l, i in tiles)
ch = max(i.height for _l, i in tiles) + 14
sheet = Image.new('RGB', (cw * cols, ch * rows), (22, 22, 26))
dr = ImageDraw.Draw(sheet)
for i, (lbl, im) in enumerate(tiles):
    x = (i % cols) * cw
    y = (i // cols) * ch
    sheet.paste(im, (x, y + 14))
    dr.text((x + 3, y + 2), lbl, fill=(230, 230, 230))
sheet.save(OUT)
print('sheet -> %s   (%d tiles)' % (OUT, len(tiles)))
