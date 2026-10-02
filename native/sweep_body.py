# -*- coding: utf-8 -*-
"""Dense frame sweep of the FULL body, to find any frame where the character inverts."""
import os, sys
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nlp_pack as N, nlp_model as M, nlp_mot as MOT, nlp_render as R
from compose import character_parts, LAYERS

IMG = r'D:/dsh/nlpp/img.bin'
OUT = r'D:/dsh/nlpp/out'

objs, bones = [], None
for nm in character_parts('m', 'm_00_200', 'm_01_012'):
    try:
        m = M.load_model(IMG, nm)
    except Exception:
        continue
    if bones is None:
        bones = m.bones
    for me in m.meshes:
        mtl = m.materials[me.material_index] if me.material_index < len(m.materials) else None
        mp = mtl.diffuse_mapper() if mtl else None
        tex = None
        if mp and mp.texture:
            t = m.textures.get(mp.texture)
            if t and 'rgba' in t:
                tex = np.frombuffer(t['rgba'], np.uint8).reshape(t['h'], t['w'], 4)
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
                             pos=np.array([v['pos'] for v in verts], np.float64),
                             nrm=np.array([v['nrm'] for v in verts], np.float64),
                             uv=np.array([v['uv'] for v in verts], np.float64),
                             indices=[remap[i] for i in pr.indices if i in remap],
                             texture=tex, wrap_u=mp.wrap_u if mp else 2,
                             wrap_v=mp.wrap_v if mp else 2, base_color=(1.0, 1.0, 1.0),
                             alpha_func=getattr(mtl, 'alpha_test_func', None),
                             alpha_ref=(getattr(mtl, 'alpha_test_ref', 0)
                                        if getattr(mtl, 'alpha_test_enabled', False) else None)))

BIND = {b.index: list(b.translation) for b in bones}
mots, got = [], set()
MOTION = os.environ.get('LP_SWEEP_MOTION', LAYERS['default'][0][0])
# take base/per from the LAYERS body entry -- hardcoding per=1 here silently ignored
# whatever compose.py was configured with, so an earlier "per=3 fixed the twist" change
# never reached this render.
spec = [(MOTION, LAYERS['default'][0][1], LAYERS['default'][0][2])] + list(LAYERS['default'][1:])
with open(IMG, 'rb') as f:
    for p in N.iter_packs(IMG):
        for pf in p.files:
            for name, base, per in spec:
                if pf.name == name and name not in got:
                    got.add(name)
                    mots.append(MOT.load_mot_final(N.pack_bytes(f, pf), name, base=base, per=per,
                                               bones_bind=BIND))
        if len(got) == len(spec):
            break

bind_t = {b.index: list(b.translation) for b in bones}
MOT.BIND_ROT = {b.index: tuple(b.rotation) for b in bones}
wb = R.world_matrices(bones, {})
binv = [np.linalg.inv(m) if m is not None else None for m in wb]


def pose_at(t):
    flat = []
    for mm in mots:
        p = MOT.sample_present(mm, t)
        d = {}
        for bone, ch in p.items():
            dd = {}
            for ax, v in ch.get('T', {}).items():
                dd.setdefault('T', {})[ax] = (v - bind_t.get(bone, [0, 0, 0])[ax]
                                              if mm.absolute_translation else v)
            for ax, v in ch.get('R', {}).items():
                dd.setdefault('R', {})[ax] = v
            if dd:
                d[bone] = dd
        flat.append(d)
    st = MOT.stack_poses(flat)
    out = {}
    for bone, ch in st.items():
        d = [0.0] * 9
        for ax, v in ch.get('T', {}).items():
            d[ax] = v
        for ax, v in ch.get('R', {}).items():
            d[3 + ax] = v
        d[6] = d[7] = d[8] = 1.0
        out[bone] = tuple(d)
    return out


def render(t, w=200, h=300):
    pose = pose_at(t)
    wp = R.world_matrices(bones, pose)
    world = [(wp[i] @ binv[i]) if (wp[i] is not None and binv[i] is not None) else wp[i]
             for i in range(len(wp))]
    fobj = []
    for o in objs:
        pos, nrm = R.skin_positions(o['verts'], bones, world)
        g = dict(o); g['pos'] = pos; g['nrm'] = nrm
        fobj.append(g)
    # FIXED camera: framed once from the BIND pose, then held for every frame.
    # Re-framing on the posed bbox made the camera dive in whenever the motion bent
    # the body (height 158 -> 52) and the head then filled the screen.
    if not hasattr(render, '_cam'):
        bind_lo = np.vstack([o['pos'] for o in objs]).min(axis=0)
        bind_hi = np.vstack([o['pos'] for o in objs]).max(axis=0)
        render._cam = ((bind_lo + bind_hi) / 2.0,
                       float(max(bind_hi[1] - bind_lo[1], 20.0)))
    ctr, hgt = render._cam
    fb = R.Framebuffer(w, h, bg=(0.12, 0.13, 0.16))
    R.render_scene(fb, fobj, ctr + np.array([0.0, 0.0, 1.0]) * (2.6 * hgt),
                   ctr, up=(0, 1, 0), fovy=30.0)
    allp = np.vstack([g['pos'] for g in fobj])
    lo, hi = allp.min(axis=0), allp.max(axis=0)
    return Image.fromarray((np.clip(fb.color, 0, 1) * 255).astype(np.uint8)), lo, hi


times = list(range(0, 96, 8))
tiles = []
print('%-6s %-18s %-10s %s' % ('t', 'bbox Y', 'height', 'head-vs-feet'))
for t in times:
    im, lo, hi = render(t)
    # which end is the head? the head bone is high in bind; check the hair's Y extent
    tiles.append(('t=%d Y=[%.0f,%.0f]' % (t, lo[1], hi[1]), im))
    print('%-6d [%7.1f,%7.1f] %-10.1f' % (t, lo[1], hi[1], hi[1] - lo[1]))

cols = 6
rows = (len(tiles) + cols - 1) // cols
cw = max(i.width for _l, i in tiles)
ch = max(i.height for _l, i in tiles) + 16
sheet = Image.new('RGB', (cw * cols, ch * rows), (24, 24, 28))
dr = ImageDraw.Draw(sheet)
for i, (lbl, im) in enumerate(tiles):
    x = (i % cols) * cw
    y = (i // cols) * ch
    sheet.paste(im, (x, y + 16))
    dr.text((x + 4, y + 3), lbl, fill=(235, 235, 235))
sheet.save(os.path.join(OUT, 'sweep_%s.png' % MOTION.replace('.mot', '')))
print('MOTION=%s' % MOTION)
print('sheet -> sweep_%s.png' % MOTION.replace('.mot', ''))
