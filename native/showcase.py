# -*- coding: utf-8 -*-
"""Showcase render: one still (PNG) and one animation (GIF), both with a real game
background, so the current state can be judged by eye.

    python showcase.py

Writes  out/showcase.png  and  out/showcase.gif
Everything is the game's own data: .smes / .smat / .bone / .texi / .mot / .jpg.
"""
import io
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nlp_pack as N          # noqa: E402
import nlp_model as M         # noqa: E402
import nlp_mot as MOT         # noqa: E402
import nlp_render as R        # noqa: E402
import nlp_bg                 # noqa: E402
from compose import character_parts, LAYERS   # noqa: E402

IMG = r'D:/dsh/nlpp/img.bin'
OUT = r'D:/dsh/nlpp/out'

CHAR, HAIR, OUTFIT = 'm', 'm_00_200', 'm_01_012'
BACKGROUND = 'bs_0000_00_00.jpg'
MOTION = os.environ.get('LP_SHOW_MOTION', 'm_00000_40.mot')

# ---------------------------------------------------------------- geometry
objs = []
bones = None
for nm in character_parts(CHAR, HAIR, OUTFIT):
    try:
        m = M.load_model(IMG, nm)
    except Exception as e:
        print('  skip %s (%s)' % (nm, str(e)[:40]))
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
            objs.append(dict(
                verts=verts,
                pos=np.array([v['pos'] for v in verts], np.float64),
                nrm=np.array([v['nrm'] for v in verts], np.float64),
                uv=np.array([v['uv'] for v in verts], np.float64),
                indices=[remap[i] for i in pr.indices if i in remap],
                texture=tex, wrap_u=mp.wrap_u if mp else 2, wrap_v=mp.wrap_v if mp else 2,
                base_color=(1.0, 1.0, 1.0),
                alpha_func=getattr(mtl, 'alpha_test_func', None),
                alpha_ref=(getattr(mtl, 'alpha_test_ref', 0)
                           if getattr(mtl, 'alpha_test_enabled', False) else None)))
print('primitives: %d   bones: %d' % (len(objs), len(bones)))

# ---------------------------------------------------------------- motion
mots = []
spec = LAYERS['default']
got = set()
with open(IMG, 'rb') as f:
    for p in N.iter_packs(IMG):
        for pf in p.files:
            for name, base, per in spec:
                if pf.name == name and name not in got:
                    got.add(name)
                    mots.append(MOT.load_mot_final(N.pack_bytes(f, pf), name,
                                                   base=base, per=per))
        if len(got) == len(spec):
            break
print('motion layers: %d' % len(mots))

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


BG = nlp_bg.load_background(BACKGROUND, IMG)
print('background: %s -> %s' % (BACKGROUND, 'ok' if BG is not None else 'MISSING'))


def render(w, h, t, zoom=2.6):
    pose = pose_at(t)
    wp = R.world_matrices(bones, pose)
    world = [(wp[i] @ binv[i]) if (wp[i] is not None and binv[i] is not None) else wp[i]
             for i in range(len(wp))]
    fobj = []
    for o in objs:
        pos, nrm = R.skin_positions(o['verts'], bones, world)
        g = dict(o); g['pos'] = pos; g['nrm'] = nrm
        fobj.append(g)
    # fixed camera from the bind pose (see sweep_body.py)
    if not hasattr(render, '_cam'):
        blo = np.vstack([o['pos'] for o in objs]).min(axis=0)
        bhi = np.vstack([o['pos'] for o in objs]).max(axis=0)
        render._cam = ((blo + bhi) / 2.0, float(max(bhi[1] - blo[1], 20.0)))
    ctr, hgt = render._cam
    fb = R.Framebuffer(w, h, bg=(0.12, 0.13, 0.16), background=BG)
    R.render_scene(fb, fobj, ctr + np.array([0.0, 0.0, 1.0]) * (zoom * hgt),
                   ctr, up=(0, 1, 0), fovy=30.0)
    col = np.clip(fb.color, 0, 1)
    return Image.fromarray((col * 255).astype(np.uint8))


# ---------------------------------------------------------------- still
still = render(600, 860, 0)
p1 = os.path.join(OUT, 'showcase.png')
still.save(p1)
print('PNG -> %s' % p1)

# ---------------------------------------------------------------- animation
n_frames, step, fps = 48, 2, 15
frames = []
for i in range(n_frames):
    im = render(340, 480, i * step)
    frames.append(im.convert('P', palette=Image.ADAPTIVE, colors=128))
p2 = os.path.join(OUT, 'showcase.gif')
frames[0].save(p2, 'GIF', save_all=True, append_images=frames[1:],
               duration=int(1000.0 / fps), loop=0, optimize=True)
print('GIF -> %s  (%d frames, %d fps)' % (p2, n_frames, fps))
