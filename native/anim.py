# -*- coding: utf-8 -*-
"""Animated render that reuses the SWEEP's pose and camera path exactly.

sweep_body.py is the verified path: single motion, camera framed once from the bind pose,
frames sampled evenly across the duration.  showcase.py drifted from it (different camera
and a layer stack that bent the hands), so this script builds the animation the same way
sweep does and writes both a still and a GIF.

    python anim.py                 # LAYERS['default'], 48 frames
    LP_ANIM_MOTION=m_00000_40.mot LP_ANIM_FRAMES=24 python anim.py
"""
import os
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
OUT = r'D:/dsh/nlpp/out'
SPEC = list(LAYERS['default'])
if os.environ.get('LP_ANIM_MOTION'):
    SPEC[0] = (os.environ['LP_ANIM_MOTION'], SPEC[0][1], SPEC[0][2])
NFR = int(os.environ.get('LP_ANIM_FRAMES') or 48)
W, H = 340, 480
BG = 'bs_0000_00_00.jpg'

# ---- geometry (same assembly as sweep_body.py) ------------------------------
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
MOT.BIND_ROT = {b.index: tuple(b.rotation) for b in bones}
wb = R.world_matrices(bones, {})
binv = [np.linalg.inv(m) if m is not None else None for m in wb]

# bind-pose framing: fixed for the WHOLE animation (a per-frame refit made the camera
# swing wildly and was one of the earliest bugs)
bind_pts = np.vstack([R.skin_positions(o['verts'], bones, wb)[0] for o in objs])
LO, HI = bind_pts.min(axis=0), bind_pts.max(axis=0)
CTR = (LO + HI) / 2.0
HGT = float(max(HI[1] - LO[1], 20.0))
FLOOR = float(LO[1])
EYE = CTR + np.array([0.0, 0.0, 1.0]) * (2.4 * HGT)

# ---- background -------------------------------------------------------------
bg = None
try:
    import nlp_bg
    b = nlp_bg.load_background(IMG, BG)
    if b is not None:
        bg = b
except Exception:
    try:
        import nlp_pack as _N
        with open(IMG, 'rb') as f:
            for p in _N.iter_packs(IMG):
                for pf in p.files:
                    if pf.name == BG:
                        bg = _N.decode_background(_N.pack_bytes(f, pf))
                        break
                if bg is not None:
                    break
    except Exception:
        bg = None
print('background: %s' % ('ok' if bg is not None else 'none'))

# ---- motions -----------------------------------------------------------------
mots, got = [], set()
with open(IMG, 'rb') as f:
    for p in N.iter_packs(IMG):
        for pf in p.files:
            for name, base, per in SPEC:
                if pf.name == name and name not in got:
                    got.add(name)
                    mots.append(MOT.load_mot_final(N.pack_bytes(f, pf), name, base=base, per=per))
        if len(got) == len(SPEC):
            break
print('motions: %s' % [m.name for m in mots])
dur = max([int(m.duration) for m in mots] or [60])
print('duration=%d  frames=%d' % (dur, NFR))


def pose_at(t):
    flat = []
    for mm in mots:
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


frames, ys = [], []
for k in range(NFR):
    t = int(dur * k / float(NFR))
    wp = R.world_matrices(bones, pose_at(t))
    world = [(wp[i] @ binv[i]) if (wp[i] is not None and binv[i] is not None) else wp[i]
             for i in range(len(wp))]
    fobj = []
    for o in objs:
        pos, nrm = R.skin_positions(o['verts'], bones, world)
        g = dict(o); g['pos'] = pos; g['nrm'] = nrm
        fobj.append(g)
    pts = np.vstack([g['pos'] for g in fobj])
    ys.append((k, t, float(pts[:, 1].min()), float(pts[:, 1].max())))
    fb = R.Framebuffer(W, H, bg=(0.12, 0.14, 0.18), background=bg)
    # put the character's lowest point on the floor line of the bind pose
    shift = np.array([0.0, FLOOR - pts[:, 1].min(), 0.0])
    for g in fobj:
        g['pos'] = g['pos'] + shift
    R.render_scene(fb, fobj, EYE + shift, CTR + shift, up=(0, 1, 0), fovy=30.0)
    frames.append(Image.fromarray((np.clip(fb.color, 0, 1) * 255).astype(np.uint8)))

print('lowest Y per frame: min=%.1f max=%.1f (floor=%.1f)'
      % (min(y[2] for y in ys), max(y[2] for y in ys), FLOOR))
frames[0].save(os.path.join(OUT, 'anim.png'))
frames[0].save(os.path.join(OUT, 'anim.gif'), save_all=True,
               append_images=frames[1:], duration=66, loop=0)
print('PNG -> %s/anim.png' % OUT)
print('GIF -> %s/anim.gif  (%d frames)' % (OUT, len(frames)))
