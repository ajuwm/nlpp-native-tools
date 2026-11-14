# -*- coding: utf-8 -*-
r"""ab_light.py — 光照 A/B 渲染：LPW 现有拟合光照 vs 实机真值（Citra 读出）

    # 现有拟合光照（LPW 默认参数）
    python ab_light.py
    # 实机真值（2 盏灯 + 真实 ambient，来自 out/live/live_lights.json）
    set LP_LIGHTS_JSON=..\out\live\true_lights_env.json
    python ab_light.py

输出：out/ab/ab_<name>.png（名字由 LP_AB_NAME 指定）
模型/贴图/动作/背景全部来自工作区 data/img.bin（游戏原始数据）。
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

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
IMG = os.path.join(_ROOT, 'data', 'img.bin')
OUT = os.path.join(_ROOT, 'nlpp-re', 'out', 'ab')
os.makedirs(OUT, exist_ok=True)

CHAR, HAIR, OUTFIT = 'm', 'm_00_200', 'm_01_012'
BACKGROUND = 'bs_0000_00_00.jpg'
MOTION = os.environ.get('LP_SHOW_MOTION', 'm_00000_40.mot')

# 打印当前使用的光照，便于确认 A/B 身份
try:
    _l = R.LIGHTS
    print('[AB] shade_mode=%s  multi=%s' % (R.SHADE_MODE,
          'None' if getattr(_l, 'multi', None) is None else len(_l.multi)))
    print('[AB] ambient=%s' % (np.round(_l.ambient, 4),))
    if getattr(_l, 'multi', None):
        for i, lt in enumerate(_l.multi):
            print('[AB]   light%d dir=%s color=%s spec=%s' %
                  (i, np.round(lt['dir'], 4), np.round(lt['color'], 4),
                   np.round(lt['spec'], 4)))
    else:
        print('[AB] light_dir=%s light_color=%s spec_color=%s' %
              (np.round(_l.light_dir, 4), np.round(_l.light_color, 4),
               np.round(_l.spec_color, 4)))
    print('[AB] out_tint=%s fresnel=%.3f spec=%.3f difcloth=%.3f' %
          (np.round(_l.out_tint, 4), _l.fresnel_strength, _l.spec_strength,
           float(os.environ.get('LP_DIFFCLOTH', 0.45))))
except Exception as _e:
    print('[AB] lighting info error:', _e)

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
name = os.environ.get('LP_AB_NAME', 'fitted')
still = render(600, 860, 0)
p1 = os.path.join(OUT, 'ab_%s.png' % name)
still.save(p1)
print('[AB] PNG -> %s' % p1)
