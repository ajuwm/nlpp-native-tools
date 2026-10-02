# -*- coding: utf-8 -*-
"""The composer engine: build and render a scene from the game's own assets.

    python compose.py --char m --outfit m_01_004 --hair m_00_200 \
                      --motion m_00011_40 --frames 0,16,32,48 --out D:/dsh/nlpp/out/compose.png

Everything stays in the game's native formats (.smes / .smat / .bone / .texi / .mot);
nothing is converted to OBJ or glTF.  This is the core the interactive composer will
drive -- character, hair, outfit and motion are all just parameters here.

Character = hair(_00_XXX) + outfit(_01_XXX) + legs(_04_000_XX) + face
Motion    = body(_40.._43) + hand + lower + skirt, stacked on the shared 254-bone rig
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nlp_pack as N          # noqa: E402
import nlp_model as M         # noqa: E402
import nlp_mot as MOT         # noqa: E402
import nlp_render as R        # noqa: E402

IMG = r'D:/dsh/nlpp/img.bin'

# motion layers: (filename, base bone, records per bone)
# Body motions animate the SHARED 254-bone rig by bone index and all live under the
# m_ prefix -- which is the MOTION category, not the character m.  The r_/n_ files
# are facial expressions only (147 of them).  So the same list serves every character.
LAYERS = {
    # per=3 for the BODY motion.  Measured with the fixed diagnostics:
    #   per=1 puts the ~80.9 records on bones 2,3,4,5,7,8 -- which includes two SPINE
    #         bones, so the chest is turned by hip+spine = 161.8 deg: an impossible
    #         waist twist (this is what the user reported).
    #   per=3 puts them on bones 2,3,4 only (hip + both thighs) and the measured
    #         hip-vs-spine twist never exceeds 0.22 deg over the whole motion.
    # An earlier round rejected per=3 on the strength of a broken measurement (bone
    # positions read out of the skinning matrix); that conclusion was wrong.
    # Only the layers whose target bones have been VERIFIED are kept.
    #
    # bone -> mesh, measured (native/bone_owner.py):
    #   bones 49..63  -> the m_hand mesh            => m_handNNN is the hand layer
    #   bones 78..96  -> the m_00_200 HAIR mesh and the m_face mesh
    #                    (m_face uses 80..88 and 91..96)
    #   bones 12..18  -> the leg meshes m_04_000_05/06
    #
    # That kills two earlier assumptions: 78..96 is NOT "lower body / skirt" (the layer
    # file names misled me), and m_lower011 / m_skirt_000 were therefore animating the
    # HAIR.  That is what put a spike on the head in m_31235_32_skirt -- the motion
    # drives 78..96 directly.  Both layers are dropped until their real target is known.
    # The ARM chain is bones 30/33/42/191/194/203/210..224 (measured from the mesh
    # weights, native/arm_bones.py).  No m_upperNNN layer drives those -- they all drive
    # 34..48, which the arm mesh barely uses, which is why each of them folded the arm.
    # The only motions covering the whole arm set are the _10 variants
    # (m_32510_10 / m_31834_10 / m_31450_10).  Trying m_32510_10 here made it WORSE --
    # the head turned away and the legs folded -- because it belongs to a different
    # situation and its absolute values fight this body motion.  So the arms stay as the
    # body motion leaves them (roughly horizontal) until the arm motion is paired by the
    # game's own rule rather than by guesswork.
    # The body motion matters more than any layer.  Bone coverage measured over all
    # 1701 m_NNNNN_VV motions (native/ scan): 1663 of them drive BOTH the arm set and
    # the lower legs, but m_00000_40 -- the file I happened to pick first -- drives only
    # 2 arm bones, i.e. it is a legs-only motion.  That is why the arms never moved:
    # not a decode problem, a motion-selection problem.
    #   m_31003_40 : 21 arm bones + 7 lower-leg bones, 94 frames
    # Rendered it gives a proper walk cycle with the height pinned at 158 in every
    # sampled frame, against 127..167 for m_00000_40.
    # The body motion alone now renders correctly.  Two things had to be fixed first:
    #   1. the keyframe stride is 2*nch, not nch (nlp_mot.load_mot_final).  With stride
    #      nch, channel 0 interleaved two different quantities -- that is what produced
    #      the "six leg bones all carry ~80.9" mystery and the permanent crouch.
    #   2. the layered motions are NOT safe to stack here yet: m_hand000 bends the hand
    #      into a flat plate, and m_lower011 / m_skirt_000 actually drive the HAIR bones
    #      (78..96), not the lower body.
    # With just the body motion the character stands with straight legs and a natural
    # step.  Re-add a layer only once its target bones have been verified.
    'default': [('m_00010_40.mot', 2, 3)],
}


# The full character is a BASE BODY plus clothing.  The earlier session scripts
# (build_v12.py / truechar.py) already used this assembly; compose/viewer/server had
# regressed to five parts, which is why the character had no hands and legs that
# stopped at the knee:
#   <char>_face        head / face features
#   <char>_04_000_00..06   SEVEN consecutive leg segments, _06 reaching the ground
#   <char>_hand        both hands
# m_04_000_00 Y=[77.4,89.5]  _01 [72.7,88.8]  _02 [66.0,74.5]  _03 [58.1,68.6]
# m_04_000_04 Y=[44.2,59.5]  _05 [10.7,46.3]  _06 [0.0,11.3]   <- foot at ground level
def character_parts(pre, hair, outfit, legs=7, hand=True):
    """Full part list for one character.  legs=2 reproduces the old (truncated) look."""
    parts = [hair, outfit]
    parts += ['%s_04_000_%02d' % (pre, i) for i in range(max(1, min(7, legs)))]
    if hand:
        parts.append('%s_hand' % pre)
    parts.append('%s_face' % pre)
    return parts


def grab_models(names, img=IMG):
    out, bones = [], None
    for nm in names:
        try:
            m = M.load_model(img, nm)
        except Exception as e:
            print('  (skip %s: %s)' % (nm, str(e)[:44]))
            continue
        if not m.meshes:
            continue
        if bones is None:
            bones = m.bones
        out.append((nm, m))
    return out, bones


def collect_geometry(models):
    objs = []
    for nm, m in models:
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
                        v['weights'] = [x / s for x in w]      # normalise
                    verts.append(v)
                objs.append(dict(
                    verts=verts,
                    pos=np.array([v['pos'] for v in verts], np.float64),
                    nrm=np.array([v['nrm'] for v in verts], np.float64),
                    uv=np.array([v['uv'] for v in verts], np.float64),
                    indices=[remap[i] for i in pr.indices if i in remap],
                    texture=tex,
                    wrap_u=mp.wrap_u if mp else 2, wrap_v=mp.wrap_v if mp else 2,
                    base_color=(1.0, 1.0, 1.0),
                    alpha_func=getattr(mtl, 'alpha_test_func', None),
                    alpha_ref=(getattr(mtl, 'alpha_test_ref', 0)
                               if getattr(mtl, 'alpha_test_enabled', False) else None)))
    return objs


def load_layers(pre, img=IMG, bones_bind=None):
    spec = LAYERS.get(pre) or LAYERS.get('default')
    if not spec:
        return []
    got, mots = set(), []
    with open(img, 'rb') as f:
        for p in N.iter_packs(img):
            for pf in p.files:
                for name, base, per in spec:
                    if pf.name == name and name not in got:
                        got.add(name)
                        mots.append(MOT.load_mot_final(N.pack_bytes(f, pf), name,
                                                       base=base, per=per,
                                                       bones_bind=bones_bind))
            if len(got) == len(spec):
                break
    return mots


def pose_at(mots, bind_t, t):
    flat = []
    for mm in mots:
        pose = MOT.sample_present(mm, t)
        d = {}
        for bone, ch in pose.items():
            dd = {}
            for ax, v in ch.get('T', {}).items():
                dd.setdefault('T', {})[ax] = (v - bind_t.get(bone, [0, 0, 0])[ax]
                                              if mm.absolute_translation else v)
            for ax, v in ch.get('R', {}).items():
                dd.setdefault('R', {})[ax] = v
            if dd:
                d[bone] = dd
        flat.append(d)
    stacked = MOT.stack_poses(flat)
    out = {}
    for bone, ch in stacked.items():
        d = [0.0] * 9
        for ax, v in ch.get('T', {}).items():
            d[ax] = v
        for ax, v in ch.get('R', {}).items():
            d[3 + ax] = v
        d[6] = d[7] = d[8] = 1.0
        out[bone] = tuple(d)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--char', default='m', choices=['m', 'r', 'n'])
    ap.add_argument('--hair', default=None)
    ap.add_argument('--outfit', default=None)
    ap.add_argument('--frame', type=float, default=0.0)
    ap.add_argument('--frames', default=None, help='comma separated; renders a filmstrip')
    ap.add_argument('--w', type=int, default=300)
    ap.add_argument('--h', type=int, default=430)
    ap.add_argument('--zoom', type=float, default=2.35)
    ap.add_argument('--out', default=r'D:/dsh/nlpp/out/compose.png')
    ap.add_argument('--bg', default=None, help='background asset name (bs_*.jpg)')
    ap.add_argument('--img', default=IMG)
    a = ap.parse_args()

    pre = a.char
    hair = a.hair or {'m': 'm_00_200', 'r': 'r_00_000', 'n': 'n_00_000'}[pre]
    outfit = a.outfit or {'m': 'm_01_012', 'r': 'r_01_000', 'n': 'n_01_000'}[pre]
    parts = character_parts(pre, hair, outfit,
                            legs=int(os.environ.get('LP_LEGS', '7')),
                            hand=os.environ.get('LP_HAND', '1') != '0')

    models, bones = grab_models(parts, a.img)
    if bones is None:
        print('!! nothing loaded')
        return 1
    objs = collect_geometry(models)
    print('parts: %s' % ', '.join(nm for nm, _ in models))
    print('primitives: %d   bones: %d' % (len(objs), len(bones)))

    mots = load_layers(pre, a.img)
    print('motion layers: %d' % len(mots))
    bind_t = {b.index: list(b.translation) for b in bones}
    wb = R.world_matrices(bones, {})
    bind_inv = [np.linalg.inv(m) if m is not None else None for m in wb]

    times = [float(x) for x in a.frames.split(',')] if a.frames else [a.frame]
    tiles = []
    for t in times:
        pose = pose_at(mots, bind_t, t) if mots else {}
        wp = R.world_matrices(bones, pose)
        world = [(wp[i] @ bind_inv[i]) if (wp[i] is not None and bind_inv[i] is not None)
                 else wp[i] for i in range(len(wp))]
        fobj = []
        for o in objs:
            pos, nrm = R.skin_positions(o['verts'], bones, world)
            g = dict(o)
            g['pos'] = pos
            g['nrm'] = nrm
            fobj.append(g)
        allp = np.vstack([g['pos'] for g in fobj])
        lo, hi = allp.min(axis=0), allp.max(axis=0)
        ctr = (lo + hi) / 2.0
        hgt = float(max(hi[1] - lo[1], 20.0))
        _bimg = None
        if a.bg:
            import nlp_bg
            _bimg = nlp_bg.load_background(a.bg, a.img)
            if _bimg is None:
                print('  (background %s not found)' % a.bg)
        fb = R.Framebuffer(a.w, a.h, bg=(0.12, 0.13, 0.16), background=_bimg)
        R.render_scene(fb, fobj, ctr + np.array([0.0, 0.0, 1.0]) * (a.zoom * hgt),
                       ctr, up=(0, 1, 0), fovy=30.0)
        p = a.out.replace('.png', '_t%d.png' % int(t))
        R.save_png(p, fb)
        tiles.append((t, p))
        print('  t=%-6.0f bbox Y=[%6.1f,%6.1f] -> %s' % (t, lo[1], hi[1], os.path.basename(p)))

    ims = [(t, Image.open(p).convert('RGB')) for t, p in tiles]
    if len(ims) == 1:
        ims[0][1].save(a.out)
    else:
        cw = max(i.width for _t, i in ims)
        ch = max(i.height for _t, i in ims) + 16
        sheet = Image.new('RGB', (cw * len(ims), ch), (26, 26, 30))
        dr = ImageDraw.Draw(sheet)
        for i, (t, im) in enumerate(ims):
            sheet.paste(im, (i * cw, 16))
            dr.text((i * cw + 4, 3), 't=%d' % t, fill=(235, 235, 235))
        sheet.save(a.out)
    print('-> %s' % a.out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
