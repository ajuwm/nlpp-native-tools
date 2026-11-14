# -*- coding: utf-8 -*-
"""The composer engine: build and render a scene from the game's own assets.

    python compose.py --char m --outfit m_01_004 --hair m_00_200 \
                      --motion m_00010_41 --frames 0,16,32,48 --out out/compose.png

Everything stays in the game's native formats (.smes / .smat / .bone / .texi / .mot);
nothing is converted to OBJ or glTF.  This is the core the interactive composer will
drive -- character, hair, outfit and motion are all just parameters here.

Character = hair(_00_XXX) + outfit(_01_XXX) + legs(_04_000_XX) + face
Motion    = body(_40.._43) + hand + lower + skirt, stacked on the shared 254-bone rig
"""
import argparse
import json
import os
import re
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nlp_pack as N          # noqa: E402
import nlp_model as M         # noqa: E402
import nlp_mot as MOT         # noqa: E402
import nlp_render as R        # noqa: E402
import nlp_shade as SH        # noqa: E402

IMG = os.environ.get('LP_IMG', 'data/img.bin')

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

# LP_MOTION 覆盖默认动作（base/per 只是没有骨骼表时的兜底，现在有 ID 表就用不上了）
import os as _os_mot
if _os_mot.environ.get('LP_MOTION'):
    LAYERS['default'] = [(_os_mot.environ['LP_MOTION'], 2, 3)]


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


def _needs_blend(m, mtl):
    """该材质的贴图是否含明显半透明 → 需要 alpha 混合而不是硬切。

    实测：脸部 `m_face_col` 层（腮红/眼影）贴图有 35.9% 半透明像素、
    且材质没开 alpha 测试；只做 alpha 测试会把整层切掉，画面"少一层" ✗ ✓

    ★ 2026-10-17 补充：`m_face_col`（rgba5551）整张是 `00 1f` 全平 ——
      R/G/B 近黑、**alpha 全 0** ⇒ 这一层本应【完全不可见】✓。
      以前它被解码成噪点、又被噪点过滤器丢掉，所以看不出来 ✗；
      修好格式后如果仍按【不透明】渲染，就会把整张脸压暗 ✗（用户看到的"眼睛又变回去了"）
      ⇒ 判据要同时覆盖"半透明"和"几乎全透明"两种情况 ✓
    """
    try:
        mp = mtl.mappers.get(0)
        t = m.textures.get(mp.texture) if (mp and mp.texture) else None
        if not t or 'rgba' not in t:
            return False
        a = np.frombuffer(t['rgba'], np.uint8).reshape(t['h'], t['w'], 4)[:, :, 3]
        semi = float(((a > 8) & (a < 248)).mean())
        return semi > 0.05 or float(a.mean()) < 0.9
    except Exception:
        return False


# 高光层的动画相位（0/1/2 → 取该层的 R/G/B 通道）✓
# 实测：m_eye_light 把【同一处高光的三个动画帧】分别打包进 R/G/B ✓
# ⇒ 每帧只取一个通道；取 max 会把三帧糊成三个点 ✗
_PHASE = [0]


def _specular_unit(m, mtl):
    """unit1..N 里名为 `*_spe` 的贴图 = 高光强度贴图（不是 albedo 叠加层）。

    实测 m_00_200（头发）的第二张贴图 m_00_200_spe 是黑底亮条纹的灰度图，
    条纹位置正好对应实机头发上那条高光带 ⇒ 它必须乘在**光照的高光项**上
    （D0 × spec map），而不是像以前那样按 albedo 加法叠进头发颜色 ✗
    —— 那样只会让头发整体发灰、且没有那条亮带。
    LP_SPECMAP=0 可关闭该贴图（A/B 对照用）。
    """
    if os.environ.get('LP_SPECMAP', '0') == '0':
        return None
    for u in sorted(mtl.mappers.keys()):
        if u == 0:
            continue
        mp = mtl.mappers[u]
        nm = (mp.texture or '').lower()
        if '_spe' not in nm:
            continue
        t = m.textures.get(mp.texture)
        if not t or 'rgba' not in t:
            continue
        a = np.frombuffer(t['rgba'], np.uint8).reshape(t['h'], t['w'], 4)
        return a, mp.wrap_u, mp.wrap_v
    return None


def _extra_units(m, mtl):
    """材质的 unit1..N 贴图（多贴图 TEV 层）。

    ★ 实测（2026-10-17）：脸部材质是多贴图的 ——
        m_eye  = m_eye + m_eye_uru(水润) + m_eye_light(高光)   三层 ✓
        m_face_01(第2个) = 脸 + m_cheek_col(腮红)
        m_face_col      = 阴影 + m_face_02(眉毛)
      以前只渲染 unit0 ✗ ⇒ 眼睛的高光/水润整层丢失 ✓ 正是"眼睛少一层"的根因 ✓

    返回 [(rgba 数组, wrap_u, wrap_v, 'add'|'mul'), ...]
      * 层图若是"黑底亮点"（均值暗、最大亮）⇒ 加法叠加 ✓（高光/反光都是这样）
      * 否则按相乘（细节/阴影叠加）✓
    """
    if getattr(mtl, "drop_skin_extra", False):
        return []
    out = []
    for u in sorted(mtl.mappers.keys()):
        if u == 0:
            continue
        mp = mtl.mappers[u]
        t = m.textures.get(mp.texture) if mp.texture else None
        if not t or 'rgba' not in t:
            continue
        a = np.frombuffer(t['rgba'], np.uint8).reshape(t['h'], t['w'], 4)
        gain = 1.0
        if '_spe' in (mp.texture or '').lower() and \
                os.environ.get('LP_SPECMAP', '0') != '0':
            continue          # 高光贴图交给光照的高光项，不当 albedo 叠层 ✓
        if '_spe' in (mp.texture or '').lower():
            # ★ 2026-10-18：发丝 sheen 用加法叠层时带一点蓝灰偏色（实机 sheen ≈(73,69,77)，
            #   比 base 多 +27/+24/+28 ⇒ 蓝、红都比绿多），否则会是中性灰、饱和度偏低。
            tint = np.array([float(x) for x in os.environ.get(
                'LP_SHEENTINT', '1.10,1.0,1.14').split(',')], np.float64)
            scaled = a.astype(np.float64).copy()
            scaled[:, :, :3] = np.clip(scaled[:, :, :3] * tint[None, None, :], 0, 255)
            a = scaled.astype(np.uint8)
            gain = float(os.environ.get('LP_SHEENGAIN', '1.05'))
        rgb = a[:, :, :3].astype(np.float32) / 255.0
        # ★ 排除"解出来是噪点"的层：相邻像素差很大说明数据不对（例如 m_face_col 的 LA4
        #   这张在某处取错了数据 ✗）。直接把这种层丢掉，总比把噪点加进画面好 ✓
        g = rgb.mean(axis=2)
        if float(np.abs(np.diff(g, axis=1)).mean()) > 12.0:
            continue
        mean, mx = float(rgb.mean()), float(rgb.max())
        if '_spe' in (mp.texture or '').lower():
            mode = 'addrgb'   # 发丝 sheen：彩色加法（三个通道是同一张灰度高光，带偏色）
        else:
            mode = 'add' if (mean < 0.25 and mx > 0.6) else 'mul'
        # ★ 高光层把【多处高光分别打包进 R/G/B 三个通道】—— 例如 m_eye_light 是黑底上
        #   三个分开的红/绿/蓝点。它们不是"三个同时存在的点"✗，而是**同一处高光的三个
        #   动画帧** ✓（省贴图的做法）⇒ 必须【每帧只取一个通道】✓，否则三帧会糊在一起 ✗。
        #   原贴图原样带过去，由渲染器按当前帧号挑通道 ✓
        out.append((a, mp.wrap_u, mp.wrap_v, mode, gain))
    return out


# ------------------------------------------------------------------ skin materials
# 一些服装模型自带"裸皮"部件（手臂 ar / 脖子 ne / 大腿 wa / 躯干 bo）：
# 它们的 unit0 贴图就是角色皮肤贴图（和 m_04_000_wa/_le/_fo 统计上一致），
# 但材质名不是身体那套 m_04_ / m_face / m_hand ⇒ 现有的按名前缀增益覆盖不到。
# 症状：同一块皮肤，身体部分有色温校正，服装自带的那几块却按 1.0 渲染 ⇒
# 亮出 10% 左右，形成"没贴图的白平面"（m_01_002 的 ar/ne/wa3/bo1/wa1 就是这样）。
#
# 判据（数据驱动，不看服装编号）：
#   1. unit0 贴图均值接近实测身体皮肤（m_04_000_wa 253.9/225.9/217.6，
#      _le 254.9/226.7/218.4，_fo 254.9/228.7/220.7）；容差 12 级。
#   2. 局部对比度低（<1.5）：皮肤贴图基本是渐变；波点/条纹布料的相邻像素差更大，
#      例如 m_01_002_bo2 的 contrast=2.25、wa2=5.98 ⇒ 不会被误判成皮肤。
#   3. SMAT 的 specular 非零：实测所有身体皮肤材质（m_04_/m_face/m_hand 以及
#      服装自带的 ar/ne/wa/bo 裸皮）都带 (0.349,0.447,0.639) 的高光色，
#      布料材质是 (0,0,0) ⇒ 这条可以挡住"肤色布料"误判。
#   4. 只有"整套裸皮"才启用：同一模型里同时存在皮肤色的 bo* 和 wa* 部件
#      （泳装/内衣这类把躯干和腿都做成裸皮的服装）。普通服装只在领口/小臂等
#      局部露出皮肤，颜色已按各自材质校准，保持原样不动 ✓。
# 满足这四条时，该模型所有皮肤色材质统一乘上身体皮肤的增益（与 m_04_ 相同）。
SKIN_GAIN = (0.996, 0.898, 0.813)
SKIN_REF_RGB = (254.6, 227.1, 218.9)
_SKIN_FLAG_CACHE = {}


def _texture_is_skin(m, mtl, ref_rgb=SKIN_REF_RGB):
    """SMAT specular 非零 + unit0 贴图低对比且接近身体皮肤均值。"""
    ref = np.asarray(ref_rgb, np.float64)
    key = (getattr(m, "name", ""), mtl.index,
           tuple(np.round(ref, 2).tolist()))
    if key in _SKIN_FLAG_CACHE:
        return _SKIN_FLAG_CACHE[key]
    flag = False
    try:
        if not any(float(v) > 0.0 for v in mtl.specular_rgb()):
            _SKIN_FLAG_CACHE[key] = False
            return False
        mp = mtl.mappers.get(0)
        t = m.textures.get(mp.texture) if (mp and mp.texture) else None
        if t and "rgba" in t:
            a = np.frombuffer(t["rgba"], np.uint8).reshape(t["h"], t["w"], 4)
            rgb = a[:, :, :3].astype(np.float32)
            mean = rgb.reshape(-1, 3).mean(axis=0)
            g = rgb.mean(axis=2)
            contrast = float(np.abs(np.diff(g, axis=1)).mean())
            flag = (float(np.abs(mean - ref).max()) <= 12.0
                    and contrast < 1.5)
    except Exception:
        flag = False
    _SKIN_FLAG_CACHE[key] = flag
    return flag


def _texture_mean(m, mtl):
    mp = mtl.mappers.get(0)
    t = m.textures.get(mp.texture) if (mp and mp.texture) else None
    if not t or "rgba" not in t:
        return None
    a = np.frombuffer(t["rgba"], np.uint8).reshape(t["h"], t["w"], 4)
    return a[:, :, :3].astype(np.float32).reshape(-1, 3).mean(axis=0)


def _material_points(m, mtl):
    pts = []
    for me in m.meshes:
        if me.material_index != mtl.index:
            continue
        for pr in me.prims:
            pts.extend(v["pos"] for v in pr.vertices.values())
    return np.asarray(pts, np.float64) if pts else np.zeros((0, 3), np.float64)


def _nearest_stats(points, ref, radius=0.5, chunk=1024):
    """Return (fraction within radius, median, p90) nearest distances."""
    if len(points) == 0 or len(ref) == 0:
        return 0.0, float("inf"), float("inf")
    ds = []
    for i in range(0, len(points), chunk):
        p = points[i:i + chunk]
        d2 = ((p[:, None, :] - ref[None, :, :]) ** 2).sum(axis=2)
        ds.append(np.sqrt(d2.min(axis=1)))
    d = np.concatenate(ds)
    return (float((d < radius).mean()), float(np.median(d)),
            float(np.percentile(d, 90)))


def _nearest_fraction(points, ref, radius=0.5, chunk=1024):
    """Return (fraction within radius, median nearest distance)."""
    frac, med, _p90 = _nearest_stats(points, ref, radius, chunk)
    return frac, med


def _skip_outfit_skin_enabled():
    return os.environ.get("LP_SKIPOUTFITSKIN", "1") != "0"


def _base_skin_ctx(models):
    """Skin reference/gain/leg points from the SAME character's base body."""
    points, means, gains = [], [], []
    for nm, m in models:
        if not re.match(r"^[mnr]_04_000_\d\d$", nm):
            continue
        for mtl in m.materials:
            if not any(float(v) > 0.0 for v in mtl.specular_rgb()):
                continue
            mean = _texture_mean(m, mtl)
            if mean is not None:
                means.append(mean)
            gains.append(np.asarray(SH.material_gain(mtl), np.float64))
        for me in m.meshes:
            for pr in me.prims:
                points.extend(v["pos"] for v in pr.vertices.values())
    ref = np.mean(means, axis=0) if means else np.asarray(SKIN_REF_RGB)
    gain = np.mean(gains, axis=0) if gains else np.asarray(SKIN_GAIN)
    return dict(ref=ref, gain=gain,
                usable=bool(np.abs(gain - 1.0).max() > 1e-6),
                points=np.asarray(points, np.float64) if points
                else np.zeros((0, 3)),
                enabled=not bool(os.environ.get("LP_NOSKINGAIN")))


def _vflip_for(mtl):
    return (False if any(k and k in (getattr(mtl, "name", "") or "").lower()
                         for k in os.environ.get("LP_NOFLIP", "").split(","))
            else None)


def _uv_sample_mean(texture, uv, vflip=True):
    """Mean/std of the texels sampled by one primitive's vertex UVs."""
    if texture is None or uv is None or not len(uv):
        return None, 0.0
    h, w = texture.shape[:2]
    u = uv[:, 0] - np.floor(uv[:, 0])
    v = uv[:, 1] - np.floor(uv[:, 1])
    if vflip:
        v = 1.0 - v
    x = np.clip((u * w).astype(int), 0, w - 1)
    y = np.clip((v * h).astype(int), 0, h - 1)
    px = texture[y, x, :3].astype(np.float32)
    return px.mean(axis=0), float(px.std(axis=0).mean())


def _object_skin_gain(texture, uv, pos, mtl, ctx):
    """Object-level skin test: a mixed atlas can hold fabric and skin regions."""
    if (not ctx["enabled"] or not ctx["usable"] or texture is None or
            mtl is None or not any(float(v) > 0 for v in mtl.specular_rgb())):
        return None
    if not np.allclose(SH.material_gain(mtl), 1.0, atol=2e-3):
        return None
    vflip = _vflip_for(mtl)
    mean, std = _uv_sample_mean(texture, uv, True if vflip is None else vflip)
    if mean is None:
        return None
    dref = float(np.abs(mean - ctx["ref"]).max())
    if dref > 14.0 or std >= 15.0:
        return None
    frac, med = _nearest_fraction(pos, ctx["points"])
    if med < 0.5 or frac > 0.3:
        return tuple(float(x) for x in ctx["gain"])
    return None


def _tint_gray_skin_texture(tex, ref):
    """Colorize a grayscale skin highlight mask to the body's skin tone."""
    if tex is None or tex.ndim != 3 or tex.shape[2] < 3:
        return tex
    rgb = tex[:, :, :3].astype(np.float32)
    if float((rgb.max(axis=2) - rgb.min(axis=2)).mean()) > 1.0:
        return tex
    gray = rgb.mean(axis=2, keepdims=True)
    gm = float(gray.mean())
    if gm <= 1.0:
        return tex
    scale = np.asarray(ref, np.float64) / gm
    out = tex.copy()
    out[:, :, :3] = np.clip(gray * scale[None, None, :], 0, 255).astype(np.uint8)
    return out




SKIN_TINT = np.array([254.0 / 255.0, 204.0 / 255.0, 174.0 / 255.0])


def _neck_lining_objects(models):
    """找出服装里"白色领口内衬"这种独立小件（游戏里被外层领口挡住）。

    用户 2026-10-18 反馈：「脖子下方有一片白」。
    m_01_005 的 mesh1（mat6 / bo 白贴图 / 64 tris）就是这样的内衬：
    游戏里被外层领口和脖子挡住，我们没做遮挡，于是露出一块纯白平片。
    判据（保守，只动明显是内衬的小件）：
      * 先用该服装自己的 `_ne`（neck）材质圈出脖子区域；
      * 非皮肤材质（specular=0）、三角形数 20~160、bbox 中心在脖子区域；
      * unit0 贴图均值 > 225（接近纯白）。
    命中的件把 base_color 乘上肤色比例（白贴图 x 肤色 ⇒ 读起来像脖子皮肤），
    而不是直接隐藏 —— 直接隐藏会在下巴/领口处露黑洞（实测）。
    `LP_NECKLINING=0` 可关闭。
    """
    found = {}
    if os.environ.get('LP_NECKLINING', '1') == '0':
        return found
    for nm, m in models:
        if not re.match(r"^[mnr]_01_\d{3}$", nm):
            continue
        neck_pts = []
        for me in m.meshes:
            mi = me.material_index
            mtl = m.materials[mi] if 0 <= mi < len(m.materials) else None
            cname = (getattr(mtl, 'name', '') or '').lower()
            if cname.endswith('_ne'):
                for pr in me.prims:
                    for v in pr.vertices.values():
                        neck_pts.append(v['pos'])
        if not neck_pts:
            continue
        arr = np.asarray(neck_pts, np.float64)
        y0, y1 = float(arr[:, 1].min()), float(arr[:, 1].max())
        xc = 0.5 * (float(arr[:, 0].min()) + float(arr[:, 0].max()))
        half = 0.5 * (float(arr[:, 0].max()) - float(arr[:, 0].min()))
        ytol = max(3.0, 0.55 * (y1 - y0))
        xtol = max(6.0, 1.2 * half)
        for me in m.meshes:
            mi = me.material_index
            if not (0 <= mi < len(m.materials)):
                continue
            mtl = m.materials[mi]
            cname = (getattr(mtl, 'name', '') or '').lower()
            if cname.endswith('_ne'):
                continue                       # 脖子皮肤本身不处理
            mp = mtl.mappers.get(0)
            t = m.textures.get(mp.texture) if (mp and mp.texture) else None
            if not t or 'rgba' not in t:
                continue
            tex = np.frombuffer(t['rgba'], np.uint8).reshape(t['h'], t['w'], 4)[:, :, :3]
            for pr in me.prims:
                if not pr.vertices or not pr.indices:
                    continue
                if not (20 <= pr.tris <= 160):
                    continue
                pos = np.asarray([v['pos'] for v in pr.vertices.values()], np.float64)
                uv = np.asarray([v['uv'] for v in pr.vertices.values()], np.float64)
                cy, cx = float(pos[:, 1].mean()), float(pos[:, 0].mean())
                if abs(cy - 0.5 * (y0 + y1)) > ytol or abs(cx - xc) > xtol:
                    continue
                # 用这个 prim 自己采样的贴图区域判断是不是"白色布料"
                fu = np.mod(uv[:, 0], 1.0)
                fv = np.mod(uv[:, 1], 1.0)
                if os.environ.get('LP_VFLIP', '1') != '0':
                    fv = 1.0 - fv          # 与渲染器的 V 翻转口径保持一致
                xi = np.clip((fu * tex.shape[1]).astype(int), 0, tex.shape[1] - 1)
                yi = np.clip((fv * tex.shape[0]).astype(int), 0, tex.shape[0] - 1)
                sampled = tex[yi, xi].astype(np.float64)
                if float(sampled.mean()) < 225.0:
                    continue
                found[(nm, me.index, pr.index)] = True
    return found

def _mark_skin_materials(models, ctx=None):
    """Mark body-covering outfit skin for skipping; keep gain on the rest."""
    disabled = bool(os.environ.get("LP_NOSKINGAIN"))
    skip_enabled = _skip_outfit_skin_enabled()
    if ctx is None:
        ctx = _base_skin_ctx(models)
    for nm, m in models:
        is_outfit = bool(re.match(r"^[mnr]_01_\d{3}$", nm))
        flags = {}
        covered = {}
        has_bo = has_wa = False
        near_legs = False
        for mtl in m.materials:
            ok = _texture_is_skin(m, mtl, ctx["ref"])
            flags[mtl.index] = ok
            name = (getattr(mtl, "name", "") or "").lower()
            if ok and "_bo" in name:
                has_bo = True
            if ok and "_wa" in name:
                has_wa = True
            if ok and len(ctx["points"]):
                frac, med, p90 = _nearest_stats(_material_points(m, mtl),
                                                ctx["points"])
                if med < 0.5 or frac > 0.3:
                    near_legs = True
                covered[mtl.index] = bool(p90 <= 1.5 and frac >= 0.5)
        bare_body = has_bo and has_wa
        for mtl in m.materials:
            own = np.asarray(SH.material_gain(mtl), np.float64)
            need = bare_body or near_legs
            same = bool(np.allclose(own, ctx["gain"], atol=2e-3))
            is_skin = bool(need and flags.get(mtl.index))
            skip = bool(is_outfit and is_skin and skip_enabled and
                        covered.get(mtl.index, False))
            mtl.skip_skin = skip
            # ★ 2026-10-18：记录"这套服装自带的裸皮件"，渲染时把它们稍微压后，
            #   避免它们和泳装/紧身衣布片共面打架（波点泳装裆部的米色多边形块）。
            mtl.is_outfit_skin = bool(is_skin)
            # Bare-skin atlases carry a grayscale shadow mask.  On skin it
            # paints a dirty band; body shading already supplies that detail.
            mtl.drop_skin_extra = bool(is_outfit and is_skin and bare_body)
            mtl.skin_gain = (tuple(float(x) for x in ctx["gain"])
                             if (is_skin and not skip and
                                 ctx["usable"] and not same and not disabled)
                             else None)


def collect_geometry(models):
    objs = []
    skin_ctx = _base_skin_ctx(models)
    _mark_skin_materials(models, skin_ctx)
    neck_lining = _neck_lining_objects(models)
    # ★ 统一皮肤的 LUT 曲线：脸/手的材质有的没有 LUT（走线性回退），
    #   于是同一个 n·L 下脸和手臂的明暗不一样 ✗（中位数能对上，观感对不上）。
    #   这里让"没有 LUT 的皮肤类材质"借用身体（m_04_*）那条曲线。
    ref_lut = None
    for nm, m in models:
        if re.match(r'^[mnr]_04_', nm):
            for mt in m.materials:
                if getattr(mt, 'lut_values', None):
                    ref_lut = mt.lut_values
                    break
        if ref_lut:
            break
    SKINISH = ('_face', '_hand', '_04_')
    if ref_lut:
        for nm, m in models:
            for mt in m.materials:
                if not getattr(mt, 'lut_values', None) and \
                        any(k in (getattr(mt, 'name', '') or '').lower() for k in SKINISH):
                    mt.lut_values = ref_lut
    for nm, m in models:
        for me in m.meshes:
            # ★ 材质索引偏移。默认 0（**不要默认加一** ✗）：
            #   LP_MATSHIFT=1 能让胸口出现白胸兜 ✓，但会把裙摆换成皮肤贴图 ✗ ——
            #   那只是"换了一张贴图碰巧盖住"，不是正确的对应关系 ✓。
            #   真正的病根在 **UV 方向**：LP_NOFLIP=m_01_005（服装不做 V 翻转）时
            #   胸口就出现白色 ✓，但脸必须保持翻转（否则眼睛错位 ✗）。
            shift = int(os.environ.get('LP_MATSHIFT', '0'))
            mi = max(0, min(len(m.materials) - 1, me.material_index + shift))
            mtl = m.materials[mi] if mi < len(m.materials) else None
            if mtl is not None and getattr(mtl, "skip_skin", False):
                continue
            mp = mtl.diffuse_mapper() if mtl else None
            tex = None
            if mp and mp.texture:
                t = m.textures.get(mp.texture)
                if t and 'rgba' in t:
                    tex = np.frombuffer(t['rgba'], np.uint8).reshape(t['h'], t['w'], 4)
            if (tex is not None and
                    re.match(r"^[mnr]_04_000_\d\d$", nm) and
                    (getattr(mtl, "name", "") or "").lower().endswith("_hi")):
                tex = _tint_gray_skin_texture(tex, skin_ctx["ref"])
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
                pos_arr = np.array([v['pos'] for v in verts], np.float64)
                uv_arr = np.array([v['uv'] for v in verts], np.float64)
                obj_gain = getattr(mtl, 'skin_gain', None)
                if obj_gain is None:
                    obj_gain = _object_skin_gain(tex, uv_arr, pos_arr,
                                                 mtl, skin_ctx)
                spec_u = _specular_unit(m, mtl) if mtl is not None else None
                # ★ 2026-10-18：被衣服盖住的皮肤层（基础身体 m_04_* / 服装自带裸皮件）
                #   在深度上稍微压后，保证布片总是赢；否则共面处会按三角形随机翻面，
                #   泳装裆部/胸口就出现硬边多边形色块（实测就是 body wa/hi 盖住了布片）。
                #   4e-5 在近景（dist≈70）只相当于 ~0.1 世界单位，不够；实测 2e-4
                #   才能压住"紧身衣包住身体"时的穿插（m_01_003 下摆的锯齿皮肤块）。
                skin_bias = float(os.environ.get('LP_SKINBIAS', '2e-4'))
                bias = 0.0
                if skin_bias > 0.0:
                    if re.match(r"^[mnr]_04_000_\d\d$", nm) or \
                            getattr(mtl, 'is_outfit_skin', False):
                        bias = -skin_bias
                objs.append(dict(
                    verts=verts,
                    pos=pos_arr,
                    nrm=np.array([v['nrm'] for v in verts], np.float64),
                    uv=uv_arr,
                    indices=[remap[i] for i in pr.indices if i in remap],
                    texture=tex,
                    wrap_u=mp.wrap_u if mp else 2, wrap_v=mp.wrap_v if mp else 2,
                    # ★ 之前写死 (1,1,1)；现在用 SMAT/MATC 的 diffuse（全库 589/607 就是纯白，
                    #   所以这一步几乎不改变画面，但为后面的 TEV/光照铺路）
                    # ★ 2026-11-14 修正：领口/胸口皮肤片（如 m_01_005 mesh1，64 tris）的贴图
                    #   本身就是肤色（UV 采样 (255,228,220)），旧管线又乘一遍 SKIN_TINT(254,204,174)
                    #   把它染成橙红 ✗。现在不再乘色，并按皮肤处理（吃肤色光照/色温）。
                    base_color=(mtl.diffuse_rgb() if mtl else (1.0, 1.0, 1.0)),
                    material=mtl,
                    # ★ 多贴图层（unit1..N）：眼睛的高光/水润、脸部腮红/眉毛都在这里 ✓
                    extra=_extra_units(m, mtl),
                    # ★ 高光强度贴图（*_spe）：乘在光照高光项上（头发那条亮带）
                    spec_tex=(spec_u[0] if spec_u else None),
                    spec_wrap_u=(spec_u[1] if spec_u else 2),
                    spec_wrap_v=(spec_u[2] if spec_u else 2),
                    # 高光层的动画相位（每帧换一个通道）✓ 由调用方按时间设置
                    phase=_PHASE[0],
                    # ★ 半透明叠加层（如脸部 m_face_col）走 alpha 混合，不要硬切 ✗
                    blend=(not getattr(mtl, 'alpha_test_enabled', False)
                           and _needs_blend(m, mtl)),
                    # ★ LP_NOFLIP=<材质前缀,...>：这些材质**不做** V 翻转。
                    #   实测：脸部需要翻转（否则眼睛错位 ✗），而这套衣服的胸口需要不翻转 ✓
                    vflip=_vflip_for(mtl),
                    # ★ 材质 ID（供验收按材质生成掩膜，见 tools/accept_check.py）
                    mat_tag=len(objs) + 1,
                    mat_name='%s/%s' % (nm, getattr(mtl, 'name', '?') if mtl else '?'),
                    mesh_index=me.index,
                    alpha_func=getattr(mtl, 'alpha_test_func', None),
                    alpha_ref=(getattr(mtl, 'alpha_test_ref', 0)
                               if getattr(mtl, 'alpha_test_enabled', False) else None),
                    # ★ 服装自带的裸皮部件（m_01_002 的 ar/ne/wa/bo）走身体皮肤校正，
                    #   否则会和 m_04_ 的皮肤差 10% 左右，像没贴图的白平面 ✓
                    # ★ 2026-11-14：贴图肤色判定（服装自带的裸皮件 ar/ne/wa 也包含在内），
                    #   供渲染器的"色温仅皮肤"规则使用；布料（红裙/蓝裙/白围裙）为 False。
                    #   领口/胸口皮肤片（neck_lining）强制按皮肤处理。
                    skin_like=(((nm, me.index, pr.index) in neck_lining)
                               or (_texture_is_skin(m, mtl, skin_ctx["ref"])
                                   if mtl is not None else False)
                               or any(k in (getattr(mtl, "name", "") or "").lower()
                                      for k in ("_04_", "face", "hand"))
                               # ★ 2026-11-14：皮肤的 specular 签名 (89,114,163)——
                               #   有些服装自带的脖子皮肤片贴图取自衣服图集，纹理均值不像皮肤，
                               #   但材质高光仍是皮肤色 ⇒ 按皮肤处理（用户反馈：脖子那块是皮肤，太白了）
                               or (mtl is not None and all(
                                   abs(a - b) <= 0.06 for a, b in zip(
                                       [float(x) for x in mtl.specular_rgb()],
                                       [89 / 255.0, 114 / 255.0, 163 / 255.0])
                                   ) and any(float(x) > 0.0 for x in mtl.specular_rgb()))),
                    obj_gain=obj_gain,
                    depth_bias=bias))
    return objs


def load_layers(pre, img=IMG, bones_bind=None):
    """Load the body motion plus its paired layer(s).

    ★ 配对规则（2026-10-11 确证）：PACK 容器本身就是"动作集" ——
      同一个包里出现的 `m_<动作>_<变体>_skirt` 就是该身体动作的裙摆层。
      img.bin 里有 167 个这样的包；276 个 `_skirt` 文件全部能找到同名身体动作。
      `m_upperNNN / m_lowerNNN / m_skirt_NNN / m_handNNN` 是**按服装**的变体集，
      各自独立成包，选哪一套由服装决定（选择表尚未定位），所以这里先只叠加
      同包的 `_skirt` 层，并用 LP_NOSKIRT=1 关闭以做对照。
    """
    spec = LAYERS.get(pre) or LAYERS.get('default')
    if not spec:
        return []
    want_skirt = _os_mot.environ.get('LP_NOSKIRT', '') == ''
    got, mots = set(), []
    with open(img, 'rb') as f:
        for p in N.iter_packs(img):
            names = {pf.name: pf for pf in p.files}
            for name, base, per in spec:
                if name not in names or name in got:
                    continue
                got.add(name)
                mots.append(MOT.load_mot_final(N.pack_bytes(f, names[name]), name,
                                               base=base, per=per,
                                               bones_bind=bones_bind))
                # 同包里的 `_skirt` 伙伴（可用 LP_SKIRTLAYER 强制指定某个 m_skirt_NNN 变体）
                force = _os_mot.environ.get('LP_SKIRTLAYER')
                sk = force or (name[:-4] + '_skirt.mot')
                if want_skirt and sk in names and sk not in got:
                    got.add(sk)
                    mots.append(MOT.load_mot_final(N.pack_bytes(f, names[sk]), sk,
                                                   bones_bind=bones_bind))
            if len(got) >= len(spec):
                break
    return mots


SKIRT_BONES = set(range(6, 12)) | set(range(236, 254))


def _is_skirt_layer(mm):
    recs = [r for r in mm.records if r.count]
    return bool(recs) and all(r.bone in SKIRT_BONES for r in recs)


_FACE_MOT_CACHE = {}
_FACE_IMG = None


def pose_at(mots, bind_t, t, blink_w=0.0):
    # ★ 裙摆层（同包 _skirt）存的是**单个关键帧**，也就是一个静止的裙摆形状
    #   （坐姿：面板外翻 45~55°，用来盖住盘起来的腿）。但 `m_08400_32` 这类动作是
    #   "坐→站"的过渡：站着的那几帧若继续套坐姿裙摆，就会张成一个钟形。
    #   所以按**骨盆高度**给它权重：坐着 ~1，站着 ~0。
    skirt = [mm for mm in mots if _is_skirt_layer(mm)]
    body = [mm for mm in mots if mm not in skirt]
    w = 1.0
    if skirt and _os_mot.environ.get('LP_SKIRTW', 'height') != 'off' and body:
        py = None
        for mm in body:
            ch = MOT.sample_present(mm, t).get(2, {}).get('T', {})
            if 1 in ch:
                py = ch[1]
                break
        if py is not None:
            y0 = bind_t.get(2, [0, 80.0, 0])[1]
            w = (y0 - py) / max(1e-6, (y0 - 12.0))
            w = 0.0 if w < 0.0 else (1.0 if w > 1.0 else w)
    flat = []
    for mm in mots:
        pose = MOT.sample_present(mm, t)
        scale = w if mm in skirt else 1.0
        d = {}
        for bone, ch in pose.items():
            dd = {}
            for ax, v in ch.get('T', {}).items():
                dd.setdefault('T', {})[ax] = (v - bind_t.get(bone, [0, 0, 0])[ax]
                                              if mm.absolute_translation else v) * scale
            for ax, v in ch.get('R', {}).items():
                dd.setdefault('R', {})[ax] = v * scale
            if dd:
                d[bone] = dd
        flat.append(d)
    stacked = MOT.stack_poses(flat)
    # ★ LP_ROOTLOCK=1：锁住骨盆位移（动作自带的前进/后退根位移），人物原地
    if _os_mot.environ.get('LP_ROOTLOCK'):
        ch = stacked.setdefault(2, {})
        ch['T'] = {0: 0.0, 1: 0.0, 2: 0.0}
    if _os_mot.environ.get('LP_NOROOT'):
        stacked.pop(2, None)          # 对照：忽略骨盆(骨骼2)的旋转
    # ★ 2026-11-14：真·眨眼（照搬 demo/server.py 的实现）
    #   脸部动作 = 单帧姿势目标（nk==1、时间 0、平移是绝对绑定值）
    #   用 base=0, per=1 读取，t=0 采样，转成相对绑定增量，再按权重叠加。
    _blink_w = float(blink_w or 0.0)
    if _blink_w > 0.0:
        _fm = _os_mot.environ.get('LP_BLINKPOSE', 'm__eye__m_eye_02')
        _key = _fm
        if _FACE_MOT_CACHE.get('key') != _key:
            _mot = None
            try:
                with open(_os_mot.environ.get('LP_IMG', _FACE_IMG or IMG), 'rb') as _f:
                    for _p in N.iter_packs(_os_mot.environ.get('LP_IMG', _FACE_IMG or IMG)):
                        for _pf in _p.files:
                            if _pf.name == _fm + '.mot':
                                _mot = MOT.load_mot_final(
                                    N.pack_bytes(_f, _pf), _fm,
                                    base=0, per=1)
                                break
            except Exception as _e:
                print('[BLINK] load failed:', _e)
            _cached = {}
            if _mot is not None:
                _raw = MOT.sample_present(_mot, 0)
                for _b, _chn in _raw.items():
                    _d = [0.0] * 9
                    _d[6] = _d[7] = _d[8] = 1.0
                    for _ax, _v in _chn.get('T', {}).items():
                        _d[_ax] = _v - bind_t.get(_b, [0.0, 0.0, 0.0])[_ax]
                    for _ax, _v in _chn.get('R', {}).items():
                        _d[3 + _ax] = _v
                    _cached[_b] = _d
            _FACE_MOT_CACHE['key'] = _key
            _FACE_MOT_CACHE['mot'] = _cached
        _face = _FACE_MOT_CACHE.get('mot') or {}
        _w = max(0.0, min(1.0, _blink_w))
        for _b, _d in _face.items():
            _slot = stacked.setdefault(_b, {})
            _rr = _slot.setdefault('R', {})
            for _ax in range(3):
                if _d[3 + _ax]:
                    _rr[_ax] = _rr.get(_ax, 0.0) + _d[3 + _ax] * _w
            _tt = _slot.setdefault('T', {})
            for _ax in range(3):
                if _d[_ax]:
                    _tt[_ax] = _tt.get(_ax, 0.0) + _d[_ax] * _w
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
    ap.add_argument('--yaw', type=float, default=0.0,
                    help='相机绕角色 Y 轴旋转角度：0=正面, 180=背面, 90=左侧')
    ap.add_argument('--aim', type=float, default=0.0,
                    help='相机瞄准点的垂直偏移（占身高比例）：0=包围盒中心, 0.35≈头部, -0.25≈腿')
    ap.add_argument('--out', default='out/compose.png')
    ap.add_argument('--bg', default=None, help='background asset name (bs_*.jpg)')
    ap.add_argument('--img', default=IMG)
    a = ap.parse_args()
    global _FACE_IMG
    _FACE_IMG = a.img

    pre = a.char
    hair = a.hair or {'m': 'm_00_200', 'r': 'r_00_000', 'n': 'n_00_000'}[pre]
    outfit = a.outfit or {'m': 'm_01_012', 'r': 'r_01_000', 'n': 'n_01_000'}[pre]
    parts = character_parts(pre, hair, outfit,
                            legs=int(os.environ.get('LP_LEGS', '7')),
                            hand=os.environ.get('LP_HAND', '1') != '0')
    extra = [x for x in os.environ.get('LP_EXTRA', '').split(',') if x]
    if extra:
        parts += extra
        print('LP_EXTRA -> 追加部件: %s' % ', '.join(extra))

    models, bones = grab_models(parts, a.img)
    if bones is None:
        print('!! nothing loaded')
        return 1
    # .mot 的骨骼表存的是骨骼 ID -> 扁平下标（见 nlp_model.load_bone / nlp_mot._bone_of）
    MOT.ID_TO_BONE = {b.id: b.index for b in bones}
    MOT.BIND_ROT = {b.index: tuple(b.rotation) for b in bones}
    objs = collect_geometry(models)
    only = os.environ.get('LP_ONLYMAT')
    if only:
        objs = [o for o in objs if only.lower() in (o.get('mat_name') or '').lower()]
        print('LP_ONLYMAT=%s -> 只画 %d 个对象' % (only, len(objs)))
    last = os.environ.get('LP_LASTMAT')
    if last:
        objs = ([o for o in objs if last.lower() not in (o.get('mat_name') or '').lower()] +
                [o for o in objs if last.lower() in (o.get('mat_name') or '').lower()])
        print('LP_LASTMAT=%s -> 移到最后绘制' % last)
    om = os.environ.get('LP_ONLYMESH')
    if om:
        # 格式：模型名:网格号[,网格号]，例如 m_01_005:2  或  m_01_005:0,m_01_005:6
        want = []
        for spec in om.split(','):
            spec = spec.strip()
            if ':' in spec:
                mdl, idx = spec.rsplit(':', 1)
                want.append((mdl.strip().lower(), int(idx)))
        if want:
            objs = [o for o in objs
                    if any(o.get('mat_name', '').lower().startswith(m + '/') and
                           o.get('mesh_index') == i for m, i in want)]
            print('LP_ONLYMESH=%s -> %d 个对象, 材质: %s'
                  % (om, len(objs), sorted({o.get('mat_name', '?') for o in objs})))
    sk = os.environ.get('LP_SKIPMESH')
    if sk:
        want = []
        for spec in sk.split(','):
            spec = spec.strip()
            if ':' in spec:
                mdl, idx = spec.rsplit(':', 1)
                want.append((mdl.strip().lower(), int(idx)))
        objs = [o for o in objs
                if not any(o.get('mat_name', '').lower().startswith(m + '/') and
                           o.get('mesh_index') == i for m, i in want)]
        print('LP_SKIPMESH=%s -> 剩 %d 个对象' % (sk, len(objs)))
    print('parts: %s' % ', '.join(nm for nm, _ in models))
    print('primitives: %d   bones: %d' % (len(objs), len(bones)))

    mots = load_layers(pre, a.img)
    print('motion layers: %d' % len(mots))
    bind_t = {b.index: list(b.translation) for b in bones}
    wb = R.world_matrices(bones, {})
    bind_inv = [np.linalg.inv(m) if m is not None else None for m in wb]

    times = [float(x) for x in a.frames.split(',')] if a.frames else [a.frame]
    # ★ 2026-11-14 宣传片：超采样抗锯齿 + 镜头运动 + 角色横向偏移
    #   LP_SS=2/3      ：按 2x/3x 分辨率渲染后 LANCZOS 缩回目标尺寸（消除锯齿）
    #   LP_CAM0/1      ："yaw,zoom,aim" 起止值，按帧序列线性插值（推近/横移）
    #   LP_OFFX        ：角色横向偏移（占画面宽度比例，+右 −左）
    _ss = max(1, int(os.environ.get('LP_SS', '1')))
    _ssw, _ssh = a.w * _ss, a.h * _ss
    _cam0 = os.environ.get('LP_CAM0')
    _cam1 = os.environ.get('LP_CAM1')
    _offx = float(os.environ.get('LP_OFFX', '0.0'))
    _tmin, _tmax = (min(times), max(times)) if len(times) > 1 else (0.0, 1.0)

    def _lerp_cam(tt):
        k = (tt - _tmin) / (_tmax - _tmin) if _tmax > _tmin else 0.0
        if _cam0 and _cam1:
            v0 = [float(x) for x in _cam0.split(',')]
            v1 = [float(x) for x in _cam1.split(',')]
            return tuple(v0[i] + (v1[i] - v0[i]) * k for i in range(3))
        return (a.yaw, a.zoom, a.aim)

    tiles = []
    _eye_phase_auto = os.environ.get('LP_EYEPHASE', '') == 'auto'
    _blink_sched = {}
    for _part in (os.environ.get('LP_BLINKF', '') or '').split(','):
        _part = _part.strip()
        if ':' in _part:
            _f0, _w0 = _part.split(':')
            _blink_sched[int(_f0)] = float(_w0)
        elif _part:
            _blink_sched[int(_part)] = 1.0
    _blink_frames = set()
    for _fi, t in enumerate(times):
        _blink = _fi in _blink_frames or _fi in _blink_sched
        _blink_w_now = _blink_sched.get(_fi, 0.0)
        _blink_hide = (_blink_w_now > float(os.environ.get('LP_BLINKHIDE_W', '0.2')))
        pose = pose_at(mots, bind_t, t, _blink_w_now) if mots else {}
        wp = R.world_matrices(bones, pose)
        world = [(wp[i] @ bind_inv[i]) if (wp[i] is not None and bind_inv[i] is not None)
                 else wp[i] for i in range(len(wp))]
        fobj = []
        _phase_int = int(os.environ.get('LP_EYEPHASE_INT', '30'))
        _phase_now = (_fi // max(1, _phase_int)) % 2 if _eye_phase_auto else None
        for o in objs:
            pos, nrm = R.skin_positions(o['verts'], bones, world)
            g = dict(o)
            g['pos'] = pos
            g['nrm'] = nrm
            if _phase_now is not None:
                g['phase'] = _phase_now
            # 闭眼时隐藏眼球网格（否则眼皮合不上，看起来像黑线）
            if _blink_hide and ('_eye' in (o.get('mat_name') or '').lower()):
                continue
            if False and ('_eye' in (o.get('mat_name') or '').lower()):
                _p = g['pos'].copy()
                _cy = _p[:, 1].mean()
                _p[:, 1] = _cy + (_p[:, 1] - _cy) * 0.12
                g['pos'] = _p
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
            else:
                # 背景缩放/平移（宣传片构图用）：LP_BGSCALE + LP_BGOFFX/Y
                _bs = float(os.environ.get('LP_BGSCALE', '1.0'))
                if abs(_bs - 1.0) > 1e-3:
                    _w0, _h0 = _bimg.size
                    _nw, _nh = max(1, int(_w0 * _bs)), max(1, int(_h0 * _bs))
                    _im = _bimg.resize((_nw, _nh), Image.LANCZOS)
                    _ox = int((_nw - _w0) * float(os.environ.get('LP_BGOFFX', '0.5')))
                    _oy = int((_nh - _h0) * float(os.environ.get('LP_BGOFFY', '0.5')))
                    _ox = max(0, min(_nw - _w0, _ox)); _oy = max(0, min(_nh - _h0, _oy))
                    _bimg = _im.crop((_ox, _oy, _ox + _w0, _oy + _h0))
        fb = R.Framebuffer(_ssw, _ssh, bg=(0.12, 0.13, 0.16), background=_bimg)
        _yaw, _zoom, _aim = _lerp_cam(t)
        rad = np.deg2rad(_yaw)
        off = np.array([np.sin(rad), 0.0, np.cos(rad)]) * (_zoom * hgt)
        tgt = ctr + np.array([0.0, _aim * hgt, 0.0])
        eye = tgt + off
        if _offx:
            fwd = eye - tgt
            fwd = fwd / max(float(np.linalg.norm(fwd)), 1e-9)
            right = np.cross(np.array([0.0, 1.0, 0.0]), fwd)
            right = right / max(float(np.linalg.norm(right)), 1e-9)
            d = right * (_offx * hgt)
            eye = eye + d
            tgt = tgt + d
        R.render_scene(fb, fobj, eye, tgt, up=(0, 1, 0), fovy=30.0)
        # ★ 2026-11-14：头部投在衣服上的软阴影（领口发白的根因：衣服没有头影）
        if os.environ.get('LP_HEADSHD', '1') not in ('0', '', 'false', 'off'):
            try:
                _face_objs = [g for g in fobj if 'm_face/' in (g.get('mat_name') or '')]
                if _face_objs:
                    _fp = np.vstack([g['pos'] for g in _face_objs])
                    _chin = np.array([_fp[:, 0].mean(), _fp[:, 1].min() + 0.8, _fp[:, 2].mean(), 1.0])
                    _mv = R.look_at(eye, tgt, (0, 1, 0))
                    _pr = R.perspective(30.0, fb.w / float(fb.h), 1.0, 10000.0)

                    def _pj(p):
                        c = _mv.dot(p); nn = _pr.dot(c)
                        if abs(nn[3]) < 1e-9:
                            return None
                        return ((nn[0] / nn[3] * 0.5 + 0.5) * fb.w,
                                (1.0 - (nn[1] / nn[3] * 0.5 + 0.5)) * fb.h)

                    _p0 = _pj(_chin)
                    _p1 = _pj(_chin + np.array([0.35 * hgt, 0.0, 0.0, 0.0]))
                    if _p0 and _p1:
                        _cx, _cy = _p0
                        _rx = max(4.0, abs(_p1[0] - _cx))
                        _ry = _rx * 0.85
                        _yy, _xx = np.mgrid[0:fb.h, 0:fb.w]
                        _d = (((_xx - _cx) / _rx) ** 2 + ((_yy - _cy) / _ry) ** 2)
                        _shd = np.clip(1.0 - _d, 0.0, 1.0) ** 1.2 * 0.30
                        # 只作用在非皮肤/非头发的片（衣服领口），皮肤有自带明暗
                        _cl = np.ones_like(fb.mat_id, np.float32)
                        _skip = set()
                        for _g in fobj:
                            _mn = (_g.get('mat_name') or '').split('/')[-1].lower()
                            if ('face' in _mn) or ('ha1' in _mn) or ('ha2' in _mn) or ('eye' in _mn):
                                _skip.add(_g.get('mat_tag', 0))
                        _mask = ~np.isin(fb.mat_id, list(_skip))
                        fb.color *= (1.0 - _shd[:, :, None] * _mask[:, :, None]).astype(np.float32)
            except Exception:
                pass
        # ★ 2026-11-14：接地软阴影（LP_SHADOW=1）——投影脚底中心，画柔和椭圆
        if os.environ.get('LP_SHADOW', '0') not in ('0', '', 'false', 'off'):
            try:
                _mv = R.look_at(eye, tgt, (0, 1, 0))
                _pr = R.perspective(30.0, fb.w / float(fb.h), 1.0, 10000.0)
                _foot = np.array([ctr[0], lo[1], ctr[2], 1.0])
                _side = np.array([ctr[0] + 0.34 * hgt, lo[1], ctr[2], 1.0])

                def _proj(p):
                    c = _mv.dot(p)
                    n = _pr.dot(c)
                    if abs(n[3]) < 1e-9:
                        return None
                    return ((n[0] / n[3] * 0.5 + 0.5) * fb.w,
                            (1.0 - (n[1] / n[3] * 0.5 + 0.5)) * fb.h)

                p0 = _proj(_foot)
                p1 = _proj(_side)
                if p0 and p1:
                    cx, cy = p0
                    rx = max(2.0, abs(p1[0] - cx))
                    ry = max(1.5, rx * 0.32)
                    yy, xx = np.mgrid[0:fb.h, 0:fb.w]
                    d = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
                    m = np.clip(1.0 - d, 0.0, 1.0) ** 1.5 * 0.42
                    fb.color *= (1.0 - m[:, :, None]).astype(np.float32)
            except Exception:
                pass
        # ★ 三渲二描边（可选）：LP_OUTLINE=1 时按深度缓冲加描边
        if os.environ.get('LP_OUTLINE', '0') not in ('0', '', 'false', 'off'):
            R.add_outline(fb,
                          width=int(os.environ.get('LP_OUTLINE_W', '2')),
                          color=tuple(float(x) for x in
                                      os.environ.get('LP_OUTLINE_COLOR', '0.10,0.075,0.07').split(',')),
                          strength=float(os.environ.get('LP_OUTLINE_STRENGTH', '0.85')))
        p = a.out.replace('.png', '_f%03d.png' % _fi)
        _exp = float(os.environ.get('LP_EXPOSURE', '1.0'))
        if _exp != 1.0:
            fb.color = np.clip(fb.color * _exp, 0.0, 1.0)
        _img = (np.clip(fb.color, 0, 1) * 255).astype(np.uint8)
        if _ss > 1:
            _img = np.asarray(Image.fromarray(_img).resize((a.w, a.h), Image.LANCZOS))
        Image.fromarray(_img, 'RGB').save(p)
        tiles.append((t, p))
        # ★ 落一份材质 ID 缓冲 + 名字表（验收按材质取掩膜用，不依赖颜色阈值）
        if os.environ.get('LP_MATID'):
            _mid = fb.mat_id
            if _ss > 1:
                _mid = np.asarray(Image.fromarray(_mid.astype(np.uint8)).resize(
                    (a.w, a.h), Image.NEAREST)).astype(np.int32)
            np.save(p.replace('.png', '_matid.npy'), _mid)
            with open(p.replace('.png', '_matnames.json'), 'w', encoding='utf-8') as fh:
                json.dump({str(o.get('mat_tag', 0)): o.get('mat_name', '?') for o in fobj}, fh)
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
