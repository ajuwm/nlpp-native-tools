# -*- coding: utf-8 -*-
"""
Native-format renderer for New Love Plus models (software rasteriser, no GPU deps).

It consumes ONLY the game's own data: SMES vertices, SMAT materials (including the
native wrap modes), BONE skeleton and the PICA200 tiled TEXI textures.
"""
import math
import os
import numpy as np


# ------------------------------------------------------------------ math helpers
def look_at(eye, target, up):
    eye = np.asarray(eye, np.float64)
    target = np.asarray(target, np.float64)
    up = np.asarray(up, np.float64)
    f = target - eye
    f /= np.linalg.norm(f)
    s = np.cross(f, up)
    n = np.linalg.norm(s)
    if n < 1e-9:
        s = np.array([1.0, 0.0, 0.0])
    else:
        s /= n
    u = np.cross(s, f)
    m = np.eye(4)
    m[0, :3] = s
    m[1, :3] = u
    m[2, :3] = -f
    m[0, 3] = -s.dot(eye)
    m[1, 3] = -u.dot(eye)
    m[2, 3] = f.dot(eye)
    return m


def perspective(fovy_deg, aspect, znear, zfar):
    f = 1.0 / math.tan(math.radians(fovy_deg) / 2.0)
    m = np.zeros((4, 4))
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (zfar + znear) / (znear - zfar)
    m[2, 3] = (2 * zfar * znear) / (znear - zfar)
    m[3, 2] = -1.0
    return m


def translate(x, y, z):
    m = np.eye(4)
    m[:3, 3] = (x, y, z)
    return m


def rotate_axis(axis, deg):
    a = np.asarray(axis, np.float64)
    a = a / np.linalg.norm(a)
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    x, y, z = a
    m = np.eye(4)
    m[:3, :3] = np.array([
        [c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
        [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
        [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)],
    ])
    return m


def euler_xyz(rx, ry, rz):
    """三角度 -> 旋转矩阵。

    ★ 组合顺序可用 LP_EULER 切换（默认 XYZ = Rx·Ry·Rz）。
      小角度下手骨/手臂几乎分不出顺序，但**大角度**（骨盆转到 90° 以上）差别巨大：
      用 XYZ 时骨盆的最大转动会让角色**整个翻过去**，而 galgame 里角色不会翻跟头。
      用"头相对骨盆的高度"做判据扫六种顺序，ZYX 在多个动作上一致最优。
    """
    order = _os.environ.get('LP_EULER', 'ZYX').upper()
    mats = {'X': rotate_axis((1, 0, 0), rx),
            'Y': rotate_axis((0, 1, 0), ry),
            'Z': rotate_axis((0, 0, 1), rz)}
    out = np.eye(4)
    for ch in order:
        if ch in mats:
            out = out @ mats[ch]
    return out


# ------------------------------------------------------------------ skeleton
def world_matrices(bones, pose=None):
    """Local matrices from the bind pose (+ optional pose overrides) -> world matrices.

    pose: {bone_index: (dx,dy,dz, rx,ry,rz, sx,sy,sz)} deltas in the bone's own space.
    """
    count = len(bones)
    local = [None] * count
    for b in bones:
        t = list(b.translation)
        r = list(b.rotation)
        s = list(b.scale)
        if pose and b.index in pose:
            d = pose[b.index]
            t = [t[i] + d[i] for i in range(3)]
            r = [r[i] + d[3 + i] for i in range(3)]
            s = [s[i] * d[6 + i] for i in range(3)]
        m = translate(*t) @ euler_xyz(r[0], r[1], r[2])
        m[0, 0] *= s[0]; m[1, 1] *= s[1]; m[2, 2] *= s[2]
        local[b.index] = m

    world = [None] * count
    # resolve parents first (bones are stored with parents generally before children)
    for b in bones:
        p = b.parent
        if p < 0 or p >= count or world[p] is None:
            world[b.index] = local[b.index]
        else:
            world[b.index] = world[p] @ local[b.index]
    # second pass for any that were resolved out of order
    for _ in range(4):
        changed = False
        for b in bones:
            p = b.parent
            if 0 <= p < count and world[p] is not None:
                m = world[p] @ local[b.index]
                if not np.allclose(m, world[b.index]):
                    world[b.index] = m
                    changed = True
        if not changed:
            break
    return world


def skin_positions(verts, bones, world, use_skin=True):
    """Skin vertex positions/normals with the standard linear blend formula."""
    n = len(verts)
    out_pos = np.zeros((n, 3), np.float64)
    out_nrm = np.zeros((n, 3), np.float64)
    if not use_skin:
        for i, v in enumerate(verts):
            out_pos[i] = v['pos']
            out_nrm[i] = v['nrm']
        return out_pos, out_nrm
    for i, v in enumerate(verts):
        idx = v['bones']
        w = v['weights']
        if not idx:
            out_pos[i] = v['pos']
            out_nrm[i] = v['nrm']
            continue
        M = np.zeros((4, 4))
        for k, bi in enumerate(idx):
            wk = w[k] if k < len(w) else 0.0
            if wk == 0.0 or bi >= len(world) or world[bi] is None:
                continue
            M += wk * world[bi]
        p = M @ np.array([v['pos'][0], v['pos'][1], v['pos'][2], 1.0])
        out_pos[i] = p[:3]
        out_nrm[i] = M[:3, :3].dot(np.array(v['nrm']))
    ln = np.linalg.norm(out_nrm, axis=1)
    ln[ln == 0] = 1.0
    out_nrm /= ln[:, None]
    return out_pos, out_nrm


# ------------------------------------------------------------------ rasteriser
class Framebuffer(object):
    def __init__(self, w, h, bg=(0.08, 0.08, 0.10), background=None):
        """background: optional (H,W,3) float image in 0..1, stretched to fill.

        The game's backgrounds are plain JPEGs (bs_<scene>_<variant>_<index>.jpg,
        512x672 portrait), so a background is just the framebuffer's initial colour --
        no extra format to decode.
        """
        self.w = w
        self.h = h
        self.color = np.zeros((h, w, 3), np.float32)
        if background is not None:
            img = np.asarray(background, np.float32)
            if img.max() > 1.5:
                img = img / 255.0
            ih, iw = img.shape[0], img.shape[1]
            yi = (np.arange(h) * (ih / float(h))).astype(np.int32).clip(0, ih - 1)
            xi = (np.arange(w) * (iw / float(w))).astype(np.int32).clip(0, iw - 1)
            self.color[:, :] = img[yi][:, xi]
        else:
            self.color[:, :] = bg
        self.depth = np.full((h, w), 1e30, np.float64)
        # ★ 材质 ID 缓冲：每个像素记录"来自哪个材质 tag"。
        #   用途：验收时按材质生成掩膜，彻底不用颜色阈值（颜色阈值会自选择/跨类污染 ✗）。
        self.mat_id = np.zeros((h, w), np.int32)


# PICA200 OTestFunction: 0 never 1 always 2 equal 3 notequal 4 less 5 lessequal 6 greater 7 greaterequal
def _alpha_pass(a, func, ref, fallback_threshold):
    """Apply the material's STAT alpha test; fall back to a fixed threshold."""
    if ref is None:
        return a >= fallback_threshold * 255
    if func is None:
        func = 7
    if func == 0:
        return False
    if func == 1:
        return True
    if func == 2:
        return a == ref
    if func == 3:
        return a != ref
    if func == 4:
        return a < ref
    if func == 5:
        return a <= ref
    if func == 6:
        return a > ref
    return a >= ref


import os as _os
# The 3DS UV convention has V growing upward while the decoded texture rows grow
# downward (row 0 = first row in memory).  Sampling must therefore flip V; verified by
# the sailor collars on Rinko/Nene appearing only with the flip on.
VFLIP = _os.environ.get('LP_VFLIP', '1') != '0'

# 布光参数（与实机截图对齐用；见 docs/光照与材质.md）
#   LP_AMBIENT  环境项下限，默认 0.72（实机是"高环境"的柔和布光）
#   LP_DIFFUSE  方向光增益，默认 0.34（实机方向感很弱，靠 0.72 + 0.34·|n·L| 落在 0.72~1.06）
def _num(name, default):
    """容错取数：LP_AMBIENT 在新路径里是三元组，旧路径按标量解析时要能退回默认值。"""
    v = _os.environ.get(name)
    try:
        return float(v) if v else default
    except (TypeError, ValueError):
        return default


AMBIENT = _num('LP_AMBIENT', 0.72)
DIFFUSE_GAIN = _num('LP_DIFFUSE', 0.34)
# 着色路径：pica = 逐顶点光照 + 每材质 LUT（默认）；legacy = 旧的 ambient + k·|n·L|
SHADE_MODE = _os.environ.get('LP_SHADE', 'pica').lower()
_LIGHTS = None
_OBJECT_GAIN = None
if SHADE_MODE == 'pica':
    try:
        import nlp_shade as _nlp_shade
        _LIGHTS = _nlp_shade.LightSet()
        _OBJECT_GAIN = _nlp_shade.object_gain
    except Exception:
        _LIGHTS = None
        _OBJECT_GAIN = None
LIGHTS = _LIGHTS

# ★ 2026-11-14：PICA200 硬件语义光照（真值接回）用的场景对象（懒加载一次）
_PICA_SCENE = None


def _pica_scene():
    """加载场景光照真值（PICA 寄存器值）；LP_PICALIGHT 未开启时返回 None。"""
    global _PICA_SCENE
    if _os.environ.get('LP_PICALIGHT', '0') in ('0', '', 'false', 'off'):
        return None
    if _PICA_SCENE is None:
        try:
            import pica_light as _PL
            _PICA_SCENE = _PL.SceneLighting()
            print('[PICA] scene: ambient=%s lights=%d opts=%s' % (
                _PICA_SCENE.global_ambient.round(3).tolist(),
                len(_PICA_SCENE.lights), _PICA_SCENE.opts))
        except Exception as _e:
            print('[PICA] scene load failed:', _e)
            _PICA_SCENE = False
    return _PICA_SCENE or None


def _alpha_pass_batch(a, func, ref, fallback_threshold):
    """Vectorised _alpha_pass for an array of alpha values."""
    a = np.asarray(a)
    if ref is None:
        return a >= fallback_threshold * 255
    if func is None:
        func = 7
    if func == 0:
        return np.zeros(a.shape, bool)
    if func == 1:
        return np.ones(a.shape, bool)
    if func == 2:
        return a == ref
    if func == 3:
        return a != ref
    if func == 4:
        return a < ref
    if func == 5:
        return a <= ref
    if func == 6:
        return a > ref
    return a >= ref


def _smoothstep01(edge0, edge1, x):
    """Hermite smoothstep; soft band edges avoid the old hard-threshold look."""
    t = np.clip((x - edge0) / max(edge1 - edge0, 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _toon_ramp(x, bands=3, softness=0.08):
    """Map a normalized light term to 2..4 flat tone steps with smooth joins.

    The returned value is a *relative* tone multiplier. It is deliberately
    generated from |N.L|, not from the already colour-gained shade value:
    the latter is compressed near 1.0 and cannot produce useful bands.
    """
    levels_by_bands = {
        2: (0.92, 1.03),
        3: (0.90, 0.98, 1.04),
        4: (0.88, 0.95, 1.01, 1.05),
    }
    bands = min(4, max(2, int(bands)))
    levels = levels_by_bands[bands]
    edges = np.linspace(0.0, 1.0, bands + 1)
    out = np.full_like(np.asarray(x, np.float64), levels[0])
    for i in range(bands - 1):
        e = edges[i + 1]
        s = _smoothstep01(e - softness, e + softness, x)
        out += (levels[i + 1] - levels[i]) * s
    return out


def _apply_toon_shade(shade, nrm, light_dir, mv, strength, bands,
                      softness=0.035):
    """Apply a normalized soft-step ramp around each mesh's dominant tone.

    Centering the ramp by its median keeps the material's established median
    colour (the numeric regression target) while the shadow and lit plateaus
    become visibly separated. This is the key difference from the old
    LP_TOON quantization, which multiplied shade by q/lum and pushed almost
    every pixel into the brightest bucket.
    """
    lv = np.asarray(mv, np.float64)[:3, :3].dot(np.asarray(light_dir, np.float64))
    ln = np.linalg.norm(lv)
    if ln <= 1e-9:
        return shade
    lv /= ln
    ndl = np.clip(np.abs(np.asarray(nrm, np.float64).dot(lv)), 0.0, 1.0)
    ramp = _toon_ramp(ndl, bands=bands, softness=softness)
    center = float(np.median(ramp))
    if center <= 1e-9:
        return shade
    ratio = 1.0 + float(strength) * (ramp / center - 1.0)
    # Replace the continuous base gradient with a flat per-mesh tint. Merely
    # multiplying the old smooth shade by a step still looks like a gradient
    # inside every step; using the median tint keeps the material's colour and
    # median brightness while making the ramp plateaus genuinely flat.
    tint = np.median(np.asarray(shade, np.float64), axis=0)
    return tint[None, :] * ratio[:, None]


def _is_skin_material(material):
    name = (getattr(material, 'name', '') or '').lower()
    return any(k in name for k in ('_04_', 'face', 'hand', '_ne', '_ar'))


def _wrapf_batch(x, mode):
    if mode == 0:                    # clampToEdge
        return np.clip(x, 0.0, 1.0)
    if mode == 1:                    # clampToBorder (out of range -> NaN -> black)
        out = np.array(x, np.float64)
        out[(x < 0.0) | (x > 1.0)] = np.nan
        return out
    if mode == 2:                    # repeat
        return x - np.floor(x)
    y = np.abs(x) % 2.0              # mirroredRepeat
    return np.where(y <= 1.0, y, 2.0 - y)


def _sample_batch(texture, u, v, wrap_u, wrap_v, vflip=None):
    """Vectorised _sample: u, v are arrays -> (N, 4) float32."""
    n = u.shape[0]
    if VFLIP if vflip is None else vflip:
        v = 1.0 - v
    h, w = texture.shape[0], texture.shape[1]
    fu = _wrapf_batch(u, wrap_u)
    fv = _wrapf_batch(v, wrap_v)
    bad = ~(np.isfinite(fu) & np.isfinite(fv))
    fu = np.where(bad, 0.0, fu)
    fv = np.where(bad, 0.0, fv)
    x = fu * w - 0.5
    y = fv * h - 0.5
    x0 = np.floor(x)
    y0 = np.floor(y)
    fx = x - x0
    fy = y - y0
    x0i = x0.astype(np.int64)
    y0i = y0.astype(np.int64)
    if wrap_u == 2:
        xa = x0i % w
        xb = (x0i + 1) % w
    else:
        xa = np.clip(x0i, 0, w - 1)
        xb = np.clip(x0i + 1, 0, w - 1)
    if wrap_v == 2:
        ya = y0i % h
        yb = (y0i + 1) % h
    else:
        ya = np.clip(y0i, 0, h - 1)
        yb = np.clip(y0i + 1, 0, h - 1)
    c00 = texture[ya, xa].astype(np.float32)
    c10 = texture[ya, xb].astype(np.float32)
    c01 = texture[yb, xa].astype(np.float32)
    c11 = texture[yb, xb].astype(np.float32)
    fx = fx.astype(np.float32)[:, None]
    fy = fy.astype(np.float32)[:, None]
    top = c00 * (1 - fx) + c10 * fx
    bot = c01 * (1 - fx) + c11 * fx
    out = top * (1 - fy) + bot * fy
    out[bad] = 0.0
    return out


def _sample(texture, u, v, wrap_u, wrap_v):
    """Bilinear sample with the native PICA200 wrap mode."""
    if VFLIP:
        v = 1.0 - v
    h, w = texture.shape[0], texture.shape[1]

    def wrapf(x, mode, size):
        if mode == 0:      # clampToEdge
            return min(max(x, 0.0), 1.0)
        if mode == 1:      # clampToBorder
            return x if 0.0 <= x <= 1.0 else None
        if mode == 2:      # repeat
            return x - math.floor(x)
        # mirroredRepeat
        y = abs(x) % 2.0
        return y if y <= 1.0 else 2.0 - y

    fu = wrapf(u, wrap_u, w)
    fv = wrapf(v, wrap_v, h)
    if fu is None or fv is None:
        return np.array([0.0, 0.0, 0.0], np.float32)
    x = fu * w - 0.5
    y = fv * h - 0.5
    x0, y0 = math.floor(x), math.floor(y)
    fx, fy = x - x0, y - y0
    x0i, y0i = int(x0), int(y0)
    x1i, y1i = x0i + 1, y0i + 1
    if wrap_u == 2:
        x0i %= w; x1i %= w
    else:
        x0i = min(max(x0i, 0), w - 1); x1i = min(max(x1i, 0), w - 1)
    if wrap_v == 2:
        y0i %= h; y1i %= h
    else:
        y0i = min(max(y0i, 0), h - 1); y1i = min(max(y1i, 0), h - 1)
    c00 = texture[y0i, x0i].astype(np.float32)
    c10 = texture[y0i, x1i].astype(np.float32)
    c01 = texture[y1i, x0i].astype(np.float32)
    c11 = texture[y1i, x1i].astype(np.float32)
    top = c00 * (1 - fx) + c10 * fx
    bot = c01 * (1 - fx) + c11 * fx
    return top * (1 - fy) + bot * fy


def draw(fb, verts_pos, verts_nrm, verts_uv, indices, texture, mv, proj,
         wrap_u=2, wrap_v=2, light_dir=(0.3, 0.55, 0.78), ambient=0.40,
         base_color=(1, 1, 1), alpha_test=0.35, double_sided=True, alpha_func=None,
         alpha_ref=None, material=None, mat_tag=0, vflip=None, blend=False,
         depth_bias=0.0, extra=(), phase=0, spec_tex=None, spec_wrap_u=2,
         spec_wrap_v=2,
         mat_name=None, obj_gain=None, mesh_index=None, skin_like=False):
    """Draw one mesh into the framebuffer."""
    if len(verts_pos) == 0 or len(indices) == 0:
        return
    P = np.hstack([verts_pos, np.ones((len(verts_pos), 1))]) @ mv.T
    V = np.hstack([verts_nrm, np.zeros((len(verts_nrm), 1))]) @ mv.T
    ndc = P @ proj.T
    wclip = ndc[:, 3].copy()
    wclip[np.abs(wclip) < 1e-9] = 1e-9
    ndc = ndc / wclip[:, None]

    sx = (ndc[:, 0] * 0.5 + 0.5) * fb.w
    sy = (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * fb.h
    sz = ndc[:, 2]
    valid = np.isfinite(sx) & np.isfinite(sy) & (wclip > 1e-6)

    L = np.asarray(light_dir, np.float64)
    L /= np.linalg.norm(L)
    base = np.asarray(base_color, np.float32)
    # ★ 2026-10-18：PICA200 是**逐像素**光照（fragment lighting），不是逐顶点插值。
    #   旧 pica 路径把 shade_vertices() 的结果按顶点插值，低模脸上会被摊平；
    #   现在改为光栅化时插值"视空间位置 + 法线"，在像素上做光照（见下方 shade_pixels）。
    use_pica = (SHADE_MODE == 'pica' and material is not None and LIGHTS is not None)
    light_dir_view = None
    if use_pica:
        import nlp_shade
        multi = getattr(LIGHTS, 'multi', None)
        if multi:
            # ★ 实机真值多光源：每盏灯的世界方向都转到视空间
            mv3 = np.asarray(mv, np.float64)[:3, :3]
            dirs = []
            for lt in multi:
                dv = mv3.dot(np.asarray(lt['dir'], np.float64))
                nn = np.linalg.norm(dv)
                dirs.append(dv / nn if nn > 1e-9 else dv)
            light_dir_view = dirs
        else:
            light_dir_view = np.asarray(mv, np.float64)[:3, :3].dot(
                np.asarray(LIGHTS.light_dir, np.float64))
            ln_l = np.linalg.norm(light_dir_view)
            if ln_l > 1e-9:
                light_dir_view = light_dir_view / ln_l

    idx = np.asarray(indices, np.int32).reshape(-1, 3)
    # ★ 2026-10-18 性能重构：三角形光栅化**整体向量化**（纯 numpy，输出逐像素一致）。
    #   旧实现：9246 个三角形逐个跑 numpy 小调用 ⇒ 每帧约 1.9s 固定开销，
    #   而且几乎与分辨率无关（256x384 只比 512x768 快 37%）。
    #   新实现：一次算完所有三角形的包围盒/边函数/深度测试；
    #   并且**先定每像素唯一胜者，再做透视插值/采样/着色**，
    #   省掉重叠三角形的重复插值。候选顺序 = 三角形顺序 + 行优先（同深度后者胜）。
    tv = valid[idx].all(axis=1)
    if not tv.any():
        return
    T = idx[tv]
    X = sx[T]; Y = sy[T]; Z = sz[T]; Wc = wclip[T]
    xmin = np.floor(X.min(axis=1)).astype(np.int64)
    xmax = np.ceil(X.max(axis=1)).astype(np.int64)
    ymin = np.floor(Y.min(axis=1)).astype(np.int64)
    ymax = np.ceil(Y.max(axis=1)).astype(np.int64)
    np.clip(xmin, 0, fb.w - 1, out=xmin); np.clip(xmax, 0, fb.w - 1, out=xmax)
    np.clip(ymin, 0, fb.h - 1, out=ymin); np.clip(ymax, 0, fb.h - 1, out=ymax)
    area = (X[:, 1] - X[:, 0]) * (Y[:, 2] - Y[:, 0]) - (X[:, 2] - X[:, 0]) * (Y[:, 1] - Y[:, 0])
    tk = (xmin <= xmax) & (ymin <= ymax) & (np.abs(area) > 1e-9)
    if not tk.any():
        return
    T = T[tk]; X = X[tk]; Y = Y[tk]; Z = Z[tk]; Wc = Wc[tk]; area = area[tk]
    xmin = xmin[tk]; xmax = xmax[tk]; ymin = ymin[tk]; ymax = ymax[tk]
    bw = xmax - xmin + 1
    cnt = bw * (ymax - ymin + 1)
    uv_all = np.asarray(verts_uv, np.float64)
    out = []
    CHUNK = 1500000
    ntri = len(T)
    start = 0
    cum = np.cumsum(cnt)
    depth = fb.depth
    while start < ntri:
        chunk_base = int(cum[start - 1]) if start > 0 else 0
        end = int(np.searchsorted(cum, chunk_base + CHUNK, side='right'))
        end = max(start + 1, min(end, ntri))
        c = cnt[start:end]
        ncand = int(c.sum())
        if ncand <= 0:
            start = end
            continue
        sel = np.arange(start, end, dtype=np.int64)
        rep = np.repeat(sel, c)
        intra = np.arange(ncand, dtype=np.int64) - np.repeat(np.cumsum(c) - c, c)
        bwt = bw[rep]
        px = xmin[rep] + (intra % bwt)
        py = ymin[rep] + (intra // bwt)
        xs = px + 0.5
        ys = py + 0.5
        X0 = X[rep, 0]; X1 = X[rep, 1]; X2 = X[rep, 2]
        Y0 = Y[rep, 0]; Y1 = Y[rep, 1]; Y2 = Y[rep, 2]
        inv = 1.0 / area[rep]
        w0 = ((Y1 - Y2) * inv) * xs + ((X2 - X1) * inv) * ys + (X1 * Y2 - X2 * Y1) * inv
        w1 = ((Y2 - Y0) * inv) * xs + ((X0 - X2) * inv) * ys + (X2 * Y0 - X0 * Y2) * inv
        w2 = 1.0 - w0 - w1
        ins = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
        if not ins.any():
            start = end
            continue
        px = px[ins]; py = py[ins]; rep = rep[ins]
        w0 = w0[ins]; w1 = w1[ins]; w2 = w2[ins]
        z = w0 * Z[rep, 0] + w1 * Z[rep, 1] + w2 * Z[rep, 2]
        if depth_bias:
            z = z - depth_bias
        if blend:
            win = z <= depth[py, px] + 1e-6
        else:
            win = z < depth[py, px]
        if not win.any():
            start = end
            continue
        out.append((py[win], px[win], rep[win], w0[win], w1[win], w2[win], z[win]))
        start = end
    if not out:
        return
    py = np.concatenate([e[0] for e in out])
    px = np.concatenate([e[1] for e in out])
    rep = np.concatenate([e[2] for e in out])
    w0 = np.concatenate([e[3] for e in out])
    w1 = np.concatenate([e[4] for e in out])
    w2 = np.concatenate([e[5] for e in out])
    zz = np.concatenate([e[6] for e in out])
    del out
    # ★ 每像素唯一胜者（z 最小；同 z 后画者胜），在插值/采样之前完成
    lin = py * fb.w + px
    order = np.lexsort((-np.arange(lin.size), zz))
    lin = lin[order]
    _, first = np.unique(lin, return_index=True)
    sel = order[first]
    del lin, order
    py = py[sel]; px = px[sel]; rep = rep[sel]; zz = zz[sel]
    w0 = w0[sel]; w1 = w1[sel]; w2 = w2[sel]
    # ---- 只对胜者做透视校正插值 ----
    ia = 1.0 / Wc[rep, 0]; ib = 1.0 / Wc[rep, 1]; ic = 1.0 / Wc[rep, 2]
    denom = w0 * ia + w1 * ib + w2 * ic
    np.clip(denom, 1e-12, None, out=denom)
    inv_d = 1.0 / denom
    v0 = T[rep, 0]; v1 = T[rep, 1]; v2 = T[rep, 2]
    uv0 = uv_all[v0]; uv1 = uv_all[v1]; uv2 = uv_all[v2]
    u = (w0 * uv0[:, 0] * ia + w1 * uv1[:, 0] * ib + w2 * uv2[:, 0] * ic) * inv_d
    v = (w0 * uv0[:, 1] * ia + w1 * uv1[:, 1] * ib + w2 * uv2[:, 1] * ic) * inv_d
    V0 = V[v0]; V1 = V[v1]; V2 = V[v2]
    nrm = np.empty((px.size, 3), np.float64)
    nrm[:, 0] = (w0 * V0[:, 0] * ia + w1 * V1[:, 0] * ib + w2 * V2[:, 0] * ic) * inv_d
    nrm[:, 1] = (w0 * V0[:, 1] * ia + w1 * V1[:, 1] * ib + w2 * V2[:, 1] * ic) * inv_d
    nrm[:, 2] = (w0 * V0[:, 2] * ia + w1 * V1[:, 2] * ib + w2 * V2[:, 2] * ic) * inv_d
    P0 = P[v0]; P1 = P[v1]; P2 = P[v2]
    posv = np.empty((px.size, 3), np.float64)
    posv[:, 0] = (w0 * P0[:, 0] * ia + w1 * P1[:, 0] * ib + w2 * P2[:, 0] * ic) * inv_d
    posv[:, 1] = (w0 * P0[:, 1] * ia + w1 * P1[:, 1] * ib + w2 * P2[:, 1] * ic) * inv_d
    posv[:, 2] = (w0 * P0[:, 2] * ia + w1 * P1[:, 2] * ib + w2 * P2[:, 2] * ic) * inv_d

    shade = None
    alpha_src = None
    spec_mask = None
    if texture is not None:
        tex = _sample_batch(texture, u, v, wrap_u, wrap_v, vflip)
        # ★ 高光强度贴图（*_spe）：黑底亮条纹的灰度图，乘在光照的高光项上 ✓
        if spec_tex is not None:
            stex = _sample_batch(spec_tex, u, v, spec_wrap_u, spec_wrap_v, vflip)
            spec_mask = stex[:, :3].mean(axis=1).astype(np.float64) / 255.0
        # ★ 多贴图层合成（unit1..N）：
        #   高光/反光是"黑底亮点" ⇒ 加法 ✓；其它按相乘 ✓
        if extra:
            base_c = tex[:, :3].astype(np.float32) / 255.0
            ph = int(phase) % 3
            ADDG = float(os.environ.get('LP_ADDGAIN', '0.30'))
            # ★ 2026-10-23 眼睛高光/水润层（eye_uru/eye_light）加大增益：
            #   贴图里的高光是"黑底亮点"，默认 ADDG=0.30 让眼睛看着没神（用户反馈）。
            #   眼睛材质单独放大（LP_EYEHIGAIN，默认 2.0 → 等效 0.60）。
            if '_eye' in (mat_name or ''):
                ADDG *= float(os.environ.get('LP_EYEHIGAIN', '2.0'))
            MULG = float(os.environ.get('LP_MULGAIN', '0.15'))
            for item in extra:
                et, ewu, ewv, emode = item[:4]
                egain = float(item[4]) if len(item) > 4 else 1.0
                t2 = _sample_batch(et, u, v, ewu, ewv, vflip)
                if emode == 'addrgb':
                    add = t2[:, :3].astype(np.float32) / 255.0 * (ADDG * egain)
                    base_c = base_c + add
                elif emode == 'add':
                    mask = t2[:, ph].astype(np.float32) / 255.0 * (ADDG * egain)
                    base_c = base_c + mask[:, None]
                else:
                    e = t2[:, :3].astype(np.float32) / 255.0
                    ea = t2[:, 3:4].astype(np.float32) / 255.0 * MULG
                    base_c = base_c * (1.0 - ea + e * ea)
            tex = tex.copy()
            tex[:, :3] = np.clip(base_c, 0.0, 1.0) * 255.0
        keep = _alpha_pass_batch(tex[:, 3], alpha_func, alpha_ref, alpha_test)
        if not keep.any():
            return
        if blend:
            alpha_src = (tex[:, 3].astype(np.float64) / 255.0)
        py = py[keep]; px = px[keep]; nrm = nrm[keep]; zz = zz[keep]
        posv = posv[keep]
        if spec_mask is not None:
            spec_mask = spec_mask[keep]
        if alpha_src is not None:
            alpha_src = alpha_src[keep]
        col = tex[keep, :3] / 255.0 * base
    else:
        col = np.broadcast_to(base, (py.size, 3)).astype(np.float32)

    ln = np.sqrt((nrm * nrm).sum(axis=1))
    ln[ln <= 1e-9] = 1.0
    nrm = nrm / ln[:, None]
    # ★ 2026-11-14：PICA200 硬件语义光照（LP_PICALIGHT=1）
    #   按 sw_lighting.cpp 算 primary/secondary；不再叠加 2026-10 的拟合增益/白平衡/skin cap。
    _pica_rgb = None
    _pica_scene_obj = _pica_scene() if use_pica else None
    if _pica_scene_obj is not None and material is not None:
        try:
            import pica_light as _PL
            _primary, _secondary = _PL.compute(material, _pica_scene_obj, nrm, posv)
            _pica_rgb = col * _primary
            _secmode = _os.environ.get('LP_PICASECMODE', 'mask')
            _secg = float(_os.environ.get('LP_PICASEC', '1.0'))
            if _secmode != 'none' and _secg > 0.0:
                if spec_mask is not None:
                    _pica_rgb = _pica_rgb + spec_mask[:, None] * (_secondary * _secg)
                elif _secmode == 'full':
                    _pica_rgb = _pica_rgb + _secondary * _secg
            # ★ 2026-11-14 假设验证开关：全局暖色项（疑似 TEV buffer color / screen 后处理色温）。
            #   默认关闭；LP_PICAGRADE="0.99,0.90,0.80" 用于 A/B 验证"缺的是全局色温"这条假设。
            #   实测主菜单截图的眼白是中性偏冷 (215,218,220)、皮肤是暖色 (254,205,175)，
            #   白发围裙是中性白 ⇒ 暖色项不是"全屏统一"，按材质排除（默认 '_eye'）。
            #   LP_PICAGRADESKIP="'_eye,_ha'" 可把头发也排除（头发实机是蓝黑+蓝灰 sheen，不该被染暖）。
            # ★ 2026-11-14：LP_PICATEV=1 —— 按运行时抓到的真值 TEV 程序合成（docs/08 §7.2）
            _tev = _os.environ.get('LP_PICATEV', '0') not in ('0', '', 'false', 'off')
            _grade = _os.environ.get('LP_PICAGRADE')
            if _grade and not _tev:
                _skip = _os.environ.get('LP_PICAGRADESKIP', '_eye').lower()
                if _os.environ.get('LP_PICAGRADEEYE') == '1':
                    _skip = ''                         # 兼容旧开关：强制包含眼睛
                _mn = (mat_name or '').lower()
                # LP_PICAGRADEONLY=skin：只对皮肤材质应用色温项。
                #   依据（实机主菜单截图量化）：皮肤暖 (254,205,175)，但白围裙 (230,237,238)、
                #   眼白 (215,218,220)、头发 (46,45,49, 微蓝) 都是中性/偏冷 ⇒ 暖色项只作用于皮肤。
                _only = _os.environ.get('LP_PICAGRADEONLY', '').lower()
                _is_skin = (bool(skin_like) or getattr(material, 'is_outfit_skin', False)
                            or ('_04_' in _mn) or ('face' in _mn) or ('hand' in _mn))
                _do_grade = True
                if _only == 'skin' and not _is_skin:
                    _do_grade = False                    # 非皮肤：不加色温项
                if _do_grade and not any(k and k in _mn for k in _skip.split(',')):
                    try:
                        _gv = np.array([float(x) for x in _grade.split(',')], np.float64)
                        if _gv.shape == (3,):
                            _pica_rgb = _pica_rgb * _gv[None, :]
                    except Exception:
                        pass
            _pica_rgb = np.clip(_pica_rgb, 0.0, 1.0)
            # ★ 2026-11-14：假 AO —— 朝下的面（下巴/脖子底、手臂内、裙底）压暗，
            #   否则平涂的脖子在环境光下会亮成一块白板（用户反馈脖子不对）。
            _ao = 1.0 - 0.18 * np.clip((-nrm[:, 1] - 0.25) / 0.55, 0.0, 1.0)
            _mn3 = (mat_name or '').lower()
            # 只对皮肤件/布料做朝下面压暗；脸部有贴图烘焙的明暗，别叠加（会发灰）
            if ('face' not in _mn3) and ('eye' not in _mn3) and (
                    bool(skin_like) or '_04_' in _mn3 or 'hand' in _mn3
                    or '_ne' in _mn3 or '_ar' in _mn3):
                _pica_rgb = _pica_rgb * _ao[:, None]
            if _tev:
                # ── 运行时 TEV 真值（out/live/cmd/tev_runtime.json，docs/08 §7.2）──
                # 皮肤：st1 MODULATE const (255,229,204) → st2 ADD secondary
                # 头发：st1 ×const (38,38,41) → st2 ×primary → st3 +TEX1×secondary → st4 +secondary
                # 眼睛：st0 = primary×TEX0 + secondary（其余 stage 是高光通道复用，由 extra 层模拟）
                _mn2 = (mat_name or '').lower()
                _skin = ((bool(skin_like) or getattr(material, 'is_outfit_skin', False)
                          or ('_04_' in _mn2) or ('face' in _mn2) or ('hand' in _mn2))
                         and '_eye' not in _mn2)
                _hair = ('_ha1' in _mn2) or ('_ha2' in _mn2) or ('hair' in _mn2)
                if _skin:
                    # 皮肤：st1 ×(255,229,204)；st2 的 +secondary 实际贡献≈0
                    #（TEV 的 mod 是 SRC_ALPHA，而 secondary 的 alpha 通道在 PICA 里基本为 0）
                    _SKIN_C = np.array([255.0, 229.0, 204.0]) / 255.0
                    _pica_rgb = _pica_rgb * _SKIN_C[None, :]
                elif _hair:
                    # 头发：st1 ×(38,38,41) → st2 ×primary → st3 +TEX1×secondary（=sheen）→ st4 +secondary×alpha≈0
                    _HAIR_C = np.array([38.0, 38.0, 41.0]) / 255.0
                    _pica_rgb = col * _primary * _HAIR_C[None, :]
                    if spec_mask is not None:
                        _pica_rgb = _pica_rgb + spec_mask[:, None] * _secondary
                # 眼睛：st0 = primary×TEX0 + secondary；secondary 的 alpha≈0，且眼白已偏亮，
                #       这里保持原路径（extra 高光层由 LP_ADDGAIN/LP_EYEHIGAIN 控制）
                _pica_rgb = np.clip(_pica_rgb, 0.0, 1.0)
        except Exception as _e:
            print('[PICA] lighting failed:', _e)
            _pica_rgb = None
    # ★ 2026-10-23 手机版三渲二移植：LP_TOONSTYLE=1 时改用手机 shader 的 toon ramp 公式
    _toon_on = os.environ.get('LP_TOONSTYLE', '0') not in ('0', '', 'false', 'off')
    _toon_rgb = None
    _mlow = (mat_name or '').lower()
    _is_skin_mat = (('_04_' in _mlow) or ('face' in _mlow) or ('hand' in _mlow)
                    or _mlow.endswith('_ne') or _mlow.endswith('_ar'))
    # 皮肤/眼睛保留原生 PICA 渲染（手机公式会把 3DS 已烘好明暗的亮皮肤打爆成白色）；
    # 布料/头发用手机 toon 公式 ⇒ 混合风：衣服三渲二 + 皮肤自然 + 描边
    _is_hair_mat = ('_ha1' in _mlow) or ('_ha2' in _mlow) or ('hair' in _mlow)
    if _toon_on and '_eye' not in (mat_name or '') and not _is_skin_mat and not _is_hair_mat:
        try:
            import nlp_shade as _nlp_shade
            Rm = mv[:3, :3]
            Rinv = np.linalg.inv(Rm)
            tvec = mv[:3, 3]
            nw = nrm @ Rinv.T
            pw = (posv - tvec) @ Rinv.T
            camw = -tvec @ Rinv.T
            _toon_rgb = _nlp_shade.toon_pixels(nw, pw, camw, col, mat_name)
            _tg = float(os.environ.get('LP_TOONGAIN', '1.0'))
            if _tg != 1.0:
                _toon_rgb = _toon_rgb * _tg
        except Exception:
            _toon_rgb = None
    # ★ 2026-10-18：默认走逐像素 PICA200 风格光照（环境 + N·L + D0 高光 + FR 边缘光）
    if use_pica and _toon_rgb is None and _pica_rgb is None:
        import nlp_shade as _nlp_shade
        shade = _nlp_shade.shade_pixels(posv, nrm, material, LIGHTS, light_dir_view,
                                        spec_mask=spec_mask)
    # ★ 2026-10-17 最终：默认【全部材质受光】✓✓
    #   任务006 修掉了图集级增益误伤白色件的 bug ✓ ⇒ 受光模式下围裙也是纯白 ✓
    #   实机验证（tools/regress.py，受光模式）：
    #     四部位全 PASS ✓ + 围裙上下一致性 Δ=(+2,+1,+1) PASS ✓✓
    #   想退回旧 unlit 行为：LP_UNLIT='_bo,_wa'
    unlit = os.environ.get('LP_UNLIT', '')
    # ★ 2026-10-17 从会话记录恢复的【原始且已验证】写法 ✓
    #   当时的效果：围裙纯白 ✓（用户认可 ✓，见 out/TOON_0_t20.png）
    #   关键就是 shade = None ⇒ 走下面的回退光照 ⇒ 亮 ✓
    #   我后来擅自改成乘材质增益 ✗ ⇒ 增益都 <1 ⇒ 整体变暗发蓝 ✗✓
    if unlit and shade is not None:
        for _k in unlit.split(','):
            if _k and (_k == '*' or _k in (mat_name or '')):
                shade = None
                break
    # ★ 2026-10-23 眼睛（sclera/iris）改为"自发光"：
    #   m_eye 的贴图里已经烘好了眼白/虹膜的明暗与高光（还有 eye_uru/eye_light 叠加层），
    #   再乘全局光照会把眼白压灰、偏蓝（实测 sclera 199,195,198，实机是亮白）——
    #   用户说的"眼睛这块阴影不对"。眼睛吃光照没有意义 ⇒ shade=1，直接出贴图色。
    #   LP_EYEUNLIT=0 可回退 A/B 对照。
    if shade is not None and _pica_rgb is None and \
            os.environ.get('LP_EYEUNLIT', '1') != '0' \
            and '_eye' in (mat_name or ''):
        shade = np.ones_like(shade)
    # ★ LP_DEBUG_GAIN=1：打印这个对象的增益链路，用来定位增益不生效的 bug ✓
    if os.environ.get('LP_DEBUG_GAIN'):
        try:
            import nlp_shade as _SD
            _g = _SD.material_gain(material) if material is not None else None
            print('[GAIN] mat=%-34s mat_name=%-30s gain=%s shade_is_none=%s extra=%d blend=%s'
                  % ((getattr(material, 'name', '?') or '?')[-34:],
                     (mat_name or '?')[-30:],
                     None if _g is None else np.round(_g, 3),
                     shade is None,
                     len(extra) if extra else 0,
                     blend))
        except Exception as _e:
            print('[GAIN] debug failed:', _e)
    # ★ LP_SINGLESIDED=1：强制只画正面（排查看到背面内侧用 ✓）
    if os.environ.get('LP_SINGLESIDED'):
        double_sided = False
    if _toon_rgb is not None:
        rgb = _toon_rgb
    elif _pica_rgb is not None:
        rgb = _pica_rgb
    elif shade is not None:
        # ★ 2026-10-18：旧的 LP_CONTRAST=0.85 全局压平已取消（它正是"光照太平"的来源）。
        #   明暗对比现在由 nlp_shade 的 ambient / light_color 显式控制；
        #   LP_CONTRAST 仍保留为后期微调（默认 1.0 = 不改）。
        contr = float(os.environ.get('LP_CONTRAST', '1.0'))
        if contr != 1.0:
            shade = 0.5 + (shade - 0.5) * contr
        # 三渲二分档在 nlp_shade.shade_pixels() 里对 N·L 做（LP_TOON），这里不再重复处理。
        rgb = col * shade
    else:
        d = np.abs(nrm @ L)
        sh = ambient + (1.0 - ambient) * d
        if DIFFUSE_GAIN != 1.0:
            sh = AMBIENT + DIFFUSE_GAIN * d      # 用模块级布光参数（不是函数默认值）
        rgb = col * sh[:, None]
    # ★ 对象级颜色校正属于最终光照结果，pica 与回退光照都必须且只能应用一次。
    #   增益必须按【实际走的光照路径】取：同一图集里既有受光的蓝裙网格，
    #   也可能有 unlit 的白色围裙网格 ⇒ 不能在 render_scene() 里提前定死 ✗。
    if obj_gain is None and material is not None and _OBJECT_GAIN is not None \
            and _pica_rgb is None:
        obj_gain = _OBJECT_GAIN(material, mesh_index, lit=(shade is not None))
    if obj_gain is not None:
        rgb = rgb * np.asarray(obj_gain, np.float64)[None, :]
    rgb = np.clip(rgb, 0.0, 1.0)
    # ★ 2026-10-18 脸部高光压平（用户：「脸要跟领口内衬那块一个颜色」）：
    #   实机脸部的 L75≈L90（高光被压平、偏暖），我们的脸高光区 (255,219,185) 偏白偏冷。
    #   对脸部皮肤贴图材质做 G/B 的"提亮 + 上限截断"：中调抬起来、高光压到 207/178，
    #   观感从"白脸"变成和内衬/脖子一致的暖肤色。只对 face_01/face_02 生效，
    #   不碰眼睛(_eye)、腮红叠加层(_face_col)。
    # ★ 2026-10-18 皮肤色统一（用户：脖子下方和全身颜色/光照统一）：
    #   原来只对脸部做 G/B 上限，导致脖子(_ne) 偏白、手臂(_ar) 偏深。
    #   这里改成：**所有皮肤材质**共用同一套"中调提亮 + 高光上限"，
    #   且按贴图色度判断是不是皮肤（白色围裙/领口内衬不会被误伤）。
    _cap_mode = '0' if (_toon_rgb is not None or _pica_rgb is not None) \
        else os.environ.get('LP_SKINCAP', 'all')
    _skin_mat = None
    if mat_name and _cap_mode != '0':
        _low = mat_name
        _is_face = ('face' in _low and ('_face_01' in _low or '_face_02' in _low))
        if (_is_face or (_cap_mode != 'face' and (
                '_04_' in _low or 'hand' in _low
                or _low.endswith('_ne') or _low.endswith('_ar')))):
            try:
                _cm = col.mean(axis=0)
                if _cm[0] > 0.80 and 0.60 < _cm[1] / max(_cm[0], 1e-6) < 0.90 \
                        and 0.48 < _cm[2] / max(_cm[0], 1e-6) < 0.82:
                    _skin_mat = True
            except Exception:
                _skin_mat = None
    if _skin_mat:
        try:
            fg_l = [float(x) for x in os.environ.get('LP_FACEG', '1.0,205').split(',')]
            fb_l = [float(x) for x in os.environ.get('LP_FACEB', '1.0,171').split(',')]
            g_gain, g_cap = (fg_l + [205.0])[:2]
            b_gain, b_cap = (fb_l + [171.0])[:2]
            rgb[:, 1] = np.minimum(rgb[:, 1] * g_gain, g_cap / 255.0)
            rgb[:, 2] = np.minimum(rgb[:, 2] * b_gain, b_cap / 255.0)
        except Exception:
            pass
    if blend:
        # ★ alpha 混合：脸部有多层叠加网格（如 `m_face_col` = 腮红/眼影，35.9% 半透明），
        #   只做 alpha 测试会把整层切掉 → 画面"少一层" ✗。这里按 src over dst 混合 ✓。
        al = np.clip(alpha_src, 0.0, 1.0)[:, None]
        dst = fb.color[py, px]
        fb.color[py, px] = rgb * al + dst * (1.0 - al)
    else:
        fb.color[py, px] = rgb
    fb.depth[py, px] = zz
    if mat_tag:
        fb.mat_id[py, px] = mat_tag


def render_scene(fb, objects, eye, target, up=(0, 0, 1), fovy=30.0,
                 ambient=0.40, light_dir=(0.3, 0.55, 0.78)):
    """objects: list of dicts with keys pos,nrm,uv,indices,texture,wrap_u,wrap_v,base_color."""
    mv = look_at(eye, target, up)
    proj = perspective(fovy, fb.w / float(fb.h), 1.0, 10000.0)
    for o in objects:
        draw(fb, o['pos'], o['nrm'], o['uv'], o['indices'], o.get('texture'),
             mv, proj, o.get('wrap_u', 2), o.get('wrap_v', 2),
             light_dir=light_dir, ambient=ambient,
             base_color=o.get('base_color', (1, 1, 1)),
             alpha_func=o.get('alpha_func'), alpha_ref=o.get('alpha_ref'),
             material=o.get('material'), mat_tag=o.get('mat_tag', 0),
             blend=o.get('blend', False),
             double_sided=o.get('double_sided', True),
             depth_bias=o.get('depth_bias', 0.0),
             extra=o.get('extra') or (),
             phase=o.get('phase', 0),
             spec_tex=o.get('spec_tex'),
             spec_wrap_u=o.get('spec_wrap_u', 2),
             spec_wrap_v=o.get('spec_wrap_v', 2),
             mat_name=(getattr(o.get('material'), 'name', '') or ''),
             mesh_index=o.get('mesh_index'),
             obj_gain=o.get('obj_gain'),
             skin_like=o.get('skin_like', False),
             vflip=o.get('vflip'))
    return fb


def save_png(path, fb):
    from PIL import Image
    img = (np.clip(fb.color, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(img, 'RGB').save(path)
    return path


# ------------------------------------------------------------------ 三渲二描边
def add_outline(fb, width=2, color=(0.10, 0.075, 0.07), strength=0.85):
    """基于深度缓冲的描边：剪影边界 + 明显深度断层（头发/脸部/衣领交界）。

    在 ss 超采样 framebuffer 上用，width 按最终像素给（内部会乘 ss）。
    """
    d = fb.depth
    mask = d < 1e29
    if not mask.any():
        return
    er = mask.copy()
    for _ in range(max(1, int(width))):
        er = (er & np.roll(er, 1, 0) & np.roll(er, -1, 0)
                & np.roll(er, 1, 1) & np.roll(er, -1, 1))
    edge = mask & ~er
    # 深度断层：相邻像素深度差 > 2%（且都有效）→ 内描边
    with np.errstate(invalid='ignore'):
        dx = np.abs(np.diff(d, axis=1))
        thr_x = 0.02 * np.minimum(d[:, :-1], d[:, 1:])
        ex = (dx > thr_x) & mask[:, :-1] & mask[:, 1:]
        dy = np.abs(np.diff(d, axis=0))
        thr_y = 0.02 * np.minimum(d[:-1, :], d[1:, :])
        ey = (dy > thr_y) & mask[:-1, :] & mask[1:, :]
    edge[:, :-1] |= ex
    edge[:, 1:] |= ex
    edge[:-1, :] |= ey
    edge[1:, :] |= ey
    edge &= mask
    if not edge.any():
        return
    c = np.asarray(color, np.float64)
    fb.color[edge] = fb.color[edge] * (1.0 - strength) + c[None, :] * strength
