# -*- coding: utf-8 -*-
"""
Native-format renderer for New Love Plus models (software rasteriser, no GPU deps).

It consumes ONLY the game's own data: SMES vertices, SMAT materials (including the
native wrap modes), BONE skeleton and the PICA200 tiled TEXI textures.
"""
import math
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
    return rotate_axis((1, 0, 0), rx) @ rotate_axis((0, 1, 0), ry) @ rotate_axis((0, 0, 1), rz)


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
         base_color=(1, 1, 1), alpha_test=0.35, double_sided=True, alpha_func=None, alpha_ref=None):
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

    idx = np.asarray(indices, np.int32).reshape(-1, 3)
    for tri in idx:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        if not (valid[a] and valid[b] and valid[c]):
            continue
        x0, y0 = sx[a], sy[a]
        x1, y1 = sx[b], sy[b]
        x2, y2 = sx[c], sy[c]
        minx = max(int(math.floor(min(x0, x1, x2))), 0)
        maxx = min(int(math.ceil(max(x0, x1, x2))), fb.w - 1)
        miny = max(int(math.floor(min(y0, y1, y2))), 0)
        maxy = min(int(math.ceil(max(y0, y1, y2))), fb.h - 1)
        if minx > maxx or miny > maxy:
            continue
        area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        if abs(area) < 1e-9:
            continue
        xs = np.arange(minx, maxx + 1) + 0.5
        ys = np.arange(miny, maxy + 1) + 0.5
        X, Y = np.meshgrid(xs, ys)
        w0 = ((x1 - X) * (y2 - Y) - (x2 - X) * (y1 - Y)) / area
        w1 = ((x2 - X) * (y0 - Y) - (x0 - X) * (y2 - Y)) / area
        w2 = 1.0 - w0 - w1
        if double_sided:
            inside = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
            if not inside.any():
                w0, w1, w2 = -w0, -w1, -w2
                inside = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
        else:
            inside = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
        if not inside.any():
            continue

        z = w0 * sz[a] + w1 * sz[b] + w2 * sz[c]
        sub = fb.depth[miny:maxy + 1, minx:maxx + 1]
        win = inside & (z < sub)
        if not win.any():
            continue

        # perspective-correct interpolants
        ia = 1.0 / wclip[a]; ib = 1.0 / wclip[b]; ic = 1.0 / wclip[c]
        denom = w0 * ia + w1 * ib + w2 * ic
        denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)

        u = (w0 * verts_uv[a][0] * ia + w1 * verts_uv[b][0] * ib + w2 * verts_uv[c][0] * ic) / denom
        v = (w0 * verts_uv[a][1] * ia + w1 * verts_uv[b][1] * ib + w2 * verts_uv[c][1] * ic) / denom
        nx = (w0 * V[a][0] * ia + w1 * V[b][0] * ib + w2 * V[c][0] * ic) / denom
        ny = (w0 * V[a][1] * ia + w1 * V[b][1] * ib + w2 * V[c][1] * ic) / denom
        nz = (w0 * V[a][2] * ia + w1 * V[b][2] * ib + w2 * V[c][2] * ic) / denom

        rows, cols = np.nonzero(win)
        for r, cc in zip(rows, cols):
            py = miny + r
            px = minx + cc
            uu = float(u[r, cc]); vv = float(v[r, cc])
            if texture is not None:
                tex = _sample(texture, uu, vv, wrap_u, wrap_v)
                if not _alpha_pass(tex[3], alpha_func, alpha_ref, alpha_test):
                    continue
                col = tex[:3] / 255.0 * base
            else:
                col = base
            n = np.array([nx[r, cc], ny[r, cc], nz[r, cc]])
            ln = np.linalg.norm(n)
            if ln > 1e-9:
                n = n / ln
            d = abs(float(n.dot(L)))
            shade = ambient + (1.0 - ambient) * d
            fb.color[py, px] = np.clip(col * shade, 0.0, 1.0)
            fb.depth[py, px] = z[r, cc]


def render_scene(fb, objects, eye, target, up=(0, 0, 1), fovy=30.0,
                 ambient=0.40, light_dir=(0.3, 0.55, 0.78)):
    """objects: list of dicts with keys pos,nrm,uv,indices,texture,wrap_u,wrap_v,base_color."""
    mv = look_at(eye, target, up)
    proj = perspective(fovy, fb.w / float(fb.h), 1.0, 10000.0)
    for o in objects:
        draw(fb, o['pos'], o['nrm'], o['uv'], o['indices'], o.get('texture'),
             mv, proj, o.get('wrap_u', 2), o.get('wrap_v', 2),
             light_dir=light_dir, ambient=ambient,
             base_color=o.get('base_color', (1, 1, 1)))
    return fb


def save_png(path, fb):
    from PIL import Image
    img = (np.clip(fb.color, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(img, 'RGB').save(path)
    return path
