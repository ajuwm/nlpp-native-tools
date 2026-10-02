# -*- coding: utf-8 -*-
"""最小可用示例：读游戏原生资源，渲染一帧角色。

这是理解整个工具链的入口 —— 只有 5 步，不涉及任何格式转换。

    python quickstart.py --img D:/path/to/img.bin --out frame.png

    python quickstart.py --img img.bin --motion m_00010_40.mot --time 20
    python quickstart.py --img img.bin --bg bs_0000_00_00.jpg --yaw 30

所有资源都从游戏本体读取，本仓库不含任何游戏数据。
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'native'))

import nlp_pack as N          # noqa: E402
import nlp_model as M         # noqa: E402
import nlp_mot as MOT         # noqa: E402
import nlp_render as R        # noqa: E402
from compose import character_parts   # noqa: E402

# 一个完整的角色 = 头发 + 服装 + 腿 + 脸 + 手（全部来自游戏本体）
HAIR, OUTFIT = 'm_00_200', 'm_01_012'


def build_objects(img):
    """读模型 → 顶点/法线/UV/骨骼权重/索引。返回 (物体列表, 骨架)。"""
    objs, bones = [], None
    for name in character_parts('m', HAIR, OUTFIT):
        try:
            model = M.load_model(img, name)
        except Exception:
            continue
        if bones is None:
            bones = model.bones
        for mesh in model.meshes:
            for prim in mesh.prims:
                if not prim.vertices or not prim.indices:
                    continue
                order = sorted(prim.vertices.keys())
                remap = {vi: k for k, vi in enumerate(order)}
                verts = []
                for vi in order:
                    v = dict(prim.vertices[vi])
                    w = list(v.get('weights') or [])
                    s = sum(w)
                    if s > 0:
                        v['weights'] = [x / s for x in w]   # 权重归一化（必须）
                    verts.append(v)
                objs.append(dict(verts=verts,
                                 uv=np.array([v['uv'] for v in verts], np.float64),
                                 texture=None,
                                 indices=[remap[i] for i in prim.indices if i in remap]))
    return objs, bones


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--img', required=True, help='游戏本体的 img.bin')
    ap.add_argument('--out', default='frame.png')
    ap.add_argument('--motion', default=None, help='动作名，如 m_00010_40.mot')
    ap.add_argument('--time', type=int, default=0, help='帧号')
    ap.add_argument('--bg', default=None, help='背景名，如 bs_0000_00_00.jpg')
    ap.add_argument('--yaw', type=float, default=0.0, help='相机水平角（度）')
    ap.add_argument('--size', default='600x860')
    args = ap.parse_args()

    W, H = (int(x) for x in args.size.lower().split('x'))

    print('[1/5] 读取模型 ...')
    objs, bones = build_objects(args.img)
    print('      %d 个部件, %d 根骨骼' % (len(objs), len(bones)))

    print('[2/5] 计算绑定姿势 ...')
    MOT.BIND_ROT = {b.index: tuple(b.rotation) for b in bones}
    wb = R.world_matrices(bones, {})
    binv = [np.linalg.inv(m) if m is not None else None for m in wb]

    # 相机一次性按绑定姿势取景，全程固定（按当前姿势取景会导致画面剧烈缩放）
    bind_pts = np.vstack([R.skin_positions(o['verts'], bones, wb)[0] for o in objs])
    lo, hi = bind_pts.min(axis=0), bind_pts.max(axis=0)
    ctr = (lo + hi) / 2.0
    height = float(max(hi[1] - lo[1], 20.0))

    print('[3/5] 采样动作 ...')
    pose = {}
    if args.motion:
        raw, name = None, args.motion
        with open(args.img, 'rb') as f:
            for pack in N.iter_packs(args.img):
                for pf in pack.files:
                    if pf.name == name:
                        raw = N.pack_bytes(f, pf)
                        break
                if raw:
                    break
        if raw is None:
            print('      找不到 %s' % name)
            return 1
        mot = MOT.load_mot_final(raw, name)
        print('      %s: %d 条记录, %d 帧' % (name, len(mot.records), mot.duration))
        p = MOT.sample_present(mot, args.time)
        for bone, ch in p.items():
            if bone >= len(bones):
                continue
            d = [0.0] * 9
            for ax, v in ch.get('T', {}).items():
                d[ax] = v
            for ax, v in ch.get('R', {}).items():
                d[3 + ax] = v
            d[6] = d[7] = d[8] = 1.0
            pose[bone] = tuple(d)
    else:
        print('      未指定动作，渲染绑定姿势')

    print('[4/5] 蒙皮 ...')
    wp = R.world_matrices(bones, pose)
    world = [(wp[i] @ binv[i]) if (wp[i] is not None and binv[i] is not None) else wp[i]
             for i in range(len(wp))]
    fobj = []
    for o in objs:
        pos, nrm = R.skin_positions(o['verts'], bones, world)
        g = dict(o)
        g['pos'] = pos
        g['nrm'] = nrm
        fobj.append(g)

    print('[5/5] 渲染 ...')
    bg = None
    if args.bg:
        try:
            import nlp_bg
            bg = nlp_bg.load_background(args.img, args.bg)
        except Exception as e:
            print('      背景加载失败: %s' % e)
    ang = np.radians(args.yaw)
    eye = ctr + np.array([np.sin(ang) * 2.4 * height, 0.0, np.cos(ang) * 2.4 * height])
    fb = R.Framebuffer(W, H, bg=(0.12, 0.14, 0.18), background=bg)
    R.render_scene(fb, fobj, eye, ctr, up=(0, 1, 0), fovy=30.0)
    Image.fromarray((np.clip(fb.color, 0, 1) * 255).astype(np.uint8)).save(args.out)
    print('      -> %s' % args.out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
