# -*- coding: utf-8 -*-
"""motion_sheet.py — 把若干 .mot 动作渲成一张对照图（编号 + 动作名 + 时长）。

用法：
    python scripts/motion_sheet.py --img data/img.bin \
        --motions m_00010_41,m_00022_40 --frames 4 --out out/motions.png

每个动作横向排 N 帧，纵向一行一个动作；适合快速挑动作。
"""
import argparse
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NATIVE = os.path.join(ROOT, 'native')
sys.path.insert(0, NATIVE)

import nlp_pack as N            # noqa: E402
import nlp_mot as MOT           # noqa: E402


def motion_duration(img, name):
    """返回动作的帧时长（最大关键帧时间）。"""
    with open(img, 'rb') as f:
        for p in N.iter_packs(img):
            for pf in p.files:
                if pf.name == name + '.mot':
                    mot = MOT.load_mot_final(N.pack_bytes(f, pf), name + '.mot',
                                             base=0, per=1)
                    return max(max(r.times) for r in mot.records if r.count)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--img', default='data/img.bin')
    ap.add_argument('--motions', required=True, help='逗号分隔的动作名')
    ap.add_argument('--frames', type=int, default=4)
    ap.add_argument('--size', default='150x200')
    ap.add_argument('--hair', default='m_00_000')
    ap.add_argument('--outfit', default='m_01_126')
    ap.add_argument('--out', default='out/motions.png')
    a = ap.parse_args()

    from PIL import Image, ImageDraw, ImageFont

    w, h = [int(x) for x in a.size.lower().split('x')]
    names = [x.strip() for x in a.motions.split(',') if x.strip()]
    tmp = os.path.join(ROOT, 'out', '_motion_sheet_tmp')
    os.makedirs(tmp, exist_ok=True)
    rows = []
    for idx, name in enumerate(names, 1):
        dur = motion_duration(a.img, name)
        if dur <= 0:
            print('skip (not found):', name)
            continue
        ts = [round(dur * k / float(a.frames)) for k in range(a.frames)]
        imgs = []
        for j, t in enumerate(ts):
            out = os.path.join(tmp, '%02d_%d.png' % (idx, j))
            env = dict(os.environ)
            env['LP_MOTION'] = name + '.mot'
            subprocess.run([sys.executable, os.path.join(NATIVE, 'compose.py'),
                            '--img', a.img, '--hair', a.hair, '--outfit', a.outfit,
                            '--w', str(w), '--h', str(h), '--zoom', '2.3',
                            '--aim', '0.12', '--frame', str(t), '--out', out],
                           env=env, check=False,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if os.path.exists(out):
                imgs.append(Image.open(out).convert('RGB'))
        rows.append((idx, name, dur, imgs))

    if not rows:
        print('no motions rendered')
        return 1
    try:
        font = ImageFont.truetype(r'C:\Windows\Fonts\msyh.ttc', 16)
    except Exception:
        font = ImageFont.load_default()
    label_w = 210
    sheet = Image.new('RGB', (label_w + w * a.frames, (h + 22) * len(rows)), (24, 24, 28))
    d = ImageDraw.Draw(sheet)
    for r, (idx, name, dur, imgs) in enumerate(rows):
        y = r * (h + 22)
        d.text((6, y + 4), '%d) %s' % (idx, name), fill=(255, 225, 120), font=font)
        d.text((6, y + 22), 'dur=%d' % dur, fill=(205, 205, 210), font=font)
        for j, im in enumerate(imgs):
            sheet.paste(im, (label_w + j * w, y))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    sheet.save(a.out)
    print('->', a.out, sheet.size)
    return 0


if __name__ == '__main__':
    sys.exit(main())
