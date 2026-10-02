# -*- coding: utf-8 -*-
"""Background library: the game's own bs_<scene>_<variant>_<index>.jpg assets.

Verified: 1356 files decode as ordinary 512x672 RGB JPEGs covering 549 scene numbers.
So a background needs no decoding work -- only extraction and compositing.
"""
import collections
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nlp_pack as N          # noqa: E402

IMG = os.environ.get('LP_IMG', r'D:/dsh/nlpp/img.bin')
_cache = {}


def list_backgrounds(img=IMG, limit=None):
    """-> sorted list of bs_*.jpg asset names (and bg_*.jpg), no decoding."""
    out = []
    with open(img, 'rb') as f:
        for p in N.iter_packs(img):
            for pf in p.files:
                nm = pf.name.lower()
                if nm.endswith('.jpg') and (pf.name.startswith('bs_') or pf.name.startswith('bg_')):
                    out.append(pf.name)
            if limit and len(out) >= limit:
                break
    return sorted(set(out))


def load_background(name, img=IMG):
    """-> PIL RGB image, or None."""
    if name in _cache:
        return _cache[name]
    from PIL import Image
    with open(img, 'rb') as f:
        for p in N.iter_packs(img):
            for pf in p.files:
                if pf.name == name:
                    try:
                        im = Image.open(io.BytesIO(N.pack_bytes(f, pf))).convert('RGB')
                        im.load()
                        # The 512x672 tiles are stored portrait, but the game is played
                        # with the 400x240 screen rotated, so the scene only reads
                        # upright after a quarter turn.  LP_BG_ROT selects the angle;
                        # 90 (CCW) puts sky up / ground down.
                        rot = int(os.environ.get('LP_BG_ROT', '90'))
                        if rot:
                            im = im.rotate(rot, expand=True)
                    except Exception:
                        return None
                    if len(_cache) > 8:
                        _cache.clear()
                    _cache[name] = im
                    return im
    return None
