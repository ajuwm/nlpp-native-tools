# -*- coding: utf-8 -*-
"""Which mesh do bones 78..96 actually deform?

The bind pose renders clean, so the model is fine.  m_31235_32_skirt drives ONLY bones
78..96 yet the head/hair gets a spike, and m_lower011 / m_skirt_000 both drive 78..96.
That means my assumption "78..96 = lower body / skirt" was never checked.  Ask the data:
list, per mesh, the bones its vertices are weighted to.
"""
import sys
import collections

sys.path.insert(0, r'D:/dsh/nlpp/native')
sys.stdout.reconfigure(encoding='utf-8')
import nlp_model as M         # noqa: E402
from compose import character_parts   # noqa: E402

IMG = r'D:/dsh/nlpp/img.bin'
TARGET = (78, 96)

parts = character_parts('m', 'm_00_200', 'm_01_012')
print('parts: %s' % list(parts))
print()
interest = []
for nm in parts:
    try:
        m = M.load_model(IMG, nm)
    except Exception:
        continue
    bones_used = collections.Counter()
    for me in m.meshes:
        for pr in me.prims:
            for v in pr.vertices.values():
                for b in (v.get('bones') or []):
                    bones_used[b] += 1
    if not bones_used:
        continue
    lo, hi = TARGET
    hit = sorted(b for b in bones_used if lo <= b <= hi)
    allb = sorted(bones_used)
    print('%-18s verts=%-6d bones=%-4d range=[%d..%d]' %
          (nm, sum(bones_used.values()), len(allb), allb[0], allb[-1]))
    print('%-18s   in %d..%d : %s' % ('', lo, hi, hit if hit else 'NONE'))
    print('%-18s   all bones up to 40: %s' % ('', allb[:40]))
    if hit:
        interest.append(nm)
print()
print('meshes that use bones %d..%d : %s' % (TARGET[0], TARGET[1], interest if interest else 'NONE'))
