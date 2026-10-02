# -*- coding: utf-8 -*-
"""Which layers fight over the same bones?

Layers are merged by stack_poses with "later overrides earlier" per bone+component.  If
the arm layer (m_upperNNN) and the hand layer (m_handNNN) both drive bones 49..63, the
one listed LAST wins -- and if that is the arm layer the hand collapses into a flat plate
(the artifact seen in the last render).  Print each layer's bone set, the pairwise
overlaps, and how many bones shift depending on the order.
"""
import sys

sys.path.insert(0, r'D:/dsh/nlpp/native')
sys.stdout.reconfigure(encoding='utf-8')
import nlp_pack as N          # noqa: E402
import nlp_mot as MOT         # noqa: E402
from compose import LAYERS    # noqa: E402

IMG = r'D:/dsh/nlpp/img.bin'
spec = LAYERS['default']

mots, got = [], set()
with open(IMG, 'rb') as f:
    for p in N.iter_packs(IMG):
        for pf in p.files:
            for name, base, per in spec:
                if pf.name == name and name not in got:
                    got.add(name)
                    mots.append(MOT.load_mot_final(N.pack_bytes(f, pf), name, base=base, per=per))
        if len(got) == len(spec):
            break

print('layer order (later overrides earlier):')
sets = []
for mm in mots:
    s = set(r.bone for r in mm.records if r.channels)
    sets.append(s)
    kind = 'abs' if mm.absolute_translation else 'delta'
    print('   %-36s bones=%-3d %-5s  %s' % (mm.name, len(s), kind, sorted(s)[:14]))
print()
print('pairwise overlap:')
for i in range(len(mots)):
    for j in range(i + 1, len(mots)):
        ov = sets[i] & sets[j]
        if ov:
            print('   %-30s vs %-30s  %2d bones  %s'
                  % (mots[i].name, mots[j].name, len(ov), sorted(ov)))
print()
print('bones whose last writer is NOT the body motion:')
for mm, s in zip(mots, sets):
    print('   %-36s owns %d distinct bones' % (mm.name, len(s)))
last = {}
for mm, s in zip(mots, sets):
    for b in s:
        last[b] = mm.name
import collections
c = collections.Counter(last.values())
for k, v in c.most_common():
    print('   final owner %-36s %d bones' % (k, v))
