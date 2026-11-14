# -*- coding: utf-8 -*-
"""
Native NLPP/NLP motion (.mot) reader + skeleton animation.

Container layout:
    +0x00  'MOT '   (4cc)
    +0x08  u32  bone record count
    +0x14  u32  float pool A offset
    +0x18  u32  float pool B offset
    +0x1C  u32[count] record offsets
  Each record:
    +0x00  u16  keyframe count
    +0x02  u16  channel mask -- bits start at 3:
                  bits 3,4,5 -> T xyz   bits 6,7,8 -> R xyz   bits 9,10,11 -> S xyz
    +0x04  u16[count] keyframe times
    then   u32[channels] value references (4-byte aligned, +2 padding if needed)
             r <  0x80000000 -> float index r        in pool A
             r >= 0x80000000 -> float index r & 0x7fffffff in pool B
           channel c occupies the `count` consecutive floats starting at its index.
"""
import struct
import math
import numpy as np

CHMAP = {3: ('T', 0), 4: ('T', 1), 5: ('T', 2),
         6: ('R', 0), 7: ('R', 1), 8: ('R', 2),
         9: ('S', 0), 10: ('S', 1), 11: ('S', 2)}


class MotRecord(object):
    __slots__ = ('bone', 'count', 'mask', 'channels', 'times', 'values')

    def __init__(self, bone, count, mask, times, values, channels):
        self.bone = bone
        self.count = count
        self.mask = mask
        self.channels = channels      # list of (component, axis)
        self.times = times
        self.values = values          # list[count][len(channels)]

    def __repr__(self):
        return '<MotRecord bone=%d keys=%d mask=0x%X ch=%r>' % (
            self.bone, self.count, self.mask, self.channels)


class Motion(object):
    def __init__(self, name='motion'):
        self.name = name
        self.records = []
        self.duration = 0.0

    def __repr__(self):
        return '<Motion %r bones=%d duration=%.2f>' % (
            self.name, len(self.records), self.duration)

    def sample(self, t):
        """Return {bone: dict(T=(x,y,z), R=(x,y,z), S=(x,y,z))} at time t.

        Keyframes are linearly interpolated; missing channels keep the bind value.
        """
        out = {}
        for r in self.records:
            n = r.count
            if n == 0:
                continue
            if t <= r.times[0]:
                k0 = k1 = 0
                f = 0.0
            elif t >= r.times[-1]:
                k0 = k1 = n - 1
                f = 0.0
            else:
                lo, hi = 0, n - 1
                while hi - lo > 1:
                    mid = (lo + hi) // 2
                    if r.times[mid] <= t:
                        lo = mid
                    else:
                        hi = mid
                k0, k1 = lo, hi
                span = r.times[k1] - r.times[k0]
                f = 0.0 if span == 0 else (t - r.times[k0]) / float(span)
            a, b = r.values[k0], r.values[k1]
            ch = {}
            for ci, (comp, axis) in enumerate(r.channels):
                v = a[ci] + (b[ci] - a[ci]) * f
                default = 1.0 if comp == 'S' else 0.0
                ch.setdefault(comp, [default, default, default])[axis] = v
            out[r.bone] = ch
        return out


def load_mot(data, name='motion'):
    if data[:3] != b'MOT':
        raise ValueError('not a MOT file: %r' % data[:4])
    m = Motion(name)
    count = struct.unpack_from('<I', data, 8)[0]
    if count <= 0 or count > 8192:
        raise ValueError('implausible bone count %d' % count)
    pool_a = struct.unpack_from('<I', data, 0x14)[0]
    pool_b = struct.unpack_from('<I', data, 0x18)[0]
    offs = list(struct.unpack_from('<%dI' % count, data, 0x1C))
    if len(offs) > count:
        offs = offs[:count]
    for i in range(len(offs)):
        s = offs[i]
        if s + 4 > len(data) or s >= pool_a:
            m.records.append(MotRecord(i, 0, 0, [], [], []))
            continue
        nk, mask = struct.unpack_from('<HH', data, s)
        times = [struct.unpack_from('<H', data, s + 4 + k * 2)[0] for k in range(nk)]
        body = s + 4 + nk * 2
        if body % 4:
            body += 2
        bits = [b + 3 for b in range(16) if (mask >> 3 >> b) & 1]
        chans = [CHMAP[b] for b in bits if b in CHMAP]
        nch = len(bits)
        # refs holds one u32 per (keyframe, channel): nk*nch entries, NOT nch.
        # Measured: m_00000_40 rec1 has refs 0xB4..0xDD = 42 = 14*3, and the record
        # size 4 + nk*2 + pad4 + 4*nk*nch then matches the next record exactly.
        nrefs = nk * nch
        refs = []
        if nrefs and body + 4 * nrefs <= len(data):
            refs = list(struct.unpack_from('<%dI' % nrefs, data, body))
        vals = []
        for ci, bit in enumerate(bits):
            if bit not in CHMAP or nk == 0 or not refs:
                continue
            col = []
            for k in range(nk):
                r = refs[k * nch + ci]
                base = pool_b if (r & 0x80000000) else pool_a
                idx = (r & 0x7FFFFFFF) if (r & 0x80000000) else r
                o = base + idx * 4
                if o < 0 or o + 4 > len(data):
                    col = []
                    break
                col.append(struct.unpack_from('<f', data, o)[0])
            if col:
                vals.append(col)
        per_key = [[vals[c][k] for c in range(len(vals))] for k in range(nk)]
        bone = bone_table[i] if bone_table else (base + i // per)
        assert 0 <= bone < 4096, 'bone index out of range: %d' % bone
        m.records.append(MotRecord(bone, nk, mask, times, per_key, chans))
    m.duration = max((r.times[-1] for r in m.records if r.times), default=0.0)
    # Layered motions (hand / lower / skirt) are made of single-keyframe records and
    # store ABSOLUTE bone translations (verified: their values equal the bind pose, and
    # treating them as deltas doubles the bind).  Body motions (nk>1) store DELTAS.
    m.absolute_translation = all(r.count <= 1 for r in m.records if r.count)
    return m


# ------------------------------------------------------------------ posing
def _rot_matrix(rx, ry, rz, order='XYZ'):
    cx, sy = math.cos(rx), math.sin(ry)
    sx, cy = math.sin(rx), math.cos(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], np.float64)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], np.float64)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], np.float64)
    o = {'XYZ': (Rx, Ry, Rz), 'ZYX': (Rz, Ry, Rx),
         'YXZ': (Ry, Rx, Rz), 'ZXY': (Rz, Rx, Ry),
         'YZX': (Ry, Rz, Rx), 'XZY': (Rx, Rz, Ry)}[order]
    R = np.eye(3)
    for M in o:
        R = R @ M
    return R


def local_matrices(bones, pose=None, order='XYZ', absolute=False):
    """Local bind matrices, optionally overridden by a pose sample.

    `absolute=False` (default) treats .mot values as deltas on the bind pose -- the
    only interpretation that keeps the figure standing at frame 0.
    """
    out = []
    for b in bones:
        t = list(b.translation)
        r = list(b.rotation)
        s = list(b.scale)
        if pose and b.index in pose:
            p = pose[b.index]
            if 'T' in p:
                t = list(p['T']) if absolute else [t[i] + p['T'][i] for i in range(3)]
            if 'R' in p:
                r = list(p['R']) if absolute else [r[i] + p['R'][i] for i in range(3)]
            if 'S' in p:
                s = list(p['S']) if absolute else [s[i] * p['S'][i] for i in range(3)]
        m = np.eye(4)
        m[:3, :3] = _rot_matrix(r[0], r[1], r[2], order)
        m[:3, :3] = m[:3, :3] @ np.diag(s)
        m[:3, 3] = t
        out.append(m)
    return out


def world_matrices_posed(bones, pose=None, order='XYZ'):
    n = len(bones)
    local = local_matrices(bones, pose, order)
    world = [None] * n
    for _ in range(8):
        changed = False
        for b in bones:
            p = b.parent
            m = local[b.index] if (p < 0 or p >= n or world[p] is None) \
                else world[p] @ local[b.index]
            if world[b.index] is None or not np.allclose(m, world[b.index]):
                world[b.index] = m
                changed = True
        if not changed:
            break
    for i in range(n):
        if world[i] is None:
            world[i] = local[i]
    return world


def bind_inverse(world):
    """Inverse bind matrices (bind pose -> bone space)."""
    return [np.linalg.inv(w) for w in world]


# ============================================================================
# FINAL, VERIFIED .mot READER  (supersedes load_mot / CHMAP above)
# ----------------------------------------------------------------------------
# Derived and validated by measurement; see out/阶段2_动作格式_进展.md s.25.
#   * record: u16 nk | u16 mask | u16 times[nk] | pad4 | u32 refs[nk*nch]
#   * channel count C = nch (nk==1) else nch // 3
#   * nk==1 : value = refs[k*nch + ci]
#     nk>1  : each channel's keyframe is a triple (A, B, B); the VALUE is the
#             MIDDLE element B   (A and the trailing B are tangents)
#   * component types are ordered by TYPE: translation first, then rotation
#     (bits 6,7,8 = translation axes ; bits 3,4,5 = rotation ; 9,10,11 = scale)
#   * within a type the bits go ASCENDING; i-th value's axis is
#     cycle[(start+i)%3] with cycle=(x,y,z) and start = (8 - record max bit) % 3
#   * translations are DELTAS on the bind pose
#   * rotations are DELTAS on the bind pose, stored in DEGREES.
#
# !! rot_degrees MUST stay False for this pipeline.
#    nlp_render.rotate_axis(axis, deg) takes DEGREES, but this loader used to convert
#    to radians, so every body-motion rotation reached the renderer ~57x too small.
#    The character then only drifted: max frame-to-frame movement was 4.4 units where
#    a real action gives 100+.  Found by grid-searching the mapping with an AMPLITUDE
#    criterion (the old "the curve varies" criterion passed the broken version).
# ============================================================================
KEY_RANGES = (('T', 6, 8), ('R', 3, 5), ('S', 9, 11))
AXIS_CYCLE = (0, 1, 2)


# mask bit -> output component, taken from the dispatcher at VA 0x0069B26C.
# Slots 0-2 are translation (verified: bits 8/7/6 reproduce the bind T exactly for
# `m_hand000` and bits 7/6 reproduce T.y/T.z for `m_lower011`).
# The ROTATION group's slot order is NOT (x,y,z): a turn motion (m_00011_40) must keep
# its height constant, and measuring the bounding-box height over the whole motion
# gives a spread of 0.0 when the first rotation channel drives Y and 106.6 when it
# drives X.  So the first rotation channel is YAW.
_BIT_COMPONENT = {
    8: ('T', 0), 7: ('T', 1), 6: ('T', 2),
    # Rotation axes.  Ranked by a two-part criterion -- "the picture must VISIBLY change
    # AND the height must stay put" -- over three motions (native/hypo_sweep.py):
    #   R0->y, R1->x, R2->z  (previous)  visible 0.0166  height 14.1
    #   R0->x, R1->z, R2->y  (this one)  visible 0.0322  height  8.0   better on both
    #   R0->y, R1->z, R2->x              visible 0.0117  height  8.3
    # A Y rotation on a limb that points down is a spin about the limb's own axis, which
    # is invisible -- that is why "height stays constant" on its own picked a bad answer.
    5: ('R', 0), 4: ('R', 1), 3: ('R', 2),
    2: ('S', 0), 1: ('S', 1), 0: ('S', 2),
}
# 上面是「自然轴序」：bit8/7/6 -> T.x/y/z，bit5/4/3 -> R.x/y/z，bit2/1/0 -> S.x/y/z。
# 早期版本把 bit4/bit3 映射成 R.z/R.y（用"高度保持不变"这个间接判据选的），
# 在骨骼映射和条目宽度都修正之后，那个顺序会把手臂折到头顶；
# 自然轴序得到的是手臂自然下垂的站姿。LP_ROLD=1 可切回旧顺序做对照。
import os as _os0
if _os0.environ.get('LP_ROLD'):
    _BIT_COMPONENT.update({4: ('R', 2), 3: ('R', 1)})


# 骨骼 ID -> 扁平骨骼号。渲染器装好骨骼后设置一次。
# 所有部件的 .bone 共用同一张 ID 表（254 个 ID 互不相同，范围 ~10..322），
# 所以一张表对整个游戏都成立。
ID_TO_BONE = {}


def _bone_of(bone_table, i, base, per):
    """记录 i 的目标骨骼（扁平序号）。

    ★ header+0x0C 的表存的是**骨骼 ID**，不是骨骼下标，也不是「ID + 225*bank」。

    证据链：
      1. `.bone` 头部第三个字段（旧注为 unk_off）指向 u32[254] 表，`表[骨骼号] = ID`，
         各部件**逐字节一致**（全局 ID 空间）。
      2. `.mot` 的表值就落在这个 ID 空间里（274 / 303 / 137 …）。
      3. 单关键帧记录存的是**绝对绑定平移**，与 ID 映射结果逐值吻合：
         m_skirt_000 的 ID 303 -> 骨骼 6，其 T=(12.0,-2.8) 正是骨骼 6 的绑定平移；
         ID 304/305/306 -> 236/237/238，ID 84 -> 242 …… 完全对应六片三段裙摆。

    旧的 `v - 225 if v >= 225` 规则「看起来能用」，只是因为 m_hand000 的
    ID 274..288 减 225 恰好落在左手的 49..63 —— 而真值是**右手** 210..224
    （同样是 15 根连续骨，形状相同、对象不同）。
    LP_IDMAP=0 可切回旧规则做对照。
    """
    import os as _os
    if bone_table and ID_TO_BONE and _os.environ.get('LP_IDMAP', '1') != '0':
        return ID_TO_BONE.get(bone_table[i], 9999)
    if bone_table:
        v = bone_table[i]
        return v - 225 if v >= 225 else v
    return base + i // per


def channel_layout(bits, nk, bone=None, bind_x=None):
    """-> list of (component, axis, offset), one entry per set mask bit.

    READ OFF THE GAME'S OWN CODE, not inferred.  VA 0x0069B26C is the channel
    dispatcher; it tests the mask bits from HIGH to LOW and stores each result:

        tst r7,#0x100 -> [r6]      bit 8 -> slot 0  (T.x)
        tst r7,#0x080 -> [r6,#4]   bit 7 -> slot 1  (T.y)
        tst r7,#0x040 -> [r6,#8]   bit 6 -> slot 2  (T.z)
        tst r7,#0x020 -> [r6,#0x18] bit 5 -> slot 6 (R.x)
        tst r7,#0x010 -> [r6,#0x1c] bit 4 -> slot 7 (R.y)
        tst r7,#0x008 -> [r6,#0x20] bit 3 -> slot 8 (R.z)
        tst r7,#0x004 -> [r6,#0x0c] bit 2 -> slot 3 (S.x)
        tst r7,#0x002 -> [r6,#0x10] bit 1 -> slot 4 (S.y)
        tst r7,#0x001 -> [r6,#0x14] bit 0 -> slot 5 (S.z)

    Covered by two independent anchors: `m_hand000` (bits 3..8) yields
    [-7.782, 1.055, 1.6236, ...] whose first three equal the bone's bind translation
    -> bit 8/7/6 are T.x/T.y/T.z; and `m_lower011` (bits 6,7) yields [12.0, -2.8002]
    which is T.y then T.z -> bit 7/6 again.

    Channels are consumed in DESCENDING bit order, and each channel takes exactly ONE
    float per keyframe -- verified at VA 0x00740BF8 (`mul r0,r0,r6` with r6 = popcount,
    then `lsl #2`), i.e. the stride is nch*4 bytes, not nch*3*4.

    The same layout serves nk==1 and nk>1: only the number of keyframes differs.
    There is NO tangent data -- the game interpolates linearly (VA 0x00740C9C:
    vsub B-A ; vmla A + k*(B-A)).
    """
    order = sorted(bits, reverse=True)
    out = []
    for ci, b in enumerate(order):
        comp, axis = _BIT_COMPONENT[b]
        out.append((comp, axis, ci))
    return out



def load_mot_final(data, name='motion', rot_degrees=False, base=0, per=1,
                   bones_bind=None):
    """Parse a .mot with the verified rule. Returns a Motion.

    ★ 旋转值单位是**弧度**（渲染器 _rot_matrix 直接吃弧度）。
      证据：m_31003_40 里上臂骨（骨骼 30）的旋转是 1.39~1.45，而绑定姿势是 T-pose；
      1.4 rad ≈ 80°，正好是"手臂从水平垂到身侧"所需的角度 —— 按度解释时只有 1.4°
      （画面上手臂纹丝不动），按圈解释（×360）则整个人扭曲。LP_ROT_SCALE 可覆盖。
    """
    if data[:3] != b'MOT':
        raise ValueError('not a MOT file: %r' % data[:4])
    m = Motion(name)
    count = struct.unpack_from('<I', data, 8)[0]
    if count <= 0 or count > 8192:
        raise ValueError('implausible bone count %d' % count)
    pool_a = struct.unpack_from('<I', data, 0x14)[0]
    pool_b = struct.unpack_from('<I', data, 0x18)[0]
    # ---- record -> bone table -------------------------------------------------
    # Header +0x0C is NOT a size: it is the OFFSET of a u32[count] array that gives the
    # bone index of every record.  This is the table the game reads in 0x740d08 as
    #     obj->[+8]->[+0x0C][record_index]
    # Verified on m_00011_40 (60 records, offset 75600 = size-240, values 12..288),
    # m_hand000 (values 274..288 for its 15 records) and m_lower011 (303..321).
    # `base + i // per` was a guess and is now only the fallback.
    bone_table = None
    try:
        bt_off = struct.unpack_from('<I', data, 0x0C)[0]
        if 0 < bt_off and bt_off + count * 4 <= len(data):
            bone_table = struct.unpack_from('<%dI' % count, data, bt_off)
    except Exception:
        bone_table = None

    offs = struct.unpack_from('<%dI' % count, data, 0x1C)
    # ★ .mot 的旋转值也是**弧度**，而渲染器 nlp_render 收的是**度**。
    #   默认做 弧度->度 的换算；rot_degrees=True 表示"保持弧度不换算"。
    scale = 1.0 if rot_degrees else (180.0 / math.pi)
    # LP_ROT_SCALE 可覆盖（做单位对照实验用）
    import os as _os
    scale *= float(_os.environ.get('LP_ROT_SCALE') or 1.0)
    # pool_a == 0 means there is no A pool (e.g. m_lower011); records still live below pool_b.
    rec_limit = pool_a if pool_a else pool_b
    for i in range(count):
        s = offs[i]
        if s == 0 or s + 4 > len(data) or s >= rec_limit:
            m.records.append(MotRecord(i, 0, 0, [], [], []))
            continue
        nk, mask = struct.unpack_from('<HH', data, s)
        times = [struct.unpack_from('<H', data, s + 4 + k * 2)[0] for k in range(nk)]
        body = s + 4 + nk * 2
        if body % 4:
            body += 2
        # The game counts the popcount of bits 0..8 of the RAW mask (VA 0x00740A80 系列:
        # tst #0x100 ... #0x1).  The old code read (mask>>3) which happens to agree for
        # every mask observed so far but is not what the game does.
        bits = [b for b in range(9) if (mask >> b) & 1]
        nch = len(bits)
        # ---- refs: 每条记录 nk*nch 个 u32（已由记录长度公式逐条确证）-----------
        # ★ 两个池的元素宽度不同，这是全库统计出来的硬结论：
        #     refs 全部指向 pool B 的记录 = 64277 条，**全部是 nk==1**
        #     refs 指向 pool A 的记录   = 41574 条，**全部是 nk>1**
        #   pool A : 3 浮点一个条目（值 + 两条切线）  -> 浮点偏移 = 3 * ref
        #   pool B : 1 浮点一个条目                   -> 浮点偏移 = ref
        # 混合记录里同一个通道会整条复用同一个 B 下标（常量通道），
        # 动画通道则每帧取一个新的 A 条目。
        if nk == 0 or nch == 0 or body + 4 * nk * nch > len(data):
            m.records.append(MotRecord(
                _bone_of(bone_table, i, base, per),
                nk, mask, times, [], []))
            continue
        bone = _bone_of(bone_table, i, base, per)
        refs = list(struct.unpack_from('<%dI' % (nk * nch), data, body))
        layout = channel_layout(bits, nk)
        cols, chans = [], []
        for comp, axis, off in layout:
            col = []
            ok = True
            for k in range(nk):
                r = refs[k * nch + off]
                if r & 0x80000000:
                    o = pool_b + (r & 0x7FFFFFFF) * 4
                else:
                    o = pool_a + r * 12          # ★ 条目宽 3 浮点
                if o < 0 or o + 4 > len(data):
                    ok = False
                    break
                v = struct.unpack_from('<f', data, o)[0]
                col.append(v * scale if comp == 'R' else v)
            if not ok:
                continue
            cols.append(col)
            chans.append((comp, axis))
        per_key = [[cols[c][k] for c in range(len(cols))] for k in range(nk)]
        bone = _bone_of(bone_table, i, base, per)
        # Some files carry table values that are not bone indices at all (12557 in
        # m_32024_10.mot) -- a different encoding variant.  Skip those records rather
        # than crash; 9999 is also the LP_BANK ablation's sentinel.
        if bone >= 4096:
            bone = 9999
        m.records.append(MotRecord(bone, nk, mask, times, per_key, chans))
    m.duration = max((r.times[-1] for r in m.records if r.times), default=0.0)
    # ★ 平移一律是**绝对位置**，不是增量。
    # 证据：m_00010_40 记录 0（骨盆，骨骼 2）第一帧的 6 个通道读出来是
    #   T=(7.48, 80.13, -4.28)   R=(-0.04, 0.18, 0.10)
    # 而骨骼 2 的绑定平移是 (0, 80.91, 0.77) —— 80.13 正是"身高"，
    # 只有当平移是绝对坐标时才讲得通。分层动作（nk==1）同样如此。
    # 旧代码把 nk>1 的平移当增量加到绑定上，角色因此被抬高约一倍（悬空）。
    m.absolute_translation = True
    return m


def stack_poses(poses):
    """Combine layered poses. Later entries override earlier ones per bone+component.

    Layering is how the game builds a full pose: a body motion plus hand / lower /
    skirt overlays. The layers overlap on some bones (the body motion covers bones
    2..61 while lower and skirt cover 6..11), so an explicit priority is required --
    whatever comes later in `poses` wins.
    """
    out = {}
    for pose in poses:
        for bone, ch in pose.items():
            d = out.setdefault(bone, {})
            for comp, axes in ch.items():
                d.setdefault(comp, {}).update(axes)
    return out


# Bind rotations of the rig currently being rendered, {bone_index: (rx, ry, rz)}.
# sample_present uses it to convert the ABSOLUTE rotations of layered motions
# (nk == 1) into the bind-relative deltas that world_matrices expects.  The renderer
# sets it once after loading the bones.
BIND_ROT = {}


def sample_present(mot, t):
    """Pose at time t using ONLY the components each record actually animates.

    Motion.sample() fills absent axes with 0, which is harmless when values are
    deltas but ZEROES a bone's un-animated axis when translations are absolute
    (the layered motions).  Use this one for layered/absolute motions.
    """
    out = {}
    for r in mot.records:
        if r.count == 0:
            continue
        if t <= r.times[0]:
            k0 = k1 = 0; f = 0.0
        elif t >= r.times[-1]:
            k0 = k1 = r.count - 1; f = 0.0
        else:
            lo, hi = 0, r.count - 1
            while hi - lo > 1:
                mid = (lo + hi) // 2
                if r.times[mid] <= t:
                    lo = mid
                else:
                    hi = mid
            k0, k1 = lo, hi
            span = r.times[k1] - r.times[k0]
            f = 0.0 if span == 0 else (t - r.times[k0]) / float(span)
        # a bone may own SEVERAL records (that is exactly the layered-motion layout:
        # m_skirt_000 has 4 records per bone), so MERGE -- never overwrite.
        ch = out.setdefault(r.bone, {})
        for ci, (comp, axis) in enumerate(r.channels):
            a, b = r.values[k0], r.values[k1]
            if ci >= len(a) or ci >= len(b):
                continue
            ch.setdefault(comp, {})[axis] = a[ci] + (b[ci] - a[ci]) * f
    # Layered motions (single keyframe) store ABSOLUTE poses: their translation is
    # already treated as absolute, and the rotation has to be too.  Adding the stored
    # rotation on top of the bind left the hands bent 90 deg at the wrist and the arms
    # frozen; replacing the bind rotation makes the hands point along the arm and the
    # whole pose read correctly (native/absolute_layers.py renders both side by side).
    # BIND_ROT is set once by the renderer: {bone_index: (rx, ry, rz)} of the bind pose.
    if BIND_ROT and getattr(mot, 'absolute_translation', False):
        for bone, ch in out.items():
            r = ch.get('R')
            br = BIND_ROT.get(bone)
            if r and br:
                for ax in list(r):
                    r[ax] = r[ax] - br[ax]
    return out
