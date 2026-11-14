# -*- coding: utf-8 -*-
"""
NLP native model loader: SERI descriptor -> SMES mesh + SMAT materials + BONE skeleton
+ TEXI textures.  Never leaves the game's own formats.

Formats (authoritative refs, gdkchan):
  Ohana3DS-Rebirth/Ohana/Models/NewLovePlus/Model.cs   (SMAT, BONE)
  Ohana3DS-Rebirth/Ohana/Models/NewLovePlus/Mesh.cs    (SMES)
  NLPUnpacker/NLPUnpacker/Program.cs                   (PACK, SERI)
"""
import math
import struct
import os

import nlp_pack as N
import nlp_tex as T

WRAP_NAMES = {0: 'clampToEdge', 1: 'clampToBorder', 2: 'repeat', 3: 'mirroredRepeat'}


# --------------------------------------------------------------------------- bone
class Bone(object):
    __slots__ = ('index', 'name', 'parent', 'translation', 'rotation', 'scale', 'id')

    def __init__(self, index, name):
        self.index = index
        self.name = name
        self.parent = -1
        self.translation = (0.0, 0.0, 0.0)
        self.rotation = (0.0, 0.0, 0.0)
        self.scale = (1.0, 1.0, 1.0)
        self.id = index          # 骨骼 ID（见 load_bone）

    def __repr__(self):
        return '<Bone %d id=%d %r parent=%d>' % (self.index, self.id, self.name, self.parent)


def load_bone(data):
    """BONE file -> list of Bone (flat, in file order).

    头部第三个字段（此前记作 `unk_off`）指向一张 **u32[count] 骨骼 ID 表**：
        unk[i] = 骨骼 i 的 ID
    `img.bin` 里所有部件的这张表**逐字节一致**（全局 ID 空间，254 个 ID 互不相同），
    而 `.mot` 每条记录的目标骨骼存的就是这个 **ID**，不是扁平下标。
    ID 范围 10..322，直接当下标用会把动作套到完全错误的骨头上
    （例：ID 303 -> 骨骼 6「腰」，ID 274 -> 骨骼 210「右手」）。
    """
    if data[:4] != b'BONE':
        raise ValueError('not a BONE file: %r' % data[:4])
    count, id_off, parent_off, bone_off = struct.unpack_from('<IIII', data, 4)
    bones = []
    for i in range(count):
        b = Bone(i, 'bone_%d' % i)
        b.id = struct.unpack_from('<I', data, id_off + i * 4)[0]
        b.parent = struct.unpack_from('<i', data, parent_off + i * 4)[0]
        o = bone_off + i * 0x24
        b.translation = struct.unpack_from('<3f', data, o)
        # ★ BONE 文件里的旋转是**弧度**，而渲染器 nlp_render.euler_xyz/rotate_axis
        #   收的是**度**（内部自己做 math.radians）。这里统一转成度。
        #   证据：手骨 49..63 / 210..224 的绑定旋转是 (±0.25, 0, ∓1.709)；
        #   当 1.7° 用时整套 hand rig 等于没转，动作一摆手臂手就相对手腕扭掉
        #   （用户的"手是扭曲的"）；当 1.7 rad = 98° 用时才是正常的手骨朝向。
        b.rotation = tuple(math.degrees(v) for v in struct.unpack_from('<3f', data, o + 0x0C))
        b.scale = struct.unpack_from('<3f', data, o + 0x18)
        bones.append(b)
    return bones


def build_node_binding(bones):
    """Ohana Mesh.buildNodeBinding: node index -> bone tree index (root first, DFS)."""
    children = {}
    for b in bones:
        children.setdefault(b.parent, []).append(b.index)
    order = []
    # roots: parent that is not itself a valid bone index
    valid = set(b.index for b in bones)
    roots = [b.index for b in bones if b.parent not in valid]
    if not roots:
        roots = [0]

    def walk(i):
        order.append(i)
        for c in children.get(i, []):
            walk(c)

    for r in roots:
        walk(r)
    # any bone not reached (cycles / orphans) appended at the end
    for b in bones:
        if b.index not in order:
            order.append(b.index)
    return order


# --------------------------------------------------------------------------- smat
class TextureMapper(object):
    __slots__ = ('border_color', 'min_filter', 'mag_filter', 'wrap_u', 'wrap_v', 'texture')

    def __init__(self):
        self.border_color = (0, 0, 0, 0)
        self.min_filter = 1
        self.mag_filter = 0
        self.wrap_u = 0
        self.wrap_v = 0
        self.texture = None

    def __repr__(self):
        return '<TexMapper %s wrapU=%s wrapV=%s>' % (
            self.texture, WRAP_NAMES.get(self.wrap_u, self.wrap_u),
            WRAP_NAMES.get(self.wrap_v, self.wrap_v))


class Material(object):
    def __init__(self, index):
        self.index = index
        self.name = 'material_%d' % index
        self.material_color = {}
        self.mappers = {}          # unit -> TextureMapper
        self.tex_indices = []
        # ★ 2026-10-15 新增：把原来丢弃的块留下来（结构已解，语义逐步补）
        self.combiner = []         # COMB: [(w0,w1,w2,w3,w4), ...] 每个 TEV stage 5 个 u32
        self.combiner_head = 0     # COMB 头部 u32（全库恒为 0x000000ff）
        self.lut = ()              # LUTS: 7 个 u32
        self.dump = b''            # DUMP: 内层载荷（PICA200 GPU 命令流）
        # ★ 2026-10-18 重解：DUMP 不是"每材质一条漫反射渐变"，而是一小段
        #   PICA200 GPU 命令流（[参数...][命令头]，头 = reg|mask<<16|count<<20|consecutive<<31）。
        #   实测每个材质写入的是 LIGHTING_CONFIG0/1、LUT_INDEX、某个 LUT 的 256 项数据、
        #   以及 LUTINPUT_ABS/SELECT/SCALE —— 具体写的是哪张 LUT（D0/FR/...）由文件决定。
        self.light_regs = {}       # 原始寄存器值 {reg: value}
        self.light_cfg = {}        # 语义化后的光照配置（见 _decode_light_cfg）
        self.luts = {}             # LUT 名 -> 256 项 float 列表（value/4096；未写入处 0.0）
        self.lut_written = {}      # LUT 名 -> 256 项 bool
        self.lut_d0 = []           # 兼容旧字段：第一张写过的 LUT 的 256 项
        self.lut_d1 = []
        self.lut_values = []       # 兼容旧字段：= FR（没有则第一张）曲线，供 compose 借曲线
        self.shader = None         # SHADER 块（若有）

    # --- 颜色便捷访问（MATC 顺序是 A,R,G,B，已转成 RGBA）---
    def color(self, which):
        return self.material_color.get(which, (0, 0, 0, 0))

    def diffuse_rgb(self):
        return tuple(v / 255.0 for v in self.color('diffuse')[:3])

    def ambient_rgb(self):
        return tuple(v / 255.0 for v in self.color('ambient')[:3])

    def specular_rgb(self):
        return tuple(v / 255.0 for v in self.color('specular0')[:3])

    def emission_rgb(self):
        return tuple(v / 255.0 for v in self.color('emission')[:3])

    @property
    def tev_stage_count(self):
        return len(self.combiner)

    def diffuse_mapper(self):
        """The mapper to draw as the base colour.

        Some face materials stack a `*_col` colour/region map on unit 0 and put the
        actual shape-and-alpha texture on unit 1 (m_face: unit0 = m_face_col, a striped
        ramp; unit1 = m_face_02, clean eyebrows with alpha).  Sampling unit 0 directly
        produces stripes, so when unit 0 is a `*_col` map prefer the next unit that is
        not.  Replace this heuristic once the TEV combiner is known.
        """
        m0 = self.mappers.get(0)
        if m0 and m0.texture and '_col' in m0.texture:
            for u in sorted(self.mappers):
                if u == 0:
                    continue
                mp = self.mappers[u]
                if mp.texture and '_col' not in mp.texture:
                    return mp
        return m0

    def __repr__(self):
        return '<Material %d %r tex=%r>' % (self.index, self.name, self.tex_indices)

    # ---- 光照 LUT 便捷访问（2026-10-18 重解 DUMP 后新增）----
    def light_lut(self, name):
        """取某张已启用 LUT 的 256 项曲线（value/4096），否则 None。

        会按 LIGHTING_CONFIG0 的"光环境配置"和 CONFIG1 的 disable 位判断是否真的可用：
        不可用的 LUT 在 PICA200 上读出来恒为 1.0，不能拿曲线去乘。
        """
        cfg = self.light_cfg or {}
        if not cfg:
            return None
        if name not in set(cfg.get('lut_available') or ()):
            return None
        if cfg.get('lut_disabled', {}).get(name):
            return None
        return self.luts.get(name)

    def lut_input(self, name):
        """(abs?, selector, scale) —— 1 表示不做 abs（PICA200 语义：0=启用 abs）。"""
        cfg = self.light_cfg or {}
        if not cfg:
            return True, 0, 0
        fld = {'D0': 0, 'D1': 1, 'SP': 2, 'FR': 3, 'RB': 4, 'RG': 5, 'RR': 6}.get(name, 0)
        abs_flag = (cfg.get('lutinput_abs', 0) >> (1 + 4 * fld)) & 1
        sel = (cfg.get('lutinput_select', 0) >> (4 * fld)) & 0x7
        scl = (cfg.get('lutinput_scale', 0) >> (4 * fld)) & 0x7
        return bool(abs_flag), sel, scl


LUT_TABLE_NAMES = {0: 'D0', 1: 'D1', 3: 'FR', 4: 'RB', 5: 'RG', 6: 'RR'}
for _i in range(8):
    LUT_TABLE_NAMES[8 + _i] = 'SP%d' % _i
    LUT_TABLE_NAMES[16 + _i] = 'DA%d' % _i


def _mask_bytes(mask):
    """GPU 命令头 bits16-19：每个 bit 表示参数的一个字节被写入。"""
    v = 0
    for i in range(4):
        if mask & (1 << i):
            v |= 0xFF << (8 * i)
    return v


def _parse_dump_commands(blob):
    """DUMP 载荷（PICA200 GPU 命令流）-> (寄存器 dict, LUT 曲线, 写入 mask)。

    命令格式（3dbrew）：[参数0][命令头]，命令头 bits0-15=寄存器号，
    bits16-19=参数按字节写 mask，bits20-27=额外参数量，bit31=连续写模式
    （寄存器号逐个 +1；否则所有参数重复写同一寄存器，LUT_DATA 就是 FIFO）。
    LUT 数据条目 24 位：bits0-11 = 值（fixed0.0.12），bits12-23 = 与下一项的绝对差
    （fixed1.0.11，用于硬件线性插值）。LUT 共 256 项，输入域 [-1,1]。
    """
    if len(blob) < 8:
        return {}, {}, {}
    words = struct.unpack_from('<%dI' % (len(blob) // 4), blob, 0)
    regs, luts, written = {}, {}, {}
    lut_index = 0
    i, n = 0, len(words)
    while i + 1 < n:
        param0, header = words[i], words[i + 1]
        reg = header & 0xFFFF
        mask = (header >> 16) & 0xF
        count = (header >> 20) & 0xFF
        consecutive = (header >> 31) & 1
        if not (0x100 <= reg <= 0x2FF) or mask == 0 or i + 2 + count > n:
            i += 1
            continue
        params = (param0,) + words[i + 2:i + 2 + count]
        mask_bits = _mask_bytes(mask)
        for k, val in enumerate(params):
            r = reg + k if consecutive else reg
            eff = val & mask_bits
            if r == 0x1C5:
                lut_index = eff
            elif 0x1C8 <= r <= 0x1CF:
                table = (lut_index >> 8) & 0x1F
                start = lut_index & 0xFF
                name = LUT_TABLE_NAMES.get(table, 'T%d' % table)
                arr = luts.get(name)
                if arr is None:
                    arr = [0.0] * 256
                    luts[name] = arr
                    written[name] = [False] * 256
                arr[start] = (eff & 0xFFF) / 4096.0
                written[name][start] = True
                lut_index = (lut_index & ~0xFF) | ((start + 1) & 0xFF)
            else:
                old = regs.get(r, 0)
                regs[r] = (old & ~mask_bits) | eff
        i += 2 + count
    return regs, luts, written


def _decode_light_cfg(regs):
    """把 LIGHTING_CONFIG0/1 / LUTINPUT_* 解释成人类可读的光照配置。"""
    c0 = regs.get(0x1C3, 0)
    c1 = regs.get(0x1C4, 0)
    env = (c0 >> 4) & 0xF
    available = {
        0: ('D0', 'RR', 'SP', 'DA'),
        1: ('FR', 'RR', 'SP', 'DA'),
        2: ('D0', 'D1', 'RR', 'DA'),
        3: ('D0', 'D1', 'FR', 'DA'),
        4: ('D0', 'D1', 'FR', 'RB', 'RG', 'RR', 'SP', 'DA'),
        5: ('D0', 'FR', 'RB', 'RG', 'RR', 'SP', 'DA'),
        6: ('D0', 'D1', 'FR', 'SP', 'DA'),
        8: ('D0', 'D1', 'FR', 'RB', 'RG', 'RR', 'SP', 'DA'),
    }.get(env, ())
    disabled = {
        'D0': bool((c1 >> 16) & 1), 'D1': bool((c1 >> 17) & 1), 'FR': bool((c1 >> 19) & 1),
        'RB': bool((c1 >> 20) & 1), 'RG': bool((c1 >> 21) & 1), 'RR': bool((c1 >> 22) & 1),
    }
    return dict(
        config0=c0, config1=c1, env=env,
        fresnel_selector=(c0 >> 2) & 3,
        shadow_enabled=bool(c0 & 1),
        lut_available=available, lut_disabled=disabled,
        lut_index=regs.get(0x1C5, 0),
        lutinput_abs=regs.get(0x1D0, 0),
        lutinput_select=regs.get(0x1D1, 0),
        lutinput_scale=regs.get(0x1D2, 0),
    )


def load_smat(data, texture_names):
    """SMAT file -> list of Material.  texture_names = SERI `texi` list, in order."""
    if data[:4] != b'SMAT':
        raise ValueError('not a SMAT file: %r' % data[:4])
    count = struct.unpack_from('<I', data, 4)[0]
    offsets = struct.unpack_from('<%dI' % count, data, 8)
    materials = []
    for mtl in range(count):
        m = Material(mtl)
        p = offsets[mtl]
        has_data = True
        while has_data and p + 8 <= len(data):
            magic = data[p:p + 4].decode('latin1')
            length = struct.unpack_from('<I', data, p + 4)[0]
            start = p + 8
            q = start
            if magic == 'STAT':
                # 36 bytes = 9 u32.  Word 1 is the PICA200 alpha-test config
                # (register 0x104): bit0 = enabled, bits 4-7 = compare function,
                # bits 8-15 = reference.  Verified against real data:
                #   hair   m_00_200 -> 0x0000067F = on,  greaterOrEqual, ref=6   (cutout)
                #   cloth  m_01_012 -> 0x00000100 = off                          (opaque)
                #   face   m_face   -> 0x00000100 = off  (all 4 materials)
                if length >= 8:
                    w = struct.unpack_from('<I', data, start + 4)[0]
                    m.alpha_test_enabled = bool(w & 1)
                    m.alpha_test_func = (w >> 4) & 0xF
                    m.alpha_test_ref = (w >> 8) & 0xFF
            elif magic == 'MATC':
                names = ('emission', 'ambient', 'diffuse', 'specular0', 'specular1',
                         'constant0', 'constant1', 'constant2', 'constant3',
                         'constant4', 'constant5')
                for nm in names:
                    # SMAT stores colors as A,R,G,B (verified: `00ffffff` is white,
                    # not the cyan you get by reading it as R,G,B,A; `ff000000` is
                    # black, which is a sane default for constant0..5).
                    a, r, g, b = data[q:q + 4]
                    m.material_color[nm] = (r, g, b, a)
                    q += 4
            elif magic == 'TEXU':
                units = struct.unpack_from('<I', data, q)[0]
                q += 4
                for unit in range(units):
                    ti = struct.unpack_from('<I', data, q)[0]
                    q += 4
                    mapper = TextureMapper()
                    a, r, g, b = data[q:q + 4]   # stored A,R,G,B, same as MATC
                    mapper.border_color = (r, g, b, a)
                    q += 4
                    mapper.min_filter = data[q]; q += 1
                    mapper.mag_filter = data[q]; q += 1
                    mapper.wrap_u = data[q]; q += 1
                    mapper.wrap_v = data[q]; q += 1
                    q += 4   # 0x0
                    q += 4   # 0x0
                    name = texture_names[ti] if ti < len(texture_names) else None
                    mapper.texture = name
                    if unit == 0:
                        m.name = (name or '').replace('.texi', '').replace('model__Textures__', '') or m.name
                    m.mappers[unit] = mapper
                    m.tex_indices.append(ti)
            elif magic == 'COMB':
                # ★ 实测结构：长度恒为 (1 + 5N) 个 u32，N = TEV stage 数（1/2/3）。
                #   全库分布 24/44/64 字节 -> 6/11/16 个 u32。w[0] 全库恒为 0x000000ff。
                w = struct.unpack_from('<%dI' % (length // 4), data, q)
                m.combiner_head = w[0] if w else 0
                m.combiner = [tuple(w[1 + 5 * i:6 + 5 * i]) for i in range((len(w) - 1) // 5)]
            elif magic == 'LUTS':
                if length >= 28:
                    m.lut = struct.unpack_from('<7I', data, q)
            elif magic == 'DUMP':
                # ★ 2026-10-18 重解：DUMP 是**一小段 PICA200 GPU 命令流**（嵌套容器：
                #   'DUMP' + u32 内长 + 载荷），不是"每材质一条漫反射渐变曲线"。
                #   命令 = [参数0][命令头]；命令头 bits0-15=寄存器号，
                #   bits16-19=按字节写 mask，bits20-27=额外参数量，bit31=连续写。
                #   实测每个材质写：LIGHTING_CONFIG0(0x1C3)/CONFIG1(0x1C4)、
                #   LUT_INDEX(0x1C5)、某张 LUT 的 256 项（0x1C8 FIFO）、
                #   LUTINPUT_ABS/SELECT/SCALE(0x1D0/0x1D1/0x1D2)。
                #   旧解析把 LUT 数据的前两个 u32 当成寄存器、并丢掉末尾的 LUTINPUT
                #   命令，且把 24 位条目当成"低12位值 + 高4位半差分"——都是错的。
                if length >= 8 and data[q:q + 4] == b'DUMP':
                    inner = struct.unpack_from('<I', data, q + 4)[0]
                    m.dump = data[q + 8:q + 8 + min(inner, length - 8)]
                else:
                    m.dump = data[q:q + length]
                regs, luts, written = _parse_dump_commands(m.dump)
                m.light_regs = regs
                m.light_cfg = _decode_light_cfg(regs)
                m.luts = luts
                m.lut_written = written
                # 兼容旧字段：优先 FR（PICA200 里真正被这套引擎大量写的表），
                # 否则第一张写过的表。仅供 compose 的"皮肤借用曲线"逻辑使用。
                compat = luts.get('FR') or (next(iter(luts.values()), []) if luts else [])
                m.lut_values = list(compat)
                m.lut_d0 = list(compat)
                m.lut_d1 = []
            elif magic == 'SHADER':
                m.shader = data[q:q + length]
            else:
                has_data = False
            p = start + length
        materials.append(m)
    return materials


# --------------------------------------------------------------------------- smes
class Primitive(object):
    """One (SKIN, BONI, IDX) group.  Every group carries its OWN bone palette, so a
    mesh is really a list of independently-skinned draws."""
    __slots__ = ('index', 'skinning_mode', 'nodes', 'indices', 'vertices')

    def __init__(self, index, skinning_mode, nodes, indices):
        self.index = index
        self.skinning_mode = skinning_mode
        self.nodes = nodes          # BONI node ids (index into the mesh's palette)
        self.indices = indices      # u16 indices into the mesh vertex buffer
        self.vertices = {}          # vertex-buffer index -> vertex dict

    @property
    def tris(self):
        return len(self.indices) // 3

    def __repr__(self):
        return '<Prim %d tris=%d skin=%d pal=%d>' % (
            self.index, self.tris, self.skinning_mode, len(self.nodes))


class Mesh(object):
    def __init__(self, index):
        self.index = index
        self.name = 'mesh_%d' % index
        self.material_index = 0
        self.render_priority = 0
        self.stride = 0
        self.vertex_buffer_offset = 0
        self.prims = []

    @property
    def tris(self):
        return sum(p.tris for p in self.prims)

    @property
    def vertex_indices(self):
        s = set()
        for p in self.prims:
            s.update(p.indices)
        return s

    def __repr__(self):
        return '<Mesh %d tris=%d prims=%d mat=%d>' % (
            self.index, self.tris, len(self.prims), self.material_index)


def read_vertex(data, vbuf_off, stride, vi, nodes, skinning_mode, node_binding, bones):
    vo = vbuf_off + vi * stride
    if vo + 44 > len(data):
        return None
    vert = {'pos': struct.unpack_from('<3f', data, vo),
            'nrm': struct.unpack_from('<3f', data, vo + 12),
            'tan': struct.unpack_from('<3f', data, vo + 24),
            'uv': struct.unpack_from('<2f', data, vo + 36),
            'bones': [], 'weights': []}
    if bones and nodes:
        # ★ 顶点步长 0x40 = 位置12 + 法线12 + 切线12 + UV8 (=44) + **4 个骨骼字节** + **4 个 float 权重**
        #   （44 + 4 + 16 = 64 ✓）。实测每个顶点的 4 个权重和恰为 1.0，例如
        #       boneBytes=[0,1,2,0]  weights=[0.230, 0.750, 0.020, 0.000]
        #   旧代码只读第 1 组（weight 0.23），归一化后让整个顶点只跟 1 根骨 ——
        #   绑定姿势下蒙皮恒等（wp·binv = I）所以完全看不出来，一摆动作关节就塌成尖刺。
        nw = max(1, min(4, (stride - 48) // 4)) if stride > 48 else 1
        if os.environ.get('LP_OLDWEIGHTS'):
            nw = 1                      # 复现旧行为：只取第 1 组权重（用于 A/B 对照）
        b = data[vo + 44:vo + 48]
        for k in range(nw):
            if b[k] < len(nodes):
                vert['bones'].append(node_binding[nodes[b[k]]])
            else:
                vert['bones'].append(0)
        for k in range(nw):
            vert['weights'].append(struct.unpack_from('<f', data, vo + 48 + k * 4)[0])
    return vert


def load_smes(data, bones=None, want_vertices=True):
    """SMES file -> list of Mesh.  `bones` (optional) enables skinning indices."""
    if data[:4] != b'SMES':
        raise ValueError('not a SMES file: %r' % data[:4])
    data_table_offset, mesh_count = struct.unpack_from('<II', data, 4)
    node_binding = build_node_binding(bones) if bones else []

    meshes = []
    for mi in range(mesh_count):
        o = 0x0C + mi * 0x20
        (vec3_entries, vec3_off, attr_count, prim_count,
         vbuf_off, idx_off, render_prio, mat_index) = struct.unpack_from('<IIIIIIII', data, o)

        mesh = Mesh(mi)
        mesh.material_index = mat_index
        mesh.render_priority = render_prio
        mesh.stride = ((idx_off - vbuf_off) // prim_count) if prim_count else 0x40
        mesh.vertex_buffer_offset = vbuf_off

        # A mesh is a stream of (SKIN, BONI, IDX) groups -- accumulate them all.
        p = idx_off
        cur_mode = 0
        cur_nodes = []
        guard = 0
        while p + 8 <= len(data) and guard < 4096:
            guard += 1
            magic = data[p:p + 4].decode('latin1', 'replace')
            length = struct.unpack_from('<I', data, p + 4)[0]
            start = p + 8
            if magic == 'SKIN':
                cur_mode = struct.unpack_from('<I', data, start)[0]
            elif magic == 'BONI':
                n = struct.unpack_from('<H', data, start)[0]
                cur_nodes = list(struct.unpack_from('<%dH' % n, data, start + 2))
            elif magic == 'IDX ':
                n = struct.unpack_from('<H', data, start + 2)[0]
                idx = list(struct.unpack_from('<%dH' % n, data, start + 4))
                mesh.prims.append(Primitive(len(mesh.prims), cur_mode,
                                            list(cur_nodes), idx))
            else:
                break
            p = start + length

        if want_vertices and mesh.stride:
            for pr in mesh.prims:
                vmap = {}
                for vi in sorted(set(pr.indices)):
                    v = read_vertex(data, vbuf_off, mesh.stride, vi,
                                    pr.nodes, pr.skinning_mode, node_binding, bones)
                    if v:
                        vmap[vi] = v
                pr.vertices = vmap
        meshes.append(mesh)
    return meshes


# --------------------------------------------------------------------------- model
class NativeModel(object):
    def __init__(self, name):
        self.name = name
        self.bones = []
        self.node_binding = []
        self.meshes = []
        self.materials = []
        self.textures = {}      # name -> dict(w,h,fmt,rgba)
        self.tex_order = []
        self.descriptor = {}

    def stats(self):
        tris = sum(m.tris for m in self.meshes)
        verts = sum(len(p.vertices) for m in self.meshes for p in m.prims)
        return dict(meshes=len(self.meshes), prims=sum(len(m.prims) for m in self.meshes),
                    tris=tris, verts=verts, bones=len(self.bones),
                    materials=len(self.materials), textures=len(self.textures))

    def __repr__(self):
        return '<NativeModel %r %r>' % (self.name, self.stats())


def load_model(img_path, model_name, load_textures=True):
    """Load a full model straight out of img.bin."""
    mdl_name = model_name + '.mdl'
    pack = None
    with open(img_path, 'rb') as f:
        for p in N.iter_packs(img_path):
            if any(pf.name == mdl_name for pf in p.files):
                pack = p
                break
    if pack is None:
        raise ValueError('model descriptor %r not found in %s' % (mdl_name, img_path))

    with open(img_path, 'rb') as f:
        f.seek(pack.strptrs_off)
        strptrs = struct.unpack_from('<%dI' % pack.file_count, f.read(pack.file_count * 4), 0)
        f.seek(pack.strtab_off)
        strtab = f.read(max(0x1000, min(0x100000, max(0, pack.end - pack.strtab_off))))

        by_name = {pf.name: pf for pf in pack.files}
        seri = N.parse_seri_block(N.pack_bytes(f, by_name[mdl_name]), strptrs, strtab)
        if seri is None:
            raise ValueError('failed to parse %s' % mdl_name)

        m = NativeModel(model_name)
        m.descriptor = seri.params
        tex_order = [t[1] if isinstance(t, tuple) else t for t in (seri.get('texi') or [])]
        m.tex_order = tex_order

        if seri.get('bone') and seri.get('bone') in by_name:
            m.bones = load_bone(N.pack_bytes(f, by_name[seri.get('bone')]))
            m.node_binding = build_node_binding(m.bones)
        if seri.get('smes') and seri.get('smes') in by_name:
            m.meshes = load_smes(N.pack_bytes(f, by_name[seri.get('smes')]), m.bones)
        if seri.get('smat') and seri.get('smat') in by_name:
            m.materials = load_smat(N.pack_bytes(f, by_name[seri.get('smat')]), tex_order)

        if load_textures:
            descs, payloads = {}, {}
            for pf in pack.files:
                if pf.signature_str == 'TEXI':
                    s = N.parse_seri_block(N.pack_bytes(f, pf), strptrs, strtab)
                    if s:
                        descs[pf.name] = s.params
                elif pf.signature_str == 'TEX ':
                    payloads[pf.name] = N.pack_bytes(f, pf)
            for nm in tex_order:
                d = descs.get(nm)
                if not d or nm not in payloads:
                    continue
                try:
                    fmt = d['format']
                    # ★ 用**数据量反推格式**（不靠声明 ✓）：
                    #   TEX 的数据量 = w*h*bpp*1.25（多带一级 mipmap）。
                    #   实测 m_face_col 声明为 LA4（1 字节/像素）✗，但数据量 163840 对应
                    #   256*256*2*1.25 ✓ ⇒ 实际是 2 字节格式 ✗ —— 按声明解只会读到一半数据，
                    #   结果就是一张噪点图 ✓。所以声明与数据量矛盾时，以数据量为准 ✓。
                    if d.get('w') and d.get('h'):
                        bpp_dec = T._BPP.get(T.FMT.get(fmt, ''), 0)
                        exp = int(d['w'] * d['h'] * bpp_dec * 1.25)
                        got = len(payloads[nm])
                        if bpp_dec and got > exp * 1.2:
                            real = got / float(d['w'] * d['h'] * 1.25)
                            cand = min(T._BPP.items(), key=lambda kv: abs(kv[1] - real))
                            if cand[1] > bpp_dec:
                                for k, v in T.FMT.items():
                                    if v == cand[0]:
                                        fmt = k
                                        break
                    m.textures[nm] = dict(w=d['w'], h=d['h'], fmt=fmt,
                                          rgba=T.decode(payloads[nm], d['w'], d['h'], fmt))
                except Exception as e:
                    m.textures[nm] = dict(w=d.get('w'), h=d.get('h'), fmt=d.get('format'), error=str(e))
    return m
