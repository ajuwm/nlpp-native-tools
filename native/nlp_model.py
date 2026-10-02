# -*- coding: utf-8 -*-
"""
NLP native model loader: SERI descriptor -> SMES mesh + SMAT materials + BONE skeleton
+ TEXI textures.  Never leaves the game's own formats.

Formats (authoritative refs, gdkchan):
  Ohana3DS-Rebirth/Ohana/Models/NewLovePlus/Model.cs   (SMAT, BONE)
  Ohana3DS-Rebirth/Ohana/Models/NewLovePlus/Mesh.cs    (SMES)
  NLPUnpacker/NLPUnpacker/Program.cs                   (PACK, SERI)
"""
import struct
import os

import nlp_pack as N
import nlp_tex as T

WRAP_NAMES = {0: 'clampToEdge', 1: 'clampToBorder', 2: 'repeat', 3: 'mirroredRepeat'}


# --------------------------------------------------------------------------- bone
class Bone(object):
    __slots__ = ('index', 'name', 'parent', 'translation', 'rotation', 'scale')

    def __init__(self, index, name):
        self.index = index
        self.name = name
        self.parent = -1
        self.translation = (0.0, 0.0, 0.0)
        self.rotation = (0.0, 0.0, 0.0)
        self.scale = (1.0, 1.0, 1.0)

    def __repr__(self):
        return '<Bone %d %r parent=%d>' % (self.index, self.name, self.parent)


def load_bone(data):
    """BONE file -> list of Bone (flat, in file order)."""
    if data[:4] != b'BONE':
        raise ValueError('not a BONE file: %r' % data[:4])
    count, unk_off, parent_off, bone_off = struct.unpack_from('<IIII', data, 4)
    bones = []
    for i in range(count):
        b = Bone(i, 'bone_%d' % i)
        b.parent = struct.unpack_from('<i', data, parent_off + i * 4)[0]
        o = bone_off + i * 0x24
        b.translation = struct.unpack_from('<3f', data, o)
        b.rotation = struct.unpack_from('<3f', data, o + 0x0C)
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
            elif magic in ('COMB', 'LUTS'):
                pass
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
        b = data[vo + 44:vo + 48]
        nw = skinning_mode if skinning_mode > 0 else 1
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
                    m.textures[nm] = dict(w=d['w'], h=d['h'], fmt=d['fmt'] if 'fmt' in d else d['format'],
                                          rgba=T.decode(payloads[nm], d['w'], d['h'], d['format']))
                except Exception as e:
                    m.textures[nm] = dict(w=d.get('w'), h=d.get('h'), fmt=d.get('format'), error=str(e))
    return m
