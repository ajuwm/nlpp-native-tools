# -*- coding: utf-8 -*-
"""PICA200 风格光照着色：每材质自己的 LUT 渐变 + 环境项 + 方向光。

为什么是这个模型
----------------
SMAT 的 `DUMP` 块里实测到（`nlp_model.load_smat` 解析）：

    reg 0x01C3 LIGHTING_CONFIG0   = 0x000F   （12/12 个材质一致）
    reg 0x01C4 LIGHTING_CONFIG1   = 0x0004
    reg 0x01C5 LIGHTING_LUT_INDEX = 0x000F
    reg 0x01C8 LIGHTING_LUT_DATA0 = 0x07FF
    + 每材质一张 16 位定点渐变表 (D0, D1)

也就是说：**光照模式是全局统一的一套，每个材质只是喂给它的曲线不同**。
曲线（实测）：
    m_00_200 头发 58 组，起点 0x3002, 0x5011, 0x802c, ...   （陡）
    m_01_005 蓝裙 31 组，起点 0x1239, 0x2233, 0x222a, ...   （缓）
    m_face   脸部  无 LUT  → 走线性回退

PICA200 的逐顶点光照语义（3dbrew `GPUREG_LIGHTi_*` / `LIGHTING_*`）：
    color = 全局环境 + Σ_i ( 材质漫反射 × 灯光漫反射 × D0(n·L_i) )
            + Σ_i ( 材质高光 × 灯光高光 × D1(n·H_i) )   ← 本实现暂不启用 D1
其中 D0/D1 就是上面那两条 LUT 曲线（PICA200 用 "LUT 输入" 把 n·L 映射成一条可控曲线，
这就是实机那种"亮而柔、明暗分层"的来源）。

本模块只做数学，不碰光栅化；`nlp_render.draw()` 在 `LP_SHADE=pica` 时调用它。
"""
import os

import numpy as np

# ------------------------------------------------------------------ 布光参数
# ★ 2026-10-18 重做：DUMP 重新解出后，真实机制是
#     color = mat_ambient·全局环境 + mat_diffuse·light·max(N·L,0)
#             + mat_specular0·light_spec·D0(N·H)     （头发等材质才有 D0）
#             + mat_specular0·light_spec·FR(N·V)·2   （皮肤/布料上的 Fresnel 边缘光）
#   D0/FR 曲线来自 DUMP 里真实写入的 256 项 LUT（值 /4096），不再当"漫反射明暗阶"用。
#   方向光/环境光强度游戏里由场景设置（资源里没有），所以这几个值是**按实机参照拟合**的。
DEFAULTS = dict(
    # ★ 2026-10-18 参考图量化（_work/ref_light/report.md）：
    #   实机是"ambient 主导 + 极弱顶光"——face 左右亮度比 0.995、上下 0.994，
    #   布料上/下只差 3~4%，等效 ambient:diffuse ≈ 0.86~0.95 : 0.05~0.14，
    #   且没有下巴投影、没有硬 toon 色阶。所以主光要弱、方向要偏正上方。
    ambient=(0.895, 0.880, 0.860),          # 全局环境光颜色（× 材质 MATC ambient）
    light_dir=(0.02, 0.60, 0.80),          # 主光方向（世界空间，指向光源）—— 顶前 30~40°
    light_color=(0.15, 0.145, 0.14),      # 主光颜色 × 强度（只占总亮度 ~5~8%）
    spec_color=(0.55, 0.55, 0.55),         # 高光/菲涅耳的光色
    spec_map_boost=6.0,                    # 有高光强度贴图（*_spe）时 D0 高光的倍率
    sheen_tint=(1.10, 1.0, 1.14),          # 发丝 sheen 的蓝灰偏色（实机 sheen 略偏蓝）
    out_tint=(0.856, 0.9411, 1.0),        # 最终白平衡微调（按实机参照的全身均值标定）
    spec_strength=1.0,                     # D0 高光总强度
    fresnel_strength=0.30,                 # FR 边缘光总强度（掠射角过强会发白，已减半）
    toon=0.0,                              # >0 时对 N·L 做软边分档（宽度）
    toon_strength=1.0,                     # 分档幅度
    lut_input_scale=127.0,                 # 点积 -> LUT 索引的缩放（PICA200 signed 8-bit）
)


def _f(name, default):
    v = os.environ.get('LP_' + name.upper())
    if v is None:
        return default
    try:
        return float(v)
    except ValueError:
        return default


def _vec(name, default):
    v = os.environ.get('LP_' + name.upper())
    if not v:
        return default
    try:
        parts = [float(x) for x in v.replace(' ', '').split(',')]
        return tuple(parts) if len(parts) == 3 else default
    except ValueError:
        return default


class LightSet(object):
    """全局布光（可用 LP_AMBIENT / LP_LIGHTDIR / LP_LIGHTCOLOR / ... 覆盖）。"""

    def __init__(self, ambient=None, light_dir=None, light_color=None,
                 spec_color=None, spec_strength=None, fresnel_strength=None,
                 toon=None, toon_strength=None):
        self.ambient = np.array(ambient or _vec('ambient', DEFAULTS['ambient']), np.float64)
        d = np.array(light_dir or _vec('lightdir', DEFAULTS['light_dir']), np.float64)
        n = np.linalg.norm(d)
        self.light_dir = d / n if n > 1e-9 else d
        self.light_color = np.array(light_color or _vec('lightcolor',
                                                        DEFAULTS['light_color']), np.float64)
        self.spec_color = np.array(spec_color or _vec('speccolor',
                                                      DEFAULTS['spec_color']), np.float64)
        self.spec_map_boost = _f('specmapboost', DEFAULTS['spec_map_boost'])
        self.sheen_tint = np.array(_vec('sheentint', DEFAULTS['sheen_tint']), np.float64)
        self.out_tint = np.array(_vec('tint', DEFAULTS['out_tint']), np.float64)
        self.spec_strength = _f('specstrength',
                                DEFAULTS['spec_strength'] if spec_strength is None else spec_strength)
        self.fresnel_strength = _f('fresnelstrength',
                                   DEFAULTS['fresnel_strength']
                                   if fresnel_strength is None else fresnel_strength)
        self.toon = _f('toon', DEFAULTS['toon'] if toon is None else toon)
        self.toon_strength = _f('toonstrength',
                                DEFAULTS['toon_strength'] if toon_strength is None else toon_strength)
        self.lut_input_scale = _f('lutindexscale', DEFAULTS['lut_input_scale'])
        # ★ 实机真值（Citra 内存读出）：LP_LIGHTS_JSON 指向 {"ambient":[...],"lights":[...]}
        #   有它时走多光源路径（每项 dir/color/spec），覆盖拟合的单光参数。
        self.multi = None
        _lj = os.environ.get('LP_LIGHTS_JSON')
        if _lj and os.path.exists(_lj):
            try:
                import json as _json
                _d = _json.load(open(_lj, encoding='utf-8'))
                if 'ambient' in _d:
                    self.ambient = np.asarray(_d['ambient'], np.float64)
                self.multi = [dict(dir=np.asarray(l['dir'], np.float64),
                                   color=np.asarray(l['color'], np.float64),
                                   spec=np.asarray(l.get('spec', [1, 1, 1]), np.float64))
                              for l in _d.get('lights', [])]
                if not self.multi:
                    self.multi = None
            except Exception:
                self.multi = None


def _lut_array(d0):
    """把 16 位定点 LUT 归一化成 0..1 的查找表；空表返回 None（走线性回退）。"""
    if not d0:
        return None
    a = np.asarray(d0, np.float64) / 65535.0
    return np.clip(a, 0.0, 1.0)


_LUT_CACHE = {}


def lut_of(material):
    """材质 -> 归一化 LUT 曲线（带缓存）。没有 LUT 的材质返回 None（走线性回退）。"""
    if material is None:
        return None
    key = id(material)
    if key not in _LUT_CACHE:
        vals = getattr(material, 'lut_values', None)
        if not vals:
            _LUT_CACHE[key] = None
        else:
            a = np.asarray(vals, np.float64)
            if os.environ.get('LP_LUTRAW', '1') != '1':
                lo, hi = float(a.min()), float(a.max())
                a = (a - lo) / (hi - lo) if hi - lo > 1e-9 else np.zeros_like(a)
            # ★ 默认保留 LUT 的**绝对量级**（LP_LUTRAW=1）：
            #   裙子最大 0.139、头发最大 0.261 —— 这个差别本身就是实机"不同材质亮度不同"的来源，
            #   按形状归一化会把它抹掉 ✗。绝对量级由 LP_LIGHTCOLOR（全局增益）统一缩放。
            _LUT_CACHE[key] = np.clip(a, 0.0, 1.0)
    return _LUT_CACHE[key]


def sample(table, t):
    """在 LUT 上按 t∈[0,1] 线性插值；table=None 时线性回退。"""
    t = np.clip(t, 0.0, 1.0)
    if table is None:
        return t
    n = table.shape[0]
    if n == 1:
        return np.full_like(t, table[0])
    x = t * (n - 1)
    i0 = np.floor(x).astype(np.int64)
    i0 = np.clip(i0, 0, n - 2)
    f = x - i0
    return table[i0] * (1.0 - f) + table[i0 + 1] * f


def material_ambient(material):
    if material is None:
        return np.ones(3)
    return np.array(material.ambient_rgb(), np.float64)


# ------------------------------------------------------------------ 每材质增益
# 一组全局光照无法同时满足衣服和皮肤（实测：裙子要更亮、皮肤要更暗），
# 所以按**材质名/贴图名前缀**给增益，并允许**逐通道**（皮肤的暖色调就是 R 通道差）。
# 环境变量 LP_MATGAIN="前缀:g" 或 "前缀:gr,gg,gb"，多条用 ';' 分隔（按出现顺序匹配）。
_GAIN_TABLE = None

# ★ 拟合出来的默认增益（目标 = 实机截图按材质中位色差）
#   任务009：光照改为高环境光/弱主光后重新拟合（tools/regress.py 四部位全 PASS）。
#   注意：这里的增益是在 LP_CONTRAST=0.85 下拟合的，两者不能分开改 ✗。
# 皮肤材质通用增益（按种类）：_ne 颈、_ar 臂、_04_ 身体、face 脸、hand 手
_SKIN_FALLBACK = [
    ('_ne', (1.231, 0.943, 0.800)),
    ('_ar', (1.159, 0.988, 0.838)),
    ('_04_', (1.159, 0.941, 0.808)),
    ('face', (1.231, 1.018, 0.866)),
    ('hand', (1.159, 0.922, 0.784)),
]

DEFAULT_MATGAIN = (
                   'm_04_:1.159,0.941,0.808;m_face:1.231,1.018,0.866;m_00_200:1.441,1.355,1.462;m_01_005_ar:1.159,0.988,0.838;m_01_005_ne:1.231,0.943,0.800;m_hand:1.159,0.922,0.784')

# ★ 网格对象级增益。
#   `_bo` / `_wa` 图集同时包含蓝裙和白色围裙，按图集名前缀调增益会连带改白围裙 ✗。
#   键 = 去掉 `textures__` 前缀的材质名 + `#` + 网格号；网格号来自 SERI 的 mesh index。
#   该增益在最终光照结果上统一应用，pica 与回退光照路径都只乘一次 ✓。
#
#   ★ 两条路径必须分开：pica 路径的 shade_vertices() 以前还会乘图集级 MATGAIN，
#     回退光照路径不会。现在图集级增益已去掉，两条路径的校准值就不同了 ✓。
DEFAULT_OBJGAIN_UNLIT = (
    # 按【unlit 默认路径】拟合（任务004）。默认 LP_UNLIT=_bo,_wa 时使用。
    # ★ 任务009：当前 8777/最终默认走【受光路径】，这张表没有随新光照重拟合。
    #   若强制 LP_UNLIT='_bo,_wa'，请先重新拟合，不要把 LIT 表直接套过来 ✗。
    'm_01_005_bo#0:1.239,1.196,1.149;'
    'm_01_005_wa#5:1.306,1.217,1.148'
)

DEFAULT_OBJGAIN_LIT = (
    # 按【受光路径】拟合（LP_UNLIT='__none__' 时使用）。
    # 任务009：光照/对比度改完后重新拟合；蓝裙/白围裙仍按对象区分
    # （同一图集里既有蓝裙也有白围裙，不能按图集名统一给增益 ✗）。
    # 白围裙/蓝裙在精确中位拟合后又整体压了 3~4%，用于把全身均值拉回实机；
    # 两者中位差仍远小于 regress 的 ±15 阈值 ✓。
    'm_01_005_bo#0:1.239,1.196,1.149;'
    'm_01_005_wa#5:1.744,1.277,1.121;'
    # 白围裙对象（bo#1 领口白边、bo#2 胸兜、wa#6 裙摆+花边）不吃蓝裙校正。
    # 受光路径的白色布料原本偏亮 ~10%：以前的"白围裙 PASS"是靠蓝胸兜把
    # R 通道拉低的混合中位数换来的 ✗；去掉蓝染后按实机目标 (225,238,240)
    # 重新标定这两个对象，使胸兜与裙摆落在同一白色上 ✓。
    'm_01_005_bo#1:1.175,1.158,1.148;'
    'm_01_005_bo#2:1.175,1.158,1.148;'
    'm_01_005_wa#6:1.175,1.158,1.148'
)

_OBJGAIN_OVERRIDE = None
_OBJGAIN_UNLIT_TABLE = None
_OBJGAIN_LIT_TABLE = None


def _parse_gain_table(text):
    table = []
    for item in (text or '').split(';'):
        if ':' not in item:
            continue
        k, v = item.rsplit(':', 1)
        try:
            parts = [float(x) for x in v.split(',')]
        except ValueError:
            continue
        g = parts * 3 if len(parts) == 1 else parts
        if len(g) == 3:
            table.append((k.strip().lower(), np.array(g, np.float64)))
    return table


def _objgain_table(lit=True):
    """LP_OBJGAIN 覆盖两条路径；否则按 lit/unlit 各用一张默认表。"""
    global _OBJGAIN_OVERRIDE, _OBJGAIN_UNLIT_TABLE, _OBJGAIN_LIT_TABLE
    if os.environ.get('LP_OBJGAIN') is not None:
        if _OBJGAIN_OVERRIDE is None:
            _OBJGAIN_OVERRIDE = _parse_gain_table(os.environ['LP_OBJGAIN'])
        return _OBJGAIN_OVERRIDE
    if lit:
        if _OBJGAIN_LIT_TABLE is None:
            _OBJGAIN_LIT_TABLE = _parse_gain_table(DEFAULT_OBJGAIN_LIT)
        return _OBJGAIN_LIT_TABLE
    if _OBJGAIN_UNLIT_TABLE is None:
        _OBJGAIN_UNLIT_TABLE = _parse_gain_table(DEFAULT_OBJGAIN_UNLIT)
    return _OBJGAIN_UNLIT_TABLE


def _name_is_unlit(name):
    low = (name or '').lower()
    prefixes = [k.strip().lower() for k in
                os.environ.get('LP_UNLIT', '_bo,_wa').split(',') if k.strip()]
    return any(k == '*' or k in low for k in prefixes)


def object_gain(material, mesh_index, lit=None):
    """返回网格对象的逐通道增益；没有配置时返回 (1,1,1)。

    lit=None 时按 LP_UNLIT 推断该对象走哪条光照路径；`nlp_render.draw()`
    会在确定 shade 之后再调用，并显式传入 lit=shade is not None ✓。
    """
    one = np.ones(3)
    if material is None or mesh_index is None:
        return one
    name = (getattr(material, 'name', '') or '').lower().rsplit('/', 1)[-1]
    if name.startswith('textures__'):
        name = name[len('textures__'):]
    key = '%s#%d' % (name, int(mesh_index))
    if lit is None:
        lit = not _name_is_unlit(name)
    for prefix, g in _objgain_table(lit):
        if prefix == '*' or prefix == key:
            return g
    return one


def _gain_table():
    global _GAIN_TABLE
    if _GAIN_TABLE is None:
        _GAIN_TABLE = []
        for item in os.environ.get('LP_MATGAIN', DEFAULT_MATGAIN).split(';'):
            if ':' not in item:
                continue
            k, v = item.rsplit(':', 1)
            try:
                parts = [float(x) for x in v.split(',')]
            except ValueError:
                continue
            g = parts * 3 if len(parts) == 1 else parts
            if len(g) == 3:
                _GAIN_TABLE.append((k.strip().lower(), np.array(g, np.float64)))
    return _GAIN_TABLE


def material_gain(material):
    """返回逐通道增益 (3,)。"""
    one = np.ones(3)
    if material is None:
        return one
    name = (getattr(material, 'name', '') or '').lower()
    # 材质名有两种形式：`m_face/Textures__m_face_01`（带模型前缀）
    # 和 `Textures__n_face_01`（只有贴图名）⇒ 先统一剥掉目录/Textures__ 前缀再匹配。
    if '/' in name:
        name = name.rsplit('/', 1)[-1]
    if name.startswith('textures__'):
        name = name[len('textures__'):]
    for prefix, g in _gain_table():
        if prefix and prefix in name:
            return g
    # ★ 角色前缀回退：n_/r_ 的皮肤/身体材质（命名与 m_ 相同）沿用 m_ 的标定。
    #   注意：只对"同一含义"的材质有效（_04_/face/hand/_ne/_ar），服装图集不吃这条。
    if name[:2] in ('n_', 'r_'):
        alt = 'm_' + name[2:]
        for prefix, g in _gain_table():
            if prefix and (prefix in alt) and (
                    '_04_' in alt or 'face' in alt or 'hand' in alt
                    or alt.endswith('_ne') or alt.endswith('_ar')):
                return g
    # ★ 2026-10-22 皮肤材质通用回退（修"脖子颜色不对"老毛病）：
    #   `out_tint` 是按"皮肤增益已把 R 拉回来"标定的；没进增益表的服装
    #   （如校服 m_01_000）的 _ne/_ar 会只吃 out_tint ⇒ R 被压 0.856、
    #   G/B 高 ⇒ 脖子变成灰白（实测 220,211,211）。皮肤就是皮肤，
    #   按材质种类套用同一套标定增益（精确表命中时仍优先）。
    for _suf, _g in _SKIN_FALLBACK:
        if _suf == '_ne' and name.endswith('_ne'):
            return np.array(_g, np.float64)
        if _suf == '_ar' and name.endswith('_ar'):
            return np.array(_g, np.float64)
        if _suf == '_04_' and '_04_' in name:
            return np.array(_g, np.float64)
        if _suf == 'face' and 'face' in name:
            return np.array(_g, np.float64)
        if _suf == 'hand' and 'hand' in name:
            return np.array(_g, np.float64)
    return one


# ------------------------------------------------------------------ 新光照模型
_LUT_SCALE = {0: 1.0, 1: 2.0, 2: 4.0, 3: 8.0, 6: 0.25, 7: 0.5}


def _scale_mul(sel):
    return _LUT_SCALE.get(int(sel), 1.0)


def _sample_curve(curve, t, abs_input=True, index_scale=127.0):
    """PICA200 语义：点积 ∈ [-1,1] -> signed 8-bit 索引 -> 256 项 LUT。

    abs 使能时先取绝对值；索引 = round(dot*scale) & 0xFF（负值自然落到 129..255）。
    """
    a = np.asarray(curve, np.float64)
    x = np.abs(t) if abs_input else t
    idx = np.rint(np.asarray(x, np.float64) * float(index_scale)).astype(np.int64) & 0xFF
    return a[idx]


def _dot_for_selector(sel, n, v, h, ndl, nv):
    sel = int(sel)
    if sel == 0:        # N·H
        return (n * h).sum(axis=1)
    if sel == 1:        # V·H
        return (v * h).sum(axis=1)
    if sel == 2:        # N·V
        return nv
    if sel == 3:        # L·N
        return ndl
    return (n * h).sum(axis=1)


def _toon_diffuse(ndl, soft, strength):
    """把 N·L 压成两档（暗/亮）软边台阶，保持中位亮度基本不变。"""
    soft = max(1e-3, float(soft))
    b = np.clip((ndl - (0.5 - soft)) / (2.0 * soft), 0.0, 1.0)
    b = b * b * (3.0 - 2.0 * b)
    # 目标两档：暗 0.45、亮 1.0（相对连续值，幅度用 strength 控制）
    target = 0.45 + 0.55 * b
    return ndl * (1.0 - float(strength)) + target * float(strength)


def shade_pixels(pos_view, nrm, material, lights, light_dir_view, spec_mask=None):
    """逐像素光照（PICA200 fragment lighting 的近似）。

    pos_view: (N,3) 视空间位置；nrm: (N,3) 视空间法线；light_dir_view: 指向光源的视空间单位向量。
    spec_mask: (N,) 0..1，高光强度贴图（*_spe）的采样值，乘在 D0/FR 项上。
    返回 (N,3) 颜色倍率。
    """
    n = np.asarray(nrm, np.float64)
    n = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
    p = np.asarray(pos_view, np.float64)
    v = -p
    v = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)
    nv = (n * v).sum(axis=1)
    flip = nv < 0.0
    if flip.any():
        n = n.copy()
        n[flip] = -n[flip]
        nv[flip] = -nv[flip]
    multi = getattr(lights, 'multi', None)
    if multi:
        return _shade_pixels_multi(n, v, nv, material, lights, multi, spec_mask)
    ndl = np.clip(n @ np.asarray(light_dir_view, np.float64), 0.0, 1.0)
    if lights.toon > 0.0:
        ndl = _toon_diffuse(ndl, lights.toon / 10.0, lights.toon_strength)

    amb_m = np.asarray(material_ambient(material), np.float64)
    shade = lights.ambient[None, :] * amb_m[None, :]
    dif_m = np.asarray(material.diffuse_rgb(), np.float64)
    spec_m = np.asarray(material.specular_rgb(), np.float64)
    # ★ 2026-10-18 参考图量化：实机的布料明暗主要**烘焙在贴图里**，动态方向光在布料上
    #   只有 ~3% 的垂直调制；而皮肤是实打实的 Lambert（暗部 P10 需要方向光才压得下去）。
    #   ⇒ 布料（specular=0）的 N·L 权重调低，皮肤（specular≠0）保持 1.0。
    #   这样既能压住"围裙/裙子上下亮度比超标"，又能保住皮肤暗部判据。
    dif_w = 1.0
    if not np.any(spec_m > 0.0):
        dif_w = _f('diffcloth', 0.45)
    elif 'm_face' in (getattr(material, 'name', '') or ''):
        # 实机脸部几乎无方向性明暗（L75≈L90），我们的脸高光偏白 ⇒ 单独压低脸部方向光
        dif_w = _f('diffface', 1.0)
    shade = shade + lights.light_color[None, :] * (dif_m[None, :] * (ndl[:, None] * dif_w))

    if np.any(spec_m > 0.0):
        h = np.asarray(light_dir_view, np.float64)[None, :] + v
        h = h / np.maximum(np.linalg.norm(h, axis=1, keepdims=True), 1e-9)
        spec_lit = (lights.spec_color[None, :] * spec_m[None, :])
        if spec_mask is not None:
            # ★ 高光贴图：黑底亮条纹 ⇒ 只在贴图亮的地方有高光（头发的亮带）
            spec_lit = spec_lit * np.asarray(spec_mask, np.float64)[:, None]
        # --- D0：Blinn-Phong 高光分布（头发等材质把 D0 打开）---
        d0 = material.light_lut('D0')
        if d0 is not None:
            abs_flag, sel, scl = material.lut_input('D0')
            t = _dot_for_selector(sel, n, v, h, ndl, nv)
            curve = _sample_curve(d0, t, abs_input=not abs_flag,
                                  index_scale=lights.lut_input_scale)
            strength = lights.spec_strength
            d0_lit = spec_lit
            if spec_mask is not None:
                # ★ 只有带高光强度贴图（*_spe）的材质（头发）用高强度 + 蓝灰偏色：
                #   实机发丝 sheen RGB(73,69,77)、L≈1.56×base，是蓝灰不是白色高光；
                #   而 spe 贴图本身就是"哪里该亮"的 mask，所以可以放心放大。
                strength *= getattr(lights, 'spec_map_boost', 1.0)
                d0_lit = d0_lit * np.asarray(
                    getattr(lights, 'sheen_tint', (1.0, 1.0, 1.0)), np.float64)[None, :]
            shade = shade + d0_lit * (curve[:, None] * (strength * _scale_mul(scl)))
        # --- FR：Fresnel 边缘光（皮肤/布料的反射项）---
        fr = material.light_lut('FR')
        if fr is not None:
            abs_flag, sel, scl = material.lut_input('FR')
            t = _dot_for_selector(sel, n, v, h, ndl, nv)
            curve = _sample_curve(fr, t, abs_input=not abs_flag,
                                  index_scale=lights.lut_input_scale)
            shade = shade + spec_lit * (curve[:, None] * (lights.fresnel_strength * _scale_mul(scl)))

    # ★ 实机 PICA 语义：fragment lighting 输出是 8bit，主色/副色在 TEV 前 clamp 到 1.0。
    #   LP_SHADECLAMP=1 时按此截断（否则旧拟合参数下会过曝）。
    if _f('shadeclamp', 0.0) > 0.0:
        shade = np.clip(shade, 0.0, 1.0)
    g = material_gain(material)
    # ★ 上界从 1.5 提到可配置：旧管线在 shade 上截 1.5，正好把"高光头部"截平
    #   ⇒ 头发那条亮带永远出不来（基色×1.5 已经饱和）✗。高光需要 shade>1.5 才可见。
    shade_max = _f('shademax', 8.0)
    tint = getattr(lights, 'out_tint', None)
    if tint is not None:
        shade = shade * np.asarray(tint, np.float64)[None, :]
    return np.clip(shade * g[None, :], 0.0, shade_max)


def _shade_pixels_multi(n, v, nv, material, lights, multi, spec_mask=None):
    """多光源版本（实机真值：最多 8 盏灯，主菜单 2 盏）。

    multi: [{'dir': (视空间单位向量), 'color': 漫反射色, 'spec': 高光色}, ...]
    环境项只算一次（全局 ambient），漫反射按每盏灯累加；高光 LUT 用第 1 盏灯的方向。
    """
    amb_m = np.asarray(material_ambient(material), np.float64)
    dif_m = np.asarray(material.diffuse_rgb(), np.float64)
    spec_m = np.asarray(material.specular_rgb(), np.float64)
    dif_w = 1.0
    if not np.any(spec_m > 0.0):
        dif_w = _f('diffcloth', 0.45)
    elif 'm_face' in (getattr(material, 'name', '') or ''):
        dif_w = _f('diffface', 1.0)
    shade = lights.ambient[None, :] * amb_m[None, :]
    first_dir = None
    first_ndl = None
    for lt in multi:
        dirv = np.asarray(lt['dir'], np.float64)
        dnorm = np.linalg.norm(dirv)
        if dnorm > 1e-9:
            dirv = dirv / dnorm
        ndl = np.clip(n @ dirv, 0.0, 1.0)
        if lights.toon > 0.0:
            ndl = _toon_diffuse(ndl, lights.toon / 10.0, lights.toon_strength)
        shade = shade + np.asarray(lt['color'], np.float64)[None, :] * (
            dif_m[None, :] * (ndl[:, None] * dif_w))
        if first_dir is None:
            first_dir = dirv
            first_ndl = ndl
    # ★ PICA：主色(diffuse+ambient) 是 8bit，TEV 前 clamp 1.0；副色(specular) 另行相加
    if _f('shadeclamp', 0.0) > 0.0:
        shade = np.clip(shade, 0.0, 1.0)
    if np.any(spec_m > 0.0):
        h = first_dir[None, :] + v
        h = h / np.maximum(np.linalg.norm(h, axis=1, keepdims=True), 1e-9)
        spec_col = np.asarray(multi[0].get('spec', (1.0, 1.0, 1.0)), np.float64)
        spec_lit = spec_col[None, :] * spec_m[None, :]
        if spec_mask is not None:
            spec_lit = spec_lit * np.asarray(spec_mask, np.float64)[:, None]
        d0 = material.light_lut('D0')
        if d0 is not None:
            abs_flag, sel, scl = material.lut_input('D0')
            t = _dot_for_selector(sel, n, v, h, first_ndl, nv)
            curve = _sample_curve(d0, t, abs_input=not abs_flag,
                                  index_scale=lights.lut_input_scale)
            strength = lights.spec_strength
            d0_lit = spec_lit
            if spec_mask is not None:
                strength *= getattr(lights, 'spec_map_boost', 1.0)
                d0_lit = d0_lit * np.asarray(
                    getattr(lights, 'sheen_tint', (1.0, 1.0, 1.0)), np.float64)[None, :]
            shade = shade + d0_lit * (curve[:, None] * (strength * _scale_mul(scl)))
        fr = material.light_lut('FR')
        if fr is not None:
            abs_flag, sel, scl = material.lut_input('FR')
            t = _dot_for_selector(sel, n, v, h, first_ndl, nv)
            curve = _sample_curve(fr, t, abs_input=not abs_flag,
                                  index_scale=lights.lut_input_scale)
            shade = shade + spec_lit * (curve[:, None] * (lights.fresnel_strength *
                                                          _scale_mul(scl)))
    g = material_gain(material)
    shade_max = _f('shademax', 8.0)
    tint = getattr(lights, 'out_tint', None)
    if tint is not None:
        shade = shade * np.asarray(tint, np.float64)[None, :]
    return np.clip(shade * g[None, :], 0.0, shade_max)


def shade_vertices(nrm, material, lights):
    """（旧接口，保留兼容；新路径用 shade_pixels()）逐顶点近似。"""
    n = np.asarray(nrm, np.float64)
    ndl = n @ lights.light_dir
    lut_abs = getattr(lights, 'lut_abs', 1)
    if lut_abs:
        ndl = np.abs(ndl)
    t = np.clip(ndl, 0.0, 1.0) * getattr(lights, 'lut_scale', 1.0)
    ramp = sample(lut_of(material), t)                     # D0 曲线
    lut_gain = float(os.environ.get('LP_LUTGAIN', '3.0'))
    if lut_gain != 1.0:
        ramp = ramp * lut_gain
    amb = material_ambient(material) * lights.ambient       # 材质环境 × 全局环境
    diff = ramp[:, None] * lights.light_color[None, :]      # 曲线 × 光色
    g = material_gain(material)
    return np.clip((amb[None, :] + diff) * g[None, :], 0.0, 1.5)


# ============================================================ 手机版三渲二（toon）移植
# 公式来自手机版 Unity shader（FlightUnit/Chara）反编译结果（见 _work/toon_shading.py）：
#   ndl  = clamp(dot(N, key_dir))
#   toon = ToonRamp(ndl)             # 64x8 ramp，v=0.25 那一行
#   lin  = albedo^2.2
#   col  = key_scale * (lin * (max(lin,0.5) + toon*(1-max(lin,0.5)))) + key_offset
#   col /= max(1 - rim*rim_scale, .003)
#   out  = col^(1/2.2)
_TOON_CACHE = {}


def _load_toon_ramp(path):
    """加载 64x8 toon ramp（手机版贴图），取 v=0.25 行做 1D LUT。"""
    import os as _os
    from PIL import Image
    if not path or not _os.path.isfile(path):
        return None
    if path in _TOON_CACHE:
        return _TOON_CACHE[path]
    im = Image.open(path).convert('RGB')
    a = np.asarray(im, np.float32) / 255.0
    row = a[int((1.0 - 0.25) * (a.shape[0] - 1))]
    lut = np.ascontiguousarray(row[:, 0])
    _TOON_CACHE[path] = lut
    return lut


def _toon_preset():
    import os as _os
    name = _os.environ.get('LP_TOONPRESET', 'game')
    if name == 'mobile':
        name = 'game'
    return TOON_PRESETS.get(name, TOON_PRESETS['game'])


TOON_PRESETS = {
    'game': dict(
        key_dir=(0.30, 0.62, 0.66), key_scale=(1.02, 0.97, 0.90),
        key_offset=(0.06, 0.05, 0.045),
        rim_dir=(-0.30, 0.40, -0.85), rim_scale=(0.30, 0.26, 0.22),
        rim_offset=(0.0, 0.0, 0.0)),
}


def toon_pixels(nrm_world, pos_world, cam_world, albedo, mat_name):
    """逐像素手机版三渲二着色；nrm/pos/cam 都是世界空间，albedo 为 gamma 空间(0~1)。

    返回 gamma 空间颜色 (N,3)。眼睛材质不在这里处理（保留原有自发光+高光）。
    """
    import os as _os
    cfg = _toon_preset()
    n = np.asarray(nrm_world, np.float64)
    n = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
    key_dir = np.array(cfg['key_dir'], np.float64)
    key_dir = key_dir / max(np.linalg.norm(key_dir), 1e-9)
    ramp = _load_toon_ramp(_os.environ.get('LP_TOONRAMP', ''))
    skin_ramp = _load_toon_ramp(_os.environ.get('LP_TOONRAMP_SKIN', ''))
    low = (mat_name or '').lower()
    is_skin = (('_04_' in low) or ('face' in low) or ('hand' in low)
               or low.endswith('_ne') or low.endswith('_ar'))
    use = skin_ramp if (is_skin and skin_ramp is not None) else ramp
    ndl = np.clip(n @ key_dir, 0.0, 1.0)
    if use is not None:
        idx = np.clip((ndl * (len(use) - 1)).astype(np.int32), 0, len(use) - 1)
        toon = use[idx]
    else:
        toon = 0.5 + 0.5 * ndl
    lin = np.power(np.clip(albedo, 0.0, 1.0), 2.2)
    mx = np.maximum(lin, 0.5)
    col = lin * (mx + toon[:, None] * (1.0 - mx))
    col = np.asarray(cfg['key_scale'], np.float64)[None, :] * col + \
        np.asarray(cfg['key_offset'], np.float64)[None, :]
    col = np.clip(col, 0.0, 1.0)

    # rim light（手机版 shader 的 rim 项简化版）
    V = np.asarray(cam_world, np.float64)[None, :] - np.asarray(pos_world, np.float64)
    V = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-9)
    nv = np.abs(np.sum(n * V, axis=1))
    rim_dir = np.array(cfg['rim_dir'], np.float64)
    rim_dir = rim_dir / max(np.linalg.norm(rim_dir), 1e-9)
    rim_dl = n @ rim_dir
    rim_term = np.clip(rim_dl * (1.0 - nv) * 0.5 + 0.5, 0.0, 1.0)
    rim_tex = np.clip((rim_term - 0.55) * 2.2, 0.0, 1.0)
    rim_col = np.asarray(cfg['rim_scale'], np.float64)[None, :]
    col = col / np.maximum(1.0 - rim_tex[:, None] * rim_col, 0.003)
    return np.power(np.clip(col, 0.0, 1.0), 1.0 / 2.2)
