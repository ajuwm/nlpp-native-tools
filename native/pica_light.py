# -*- coding: utf-8 -*-
"""PICA200 fragment lighting —— 按硬件语义实现（移植自 Azahar/Citra 软件渲染器 sw_lighting.cpp）。

为什么要有这个模块
------------------
2026-10 的做法是"按实机截图拟合"环境光/光向/光强/材质增益，拟合出来数字对但画面发灰/发平。
2026-11 已从 Citra 运行时内存拿到主菜单真值（ambient / 两盏灯的颜色与方向），
但接回自研渲染器后画面过曝发白 —— 根因是**实现端没有按 PICA200 的合成语义算**：

  primary(主色) = clamp( global_ambient + Σ_light [ (light.diffuse·max(N·L,0) + light.ambient)
                                                    × dist_atten × spot_atten ] )
  secondary(副色) = clamp( Σ_light [ (specular_0 + specular_1) × clamp_highlights
                                      × dist_atten × spot_atten ] )
  其中 specular_0 = D0_LUT(选择器点积) × light.specular0
       specular_1 = D1_LUT(选择器点积) × reflect_LUT(RR/RG/RB) × light.specular1

要点（全部有出处，见 nlpp-re/out/ref/azahar__src_video_core_renderer_software_sw_lighting.cpp）：
  * **漫反射没有 LUT**（纯 Lambert）；LUT 只用于 specular（D0/D1/反射/FR）与衰减。
  * primary / secondary 由硬件在 8bit 域 clamp 到 [0,1]，再进 TEV 合成（texture × primary + …）。
  * LUT 输入域 [-1,1]，索引 = 有符号 8bit（abs 模式 [0,1]→0..255）。
  * LUT 值 12bit fixed（/4095），条目低 12 位=值，bits12-22=与下一项的差（/2047，bit23=负号）。
    DUMP 解析器只保留了值，这里用相邻项线性插值近似（误差 < 1/4096）。

数据来源：
  * 场景真值：`out/live/true_lights_pica.json`（由 Citra RPC 内存 dump 换算，见 docs/06）
  * 每材质光照配置/LUT：`nlp_model.Material.light_cfg / luts`（来自 SMAT 的 DUMP 命令流）
"""
import json
import os

import numpy as np

# ---------------------------------------------------------------- LUT 采样器表
# Pica LightingSampler: D0=0, D1=1, FR=3, RB=4, RG=5, RR=6, SP=8+i, DA=16+i
SAMPLER_INDEX = {'D0': 0, 'D1': 1, 'FR': 3, 'RB': 4, 'RG': 5, 'RR': 6}

_SCALE = {0: 1.0, 1: 2.0, 2: 4.0, 3: 8.0, 6: 0.25, 7: 0.5}

# Light environment configuration → 可用 LUT 集合（3dbrew GPUREG_LIGHTING_CONFIG0 bits4-7）
_AVAILABLE = {
    0: ('D0', 'RR', 'SP', 'DA'),
    1: ('FR', 'RR', 'SP', 'DA'),
    2: ('D0', 'D1', 'RR', 'DA'),
    3: ('D0', 'D1', 'FR', 'DA'),
    4: ('D0', 'D1', 'RR', 'RB', 'RG', 'SP', 'DA'),
    5: ('D0', 'FR', 'RR', 'RB', 'RG', 'SP', 'DA'),
    6: ('D0', 'D1', 'FR', 'SP', 'DA'),
    7: (),
    8: ('D0', 'D1', 'FR', 'RR', 'RB', 'RG', 'SP', 'DA'),
}

# CONFIG1 各 disable 位
_CFG1_D0 = 0x10000
_CFG1_D1 = 0x20000
_CFG1_FR = 0x80000
_CFG1_RR = 0x100000
_CFG1_RG = 0x200000
_CFG1_RB = 0x400000


def _is_sampler_supported(config, name):
    """对应 Citra LightingRegs::IsLightingSamplerSupported。"""
    if name == 'D0':
        return config != 1
    if name == 'D1':
        return config not in (0, 1, 7)
    if name == 'FR':
        return config not in (0, 2, 4)
    if name == 'RR':
        return config != 3
    if name in ('RG', 'RB'):
        return config in (4, 5, 8)
    if name == 'SP':
        return config not in (2, 3)
    if name == 'DA':
        return True
    return False


class SceneLighting(object):
    """场景光照真值（来自 Citra 内存 dump 的 PICA 寄存器值）。"""

    def __init__(self, path=None, data=None):
        if data is None:
            path = path or os.environ.get('LP_PICALIGHTS_JSON') or \
                os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             'out', 'live', 'true_lights_pica.json')
            with open(path, encoding='utf-8') as f:
                data = json.load(f)
        self.raw = data
        ga = data.get('global_ambient', [0, 0, 0])
        self.global_ambient = np.asarray(ga, np.float64) / 255.0
        self.lights = []
        for lt in data.get('lights', []):
            self.lights.append(dict(
                dir=np.asarray(lt.get('dir', [0, 0, 1]), np.float64),
                pos=np.asarray(lt.get('pos', lt.get('dir', [0, 0, 1])), np.float64),
                diffuse=np.asarray(lt.get('diffuse', [255, 255, 255]), np.float64) / 255.0,
                ambient=np.asarray(lt.get('ambient', [0, 0, 0]), np.float64) / 255.0,
                spec0=np.asarray(lt.get('spec0', lt.get('specular0', [0, 0, 0])),
                                 np.float64) / 255.0,
                spec1=np.asarray(lt.get('spec1', lt.get('specular1', [0, 0, 0])),
                                 np.float64) / 255.0,
                dist_bias=float(lt.get('dist_bias', 0.0)),
                dist_scale=float(lt.get('dist_scale', 0.0)),
                spot_dir=np.asarray(lt.get('spot_dir', [0, 0, 0]), np.float64),
            ))
        opts = data.get('options', {})
        self.opts = dict(
            directional=bool(opts.get('directional', True)),
            two_sided=bool(opts.get('two_sided', False)),
            da_default=float(opts.get('da_default', 1.0)),
            sp_default=float(opts.get('sp_default', 1.0)),
        )
        # 环境变量覆盖（A/B 用）
        for env, key in (('LP_PICADIRECTIONAL', 'directional'),
                         ('LP_PICATWOSIDED', 'two_sided')):
            v = os.environ.get(env)
            if v is not None:
                self.opts[key] = v not in ('0', '', 'false', 'off')
        for env, key in (('LP_PICADA', 'da_default'), ('LP_PICASP', 'sp_default')):
            v = os.environ.get(env)
            if v is not None:
                try:
                    self.opts[key] = float(v)
                except ValueError:
                    pass


def _lerp_lut(lut, idx_u8, delta):
    """在 256 项 LUT 上做线性插值（近似 PICA 的 value + signed_diff×delta）。"""
    a = np.asarray(lut, np.float64)
    n = a.shape[0]
    i0 = idx_u8.astype(np.int64) % n
    i1 = (i0 + 1) % n
    return a[i0] + (a[i1] - a[i0]) * delta


def _sample(material, name, sel, abs_flag, scale_enum, dot_fn, sampler):
    """取一张 LUT 的标量值（每像素）；LUT 不可用/未启用时按硬件语义返回 1.0。"""
    lut = material.luts.get(sampler)
    if lut is None:
        return 1.0
    dot = dot_fn(sel)
    if abs_flag:
        # abs 模式：输入域 [0,1]；two_sided 时取 |dot|，否则 max(dot,0)
        dot = np.abs(dot)
        flr = np.floor(dot * 256.0)
        idx = np.clip(flr, 0.0, 255.0).astype(np.int64)
        delta = dot * 256.0 - idx
    else:
        flr = np.floor(dot * 128.0)
        si = np.clip(flr, -128.0, 127.0).astype(np.int64)
        delta = dot * 128.0 - si
        idx = (si & 0xFF).astype(np.int64)
    val = _lerp_lut(lut, idx, delta)
    return val * _SCALE.get(int(scale_enum), 1.0)


def compute(material, scene, nrm, posv, opts=None):
    """计算逐像素 primary / secondary（0..1，已按硬件 clamp）。

    material : nlp_model.Material（提供 light_cfg / luts；材质颜色在 NLPP 里是白色，
               因为 LIGHTi 寄存器里存的是 material×light 的乘积，见 3dbrew）
    scene    : SceneLighting
    nrm      : (N,3) 视空间单位法线
    posv     : (N,3) 视空间位置
    返回 (primary, secondary)，各 (N,3) float。
    """
    opts = opts or scene.opts
    cfg = material.light_cfg or {}
    c0 = int(cfg.get('config0', 0))
    c1 = int(cfg.get('config1', 0))
    config = (c0 >> 4) & 0xF              # Light environment configuration
    clamp_highlights = bool((c0 >> 27) & 1)
    fresnel_selector = (c0 >> 2) & 3      # 0 none / 1 primary alpha / 2 secondary alpha / 3 both
    available = _AVAILABLE.get(config, ())

    lut_abs = int(cfg.get('lutinput_abs', 0) or 0)
    lut_sel = int(cfg.get('lutinput_select', 0) or 0)
    lut_scl = int(cfg.get('lutinput_scale', 0) or 0)

    n = np.asarray(nrm, np.float64)
    n = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
    p = np.asarray(posv, np.float64)
    view = -p
    view = view / np.maximum(np.linalg.norm(view, axis=1, keepdims=True), 1e-9)
    nv = (n * view).sum(axis=1)
    flip = nv < 0
    if flip.any():
        n = n.copy()
        n[flip] = -n[flip]
        nv[flip] = -nv[flip]

    npx = n.shape[0]
    primary = np.broadcast_to(scene.global_ambient, (npx, 3)).copy()
    secondary = np.zeros((npx, 3), np.float64)

    two_sided = bool(opts.get('two_sided', False)) or bool(
        (material.light_cfg or {}).get('two_sided', False))
    num_lights = len(scene.lights)
    if not num_lights:
        return np.clip(primary, 0.0, 1.0), secondary

    max_slot = int(scene.raw.get('max_light_index', num_lights - 1))
    max_slot = min(max_slot, 7)

    def dot_fn_factory(ndl_ref, lv_ref, half_ref):
        def dot_fn(sel):
            sel = int(sel)
            if sel == 0:    # NH
                return (n * half_ref).sum(axis=1)
            if sel == 1:    # VH
                return (view * half_ref).sum(axis=1)
            if sel == 2:    # NV
                return nv
            if sel == 3:    # LN
                return ndl_ref
            return (n * half_ref).sum(axis=1)
        return dot_fn

    for slot in range(max_slot + 1):
        if slot >= num_lights:
            break
        lt = scene.lights[slot]
        if opts.get('directional', True):
            lv = lt['dir'] / max(np.linalg.norm(lt['dir']), 1e-9)
            lv = np.broadcast_to(lv, (npx, 3))
            light_vec = lv
        else:
            # positional：light_vector = light_pos + view（view = -posv，见 sw_lighting.cpp）
            light_vec = lt['pos'][None, :] + view
            norm = np.linalg.norm(light_vec, axis=1, keepdims=True)
            light_vec = light_vec / np.maximum(norm, 1e-9)
        norm_view = view
        half = norm_view + light_vec
        half = half / np.maximum(np.linalg.norm(half, axis=1, keepdims=True), 1e-9)

        ndl = (light_vec * n).sum(axis=1)
        if two_sided:
            ndl_c = np.abs(ndl)
        else:
            ndl_c = np.maximum(ndl, 0.0)

        # 距离衰减 / 聚光衰减：主菜单的材质没有上传 DA/SP LUT，这里用可配置默认值
        dist_name = 'DA%d' % slot
        dist_atten = 1.0
        if not (c1 & (1 << (24 + slot))):
            if dist_name in material.luts:
                dist_atten = float(_sample(material, 'DA', 0, True, 0,
                                           lambda sel: np.zeros(npx), dist_name))
            else:
                dist_atten = opts.get('da_default', 1.0)
        spot_name = 'SP%d' % slot
        spot_atten = 1.0
        if not (c1 & (1 << (8 + slot))) and _is_sampler_supported(config, 'SP'):
            if spot_name in material.luts:
                spot_atten = float(_sample(material, 'SP', 0, True, 0,
                                           lambda sel: np.zeros(npx), spot_name))
            else:
                spot_atten = opts.get('sp_default', 1.0)
        atten = float(dist_atten) * float(spot_atten)

        dot_fn = dot_fn_factory(ndl_c, light_vec, half)

        # ---- specular 0：D0 LUT（不可用/未启用 → 1.0）----
        d0_val = 1.0
        if not (c1 & _CFG1_D0) and 'D0' in available and _is_sampler_supported(config, 'D0'):
            d0_val = _sample(material, 'D0', (lut_sel >> 0) & 7, ((lut_abs >> 1) & 1) == 0,
                             (lut_scl >> 0) & 7, dot_fn, 'D0')
        if isinstance(d0_val, np.ndarray):
            spec0 = lt['spec0'][None, :] * d0_val[:, None]
        else:
            spec0 = lt['spec0'][None, :] * d0_val

        # ---- reflect：RR/RG/RB（不可用/未启用 → RR 的值，否则 1.0）----
        refl = np.ones((npx, 3), np.float64)
        def _refl(name, bit_off, bit_mask, field):
            if (c1 & bit_mask) == 0 and name in available and \
                    _is_sampler_supported(config, name):
                return _sample(material, name, (lut_sel >> bit_off) & 7,
                               ((lut_abs >> (bit_off + 1)) & 1) == 0,
                               (lut_scl >> bit_off) & 7, dot_fn, name)
            return None
        rr = _refl('RR', 24, _CFG1_RR, 'rr')
        rg = _refl('RG', 20, _CFG1_RG, 'rg')
        rb = _refl('RB', 16, _CFG1_RB, 'rb')
        refl[:, 0] = 1.0 if rr is None else rr
        refl[:, 1] = refl[:, 0] if rg is None else rg
        refl[:, 2] = refl[:, 0] if rb is None else rb

        # ---- specular 1：D1 LUT ----
        d1_val = 1.0
        if not (c1 & _CFG1_D1) and 'D1' in available and _is_sampler_supported(config, 'D1'):
            d1_val = _sample(material, 'D1', (lut_sel >> 4) & 7, ((lut_abs >> 5) & 1) == 0,
                             (lut_scl >> 4) & 7, dot_fn, 'D1')
        spec1 = refl * lt['spec1'][None, :] * (
            d1_val[:, None] if isinstance(d1_val, np.ndarray) else d1_val)

        # ---- Fresnel：只影响 alpha（selector 1/2/3），RGB 不受影响 ----
        # （fresnel_selector 非 0 时 D0/D1 的 alpha 会被 FR 替换；本渲染器不做光照 alpha）

        spec = spec0 + spec1
        if clamp_highlights:
            spec = spec * (ndl_c > 0.0)[:, None]

        dif = (lt['diffuse'][None, :] * ndl_c[:, None] + lt['ambient'][None, :]) * atten
        primary += dif
        secondary += spec * atten

    primary = np.clip(primary, 0.0, 1.0)
    secondary = np.clip(secondary, 0.0, 1.0)
    return primary, secondary
