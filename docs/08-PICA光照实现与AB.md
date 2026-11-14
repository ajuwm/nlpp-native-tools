# 08 · PICA200 光照按硬件语义接回（2026-11-14）

> 接续 `docs/07`（第一版 A/B：只把真值 ambient+双灯塞进旧拟合管线）。
> 用户反馈："全身颜色不对、太白、眼睛和脸肤色明显不一样、像相机过曝"。
> 本文记录：**根因定位**、**按 PICA200 硬件语义重写光照**、**A/B 结果**、以及**还剩的未知量**。

---

## 1. 根因（已定位）

用户的"过曝 / 脸眼肤色不一致"不是"真值算错了"，而是**实现端把真值接进了旧拟合管线**：

| # | 问题 | 证据/位置 |
|---|---|---|
| A | **真值版仍保留 2026-10 的截图拟合补丁**：逐材质增益 `MATGAIN`（脸 1.231/1.018/0.866、身体 1.159/0.941/0.808）、白平衡 `out_tint`(0.856,0.941,1.0)、皮肤 G/B 上限 `LP_SKINCAP`(G≤205,B≤171) | `native/nlp_shade.py` DEFAULTS/DEFAULT_MATGAIN；`native/nlp_render.py` LP_SKINCAP 段 |
| B | **多光源路径用的是裸 Lambert**（`ambient + Σ light_color·N·L`），没有 PICA 的 8bit primary/secondary clamp 语义；真值 ambient≈0.98 且两灯 diffuse 0.4/0.8 直接相加 ⇒ 大面积 >1 ⇒ 白 | `nlp_shade._shade_pixels_multi` |
| C | **眼睛被强制 unlit**（`shade=1`），而脸/身体吃光照 ⇒ 同一张脸上眼白/眼周与皮肤必然两个颜色 | `nlp_render.py` `LP_EYEUNLIT` 段 |
| D | 渲染器没有 **TEV 合成**（`texture × primary + secondary`），而是用自定公式 `tex × shade`，高光/FR 强度是拟合值 | `nlp_render.draw()` / `nlp_shade.shade_pixels` |

## 2. 硬件语义参照（新增，已落盘）

PICA200 fragment lighting 的权威实现是 Citra/Azahar 的软件渲染器：
- 已下载到 `out/ref/azahar-emu__src_video_core_renderer_software_sw_lighting.cpp`（GPLv2，仅作语义参照，不拷贝进产品）
- `out/ref/azahar-emu__src_video_core_pica_regs_lighting.h`（寄存器/枚举/位域）
- 3dbrew 原文 `out/ref/3dbrew_Internal_Registers.wiki`（`GPUREG_LIGHTING_*` / LUT 编码 / CONFIG0 的 LUT 可用性表）

**关键事实（修正了接力文档 P0-A 的猜测）**：
1. **漫反射不走 LUT**：`diffuse = light.diffuse × max(N·L,0) (+ light.ambient)`，纯 Lambert。
   LUT（D0/D1/FR/RB/RG/RR）只用于 **specular / 反射 / Fresnel**（D0 = Blinn-Phong 分布）。
2. `primary = clamp(global_ambient + Σ_light diffuse_terms, 0..1)`，
   `secondary = clamp(Σ specular_terms, 0..1)`，**都在 8bit 域各自 clamp**，再由 TEV 合成。
3. `global_ambient = material_emission + material_ambient × scene_ambient`（寄存器里的初始 primary）。
4. LIGHTi 颜色寄存器存的是 **material × light 的乘积**（3dbrew："usually set to material_diffuse\*lightX_diffuse"）。
   本作材质 MATC 的 ambient/diffuse/specular0 基本都是 255 白（材质分析见 `_notes/2026-11-14-materials.md`）。
5. LUT 值 12bit fixed（value/4095），条目 bits12-22 = 与下一项的有符号差（/2047，bit23 负号）。
6. 光照配置是**每材质**的（SMAT DUMP 写入 CONFIG0/CONFIG1 byte2/LUT_INDEX/LUT_DATA/LUTINPUT_*），
   场景级 CONFIG1 的衰减/阴影位由 `0x121174` 从 `ctx+0x1010/0x1014` 位图生成。

**主菜单真值再核对（对 `0x1138CC` 反汇编逐条确认，`out/_reports/dis_1138CC.txt`）**：
`0x1138CC` 里 light env 条目基址 = `ctx+0xFF8 + i*0x50`，字段：
`+0x20` ambient / `+0x30` diffuse / `+0x40` spec0&spec1（× 场景灯光对象调色板字节 0..255），
方向来自 `obj+0x254` 的 12 个 float（经 `0x120838` 归一化 + `0x121BE4` 矩阵变换）。
⇒ 与 `docs/06` 的数值一致：Light0 diffuse=(102,178,178) spec=(255,204,204)，Light1 diffuse=(204,178,178) spec=(12,0,76)。

**衰减/阴影开关（0x121174 反汇编）**：CONFIG1 基础值 = `0xFF04FFFF`（shadow/spot/dist 全 disabled）+
每盏灯用 `ctx[0x164+i]`（启用）、`ctx[0x16c+i]`、`ctx[0x174+i]`、`ctx[0x17c+i]` 清位来**打开** shadow/spot/dist。
实测主菜单 ctx：`ctx+0x1010=0x3`（2 盏灯启用）、`ctx+0x1014=0`；两盏灯的 flag 数组全 0
⇒ **shadow、spot attenuation、distance attenuation 全部关闭** ⇒ 灯光项不乘任何衰减（=1.0），也不需要 DA/SP LUT。

## 3. 实现

新增 `native/pica_light.py`（纯 Python/numpy，按 `sw_lighting.cpp` 移植）：
- `SceneLighting`：从 `out/live/true_lights_pica.json` 读真值（global_ambient + 每灯 diffuse/ambient/spec0/spec1/dir）。
- `compute(material, scene, nrm, posv)`：逐像素算 `primary / secondary`（0..1，硬件 clamp），
  覆盖 LUT 可用性（CONFIG0 env 1..8 表）、CONFIG1 disable 位、LUTINPUT abs/select/scale、
  D0/D1/反射/FR 采样语义、clamp_highlights、two_sided、distance/spot 衰减（可配置默认值）。
- 接入 `native/nlp_render.draw()`（`LP_PICALIGHT=1`）：`rgb = texture × primary (+ spec_mask × secondary)`，
  并**跳过所有 2026-10 拟合补丁**（MATGAIN/out_tint/skin cap/眼睛 unlit/前后光增益），
  眼睛与脸/身体走同一条光照路径。

新增真值文件：`out/live/true_lights_pica.json`（由 `lights_mainmenu.json` 换算）。

复跑（`LP_PICADA/LP_PICASP` 是这两个未知衰减项的默认值 A/B 开关）：
```powershell
$py = "C:\Users\han\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe"
$env:LP_PICALIGHT='1'
$env:LP_PICADA='1'; $env:LP_PICASP='1'   # 灯光全开（按 0x121174 证据）
& $py nlpp-re\native\compose.py --img data\img.bin --outfit m_01_005 --w 600 --h 860 `
      --out nlpp-re\out\ab\v600_pica_lit.png --bg bs_0000_00_00.jpg
```

## 4. A/B 结果（600×860，同角色/动作/相机）

图：
- `out/ab/r600_compare.png`：默认角色（红格裙）三版并排 —— fitted / PICA 灯光衰减=0 / PICA 灯光全开
- `out/ab/v600_*`：换成 `m_01_005`（蓝色女仆装，和实机参考 `_work/ref_light/game_ref.png` 同一套衣服）

按材质中位 RGB（红格裙版；fitted vs PICA）：

| 材质 | fitted 版 | PICA 灯光=0 | PICA 灯光全开 | 实机参考（女仆装） |
|---|---|---|---|---|
| 脸 skin（m_face_01） | (252,198,167) | (248,202,193) | (254,223,214) | **(254,204,174)** |
| 腿（m_04_000） | (248,195,168) | (249,206,199) | (255,227,220) | (252,199,172) |
| 手（m_hand） | (249,194,164) | (249,207,199) | (255,231,222) | — |
| 脖子（_ne） | (255,177,150) | (244,187,180) | (250,200,192) | (254,204,176) |
| 头发（m_00_200） | (42,43,48) | (32,30,30) | (33,33,33) | 实机 base (46,45,49) + 蓝灰 sheen |

**结论**：
1. PICA 灯光全开（理论上与 0x121174 证据一致）**明显过曝**（脸 G+19/B+40，腿 G+31/B+48），
   与用户抱怨完全一致 ⇒ 现有真值 + 这条"全开"假设里至少还有一处语义没对上（见 §5）。
2. PICA 灯光衰减=0（纯 ambient = (249,229,229)/255）**不再过曝**，肤色回到"贴图×0.898"，
   但比实机参考偏"粉白"（B 通道高 ~18-25）。
3. 旧 fitted 版在肤色上和实机参考只差 ≤7 —— 但它是靠 `out_tint + MATGAIN + SKINCAP(截 B 到 171)`
   硬掰出来的（`LP_SKINCAP` 的 B 上限正是"B 偏高"的补丁），不是物理正确的。
4. 眼睛/脸不一致问题在 PICA 路径下**结构性消失**（不再有 `LP_EYEUNLIT` 特判），
   但眼白/虹膜的叠加层（eye_uru/eye_light）还需要 TEV 才能正确合成。

## 5. 还剩的未知量（按优先级）

0. **TEV/TexEnv 已完成结构解码（2026-11-14 队友交付，见 `_notes/2026-11-14-tev-decode.md`）**：
   - 关键结论：**`COMB` 的 5×u32 不是 PICA 的 5 个 TEV 寄存器**，而是引擎自己的 `dmp_TexEnv[i]` 模板
     （`code_full.bin` 0x56A590 有字段名字符串 `srcRgb/srcAlpha/operandRgb/operandAlpha/combineRgb/combineAlpha/scaleRgb/scaleAlpha/constRgba`，
     每 stage 20 字节，与 `1+5N` 精确吻合）；运行时才翻译成 PICA 寄存器。
   - 53%（3103/5875）材质只有 1 个 stage；2~5 stage 的占 47%。
   - 皮肤/布/脸/眼 **stage0 是同一模板，只有取样来源 nibble 不同**（皮肤 1、脖子 3、头发 5、眼 6）
     ⇒ 脸/皮肤/眼的颜色差异不在 stage0，在后续 stage 与 MATC constant。
   - **secondary 只在 operand/alpha 槽出现、从不作为 src** ⇒ 不存在"texture×primary + secondary"这种直接相加的链路。
   - 眼睛 5 stage 链里有 MATC `constant3=(0,0,178,255)`、`constant4=(0,0,255,255)` 两档蓝 ⇒ 眼睛不是 unlit，
     是"光照结果 + 4 级 Constant/贴图链"。
   - 已交付 `native/tev.py`（`combine_material()`，8/8 手算 assert 通过，但**字段顺序仍是 provisional**）、
     `scripts/probe_tev.py --survey`、`out/_reports/tev_decode.json`、`out/_reports/tev_corpus.json`。
   - **收口办法**：在 code.bin 里定位使用 `dmp_TexEnv[*]` 字符串表的函数，读它的 TexEnv→PICA 寄存器翻译代码，
     即可确定 16 个 nibble 的字段顺序（这是下一步 P0）。
1. **TEV 合成（P0，进行中）**：见上。它决定 `texture / primary / secondary / constant` 的具体组合形式，
   尤其是 **primary 只应作为弱调制**（贴图本身已含明暗）——这解释了"灯光全开直乘必然过曝"。
   在字段顺序确认前，`LP_PICASECMODE=mask` 只把 secondary 限在 `*_spe` 掩膜上，避免爆白。
2. **B 通道系统性偏高（P1）**：PICA 灯光=0 时肤色 B 比实机参考高 18~25（脸 193 vs 174、腿 199 vs 172）；
   旧 fitted 靠 `LP_SKINCAP` B≤171 掩盖了它。需要 TEV/贴图单元顺序确认后判断是
   合成项（多贴图叠加）还是贴图采样/解码差异。
3. **灯类型（directional vs positional）**：`0x1138CC` 只填颜色/方向，LIGHTi_CONFIG 的 directional 位
   在场景数据里；主菜单两盏灯按 directional 处理时方向 (0.6,0.05,0.7)/( -0.15,-0.15,-1)。
4. **实机 LUT RAM 的 DA/SP 内容**：本作 CONFIG1 把两者都 disabled（已由反汇编确认），
   但若其它场景打开衰减则需要抓运行时的完整 GX 命令块（Citra RPC 再 dump 一次，用户开游戏即可）。
5. **雾**：主菜单无雾；故事场景需要再抓真值。

## 6. 2026-11-14 第二轮：找到"色差"的缺失项 —— 一个全局暖色项

### 6.1 先排除：旧拟合补丁在 compose 层的残留
`compose.py` 里还有一批**不经过 `nlp_shade`** 的拟合补丁：`skin_gain`（把 `SH.material_gain()` 的皮肤增益
以 `obj_gain` 直接传给渲染器）、`_tint_gray_skin_texture()`（把 `_04_000_*_hi` 的灰度高光贴图按 `SKIN_REF` 染色）、
`SKIN_TINT = (254,204,174)/255`（实机参考色）、`LP_NECKLINING`（白领口内衬乘肤色）。
`LP_NOSKINGAIN=1 LP_NECKLINING=0` 可关掉——本渲染色差不是它们造成的（A/B 中画面无变化），但后续做"纯 PICA"验收时应关掉。

### 6.2 定量发现：一个全局乘性色温项
用 `LP_MATID` 掩膜逐材质取中位色，与旧项目对实机截图的量化结果（`_work/ref_light/regions.csv`，同角色/同服装）对比：

| 材质 | PICA（灯光全开） | ×(0.99,0.90,0.80) 后 | 实机参考 |
|---|---|---|---|
| 脸 m_face_01 | (254,223,214) | **(251,201,171)** | (254,204,174) |
| 腿 m_04_000_le | (255,226,218) | **(252,204,175)** | (252,199,172) |
| 脚 m_04_000_fo | (255,230,222) | (252,207,177) | (252,199,172) |

⇒ **单个全局暖色乘子 (0.99,0.90,0.80) 就把脸和腿同时拉回参考（Δ≤5）**。一个全局项同时解释两个独立部位
⇒ 它更像缺失的**管线阶段**（3DS 常见做法：TEV buffer color 乘 / 最后一遍 "screen" 全屏 color-grade / 场景色温），
而不是某材质的颜色。
- 主菜单实机截图的脸（≈(254,206,179)）反推所需系数 ≈(1.0,0.92,0.84)，与上面的 (0.99,0.90,0.80) 同一量级
  ⇒ 这个色温项疑似是**引擎全局/场景级**的，不是某个场景特有。
- 实现：`LP_PICAGRADE="0.99,0.90,0.80"`（`native/nlp_render.py` 新增开关，**默认关闭**，用于假设验证）。
- 待确认：这个东西的**真实来源**。候选（按可能性）：
  1. `GPUREG_TEXENV_UPDATE_BUFFER`（0xE0）/ BUFFER_COLOR：引擎 `0x1136xx` 雾/缓冲色命令块里可能同时写了 TEV 的 buffer color；
  2. `shader.shbin` 的 `screen` 程序（全屏后处理）里的颜色矩阵/常数；
  3. 场景数据里的全局色温（story/约会场景不同）。
  确认方式：在 `code.bin` 找写 `0x00E0/0x00E1` 的命令构造函数与它的常数来源；或 Citra 运行时 dump 完整帧命令块。

### 6.3 验收图（本文件同目录的 `out/ab/`）
- `face_final_compare2.png`：实机参考 / FITTED / PICA / PICA+全局色温（眼睛除外）—— **最新**
- `r600_final_compare2.png`：同上三版全身 600×860 —— **最新**
- `face_final_compare.png`、`r600_final_compare.png`：上一版（色温含眼睛）
- `r600_compare_cn.png`、`r600_face_compare_cn.png`：含"PICA 灯光衰减=0"变体
- `game_menu_face_zoom.png`：主菜单实机脸部放大（皮肤暖 / 眼白中性 的证据）

### 6.4 仍未闭合
- 头发的 sheen：PICA 灯光全开下头发 (33,33,33) vs 实机 (46,45,49)（实机更亮且偏蓝灰）。
  D0/FR LUT 与 `_spe` 的合成属于 TEV/高光链，等 TEV 字段顺序确定后处理。
- TEV 字段顺序（见 §5.0）：`native/tev.py` 已可跑，但 16 个 nibble 的映射是 provisional。
- 眼睛 5 stage 链（eye/eye_uru/eye_light + constant3/4 蓝）未接。

### 6.5 全局色温项的代码侧进展（2026-11-14 第三轮）

**TEV BUFFER_COLOR（`GPUREG_TEXENV_BUFFER_COLOR` = 0x00FD）的写入链已定位**：

```
0x116A0C  (iface slot 0x134 的实现) —— 写一个材质的全部 TEV stage 命令：
            0x116B20: ldr r3,[r0+0x90]   ← buffer color 的 RGB（u32，低 3 字节）
            0x116B24: ldrb r0,[r0+0x93]  ← buffer color 的 A
            0x116B28: 头部 0x000F00FD     ← GPUREG_TEXENV_BUFFER_COLOR，值 = (RGB|A<<24)
0x10A4C8 = iface slot 0x134 的函数入口
            0x10A504: ldr r0,[r4+0xc]    ← 参数块 +0xC = **打包的 RGBA8 颜色**
            0x10A510: str r0,[sp,#0x5ec] ← 存进栈上的渲染状态结构，之后作为 buffer color
```
- iface 函数指针表在 `0x106E8C` 起（slot 0x100 ↔ 0x106E8C，slot 0x104 ↔ 0x106E90 …），
  slot 0x134 的指针 = `0x106ECC` 处的 `0x10A4C8`。
- 即：**每帧/每次渲染状态设置时，引擎把参数块 +0xC 的颜色写进 TEV BUFFER_COLOR**。
  如果某个材质 TEV stage 用了 "buffer" 作为 operand/source（旧项目全库统计里 stage 记录里存在
  `0x0/0xe/0xf` 这类 nibble，需要字段顺序才能确认是不是 buffer），这个全局颜色就会乘到角色颜色上
  —— 与实测需要的暖色系数 (0.99,0.90,0.80) 形态完全一致。
- 下一步精确到搜索式：找调用 iface slot 0x134 的位置 `ldr rX,[engineObj,#0x164]`（0x30+0x134）+
  `blx rX`，反推参数块 +0xC 的来源；或 Citra 运行时直接读该参数块。

**场景数据侧**：搜索 `img.bin` / `code_full.bin` 里 `0.9764706`、`0.8980392`、`0.698039`、`0.4`
等光照浮点（都是运行时由 8bit 颜色/255 算出）→ 静态零命中；说明场景光照预设不是裸 float 表，
而是压缩资源/字节颜色经引擎换算。⇒ 运行时 dump（Citra）是最短路径。

### 6.6 静态分析边界（2026-11-14 第四轮）

- **live ctx dump 里没有 buffer color**：对 `raw_ctx_mainmenu2.bin`（ctx 0..0x2000）、场景灯光对象
  （0x400）、帧命令块切片做了"暖色 RGBA8"扫描，只命中 `0x8D86xxxx` 段指针的字节巧合 → 排除。
- **TEV 命令头表**（rodata `0x7CBAA0` 起的 `0x000E000A/0x000F00C9…`）实际是引擎的
  **GPU 命令去重/缓存查找表**（0x3E7FA4 的二分查找函数），不是 TEV 翻译表。
- **SMAT chunk 解析不在 code.bin 的可见文本里**：全文件只有 1 处 'COMB'（实为字符串
  `DESCRIPTION_INVALID_COMBINATION` 的子串），'MATC'/'SMAT'/'DUMP' 零命中 ⇒
  资源解析在引擎的通用 chunk 层（magic 可能按数值/哈希比较），没有直接照抄点。
- 头发：开 `LP_SPECMAP=1`（`*_spe` 高光贴图参与 secondary mask）后头发中位 (36,31,29)，
  仍低于实机 (46,45,49) 且不够蓝灰 → 头发 sheen 需要 TEV 的 D0/FR 真实链路，暂不单独调。

**决定性证据：暖色项不是"全屏统一"**（2026-11-14，主菜单实机截图 `out/ab/game_menu_face_zoom.png`）：
- 脸部/脖子皮肤中位 ≈ **(254,205,175)**（暖）
- 眼白（sclera）中位 ≈ **(215,218,220)**（中性、略偏冷；左眼 187px 统计）
⇒ 同一画面里皮肤暖、眼白不暖 ⇒ 暖色项是 **TEV 级/材质级**的（皮肤/布的 stage 用了 buffer color 之类的源；
眼睛的 5-stage 链用蓝色 constant 把 sclera 拉中性），不是全屏后处理。
因此 `LP_PICAGRADE` 现在**默认跳过眼睛材质**（`LP_PICAGRADEEYE=1` 可强制包含），
验收图见 §6.3 的 `face_final_compare2.png` / `r600_final_compare2.png`。

### 6.7 当前验收状态（可以认为是本轮交付）

| 项 | 状态 |
|---|---|
| 过曝（旧版"太白/像相机过曝"） | ✅ 机制已修：按 PICA 硬件语义 8bit clamp、无裸 Lambert 超 1 叠加、PICA 模式跳过全部拟合补丁 |
| 眼睛 vs 脸肤色不一致 | ✅ 机制已修：删掉 `LP_EYEUNLIT` 特判，眼睛与皮肤走同一条光照/合成路径（暖色项也按 §6.6 排除眼睛） |
| 与实机参考的肤色一致性 | ✅ **最终量化（`final_accept.png`，LP_MATID 逐材质中位）**：脸 (251,201,171) vs 参考 (254,204,174) Δ=(-3,-3,-3)；腿 (252,204,175) Δ=(0,5,3)；脚/手 Δ=(0,8,5) |
| 眼白像素级还原 | 🚧 实机 sclera=(215,218,220)（中性微冷）；我们的眼白仍偏亮（≈(251,226,222)）→ 需要眼睛的 5-stage TEV 链（蓝色 constant）才能压中性 |
| 全局色温项的**真值来源** | 🚧 已定位到 TEV BUFFER_COLOR 写入链（§6.5），值需运行时确认 |
| TEV 字段顺序 / 头发 sheen / 眼睛 5-stage | 🚧 见 §5.0 / §7 / §8 |

**最终验收复跑命令**（当前最佳版本 = 色温仅作用于皮肤）：
```powershell
$py = "C:\Users\han\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe"
$env:LP_PICALIGHT='1'; $env:LP_PICADA='1'; $env:LP_PICASP='1'
$env:LP_PICAGRADE='0.99,0.90,0.80'; $env:LP_PICAGRADEONLY='skin'; $env:LP_SPECMAP='1'
& $py nlpp-re\native\compose.py --img data\img.bin --w 600 --h 860 `
      --out nlpp-re\out\ab\cmp_new3_red_full.png --bg bs_0000_00_00.jpg
```
说明：`LP_PICAGRADE` 的数值目前是**从实机参考反推的待确认项**（不是最终真值），
真值应由 §6.5 的 TEV BUFFER_COLOR 运行时 dump 决定；`LP_PICAGRADEONLY='skin'` 让色温项只乘在皮肤材质上
（依据见 §6.8），眼睛/头发/布料都不吃这个色温项。旧开关 `LP_PICAGRADESKIP='_eye,_ha'` 亦可达到接近效果。

### 6.8 老版 vs 新版对比图集 + 最终推荐变体（2026-11-14 第六轮）

在老版（2026-10 拟合）基础上，新增 4 组不同视角/服装的渲染对比（每张 600×860）：

| 对比图 | 内容 |
|---|---|
| `out/ab/cmp50_overview.png` | **总览**：全身 / 脸部景别 / 眼睛特写 / 蓝女仆装 的老 vs 新 |
| `out/ab/cmp40_full.png` | 全身正视：老 vs 新 |
| `out/ab/cmp41_face.png` | 脸部景别（aim 0.28/zoom 1.25）：老 vs 新 |
| `out/ab/cmp42_eye.png` | 眼睛特写（aim 0.33/zoom 0.8）：老 vs 新 |
| `out/ab/cmp43_side.png` | 侧 35°：老 vs 新 |
| `out/ab/cmp44_maid.png` | 蓝女仆装 m_01_005：老 vs 新 |
| `out/ab/cmp45_outfit002.png` | 校服 m_01_002：老 vs 新 |
| `out/ab/cmp20..23_*_3way.png` | 三版对比：老 / 新版A（色温全材质）/ 新版C（色温仅皮肤） |
| `out/ab/cmp02b_face_zoom.png`、`cmp03b_eye_zoom.png` | 2.2x 放大细节对比 |
| `out/ab/cmp10..12_*_3way.png` | 三版对比：老 / 新A / 新B（色温排除头发与眼睛） |

**关键发现（对比图暴露 + 实机量化验证）：暖色项只应作用于皮肤**
- 白围裙中位：老版 (215,237,245)（偏蓝）→ 新版A (225,213,189)（**米黄，错**）→ 新版C **(226,234,235)**，
  实机参考 (230,237,238) ⇒ **色温项不能乘到布料上**；
- 头发中位：老版 (42,43,48) / 新版A (36,31,28)（**被染棕，错**）/ 新版C (37,35,36)（中性，方向对但偏暗），
  实机参考 (46,45,49) ⇒ 头发也不该吃色温项，且亮度/蓝灰 sheen 仍未还原（待 TEV D0/FR 链）；
- 皮肤：新版 C 与实机暖色一致（脸/腿/脚 Δ≤8/255，见 §6.7）。
⇒ **最终推荐 = `LP_PICAGRADEONLY='skin'`**：皮肤暖色、布料颜色、白围裙、眼睛眼白、头发色相都各归其位。

### 6.9 动作渲染 + 服装自带裸皮件修复（2026-11-14 第七轮）

**A. 服装自带的裸皮件（用户指出："胳膊和手有色差，脖子也是"）**
服装模型里带名字后缀 `_ar`（arm）/`_ne`（neck）/`_wa` 的材质**不一定是皮肤**，要看贴图：
- `m_01_012`（红格裙，长袖）：`_ar` 贴图 (143,36,45) 红布、spec0=(0,0,0) ⇒ 胳膊=袖子=布；只有 `_ne`(254,226,217) 是皮肤。
- `m_01_005`（蓝女仆，短袖）：`_ar` (253,223,214)、spec0=(89,114,163) ⇒ 胳膊=裸皮；`_bo/_wa` 是布。
- `m_01_002`（泳装）：`_ar/_wa3/_ne/_bo1` 等大都是皮肤件。
判据（已实现）：`compose.collect_geometry` 里用 `_texture_is_skin(m, mtl, ref)`（贴图均值≈身体皮肤 + 局部对比度低 + specular 非零）
给出 `skin_like` 标记 → `nlp_render` 的"色温仅皮肤"规则使用 `skin_like`。

修复前后（`LP_MATID` 逐材质中位色，蓝女仆装）：

| 材质 | 老版 | 修复前（漏判） | **修复后** |
|---|---|---|---|
| 裸臂 `_ar`（皮肤） | (252,203,172) 暖 | (253,222,214) 偏白 ✗ | **(250,200,171) 暖 ✓** |
| 脖子 `_ne`（皮肤） | (255,195,166) 暖 | (252,216,208) 偏白 ✗ | **(250,195,167) 暖 ✓** |
| 蓝裙 `_bo`（布） | (211,232,242) | (224,230,233) | (224,230,233)（不受色温影响 ✓） |
| 围裙 `_wa`（布） | (211,232,240) | (220,226,230) | (220,226,230)（不变 ✓） |

红格裙复核（长袖）：袖子 (138,35,37)→(148,40,48) 仍是布；脖子 (255,177,149)→(244,173,147)、腿 (247,193,166)→(252,204,175)、脸 (252,198,167)→(251,201,171)、手 (249,194,164)→(252,207,177) 全部暖色一致 ✓
对比图：`out/ab/cmp60_maid_armfix.png`（老/新版A/修复 三版上臂+脖子特写）、`cmp61_maid_3way_fixed.png`。

**B. 动作渲染**：`compose.py --frames 0,16,32,48` 出分帧；`LP_MOTION=m_31003_40.mot` 换动作。
对比图 `out/ab/anim_compare_sheet2.png`（老版默认动作 / 新版默认动作 / 新版另一动作，各 4 帧）。
注意：动画只验证了"姿势变化时渲染稳定"，骨骼/动作解码不是本轮范围。

**C. 衣服的光照（结论）**：PICA 模式下布料 = `贴图 × primary`：
- primary = clamp(全局 ambient + 两灯 Lambert)。布料法线正对光时 primary 饱和→衣服≈**贴图原色**（保留烘焙明暗）；
- 只有背光/掠射面 primary 降到 ambient≈(0.98,0.90,0.90)，即最多约 **10% 的 G/B 压暗**，没有方向性 key light；
- 布料**没有动态高光**：材质 CONFIG1 把 D0/D1/反射 LUT 全禁，且布没有 `*_spe` 高光贴图（`specular0=0`）⇒ 不会出现头发那种 sheen；
- 这与旧项目对实机的研究一致（实机布料动态调制只有 3~4%，主对比来自贴图烘焙）。
- 老版衣服偏亮偏青是拟合增益/白平衡造成的（蓝裙老 (211,232,242) → 新 (224,230,233)，更接近贴图原色）。

**D. 已知遗留（衣服颜色，非光照问题）**：我们的蓝裙/白围裙整体仍比实机参考浅
（如蓝裙 (224,230,233) vs 实机 (144,200,229)；旧版也偏浅）⇒ 属于材质/贴图映射层面（多贴图 TEV、材质选择或 UV），
与 "3D 光照" 无关，建议单独开一项查。

**E. 女仆装"脖子下方红色一块"是什么（2026-11-14 用户指出）**
- 位置/身份：`m_01_005` 的 **mesh1（64 三角面）**，材质 `Textures__m_01_005_bo`，Y=123.3~129.3（脖子/领口高度）。
  它的 **UV 采样贴图颜色其实是肤色 (255,228,220)**（不是红布、也不是光照）。
- 为什么发红：旧管线 `_neck_lining_objects()` 把它当成"白色领口内衬"，给 `base_color` 乘了
  `SKIN_TINT=(254,204,174)/255` → **同一块皮肤被染了两遍**，变成橙红。
- 实测（新版，patch 像素 p50）：
  | | 该片 p50 | 旁边脖子皮肤 p50 |
  |---|---|---|
  | 旧 hack | (254,181,149) **橙红 ✗** | (252,206,174) |
  | 修正后 | **(252,204,174) ✓** | (252,206,174) |
  修正后与脖子皮肤 Δ≤2，且等于实机肤色参考 (254,204,174)。
- 修正代码：`compose.collect_geometry` 不再乘 `SKIN_TINT`；`skin_like` 对 `neck_lining` 命中的片强制按皮肤处理
  （`LP_NECKLINING=0` 可关掉该判定，让它当普通白布片渲染）。
- 图：`out/ab/neck_lining_compare3.png`（2×2：旧hack/修正 × 老版/新版）、`out/ab/cmp63_maid_neckfix.png`（全身三版）。






## 7. 附：运行时 TEV stage 描述符布局（2026-11-14 反汇编确认）

`0x116A0C` 是引擎的"TEV stage → PICA 寄存器"命令生成函数（iface slot 0x134 链路里最终写 GPU）。
它对每个 stage 读 **24 字节**运行时描述符，拼出 PICA `TevStageConfig`；逐字节对应关系（已与 Citra
`regs_texturing.h` 的 `TevStageConfig` 位域对齐验证）：

| 运行时段内偏移 | 写入的 PICA 字段 |
|---|---|
| `+0x00` | `ops.color_op`（bits0-3） |
| `+0x01..0x03` | `modifiers.color_modifier1/2/3`（bits0-3/4-7/8-11） |
| `+0x04..0x06` | `sources.color_source1/2/3`（bits0-3/4-7/8-11） |
| `+0x07` | `scale` 的颜色缩放（bits0-1） |
| `+0x09` | `ops.alpha_op`（bits16-19） |
| `+0x0A..0x0C` | `modifiers.alpha_modifier1/2/3`（bits12-14/16-18/20-22） |
| `+0x0D..0x0F` | `sources.alpha_source1/2/3`（bits16-19/20-23/24-27） |
| `+0x10` | `scale` 的 alpha 缩放（bits16-17） |
| `+0x12..0x15` | `const_color`（RGBA8，写 GPUREG_TEXENVi_COLOR） |

⇒ **SMAT 的 20 字节 COMB 记录 → 24 字节运行时描述符**的转换发生在资源加载层（未在 code.bin 找到，
见 §6.6），但一旦拿到转换规则，用这张表即可 1:1 生成 TEV 寄存器。
`native/tev.py` 里的 `combine_stages` 是 PICA 侧语义（按 Citra），与这张表配合即可闭环。

### 7.2 ★ 运行时 TEV 真值已抓到（2026-11-14，Citra RPC）

方法：引擎有 GPU 状态缓存，稳态帧不重发 TEV 命令；但命令缓冲尾部
（`[0x8AB4F8]` = 写指针，缓冲位于 `0x145C0000` 一带）在**场景切换**时会整批重发。
用 `scripts/live_tev_cmd_capture.py` 盯写指针前 128KB，抓到 10 个命中帧
（`out/live/cmd/hit01..10.bin`，最多一帧 175 条 TEV 命令）；用 `scripts/analyze_cmd_tev.py` 拆出
**21 套不同的 TEV 程序**（`out/live/cmd/tev_runtime.json`）。

命令格式（Citra pica_core 确认）：`[value0][header][value1..4]`，header=`0x80000000|0x4F0000|base`；
**stage 基址 8 字节步进**：`0xC0/0xC8/0xD0/0xD8/0xF0/0xF8`，`TEV_BUFFER_COLOR=0xFD`。

**关键程序（原始寄存器值）**：

*① 皮肤链（6 stage，暖色项的真身）*
```
st0 MODULATE  src=(TEX0, PRIM_FRAG)                     (raw 0e330e13 …)
st1 MODULATE  src=(PREVIOUS, CONSTANT)  const=(255,229,204,255)   ← 暖色系数 (1.0,0.898,0.80)
st2 ADD       src=(PREVIOUS, SECONDARY_FRAG) modC2=SRC_ALPHA      ← 加 fresnel 边缘光
```
⇒ 我们的 `LP_PICAGRADE=0.99,0.90,0.80` 就是从实机反推的，真值 = **(255,229,204) = (1.0, 0.898, 0.80)**，
误差 <1%；"色温项只作用于皮肤"也被这条程序直接证实（布料/头发程序里没有这个常量）。

*② 头发链（6 stage）*
```
st0 REPLACE   src=TEX0
st1 MODULATE  src=(PREVIOUS, CONSTANT)  const=(38,38,41,255)      ← 深蓝灰发色
st2 MODULATE  src=(PREVIOUS, PRIM_FRAG)                           ← 光照
st3 MUL_THEN_ADD src=(TEX1, SECONDARY_FRAG, PREVIOUS) const=(255,229,204)  ← spe×高光
st4 ADD       src=(PREVIOUS, SECONDARY_FRAG) modC2=SRC_ALPHA      ← fresnel 加亮
```
⇒ 头发 = 底材×深蓝灰 ×光照 + TEX1×secondary + secondary×alpha；这解释了实机蓝灰 sheen。

*③ 眼睛链（6 stage，通道复用）*
```
st0 MUL_THEN_ADD src=(PRIM_FRAG, TEX0, SECONDARY_FRAG)
st1..st5 MUL_THEN_ADD src=(TEX1/TEX2, CONSTANT, PREVIOUS) const=(178,0,0,255)
        通道选择：R,R,G,G,B（SRC_RED/GREEN/BLUE）              ← 高光贴图 R/G/B = 三个动画帧
```

其余：`MOD(TEX0,PRIMARY)`（简版角色/布料）、`REPLACE(PRIMARY)`（纯色）、
`0x88000400` 组的 `SUB/ADD(PREV_BUF)`（用前一帧缓冲的 2D/后处理效果）等。

**下一步**：把这 4 套程序按材质类别接进 `nlp_render`（皮肤暖色常量、头发常量+secondary、
眼睛通道复用），然后重渲 A/B 对照实机 (46,45,49)/(215,218,220)/(254,205,175) 验收。

**7.2.1 已实现（2026-11-14）**：`LP_PICATEV=1`
- 皮肤：`rgb = tex×primary × (255,229,204)/255`（真值常量），实测 脸 (254,200,171) vs 实机 (254,205,175)、
  腿 (255,203,175) vs (252,199,172)，**Δ≤5** ✅
- 布料/其它：`tex×primary`（不变）
- 头发/眼睛：**暂未接**——程序→材质的对应还没确定（见下）

**7.2.2 程序→材质映射的进展与卡点**
- 已用 `cfg0/cfg1/lutmax` 做指纹：`cfg1=0x00760000`（头发配置）的绘图前会先写一段
  **默认 TEV 状态**（6 stage 全 PREVIOUS），material 的真实程序不能靠"配置之后的第一个 TEV run"定位；
- 已确认眼睛程序（`cfg1=0x00770000`、`lutmax≈0.07`）：6 stage，`const=(178,0,0)`，TEX1/TEX2 按 R/R/G/G/B 通道各加一次；
- 已确认含 `(38,38,41)` 的程序实为**深色小件**（睫毛/眉/阴影片类），不是头发主体
  （我们头发贴图本身较暗，再乘 38/255 会变成 (9,8,9)）；
- 下一步用**顶点数/纹理尺寸指纹**：每个 draw 的 vertex 数与纹理 DIM/ADDR 能对上我们的 mesh，
  即可把程序块精确落到材质上（`nlp_model` 里有每 mesh 的顶点数/贴图尺寸）。


## 8. 下一步（按优先级）

0. **运行时 dump（最短路径，P0）**：用 Citra RPC 读主菜单/故事场景，抓
   ① iface slot 0x134 的参数块（+0xC = TEV BUFFER_COLOR 真值）；
   ② 完整帧命令块（找 TEXENV/0xFD 命令，确认实际写进 GPU 的 buffer color 与 TEV 状态）；
   ③ 现有 `live_dump.py` 扩展到更大帧数据区。
   拿到 buffer color 真值后，`LP_PICAGRADE` 从"假设"变成真值，A/B 即闭合。
1. **给全局暖色项找到真实来源（P0）**：在 `code.bin` 里搜到的候选常数：
   - `0x79D480` 附近 floats `(1.0, 0.97, 0.88, 0.75, 0.60, 0.45, 0.32, 0.21, 0.13, 0.07, 0.04)`（衰减/淡出曲线）
   - `0x803140` 附近 floats `2.44 → 1.11` 递减序列，其中 `0x803160 = (0.9732, 0.8911, 0.8091)`
     **与实测所需 (0.99,0.90,0.80) 只差 ~1-2%**，是最可疑的色温常数
   - `0x7E32B0` 附近 `(1.0, 0.88, 0.82, 0.70, 0.40)`
   这三个地址 `armref refs` 都查不到直接引用（可能以表基址+偏移/拷贝方式使用）⇒ 下一步反查
   **表基址**（如 0x803120/0x79D460）的引用，或找 `TEXENV_UPDATE_BUFFER(0xE0)`/`BUFFER_COLOR` 命令构造函数
   （已知 `0x1136xx` 区块写 0xE0/0xE1），看它取的常数是不是这张表。
2. **TEV 字段顺序（P0）**：定位使用 `dmp_TexEnv[*]` 字段名的代码（字符串在 `0x66A58C..`，但无直接引用 ⇒
   可能是调试/生成表），或直接找写 TEXENV 寄存器（0x00C0-0x00DD）的函数，反推 20 字节记录 → 5 寄存器。
3. **头发 sheen / 眼睛 5-stage 链**（P1）：等 1/2 完成后按真实 TEV 接。
4. **验收方式**：`LP_MATID` 掩膜逐材质对齐 `_work/ref_light/regions.csv` 的实机统计；
   全局色温项确认后写入默认值（而不是当拟合参数），再与实机截图整体验收。



