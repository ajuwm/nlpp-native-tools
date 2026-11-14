# NLPP — 3DS《New Love Plus+》角色还原渲染工具

**English abstract** — NLPP is a from-scratch reconstruction toolkit for the character
assets of the 3DS game *New Love Plus+*: it parses the original `img.bin` container
(PACK / SMES / SMAT / TEXI), decodes the skeletal `.mot` animation format, reproduces
the PICA200 fixed-function lighting of the retail game, and renders frames with a pure
Python + numpy software rasterizer. **No game assets are included.**

---

## 这是什么

这套工具只做三件事，全部从游戏原始数据出发：

1. **资源解析**：`img.bin`（PACK 容器）→ 网格（SMES）/ 材质（SMAT）/ 贴图（TEXI）/ 背景。
2. **动作解码**：`.mot` 骨骼动画（骨骼 ID 映射、`base/per` 通道拆分、分层动作、
   裙摆配对、循环判定）。
3. **PICA200 光照复刻 + 渲染**：固定功能光照（`global_ambient + Σ light((diffuse·N·L
   + ambient) × atten)`）、TEV 阶段、高光 LUT、眼睛高光双层、真·眨眼
   （`m__eye__m_eye_02` 脸部动作按权重叠加）。

本仓库 **不包含**任何游戏数据（`img.bin`、模型、贴图、语音等），需要你自己从
合法持有的游戏本体中提取。

## 环境

- Python 3.10+
- `numpy`, `Pillow`（见 `requirements.txt`）

```bash
pip install -r requirements.txt
```

## 快速开始

把 `img.bin` 放到 `data/img.bin`（或任意路径用 `--img` 指定）：

```bash
# 单帧渲染
python native/compose.py --img data/img.bin \
    --outfit m_01_126 --hair m_00_000 \
    --bg bs_0130_21_01.jpg \
    --frame 0 --out out/frame.png

# 一个动作的多帧
python native/compose.py --img data/img.bin \
    --outfit m_01_126 --hair m_00_000 \
    --motion m_00010_41 --frames 0,10,20,30 --out out/frames.png
```

动作目录（把若干动作渲成对照图，方便挑动作）：

```bash
python scripts/motion_sheet.py --img data/img.bin \
    --motions m_00010_41,m_00022_40,m_08450_40 \
    --frames 4 --out out/motions.png
```

## 光照相关环境变量（常用）

| 变量 | 作用 |
|---|---|
| `LP_PICALIGHT=1` | 启用 PICA 固定功能光照 |
| `LP_PICADA=1` / `LP_PICASP=1` | 漫反射 / 高光 |
| `LP_PICAGRADE=r,g,b` + `LP_PICAGRADEONLY=skin` | 肤色暖色校正 |
| `LP_SPECMAP=1` | 高光贴图 |
| `LP_ROOTLOCK=1` | 锁根位移（防止动作自带的前进位移） |
| `LP_BLINKF="28:0.35,29:0.75,30:1.0"` | 眨眼调度（帧:权重） |
| `LP_BLINKPOSE=m__eye__m_eye_02` | 闭眼用的脸部动作 |
| `LP_EYEPHASE=auto` / `LP_EYEPHASE_INT=24` | 眼睛高光帧切换 |

## 目录结构

```
NLPP/
├── native/          # 解析器 + 光照 + 软件光栅化渲染器
│   ├── nlp_pack.py      # PACK 容器
│   ├── nlp_model.py     # SMES/SMAT 模型
│   ├── nlp_mot.py       # .mot 动作解码
│   ├── nlp_tex.py       # TEXI 贴图
│   ├── nlp_bg.py        # 背景
│   ├── pica_light.py    # PICA200 光照
│   ├── tev.py           # TEV 阶段解码
│   ├── nlp_shade.py     # 光栅化/着色
│   ├── nlp_render.py    # 渲染管线
│   ├── compose.py       # 命令行入口（角色+动作+相机+背景）
│   └── ab_light.py      # A/B 光照对照
├── scripts/
│   └── motion_sheet.py  # 动作目录对照图
├── docs/            # 解码与光照过程记录
├── requirements.txt
└── LICENSE
```

## 已解 / 未解

- ✅ PICA200 光照、TEV 程序（21 个程序已抓取并实现）
- ✅ `.mot` 动作格式、骨骼映射、分层动作、循环判定
- ✅ 眼睛高光 2 帧、真·眨眼（脸部动作按权重叠加）
- ⚠️ `MotionList.mobin` 的 ID → 动作名映射尚未解码（只能靠预览认动作）
- ⚠️ `Eye.arc` 等 DARC 容器中的闭眼/表情素材尚未拆出

## 文档

- `docs/MOT格式_修正.md` — 动作格式
- `docs/03-光照管线-已解部分.md` / `docs/04-shader程序.md` — 光照
- `docs/06-实机真值.md` / `docs/07-AB对比.md` — 与实机对照
- `docs/08-PICA光照实现与AB.md` — PICA 光照实现细节

## 许可

MIT，见 `LICENSE`。游戏资源版权归原权利方所有，本仓库不含任何游戏数据。
