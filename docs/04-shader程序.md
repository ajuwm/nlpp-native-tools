# 04 · PICA200 shader 程序（DVLB / DVLP / DVLE）

> 结论：**NLPP 的 shader 程序找到了**。`img.bin` 里有 9 个 `.shbin`，都是 3DS SDK 的
> **DVLB 容器**（`DVLB` 头 + 若干 `DVLE` 程序 + 一个 `DVLP` 编译代码块）。
> 角色用的主 shader 是 `shader.shbin`（6 个程序：vshader / vs_texcoord01 / vs_texcoord012 / primitive / edgeline / screen）。
> 已实现解析脚本 `scripts/parse_dvlb.py`，产物 `out/_reports/*.dvlb.json`。

---

## 1. 资源清单（从 img.bin 导出，见 `scripts/extract_res.py`）

| 文件 | 大小 | DVLB 程序数 | 用途（由符号名推断） |
|---|---|---|---|
| `shader.shbin` | 6,764 B | 6 | **角色主 shader**：蒙皮 + 法线/切线 + 2 套 UV |
| `custom.shbin` | 560 B | 1 | 自定义顶点（粒子/特效？），有 `aDir.xy` |
| `billboard_shader.shbin` | 672 B | 1 | billboard（`aIndex.x`） |
| `rain_shader.shbin` | 828 B | 1 | 雨 |
| `leaf_shader.shbin` | 904 B | 1 | 落叶 |
| `nwlyt_PaneShader.shbin` / `nwlyt_ConstColorShader.shbin` / `nwfont_*.shbin` | 328–1,060 B | 1 | 系统 UI/字体（NW4C 库） |

## 2. `shader.shbin` 结构（实测）

```
offset 0x00  'DVLB'  u32 count=6  u32 DVLE offsets[6] = 0x950, 0xd6c, 0x119c, 0x15ac, 0x1754, 0x1894
offset 0x20  'DVLP'  u32 0xf02  u32 0x28  u32 0x1de  u32 0x7a0  u32 0x26 ...   ← 编译后的 PICA200 VS 代码块
offset 0x950 'DVLE' …  程序 1（vshader.vssh）
offset 0xd6c 'DVLE' …  程序 2（vs_texcoord01.vssh）
offset 0x119c 'DVLE' … 程序 3（vs_texcoord012.vssh）
offset 0x15ac 'DVLE' … 程序 4（primitive.vssh）
offset 0x1754 'DVLE' … 程序 5（edgeline.vssh）
offset 0x1894 'DVLE' … 程序 6（screen.vssh）
```

每个 DVLE 内含：
- 程序名（`*.vssh`，在 0x910–0x950 一段字符串表里）；
- 入口标签 `main` / `endmain`；
- 输入属性表：`aPosition.xyz`、`aNormal.xyz`、`aTangent.xyz`、`aBoneIndices.xyzw`、`aWeights.xyzw`（角色 1/2/3 号程序）；
- uniform 表：`uProjection`、`uView`、`uModel`、`uBoneMatrices` 等（见 json）。
- 4/5/6 号程序的输入不同（primitive/edgeline/screen 是给辅助几何用的）。

## 3. 对渲染还原的意义（重要）

1. **PICA200 的片元光照是固定功能**（不是可编程 pixel shader）：光照计算由
   `LIGHTING_*` / `LIGHTi_*` / LUT 寄存器决定。这些寄存器我们已经在 `docs/03` 里解出构造代码。
2. 顶点 shader 负责 **蒙皮 + 法线/切线变换**，把法线交给固定功能光照；
   因此"光照到底怎么算"= `docs/03` 的寄存器值 + PICA200 固定功能公式（3dbrew/Citra 有公开语义），
   **不需要反编译片元程序**（它不存在）。
3. 真正还缺的是 **寄存器数值**（ambient / 光色 / 光位/方向 / 雾），
   即 `docs/03` 第 4 节的缺口。

## 4. DVLP 里的代码（下一步）

`DVLP` 头：`'DVLP'` + 0xf02 + 0x28 + 0x1de + 0x7a0 + 0x26 + …。
按 3DS shader binary 惯例，之后是 **PICA200 顶点 shader 指令流**（每条 32 位，成对组成 64 位指令）。
代码区里能看到典型 PICA 指令字（如 `0x60417583` 等）。

下一步（未完成）：
1. 依据 3dbrew《GPU/Shader instruction set》把 PICA VS 指令流反汇编成可读 opcode；
2. 还原 `uBoneMatrices` 的 uniform 布局（骨骼数 / 矩阵排列），用于核对我们的蒙皮；
3. 确认 `primitive/edgeline/screen` 与描边（inverted hull）的关系；
4. 把 shader 输出的法线约定（世界/视空间）与固定功能光照输入对齐。

## 5. 复现

```powershell
$py = "C:\Users\han\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe"
& $py nlpp-re\scripts\extract_res.py .shbin .olua .fsp .bgp .twbdat .btbl .mlb
& $py nlpp-re\scripts\parse_dvlb.py nlpp-re\out\resources\shader.shbin
```
