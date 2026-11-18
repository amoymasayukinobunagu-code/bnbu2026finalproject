# FYP LaTeX 编译指南 (Overleaf)

> **Last updated**: 2026-06-25
> **作者**: Vicky, Supervisor: Dr. Donglong Chen
> **编译目标**: Overleaf (云端 pdfLaTeX + BibTeX)
> **manuscript/**: 当前可上传的 Partial Thesis LaTeX 项目根目录

---

## 📁 必需文件 (上传 clean `manuscript/` 文件夹即可)

```text
manuscript/
├── main.tex                 ← Overleaf root file
├── chapter1.tex             ← Introduction
├── chapter2.tex             ← Literature Review
├── chapter3.tex             ← Research Methodology
├── chapter4.tex             ← Solution: Design and Implementation
├── chapter5.tex             ← Validation and Results
├── chapter6.tex             ← Conclusion and Future Work
├── appendixA.tex            ← Minimal supplementary note
├── ref.bib                  ← bibliography
│
├── figures/
│   ├── chapter2/            ← Chapter 2 method figures
│   ├── chapter4/
│   │   └── fig_ch4_solution_implementation_flow.png
│   └── chapter5/
│       ├── fig_ch5_kshot_adaptation_trend.png
│       └── fig_ch5_gap_to_upper_bound.png
│
└── tables/                  ← legacy table snippets, not required by current main flow
```

> 上传时只需要 clean `manuscript/`。不要上传 FYP 根目录、`results/`、`checkpoints/`、`backup/`、`材料/`、`repo/progress/` 或 `_deprecated/`。旧 standalone roots / draft backups 已从 `manuscript/` 移出，避免 Overleaf / reviewer grep 读到 stale MSA sanity-check 或 SAM-Med2D secondary-evidence wording。

---

## 🚀 Overleaf 上传步骤

### Step 1: 打包
- 将当前 clean `E:/Project/FYP/manuscript/` 文件夹打包成 zip。该目录不应包含 `lecture review.tex`、`backup/`、`_deprecated/` 或 `planning/`。
- zip 内根目录应直接包含 `main.tex`、`chapter1.tex` ... `chapter6.tex`、`ref.bib`、`figures/`。

### Step 2: 新建项目
- Overleaf → `New Project` → `Upload Project` → 上传 manuscript zip。
- Overleaf 应自动识别 `main.tex` 为 root file；如果没有，右键 `main.tex` → `Set as Main File`。

### Step 3: 检查 `main.tex` inputs
当前 root file 应包含：

```latex
\input{chapter1}
\input{chapter2}
\input{chapter3}
\input{chapter4}
\input{chapter5}
\input{chapter6}
\bibliography{ref}
\appendix
\input{appendixA}
```

### Step 4: 编译设置
- Compiler: **pdfLaTeX**
- Bibliography: **BibTeX**
- 推荐编译顺序：pdfLaTeX → BibTeX → pdfLaTeX → pdfLaTeX。

### Step 5: 成功标准
- PDF 正常生成。
- 目录、List of Figures、List of Tables 自动生成。
- 所有 citations 显示为数字引用，而不是 `[?]`。
- 所有 figures 显示正常。
- 没有 `Undefined control sequence` / `File not found` / `Citation undefined` 硬错误。

---

## 🖼 当前正文引用的 figures

静态检查确认当前正文实际引用 11 张图：

```text
figures/chapter2/fig_2.1_MedSAM.png
figures/chapter2/fig_2.2_SAMed.png
figures/chapter2/fig_2.4_ASPS.png
figures/chapter2/fig_2.6_OP-SAM.png
figures/chapter2/fig_2.5_PP-SAM.png
figures/chapter2/fig_2.8_EfficientPrompting.png
figures/chapter2/fig_2.12_Kvasir-SEG-sample.png
figures/chapter2/fig_2.10_PolypSegTrack.png
figures/chapter4/fig_ch4_solution_implementation_flow.png
figures/chapter5/fig_ch5_kshot_adaptation_trend.png
figures/chapter5/fig_ch5_gap_to_upper_bound.png
```

> `fig_ch5_scaling_efficiency_tradeoff.png` 已移入 `backup/`，因为 MobileSAM / SAMUS / SAM-Med2D 被明确舍弃，不再作为正文 evidence。

---

## 🔧 必需 LaTeX 宏包

`main.tex` 当前已包含：

```latex
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{times}
\usepackage{xcolor}
\usepackage{microtype}
\usepackage{amsmath, amssymb, amsfonts}
\usepackage{bm}
\usepackage{siunitx}
\usepackage{graphicx}
\usepackage{float}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{array}
\usepackage{caption}
\usepackage{subcaption}
\usepackage{geometry}
\usepackage{fancyhdr}
\usepackage{hyperref}
\usepackage{url}
\usepackage{bookmark}
\usepackage[numbers,sort&compress]{natbib}
```

`xcolor` 是必须的，因为 hyperref link colors 使用了 `blue!60!black` 语法。

---

## 🐛 常见错误

| Overleaf log | 处理 |
|---|---|
| `File not found` | 检查 `figures/...` 路径大小写和 zip 结构 |
| `Undefined control sequence` | 检查是否缺宏包；当前已补 `xcolor` |
| `Citation undefined` / `[?]` | 跑 BibTeX 后再跑两轮 pdfLaTeX；若仍存在，查 `ref.bib` key |
| `Overfull \hbox` | 多数是表格/长 URL 的排版 warning；先确认不是 hard error |
| 图片太大/太小 | 调整 `\includegraphics[width=...]` |

---

## ✅ 上传前静态检查结果 (2026-06-25)

| 检查 | 结果 |
|---|---|
| `\input{}` targets | PASS |
| figure paths | PASS |
| label/ref consistency | PASS |
| cite/bib consistency | PASS |
| environment pairing | PASS |
| stale placeholder / promise traps | PASS |
| scattered `.bak.*` outside `backup/` | PASS |

---

## 🗂 Backup hygiene

- `.bak.*` files are centralised under `E:/Project/FYP/backup/` with relative paths preserved. Stale manuscript-side drafts are additionally staged under `E:/Project/FYP/_deprecated/2026-06-25_stale-manuscript-drafts/`.
- Do not leave backup files beside active manuscript sources.
- Do not upload `backup/` to Overleaf.
