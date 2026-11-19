# ZHANGYingfang FYP 范本格式分析 (2026-06-02)

> **源文件**: `E:/Project/FYP/sample/Partial_thesis_ZHANGYingfang.docx`
> **作者**: Zhen LUO, Peihao ZHONG, Yingfang ZHANG (BNBU 2021)
> **题目**: Image-to-image Translation using Generative Adversarial Networks
> **作用**: 本组 (Vicky/Polyp) Chapter 2 模板参考

---

## 📐 总体结构 (5 章 + 摘要 + 致谢 + 目录 + 参考文献 + 附录)

```
0. Cover Page (title, authors, IDs, BNU-HKBU, date)
1. Declaration
2. Acknowledgement
3. Abstract (style [11] = Times New Roman 11pt, italic abstract block)
4. Contents (TOC) - 5 章 + References + Appendices
5. Chapter 1: Introduction
6. Chapter 2: Related Work  ← 你的目标
7. Chapter 3: Result Analysis
8. References (style [11] = 11pt 编号引用)
9. Appendices (optional)
```

---

## 🎨 段落样式约定 (Word 内置样式名)

| 样式 | 用途 | 备注 |
|------|------|------|
| `[1]` | 一级标题 (Chapter) | "1 Introduction", "2 Related Work" |
| `[2]` | 二级标题 (Section) | "2.1 Feature-based Method" |
| `[3]` | 三级标题 (Subsection) | "2.1.1 Domain Transfer Network" |
| `[11]` | Abstract / 参考文献条目 | 11pt |
| `[a5]` | Figure / Table 标题 | "Figure 1: Architecture of DTN." |
| `[a9]` | 占位提示 | "This part should include the acknowledgement." |
| `[12]` | 作者姓名 (Title page) | 居中大字号 |
| `[8]` | "DECLARATION" 大标题 | |
| `[5]` | 签名/日期行 | 下划线 |
| `默认` | 正文段落 | 11-12pt |

---

## ✍️ Chapter 2 Related Work 写作风格核心

### 结构模板 (从范本提取)

```
Chapter 2: Related Work
  ↓ 引言段 (介绍 + 总览, 200-300 words)
  2.1 <一级分类 1> (e.g., Feature-based Method)
    引言段 (200-300 words, 介绍本类 + 2-3 篇代表)
    2.1.1 <论文 1> (300-500 words + Figure X)
      - 论文提出背景
      - 核心架构 (配 Fig)
      - Loss/方法细节
      - 实验结果 (数据集, 数字)
      - 局限性 (150-200 words)
    2.1.2 <论文 2>
    ...
  2.2 <一级分类 2>
    2.2.1 <论文 3>
    ...
  ...
  2.X Comparison (综合对比表 + 分类讨论)
```

### 关键写作模式

#### 模式 1: 介绍段 (Section 引言)
```
"X is a huge branch of Y. By [key idea], it enables [capability]. 
[历史起源 1-2 句]. [近年发展 1-2 句]. [本节将介绍 2-3 种代表方法]."
```

#### 模式 2: 单篇论文介绍 (Subsection)
```
"<Author> et al. proposed <Method Name>, which [核心 idea]. 
[架构/方法 1-2 句]. 

[Loss / 公式 / 关键组件 1-2 句, 适度数学].

[实验设置 1-2 句]. [结果数字].

However, [局限性 1-2 句]."
```

#### 模式 3: 综合对比 (Comparison)
```
"For <任务>, a fundamental rule is that <共性>. 
[本类方法特征]. [另一类方法特征]. 
[对比表格 + 讨论]."
```

### 引用风格

- **方括号数字式**: `[1]`, `[2,3]`, `[10-12]`
- **作者+年份式**: `<Author> et al. proposed ...`, `Taigman et al. proposed DTN [3]`
- **混用**: 同一段落两种都出现

### Figure 引用
- **格式**: "Figure 1: Architecture of DTN." (句号结尾)
- **位置**: 在该 Subsection 内嵌入, 紧跟架构描述

### Table 引用
- **格式**: "Table 1: Evaluation metrics of IS, FID, VGG distance..."
- **对比**: 多用 2-3 行 (self-trained / paper) 对比格式

---

## 🎯 Chapter 2 长度参考

| 部分 | 范本字数 (估) | 我们目标 |
|------|---------------|----------|
| 引言段 | ~200 words | 200-300 words |
| 每个 Section 引言 | ~150 words | 150-200 words |
| 每个 Subsection (单篇) | ~400 words | 350-500 words |
| Comparison 段 | ~200 words | 200-300 words |
| **总 Chapter 2** | **~5000 words** | **4000-5000 words** |

---

## 📋 你 (Vicky/Polyp) Chapter 2 适配建议

### 分类 (基于 26 DEEP cards)
1. **2.1 Foundation Models for Medical Image Segmentation** (SAM, MedSAM, SAMed, MaskSAM)
2. **2.2 Polyp-Specific Adaptation** (ASPS, PP-SAM, MedSAM on polyp)
3. **2.3 Prompt Engineering for SAM** (OP-SAM, TGSAM-2, Efficient Prompting)
4. **2.4 3D and Video Extensions** (SAM-Med3D-MoE, DC-SAM, PolypSegTrack, Self-Prompting YOLO+SAM2)
5. **2.5 Datasets and Benchmarks** (Kvasir-SEG, CVC-ClinicDB, PolypGen, PICCOLO, SUN-SEG)
6. **2.6 Comparison and Research Gap** (3 大 gap 总结)

### 引用顺序
- 26 篇 PDFs → 在 `ref.bib` 中按以下顺序填:
  - Section 2.1: [1]-[6] (Foundation)
  - Section 2.2: [7]-[12] (Polyp-specific)
  - Section 2.3: [13]-[17] (Prompt)
  - Section 2.4: [18]-[23] (3D/Video)
  - Section 2.5: [24]-[26] (Datasets)
  - Comparison: 跨节引用

### LaTeX 实现建议
- 用 `\section{}` `\subsection{}` 跟样例对齐
- `\caption{}` 配 `Figure~\ref{fig:xxx}` / `Table~\ref{tab:xxx}` (LaTeX 风格)
- `\begin{figure}[H]` `\includegraphics` 放架构图
- Bib 用 `\cite{ref1,ref2}` 或 `(Author et al., 2023)`

---

## ✅ 验证清单 (写完后自检)

- [ ] 5 个一级 Section + 1 个 Comparison
- [ ] 每个 Subsection 都有 Figure 引用
- [ ] 数字都从 DEEP card 复制 (不编造)
- [ ] 每个 Subsection 末尾有 "However, [limitation]" 句式
- [ ] 引用编号连续, 跟 `ref.bib` 对应
- [ ] 总字数 4000-5000
- [ ] Comparison 段有分类讨论表
