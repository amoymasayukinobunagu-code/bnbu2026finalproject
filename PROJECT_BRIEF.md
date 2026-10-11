# FYP Project 1 — Project Brief

> **Project Title**: Promptable Medical Image Segmentation Foundation Models for Endoscopic Polyp Segmentation
> **Supervisor**: Dr. Donglong Chen (`donglongchen@bnbu.edu.cn`)
> **Source**: `reference/Donglong_Chen_2024.pdf` — FYP Projects (2026), Index 1
> **Last updated**: 2026-06-10

---

## 1. Project Snapshot

| Field | Value |
|-------|-------|
| **Project Index** | 1 (of multiple) |
| **Project Type** | Regular Project + Research Oriented |
| **Difficulty** | Challenging |
| **Team Size** | 2–3 students |
| **Prerequisites** | Machine Learning, Image Processing, Python |
| **Domain** | Medical Image Segmentation · Computer Vision · Foundation Models |
| **Application** | Endoscopic Polyp Segmentation (Colonoscopy) |
| **Expected Output** | Reproducible codebase + Thesis + (optionally) conference paper |

---

## 2. Project Description

This project focuses on **promptable medical image segmentation**, targeting **colonoscopic polyp segmentation**, and investigates how to **reproduce and improve** the segmentation performance of **SAM / MedSAM** on publicly available datasets.

An **end-to-end pipeline** will be developed, including:

1. **Data preprocessing** for colonoscopic image data
2. **Automatic generation of box/point prompts** from ground-truth annotations
3. **Model inference** using SAM, MedSAM, and adapted variants
4. **Parameter-efficient fine-tuning** (Adapter / LoRA) on medical data
5. **Result evaluation and visualization** (Dice, IoU, HD95, FPS)

### 2.1 Core Research Question

> How do **prompt quality**, **object scale**, and **weak polyp boundaries** affect the segmentation accuracy of SAM-family models, and can lightweight adaptation close the gap to SOTA?

A systematic **comparative analysis** will be conducted across three model categories:

| Model | Description | Goal |
|-------|-------------|------|
| **Original SAM** | Zero-shot SAM ViT-B/L/H | Establish the unadapted baseline |
| **Zero-shot MedSAM** | SAM fine-tuned on 1.57M medical image-mask pairs | Quantify the benefit of medical pre-training |
| **Lightweight Adapted Models** | SAM + Adapter / LoRA on polyp data | Demonstrate parameter-efficient improvement |

### 2.2 Evaluation Metrics

| Primary Metrics | Secondary Metrics |
|----------------|-------------------|
| **Dice Coefficient (DSC)** | HD95 (Hausdorff Distance 95%) |
| **Intersection over Union (IoU)** | Inference speed (FPS) |

### 2.3 Expected Deliverables

1. **Reproducible experimental scripts** (training + evaluation)
2. **A prompt generation toolkit** (automatic box/point extraction)
3. **Systematic baseline benchmarks** (SAM vs. MedSAM vs. adapted)
4. **Scalable extensions** (at least one of):
   - Pseudo-label-based **semi-supervised learning**
   - **Cross-dataset generalization** (e.g., Kvasir-SEG → CVC-ClinicDB → PolypGen)

---

## 3. Completion Criteria

> Students need to **understand, re-implement, and improve** the Polyp Segmentation to meet the SOTA:

1. ✅ Implementation codes of Polyp Segmentation (SAM/MedSAM/Adapted)
2. ✅ Thesis and paper describing the details of the Polyp Segmentation

---

## 4. Other Requirements

- 📅 **Weekly meeting** with status report (every week)
- 🌞 At least **1 month working load** in Summer vacation
- 💻 Code must be **reproducible** (portable paths, documented dependencies)
- 📚 Thesis written in **English** (BNBU FYP standard)

---

## 5. Reference Papers (5 Core + Code Repos + Datasets)

### 5.1 Core Papers

| # | Title | Venue | Year | DOI / arXiv |
|---|-------|-------|------|-------------|
| [1] | Segment Anything Model for Medical Image Analysis: An Experimental Study | Medical Image Analysis | 2023 | [10.1016/j.media.2023.102918](https://doi.org/10.1016/j.media.2023.102918) |
| [2] | Customized Segment Anything Model for Medical Image Segmentation (SAMed) | arXiv | 2023 | [2304.13785](https://arxiv.org/abs/2304.13785) |
| [3] | Segment Anything Model for Medical Images? | Medical Image Analysis | 2024 | [10.1016/j.media.2023.103061](https://doi.org/10.1016/j.media.2023.103061) |
| [4] | Segment Anything in Medical Images (MedSAM) | Nature Communications | 2024 | [10.1038/s41467-024-44824-z](https://doi.org/10.1038/s41467-024-44824-z) |
| [5] | Medical SAM Adapter: Adapting Segment Anything Model for Medical Image Segmentation | Medical Image Analysis | 2025 | [10.1016/j.media.2025.103547](https://doi.org/10.1016/j.media.2025.103547) |

### 5.2 Code Repositories

| # | Repository | URL |
|---|------------|-----|
| [6] | MedSAM (bowang-lab) | https://github.com/bowang-lab/MedSAM |
| [7] | Medical-SAM-Adapter (ImprintLab) | https://github.com/ImprintLab/Medical-SAM-Adapter |
| [8] | segment-anything-medical-evaluation (mazurowski-lab) | https://github.com/mazurowski-lab/segment-anything-medical-evaluation |

### 5.3 Datasets

| # | Dataset | URL | Status |
|---|---------|-----|--------|
| [9] | Kvasir-SEG (Polyp segmentation) | https://github.com/DebeshJha/Kvasir-SEG | ✅ Downloaded |
| [10] | CVC-612 (CVC-ClinicDB) | https://datasetninja.com/cvc-612 | ✅ Downloaded |

---

## 6. Recommended Technical Stack

| Category | Technology |
|----------|------------|
| **Deep Learning Framework** | PyTorch |
| **Foundation Models** | SAM (ViT-B/L), MedSAM, SAMed |
| **Parameter-Efficient Fine-Tuning** | LoRA, Adapter |
| **Data Processing** | OpenCV, NumPy, Albumentations |
| **Evaluation** | Dice, IoU, HD95 (SimpleITK / MedPy) |
| **Visualization** | Matplotlib, TensorBoard, Weights & Biases |
| **Experiment Management** | wandb, MLflow |

---

## 7. Project Workflow (High-Level)

```
┌──────────────────────┐
│  Colonoscopy Images  │  (Kvasir-SEG, CVC-ClinicDB)
└──────────┬───────────┘
           │
┌──────────▼───────────┐
│ Data Preprocessing   │  Resize, Normalize, Augment
│ + Prompt Generation  │  Auto box/point from mask
└──────────┬───────────┘
           │
    ┌──────┴──────┬──────────────┐
    │             │              │
┌───▼────┐  ┌─────▼─────┐  ┌─────▼──────┐
│  SAM   │  │  MedSAM   │  │  Adapted   │
│ (zero) │  │ (zero)    │  │  (LoRA)    │
└───┬────┘  └─────┬─────┘  └─────┬──────┘
    │             │              │
    └──────┬──────┴──────────────┘
           │
┌──────────▼───────────┐
│  Evaluation          │  Dice, IoU, HD95, FPS
│  + Visualization     │
└──────────┬───────────┘
           │
┌──────────▼───────────┐
│  Extensions          │  Semi-supervised (pseudo-labels)
│                      │  Cross-dataset generalization
└──────────────────────┘
```

---

## 8. Suggested Thesis Structure (BNBU FYP)

| Chapter | Title | Length |
|---------|-------|--------|
| **1** | Introduction | ~3000 words |
| **2** | Literature Review | ~5000 words |
| **3** | Methodology | ~5000 words |
| **4** | Solution / Design and Implementation | ~3000 words |
| **5** | Validation / Testing | ~4000 words |
| **6** | Conclusion and Future Work | ~2000 words |
| **Refs** | References | 20–30 entries |
| **App.** | Appendices (optional) | — |

---

## 9. Current Progress (2026-06-02)

| Phase | Deliverable | Status |
|-------|-------------|--------|
| P1 | Literature search (5 rounds) | ✅ 19 PDFs + 26 references |
| P2 | PDF verification | ✅ 19/19 verified |
| P3 | 26 DEEP note cards | ✅ 100% |
| P4 | Chapter 2 LaTeX draft | ✅ 25.4 KB · 12 figs · 2 tables |
| P5 | ref.bib (26 entries) | ✅ 9.4 KB |
| **P6** | **Compile on Overleaf** | ⏳ Pending (user to upload) |
| **P7** | **Content review with supervisor** | ⏳ Pending |
| **P8** | **Chapter 3-5 writing** | ⏳ Pending |
| **P9** | **Experimental code (SAM/MedSAM/LoRA)** | ⏳ Pending |

---

## 10. Repository Layout (Actual as of 2026-06-10)

```
E:/Project/FYP/
├── README.md                    ← 目录导航 + 关键命令
├── PROJECT_BRIEF.md             ← 本文件
├── OVERLEAF_GUIDE.md            ← LaTeX 编译指南 (含 manuscript/ 路径)
│
├── manuscript/                  ← clean Partial Thesis LaTeX upload package (main.tex + chapter1--6 + figures)
│   ├── main.tex + chapter1-6.tex + appendixA.tex + ref.bib
│   ├── figures/
│   │   ├── chapter2/  (12 张 Ch2 方法图)
│   │   └── chapter4/  (7 张 Figure 3 qualitative)
│   └── tables/
│       ├── table1_sota.tex      ← SOTA comparison
│       └── table2_ablation.tex  ← Shot ablation
│
├── code/                        ← 源代码
│   ├── sam_baseline/            ← B1
│   ├── medsam_baseline/         ← B2
│   ├── medical_sam_adapter/     ← B3 [N/A]
│   └── pp_sam_baseline/         ← B4 (train + eval + figure 脚本)
├── checkpoints/                 ← 模型权重 (1.3 GB)
│   ├── sam_vit_b.pth            358 MB
│   ├── medsam_vit_b.pth         358 MB
│   ├── medical_sam_adapter/     163 MB (4 领域 weights)
│   └── pp_sam/pp_sam_vit_b_5shot_last.pth  358 MB (5-shot ep6)
├── results/                     ← 实验产物 (387 MB)
│   ├── EXPERIMENT_SUMMARY.md    ← 论文 Table 1/2 来源
│   ├── fyp_experiments_bundle.zip (11 MB)
│   ├── medsam_new_protocol/     ← B2 评估 (Kvasir-val 120 + CVC 612)
│   ├── msa_new_protocol/        ← B3 STATUS
│   ├── phase1_medsam_*          ← B2 旧协议 (50 张, 无扰动)
│   └── pp_sam_baseline/         ← B4 训练+评估
├── reference/                   ← 21 篇 PDF + notes
├── backup/                       ← 集中备份 (.bak.* 按相对路径归档; 不上传 Overleaf)
├── 材料/                         ← datasets (NTFS ACL 写保护) + repos
└── _deprecated/, _archive/      ← 不删, 留底
```

> **维护**: 任何目录/文件结构大改必须同步更新 README.md + OVERLEAF_GUIDE.md。
> 详见 `~/.hermes/skills/maintain-docs-after-edit/`。

---

## 11. Key Results (live source: `results/EXPERIMENT_SUMMARY.md`)

| | 模型 | Kvasir-val Dice | CVC Dice |
|---|---|---|---|
| B1 | SAM ViT-B (zero-shot) | 0.7497 | 0.5182 |
| B2 | MedSAM (zero-shot) | 0.7034 | 0.5172 |
| B3 | MSA | N/A (无 polyp 权重) | N/A |
| B4 | PP-SAM 1-shot (ours, v3) | 0.7992 | 0.7516 |
| B4 | PP-SAM 5-shot (ours, v3) | 0.8512 | 0.8040 |
| B-SAMed | SAMed LoRA full 880 (v3) | 0.8955 | 0.8608 |

完整 Table 1/2 与新增 baseline 数字以 `results/EXPERIMENT_SUMMARY.md` 为准；本文档只保留 headline rows，避免 hard-coded table 再次过期。

---

*Document version: 2026-06-10 · Maintainer: Vicky · Supervisor: Dr. Donglong Chen*
