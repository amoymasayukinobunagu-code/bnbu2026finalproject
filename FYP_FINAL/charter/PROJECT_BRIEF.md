> **FYP_FINAL 写作快照（2026-11-17，Asia/Shanghai）**  
> 本文件是从 `E:\\Project\\FYP` 复制到 `E:\\Project\\FYP_FINAL` 的写作材料，原项目未移动、未删除。  
> 请以本目录 `numbers/NUMBERS_INDEX.md` 为入口。文中若仍出现 `E:\\Project\\FYP\\results\\...`，那是**原路径**，对应副本见本目录相对路径（见 SOURCE_MANIFEST）。  
> A–F 为官方章程；G 为仓库状态。数字以本目录 numbers/ 下 json 为准。

# FYP Project Brief — Index 1

> **Title / 题目**: Promptable Medical Image Segmentation Foundation Models for Endoscopic Polyp Segmentation  
> **Supervisor / 导师**: Dr. Donglong Chen (`donglongchen@bnbu.edu.cn`)  
> **Source / 权威来源**: `reference/Donglong_Chen_2024.pdf` — *FYP Projects (2026)*, **Index 1**  
> **Last updated / 更新日期**: 2026-11-17  
> **Maintainer note**: Sections **A–F** are the official project charter (faithful to Index 1). Section **G** is optional repo status and must not override the charter.

---

## A. Official Snapshot / 官方项目概览

| Field | Value |
|-------|-------|
| **Project Index** | 1 |
| **Project Type** | Regular Project · Research Oriented |
| **Difficulty** | Challenging |
| **Team Size** | 2–3 students |
| **Prerequisites** | Machine Learning, Image Processing, Python |
| **Domain** | Promptable medical image segmentation · Endoscopic (colonoscopic) polyp segmentation |
| **Required outputs** | (1) Implementation codes · (2) Thesis and paper describing the work |

---

## B. Official Description / 官方描述（忠实 paraphrasing）

本项目聚焦 **promptable medical image segmentation**，以 **结肠镜息肉分割（colonoscopic polyp segmentation）** 为应用目标，研究如何在公开数据集上 **复现并改进 SAM / MedSAM** 的分割表现。

将搭建一条 **端到端流水线（end-to-end pipeline）**，覆盖：

1. **数据预处理**（colonoscopic image data）  
2. **从标注自动生成 box / point prompts**  
3. **模型推理**（SAM、MedSAM 及轻量适配变体）  
4. **参数高效微调 PEFT**（例如 Adapter / LoRA）  
5. **结果评估与可视化**

### B.1 Core Study / 核心研究内容

对以下三类模型做 **系统对比（comparative analysis）**，量化性能差异，并理解 **prompt 质量、目标尺度（object scale）、弱边界（weak boundaries）** 对分割精度的影响：

| Category | Description |
|----------|-------------|
| **Original SAM** | 未适配的零样本 SAM 基线 |
| **Zero-shot MedSAM** | 在大规模医学图像–掩码对上微调后的零样本 MedSAM |
| **Lightweight adapted models** | 在息肉数据上的 Adapter / LoRA 等轻量适配 |

### B.2 Evaluation Metrics / 评价指标

| Primary | Secondary |
|---------|-----------|
| **Dice** · **IoU** | **HD95** · **Inference speed**（e.g. FPS） |

### B.3 Expected Outcomes / 预期成果

1. **可复现实验脚本**（reproducible experimental scripts）  
2. **Prompt 生成工具包**（prompt generation toolkit：自标注自动提 box/point）  
3. **系统基线评测**（systematic baseline benchmarks）  
4. **可扩展方向（explore）**：  
   - 基于伪标签的 **半监督学习（pseudo-label semi-supervised learning）**  
   - **跨数据集泛化（cross-dataset generalization）**

---

## C. Completion Criteria / 完成标准（Index 1 原文意图）

Students need to **understand, re-implement, and improve** Polyp Segmentation to meet the SOTA:

1. **Implementation codes** of Polyp Segmentation  
2. **Thesis and paper** describing the details of the Polyp Segmentation  

---

## D. Other Requirements / 其他要求

- **Weekly meeting** with status report  
- At least **1 month** working load in Summer vacation  

---

## E. Official References / 官方参考文献（PDF 列表）

### E.1 Papers

| # | Citation |
|---|----------|
| [1] | Mazurowski et al. Segment anything model for medical image analysis: An experimental study. *Medical Image Analysis*, 2023. doi:10.1016/j.media.2023.102918 |
| [2] | Zhang & Liu. Customized Segment Anything Model for Medical Image Segmentation (**SAMed**). arXiv:2304.13785, 2023 |
| [3] | Huang et al. Segment anything model for medical images? *Medical Image Analysis*, 2024. doi:10.1016/j.media.2023.103061 |
| [4] | Ma et al. Segment anything in medical images (**MedSAM**). *Nature Communications*, 2024. doi:10.1038/s41467-024-44824-z |
| [5] | Wu et al. Medical SAM Adapter. *Medical Image Analysis*, 2025. doi:10.1016/j.media.2025.103547 |

### E.2 Code & Datasets

| # | Resource | URL |
|---|----------|-----|
| [6] | MedSAM | https://github.com/bowang-lab/MedSAM |
| [7] | Medical-SAM-Adapter | https://github.com/ImprintLab/Medical-SAM-Adapter |
| [8] | SAM medical evaluation | https://github.com/mazurowski-lab/segment-anything-medical-evaluation |
| [9] | **Kvasir-SEG** | https://github.com/DebeshJha/Kvasir-SEG |
| [10] | **CVC-612** (CVC-ClinicDB) | https://datasetninja.com/cvc-612 |

---

## F. Charter Technical Scope / 章程技术范围（非新增研究主张）

以下仅把 Index 1 描述落成可执行边界，**不新增 PDF 未写明的研究 claim**：

| Area | In-scope per Index 1 |
|------|----------------------|
| Models | Original SAM · zero-shot MedSAM · lightweight PEFT (Adapter / LoRA) |
| Data | Public polyp datasets named in PDF: **Kvasir-SEG**, **CVC-612** |
| Pipeline | Preprocess → auto prompts → inference → PEFT → eval / viz |
| Analysis | Prompt quality · object scale · weak boundaries vs accuracy |
| Extensions | Pseudo-label SSL · cross-dataset generalization（探索项） |
| Deliverables | Reproducible code · prompt toolkit · baseline benchmarks · thesis/paper |

**Recommended stack（与章程一致、可替换实现细节）**: PyTorch；SAM / MedSAM / SAMed-style LoRA or Medical SAM Adapter；OpenCV / NumPy；Dice、IoU、HD95、FPS。

---

## G. Current Project Status / 当前仓库状态（可选；服从章程）

> **地位**: 本节记录仓库事实，便于对齐检查。**若与 A–F 冲突，以 A–F（Index 1）为准。**  
> **数字权威源**: 正式 `results/**/summary.json`。可读台账是 `results/EXPERIMENT_SUMMARY.md`，抄表索引是 `docs/NUMBERS_INDEX.md`。勿在本 brief 硬编码整表；若与 json 冲突，以 json 为准。  
> **快照日期**: 2026-11-17。

### G.1 Datasets & protocol (repo)

| Item | Status |
|------|--------|
| Kvasir-SEG · CVC-ClinicDB (CVC-612) | 已就绪 |
| Eval protocol **v3** | Kvasir 官方 `val.txt` 120；CVC 全 612；**GT 紧框 + 固定外扩**（主表默认 30 px，**不是** random jitter）；Dice/IoU/P/R/HD95（+FPS，口径不统一，需重测） |
| Train split for adapted / few-shot | Kvasir `train.txt` 880（严格不泄漏 val） |
| ± in tables | 测试图像间标准差，**不是**多种子 |

### G.2 Implementation map vs Index 1 categories

| Index 1 category | Repo evidence (high level) | Notes |
|------------------|----------------------------|-------|
| Original SAM | `code/sam_baseline/`（含 ViT-B/L/H 等） | 满足“original SAM”对照；另有 HQ-SAM、SAM2、SAM3 等扩展零样本 |
| Zero-shot MedSAM | `code/medsam_baseline/` | 满足 |
| Lightweight adapted (Adapter/LoRA) | `code/samed_baseline/`（LoRA）；`code/medsam_peft_baseline/`（MedSAM+LoRA）；`code/medical_sam_adapter/`（**已训练**的息肉 MSA-B / MSA-L，以及 MedSAM+MSA） | 息肉域 Adapter / LoRA **已有正式结果**，不再是 future work。官方 MSA zoo 仍无现成息肉权重，那一行只作负向探针 |
| Prompt toolkit / auto box·point | 评估与训练脚本中含自 mask 生成 box（评估为固定外扩） | 宜整理成独立、可复用 toolkit 交付物；点 prompt 正式表尚未做 |
| Extra baselines（超出 PDF 最小集合） | MobileSAM、SAM2、SAM3、SAM-Med2D、SAMUS、ASPS、OP-SAM、PP-SAM、Chen SAM-Adapter 等 | **扩展对照**；不得替代章程三类核心对比叙事。**PP-SAM 是否写作 Ours 未定**，先作全参 / shot 曲线对照。Chen / ASPS / OP-SAM 协议与 box-prompt 主表不同，不能直接排名 |

### G.3 Headline results (pointer only)

完整表与来源路径见 `docs/NUMBERS_INDEX.md` 和 `results/EXPERIMENT_SUMMARY.md`。下面只给便于扫一眼的 headline（Dice）：

| Line | Kvasir-val | CVC |
|------|------------|-----|
| SAM ViT-B zero-shot | 0.7497 | 0.5182 |
| MedSAM zero-shot | 0.7034 | 0.5172 |
| HQ-SAM ViT-B / SAM3 zero-shot | 0.8007 / 0.7661 | 0.6525 / 0.6287 |
| SAMed LoRA r4 full 880 | 0.8955 | 0.8608 |
| MedSAM+LoRA r4 full 880 | 0.9049 | 0.8476 |
| MSA-B / MSA-L（息肉 Adapter，880） | 0.8524 / 0.8555 | 0.7685 / 0.8234 |
| PP-SAM 5-shot / full 880（扩展对照，非已定 Ours） | 0.8512 / 0.9226 | 0.8040 / 0.8576 |

### G.4 Thesis / process

| Item | Status |
|------|--------|
| Manuscript package | `manuscript/`（Partial Thesis 与 6 月表**已过时**，见 `manuscript/README_NUMBERS.md`；正文尚未按正式结果重写） |
| Weekly meeting · summer ≥1 month | 过程要求（Index 1 Other Requirements）— 由团队自行留存记录 |
| Student / owner naming in other docs | 见 `README.md` 等；本 brief 不修改人员字段 |

### G.5 Still open relative to charter

- 将 **object scale / weak boundaries** 做成与章程对齐的系统分析（**尚未做**；A1 5-shot 烟测不能代替弱边界分析）
- Prompt quality：已有四个已训模型的固定外扩 0/10/30/50 消融；**尚未做**零样本 SAM-L、零样本 MedSAM 消融，也还没有折线图
- **多种子 / 显著性检验**：未做
- **Prompt generation toolkit** 产品化（独立脚本/文档）
- 扩展项：**伪标签半监督**；**PolypGen 等跨数据集**（Kvasir↔CVC 已覆盖一部分“跨集”）
- 论文/thesis 定稿：旧表不能用；是否把 PP-SAM 写作 Ours 仍待定；FPS 若要上报需空闲重测

---

*Document role: project charter + thin status annex. Do not treat Section G as a license to redefine the FYP topic.*
