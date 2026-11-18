# FYP Thesis Writing Plan and Study Design

Draft for author review; 2026-11-18.


## 1 Introduction

从研究动机引出可由现有实验回答的问题，明确本研究的贡献与适用条件。


### 1.1 Background and motivation
说明息肉像素级分割的任务及困难，引出可提示基础模型与领域适配的研究动机。临床意义须由原论文支持，避免将本项目结果直接写成临床收益。


### 1.2 Research problem
提出核心问题：在真实掩膜生成的框提示下，SAM 系列模型的息肉分割表现如何，以及领域适配、标注预算和框外扩范围如何影响表现。


### 1.3 Research questions
设置四个问题：零样本 SAM 与 MedSAM 的表现；全数据适配的域内与跨数据集表现；少样本方案随标注预算的变化；固定框外扩条件下的敏感性。


### 1.4 Contributions and scope
贡献聚焦复现与适配流程、明确协议下的比较评估，以及数据预算和提示敏感性的经验分析。界定真实框提示、单次训练与两数据集的范围。


### 1.5 Thesis organisation
用一个短段落概述各章内容，保持与最终章节一致。

本章不放完整性能表。背景与动机需补经核验的文献；研究问题与第5章四组分析逐一对应。


## 2 Related Work

解释选取这些方法的依据，并指出本研究能够补充的经验性比较。


### 2.1 Endoscopic polyp segmentation
概述任务、数据集及监督分割研究。引用不同论文的性能时注明划分和提示条件，不构造未经对齐的排名。


### 2.2 SAM and medical adaptation
介绍 SAM 的可提示分割机制及 MedSAM 的医学适配背景，区分预训练能力与本项目中的息肉训练。


### 2.3 Parameter efficient adaptation
介绍 LoRA、SAMed 和 Medical SAM Adapter，说明参数更新方式。具体插入位置及训练范围在核对实现后写入。


### 2.4 Limited annotation and prompt sensitivity
介绍 PP-SAM 及与标注预算、提示信息相关的研究，为少样本和框消融实验提供依据。


### 2.5 Position of this study
归纳本研究关注的比较问题和协议边界。3D、视频和大批额外模型仅保留与论点直接相关的内容。

计划表2.1：方法、适配方式、训练数据需求及提示输入。现有34条 BibTeX 是待核验来源，不等同于已核验引用。


## 3 Research Methodology

解释研究如何设计，以及每组实验能够回答什么问题。


### 3.1 Study design
以本稿后半部分的英文样章为基础，说明四组比较、共同评估条件及解释边界。


### 3.2 Datasets and partitioning
说明 Kvasir 880张训练、120张验证和 CVC 612张评估的角色。明确验证集参与权重选择，不能称为独立测试集。


### 3.3 Model groups and comparisons
区分零样本、LoRA、Adapter 和 PP-SAM 全参参照；列明骨干、初始化及可训练模块，避免把不同训练方案视为严格单因素消融。


### 3.4 Prompt construction
定义真实掩膜到紧框、固定向外扩展及图像边界裁剪的流程。区分训练时随机框变化和评估时确定性外扩。


### 3.5 Evaluation and experimental comparisons
定义 Dice、IoU、Precision、Recall 和 HD95，并核对聚合、空掩膜及单位处理；说明 full、1/5/10-shot 与0/10/30/50像素实验。


### 3.6 Interpretation and validity
解释图像间标准差、单次训练、验证集选择、提示信息和不同实现的影响。FPS 未统一前不作效率排名。

计划图3.1：整体研究框架；图3.2：掩膜到外扩框示意；表3.1：数据角色；表3.2：模型与实验设置。


## 4 Design and Implementation

提供足够的实现细节，使读者能够重现实验；不重复第3章的研究动机。


### 4.1 Experimental pipeline
描述图像、标注、划分和模型封装如何连接，以及训练、推理和指标计算的输入输出。


### 4.2 Data processing and prompt generation
核对各模型的缩放、归一化、坐标变换和预测掩膜恢复流程，明确共同协议与模型特定处理。


### 4.3 Model adaptation and training
分别说明 SAMed、MedSAM+LoRA、MSA 和 PP-SAM 的初始化、更新模块、损失、优化器、训练轮数和权重选择。以实际记录为准。


### 4.4 Inference and metric computation
描述提示输入、多个候选掩膜的选择、阈值和后处理，以及 summary 与 per-image 输出的生成。


### 4.5 Reproducibility records
列出实际环境、随机种子、配置、权重及结果路径；区分正式运行、烟测和历史版本。

计划图4.1：训练与评估数据流；必要时图4.2：经代码核验的 LoRA／Adapter 示意；表4.1：实际训练配置。


## 5 Results and Discussion

以四个研究问题组织结果，每节按观察、证据、解释和边界展开。


### 5.1 Evaluation setting
简要重申两个评估集、默认30像素外扩及指标口径，引导读者阅读结果表。


### 5.2 Zero shot performance
比较 SAM-B、SAM-L 和 MedSAM。说明当前协议下 MedSAM 未高于 SAM-B，不将观察直接归因为某种预训练数据差异。


### 5.3 Polyp specific adaptation
比较 full-880 的 SAMed、MedSAM+LoRA、MSA-B／L 和 PP-SAM，讨论域内和 CVC 排序的变化。


### 5.4 Annotation budget
展示 SAMed 与 PP-SAM 的1/5/10/full结果，保留非单调变化；没有多种子证据时，不将变化归结为确定的算法规律。


### 5.5 Box expansion sensitivity
比较0/10/30/50像素外扩，说明最优范围随方法和数据集变化。保留 SAMed 两来源差异及 MedSAM CVC 用户提供数值的标记。


### 5.6 Qualitative examples and limitations
使用正式权重预测展示正确与失败案例，明确选图规则；讨论真实框提示、验证集复用及未完成分析带来的限制。

使用现有主表、shot表和消融表。计划图5.1：标注预算曲线；图5.2：框消融曲线；图5.3：原图、真实掩膜与预测对比。不同协议结果放附录。


## 6 Conclusion and Future Work

直接回答研究问题，总结实际完成的工作和证据允许的结论。


### 6.1 Summary of findings
分别回答零样本、适配、标注预算和框外扩四个问题；不新增结果，不把单次观察写成普遍规律。


### 6.2 Contributions and limitations
总结实现与评估贡献，明确 PP-SAM 为已有方法。未做多种子、尺度分层和系统弱边界分析的事实应如实交代。


### 6.3 Future work
优先讨论重复种子与支持集抽样、独立测试、自动提示和尺度／边界分析。已完成的 MSA 训练和框外扩实验不再列为未来工作。

摘要最后撰写，按学校模板控制在250—400词。附录保存额外基线、不同提示协议结果及必要的复现细节。


## 3.1 Study design

This study evaluates SAM-based models for endoscopic polyp segmentation and examines how their performance changes with polyp-specific adaptation, annotation budget, and the spatial extent of the box prompt. The experimental comparisons address four questions: how zero-shot SAM and MedSAM perform under the selected evaluation conditions; how polyp-specific adaptation changes performance within and across datasets; how the evaluated adaptation recipes behave with different numbers of labelled training images; and how fixed outward box expansion affects segmentation accuracy. The comparisons are descriptive. They are used to characterise the evaluated models and training recipes rather than to establish a universal ranking of adaptation methods.

The data partition separates model fitting from validation. Kvasir-SEG provides 880 training images and 120 validation images, and the supplied filename lists have no overlap. Adaptation uses images from the training partition, including the smaller support sets used for the few-shot experiments. The validation partition is also used to select checkpoints during training. Its reported scores therefore describe performance on the model-selection validation set, rather than on an independent test set. CVC-ClinicDB, comprising 612 images, provides the cross-dataset evaluation set. The two evaluation sets are reported separately so that performance on the development dataset can be distinguished from performance on another dataset.

The first comparison establishes the zero-shot reference using SAM with ViT-B and ViT-L backbones and MedSAM. The second comparison evaluates polyp-specific adaptation using SAMed LoRA, MedSAM with LoRA, and Medical SAM Adapter with ViT-B and ViT-L backbones. PP-SAM is included as an existing full-parameter adaptation reference. These methods differ in their initial weights, trainable components, and training procedures. A shared evaluation partition and prompt-generation rule improve the interpretability of their results, but do not isolate the causal effect of any single architectural or optimisation choice. Implementation and training details are therefore reported for each method.

The annotation-budget comparison uses the available SAMed and PP-SAM runs with 1, 5, 10, and 880 labelled training images. The smaller support sets are drawn from the Kvasir training partition, while the full-data setting uses all 880 training images. This comparison examines the observed relationship between annotation budget and performance for the recorded runs. It does not assume that performance increases monotonically with the number of images. Because repeated training seeds and repeated support-set draws are not available, differences between budgets are interpreted as outcomes of the evaluated runs, not as estimates of the expected performance over all possible training samples.

Evaluation prompts are generated from the ground-truth masks. A tight bounding box is expanded outward by a fixed distance in original-image pixels and clipped to the image boundary. The main comparison uses an expansion of 30 pixels. Additional evaluations use expansions of 0, 10, 30, and 50 pixels for selected trained models. This procedure changes box extent deterministically; it does not simulate random displacement of a user-provided box. The results consequently describe segmentation conditional on access to a ground-truth-derived location prompt. They do not measure the performance of a fully automatic detection and segmentation system.

Dice and intersection over union quantify mask overlap, with precision, recall, and HD95 providing complementary descriptions of foreground prediction and boundary error. Where reported, the standard deviation describes variation across evaluation images rather than variation across training seeds. The main comparisons are restricted to models receiving the prescribed box prompts. Methods evaluated without those prompts are presented separately because they use different input information. Together, these design choices allow the thesis to relate the observed results to their data, training, and prompt conditions while keeping the conclusions within the available evidence.