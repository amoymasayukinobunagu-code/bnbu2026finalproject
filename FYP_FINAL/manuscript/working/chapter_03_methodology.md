# 3 Research Methodology

## 3.1 Study design

This study evaluates promptable segmentation models on two-dimensional colonoscopy images. Each prediction is made from one image and its prompt. The main comparison uses a common evaluation split, an annotation-derived box and the same outcome definitions. This setup allows the performance of zero-shot and polyp-adapted configurations to be compared under a common prompting protocol.

RQ1 examines zero-shot SAM and MedSAM. RQ2 compares zero-shot baselines with models adapted using the full Kvasir-SEG training partition. RQ3 explores variation in performance under different prompt settings and in groups defined by foreground area and boundary contrast. Table 3.1 maps these questions to the experiments.

Table 3.1 Research questions and corresponding comparisons

| Question | Comparison |
|---|---|
| RQ1 Zero-shot performance | Evaluate SAM ViT-B, SAM ViT-L and MedSAM on the same frames using the default box. |
| RQ2 Polyp-specific adaptation | Compare the full-data LoRA and Adapter configurations with zero-shot baselines. Use PP-SAM as an existing fine-tuning reference. |
| RQ3 Prompt and image conditions | Compare box expansion, box translation and point prompts. Summarise performance by relative area and boundary contrast. |

The annotation-budget comparison supplements RQ2. It uses the recorded SAMed and PP-SAM runs with 1, 5, 10 and 880 labelled training images. The smaller support sets come from the same training partition. A separate 5-shot SAMed pilot crossed three optimisation seeds with fixed and redrawn support settings, yielding six training runs. The pilot used a 12-epoch schedule at a learning rate of 0.0005 and evaluated the final checkpoint. It used the full Kvasir validation split and one frame from each of the 29 locally indexed CVC sequences. The shortened schedule and the sampled CVC set define a separate repeatability analysis.

## 3.2 Datasets and partitioning

Kvasir-SEG provides endoscopic polyp images and pixel-level masks [1]. In this study, the filename lists specify 880 training images and 120 validation images. The lists have no overlapping identifiers. Adaptation uses the training partition. The validation partition is used for model selection where applicable and for reporting within-dataset performance.

CVC-ClinicDB was used for cross-dataset evaluation [2]. The evaluated set contains 612 annotated frames. No CVC-ClinicDB images were used to fit or select the polyp-adapted checkpoints in the main comparison. Local frame metadata group these 612 images into 29 sequences for the exploratory statistical analysis in Section 3.6.1.

ETIS-LaribPolypDB [12] and CVC-ColonDB [13] provide an additional evaluation of five fixed configurations. The local TestDataset copies contained 196 and 380 files, respectively, matching the counts documented for the PraNet redistribution [14]. The screened inputs are identified by image and binary-mask hashes in the evaluation archive. The follow-up evaluated zero-shot SAM ViT-B, MSA-B, SAMed, MedSAM+LoRA and PP-SAM with fixed checkpoints and 30-pixel boxes. Exact SHA-256 checks covered the local Kvasir-SEG and CVC-ClinicDB sets and the two external sets. Repeated images with identical annotations were counted once, and duplicate groups with conflicting masks were excluded. The retained sets contained 190 and 342 images. Exact file hashing can miss re-encoded copies, and the provenance of images used in base-model pretraining is unknown.

Table 3.2 Data partitions and analysis roles

| Dataset or partition | Images | Role |
|---|---|---|
| Kvasir-SEG training | 880 | Fit adapted models and draw few-shot support sets. |
| Kvasir-SEG validation | 120 | Select checkpoints where applicable and evaluate performance within Kvasir-SEG. |
| CVC-ClinicDB | 612 | Evaluate cross-dataset performance and analyse paired prompt changes by video sequence. |
| ETIS-LaribPolypDB | 190 | Evaluate five fixed configurations on an additional dataset. |
| CVC-ColonDB | 342 | Evaluate the same five configurations on a second additional dataset. |

## 3.3 Model architectures and comparison design

### 3.3.1 Shared SAM architecture

The eight configurations share the promptable segmentation architecture shown in Figure 3.1 [3]. A Vision Transformer (ViT) encodes the RGB endoscopy frame as a spatial feature grid [30]. A separate prompt encoder converts the coordinates of a box or point into embeddings. The mask decoder combines both representations through attention and returns mask logits and an estimated intersection over union (IoU) for each candidate. When several prompts are evaluated on the same frame with a fixed checkpoint, the image embedding can be reused.

![An endoscopy frame enters the ViT image encoder. A box or point on the same frame enters the prompt encoder. Both embeddings feed the mask decoder. The example output shows a SAM-B mask under the box condition.](figures/implementation/sam_daisy_trial/sam_shared_architecture.png)
Figure 3.1 Shared SAM architecture for promptable polyp segmentation, based on Kirillov et al. [3], Fig. 4, and Ma et al. [4], Fig. 2(b). The overlay shows a saved SAM-B prediction for a Kvasir-SEG validation frame with the default box.

### 3.3.2 Zero-shot configurations

SAM-B uses the released SAM ViT-B checkpoint. Its image encoder contains 12 Transformer blocks with a token dimension of 768. SAM-L uses the released ViT-L checkpoint, with 24 blocks and a token dimension of 1,024. The two SAM configurations use the same prompt interface and image preprocessing, so their comparison describes how the two released backbones perform under matched evaluation conditions.

MedSAM also uses ViT-B, with weights obtained through medical-domain training [4]. It was evaluated through the same SAM predictor interface as SAM-B. Comparing these two ViT-B checkpoints examines how each transfers to the polyp task. All three baseline checkpoints were used directly, with no further training on the Kvasir-SEG split.

### 3.3.3 LoRA in SAMed and MedSAM+LoRA

Both configurations add LoRA to the self-attention layers of a ViT-B image encoder [5, 11]. SAMed starts from SAM ViT-B and MedSAM+LoRA starts from MedSAM ViT-B. Figure 3.2 shows how the local implementation sends the same input tokens through the frozen query–key–value projection and two independent trainable branches. In each branch, the first linear map reduces the token dimension from 768 to 4 and the second restores it to 768. The branch outputs are added to the query and value projections, respectively. The original key projection feeds attention directly. Training updates the LoRA branches, the prompt encoder and the mask decoder.

![The shared architecture and starting weights are shown above. Below, independent low-rank branches add to the frozen Q and V projection outputs while K passes directly to attention.](figures/implementation/lora_daisy_trial/lora_samed_medsam_architecture.png)
Figure 3.2 LoRA in SAMed and MedSAM+LoRA. (A) The two configurations use different ViT-B initialisations and the same adaptation structure. (B) Rank-four branches add to the query and value projections. The schematic follows Zhang and Liu [5], Figs. 2–3, and Hu et al. [11], Fig. 1.

### 3.3.4 Adapters in MSA-B and MSA-L

MSA-B and MSA-L insert the two-dimensional Medical SAM Adapter modules into ViT-B and ViT-L, respectively [6]. Figure 3.3 locates the adapters in the image encoder and enlarges one Transformer block. The Spatial Adapter processes the attention output, with an internal skip connection, before the block adds the original input. The resulting tensor is normalised and passed to both the frozen feed-forward network and the MLP Adapter. The latter output is scaled by 0.5, then both outputs are added to the feed-forward residual. Training updates both adapters and the mask decoder. The original image-encoder parameters and the prompt encoder are held fixed.

![The upper panel locates the adapted ViT blocks within SAM and shows an MSA-B prediction. The lower panel shows both adapter branches and their residual connections.](figures/implementation/msa_daisy_trial/msa_architecture_overview.png)
Figure 3.3 Medical SAM Adapter in MSA-B and MSA-L. (A) The adapted blocks and mask decoder are trainable. The overlay is a saved MSA-B prediction for a Kvasir-SEG validation frame with the default box. (B) A ViT block contains a Spatial Adapter after attention and an MLP Adapter parallel to the feed-forward layer. The block schematic draws on Wu et al. [6], Fig. 1(b).

### 3.3.5 Fine-tuning and box perturbation in PP-SAM

PP-SAM starts from SAM ViT-B and updates the full image encoder and the prompt encoder. The mask decoder combines their outputs using fixed weights [7]. As illustrated in Figure 3.4, the training pipeline draws a tight box around the annotated foreground, independently samples an outward expansion between 0 and 50 pixels for each side, and clips the expanded coordinates to the frame boundaries. The primary evaluation instead uses the common fixed expansion specified in Section 3.4.

![The upper panel shows the trainable image and prompt encoders and the frozen mask decoder. The lower panel shows the annotation mask, tight box and an example perturbed box on a training frame.](figures/implementation/ppsam_daisy_trial/ppsam_architecture.png)
Figure 3.4 PP-SAM fine-tuning and training prompt construction, based on Rahman et al. [7], Fig. 1. The tight and perturbed boxes are shown on the same Kvasir-SEG training frame. The yellow outline illustrates one set of unequal outward offsets.

### 3.3.6 Comparison design

Table 3.3 summarises the starting weights, updated modules and inference settings. The primary adapted configurations use the same 880 labelled Kvasir-SEG training images. The MSA evaluators resize each frame to 512 × 512 and select the candidate with the highest estimated IoU. The other configurations resize the longest side to 1,024 pixels and request a single mask.

Table 3.3 Architecture and inference settings for the eight primary configurations

| Model | Starting weights | Locally updated modules | Evaluation input and mask policy |
|---|---|---|---|
| SAM-B | SAM ViT-B | None | 1,024 px, one mask |
| SAM-L | SAM ViT-L | None | 1,024 px, one mask |
| MedSAM | MedSAM (ViT-B) | None | 1,024 px, one mask |
| SAMed | SAM ViT-B | LoRA on Q and V, prompt encoder, mask decoder | 1,024 px, one mask |
| MedSAM+LoRA | MedSAM (ViT-B) | LoRA on Q and V, prompt encoder, mask decoder | 1,024 px, one mask |
| MSA-B | SAM ViT-B | Spatial and MLP adapters, mask decoder | 512², top predicted IoU |
| MSA-L | SAM ViT-L | Spatial and MLP adapters, mask decoder | 512², top predicted IoU |
| PP-SAM | SAM ViT-B | Full image encoder and prompt encoder | 1,024 px, one mask |

RQ1 compares the released checkpoints under the same annotation-derived box protocol. For RQ2, SAM-B serves as the starting-weight reference for SAMed, MSA-B and PP-SAM, while MedSAM and SAM-L serve as the references for MedSAM+LoRA and MSA-L, respectively. These comparisons capture each configuration as evaluated. In the MSA comparisons, the input resolution and mask selection also differ from the reference. In PP-SAM, the adapted modules and the training prompt distribution both change. For the prompt component of RQ3, each prompt variant is evaluated within a fixed model configuration. Chapter 4 gives the training and inference settings.

## 3.4 Prompt construction and variants

The default evaluation prompt was derived from the tight bounding box of the annotated foreground in the original image. Let the box coordinates be x_min, y_min, x_max and y_max, with maxima defined by the foreground pixel indices. Each minimum is decreased by the chosen expansion and each maximum is increased by the same amount. The resulting coordinates are clipped to the image bounds using the evaluation code's coordinate convention. The default expansion is 30 pixels. Prompt generation takes place before any model-specific coordinate transformation. The same prompt is supplied to every configuration in the main comparison.

Figure 3.5 shows how the reference mask and an annotation box relate to the original frame.

![Six Kvasir-SEG examples. Each column shows a colonoscopy frame, its binary annotation and the frame with an annotation box.](legacy_reference/figures/chapter2/fig_2.12_Kvasir-SEG-sample.png)
Figure 3.5 Examples of image, mask and box annotations in Kvasir-SEG. The bottom row shows dataset boxes. The 30-pixel expansion used in this study is applied separately. Source: Jha et al. [1], Fig. 2.

The expansion experiment uses deterministic outward expansions of 0, 10, 30 and 50 pixels on selected trained configurations. Its analysis uses runs with traceable raw outputs for the plotted comparisons. The local SAMed expansion series uses its own 30-pixel rerun, while the main comparison retains the formal SAMed checkpoint evaluation. The training-time random box perturbation in PP-SAM serves a separate purpose from these fixed evaluation settings.

The point and box-translation experiments use zero-shot SAM ViT-B and MSA-B. For translations, the 30-pixel reference box is shifted in each cardinal direction. The requested displacement is 5% of the corresponding image dimension, rounded to a whole pixel. When the image edge prevents the full shift, the displacement is reduced while box width and height are preserved. The overall analysis retains these cases. A supplementary comparison records results for the subset that received the full requested displacement in each direction.

Point prompts are derived from the same annotation. One condition uses the foreground pixel farthest from the background, with ties resolved in row-major order. A second condition adds one sampled negative point from the background using base seed 24. A third condition samples one positive point uniformly from the foreground for each of three base seeds (24, 42 and 1234). The seed for a frame is derived from the base seed and a stable hash of the image identifier. Within each model, its decoder and mask-selection settings are held constant across prompt conditions.

## 3.5 Evaluation and subgroup analysis

### 3.5.1 Evaluation metrics

Metrics are computed between a binary prediction and the reference mask at native image resolution. Let TP, FP and FN denote the numbers of true-positive, false-positive and false-negative foreground pixels. The Dice coefficient is 2TP/(2TP + FP + FN), and intersection over union (IoU) is TP/(TP + FP + FN). Precision is TP/(TP + FP), and recall is TP/(TP + FN). Reported precision follows the archived implementation's convention of assigning 1 when the prediction contains no positive pixels. For an empty prediction on an image with annotated foreground, Dice, IoU, recall and boundary F1 are zero. The reported segmentation means give one weight to each evaluated image. Standard deviations, where reported, describe variation across images.

Boundary F1 measures contour agreement. A boundary pixel is obtained from the difference between a foreground mask and its four-connected erosion. The matching tolerance is 0.5% of the image diagonal, rounded to a whole pixel with a minimum of one pixel. Boundary precision is the proportion of predicted boundary pixels within the tolerance of the reference boundary. Boundary recall uses the reverse direction. Their harmonic mean gives boundary F1.

Distance metrics complement overlap metrics by describing the spatial separation between contours [27]. For surface HD95, each boundary pixel is matched to its nearest point on the other boundary. This study takes the 95th percentile separately in each direction and reports the larger value in original-image pixels. An empty prediction yields an undefined surface HD95. The same implementation recomputed the metric for 28 configuration–dataset entries in the results archive, including the few-shot runs. Tables 5.1 and 5.2 show the 16 entries in the primary comparison. The saved predictions were reconciled against the recorded overlap metrics before recomputation.

### 3.5.2 Image subgroup definitions

Relative polyp area is the fraction of original-image pixels labelled as foreground. The primary groups use area fractions of at most 5%, greater than 5% up to 20%, and greater than 20%. A sensitivity analysis also uses the area tertiles estimated from the 880 Kvasir training masks and applies those cutoffs to both primary evaluation sets. Each area group describes how much of the image is occupied by annotated foreground.

Boundary contrast is the Euclidean distance between the mean CIELAB colours in narrow bands immediately inside and outside the annotated boundary. The primary band width is 0.5% of the image diagonal, rounded to the nearest pixel with a minimum of two pixels. The lower and upper quartiles within each dataset define the low, intermediate and high contrast groups. Alternative band widths and the median grayscale gradient magnitude along the annotated boundary provide sensitivity checks. Area and contrast groups are also crossed to examine how their overlap may affect the descriptive results.

Figure 3.6 illustrates examples selected by the contrast measure within the medium-area group.

![Two rows show Kvasir-SEG and CVC-ClinicDB examples. Each row contains two lower-contrast and two higher-contrast frames with cyan reference contours.](figures/supplementary/boundary_proxy_examples.png)
Figure 3.6 Boundary-contrast examples from Kvasir-SEG validation and CVC-ClinicDB. Each row shows the two lowest and two highest proxy values within that dataset's medium-area group. Cyan indicates the annotated contour. Selection used image and annotation features.

### 3.5.3 Inference timing

SAM ViT-B, SAM ViT-L and MSA-B were benchmarked at batch size one on an NVIDIA GeForce RTX 4070 Ti SUPER. Thirty images from each primary dataset were sampled without replacement using a fixed seed. After five warm-up images per dataset and configuration, the same 30 images were used in three timed passes. GPU work was synchronised around each timer. Latency includes model-specific preprocessing, image encoding, prompt decoding and returning the mask to the host. The timed interval excludes disk reads, model loading and prompt generation. Each model retained its configured input resolution. Frames per second are computed by dividing 1,000 by the mean latency in milliseconds.

## 3.6 Statistical analysis and validity

### 3.6.1 Statistical comparisons

For the point and box-translation study, the effect for a frame is the metric under the tested prompt minus the metric under that frame's 30-pixel reference box. The three trials for the random-positive condition are averaged within each frame before aggregation. This yields a matched comparison of prompts for each configuration.

For CVC-ClinicDB, frame-level Dice changes are averaged within each of the 29 video sequences, then averaged equally across sequences. The exploratory analysis considers 21 contrasts across seven nonreference prompt conditions. These comprise the paired change within SAM ViT-B, the paired change within MSA-B and the difference between the two configurations' paired changes. Two-sided Monte Carlo sign-flip tests use 100,000 draws and assume independent, symmetric sequence-level paired differences under the null. Holm's procedure adjusts the p values across all 21 contrasts [28]. Unadjusted 95% percentile intervals are obtained by resampling entire sequences with replacement [29]. Kvasir prompt results and the image subgroup comparisons are reported descriptively.

### 3.6.2 Validity of comparisons

The annotation-derived box and points provide lesion location information during evaluation. The resulting comparisons characterise segmentation under supplied prompts. The validation scores for Kvasir-SEG also reflect checkpoint selection, and the local CVC sequence groups define the sampling unit for the statistical analysis.

The area and contrast analyses were conducted after the primary experiments. A fixed expansion in pixels changes the box size proportionally more for smaller boxes, so differences between area groups may reflect a combination of foreground scale and prompt geometry. The colour and gradient measurements are image-derived proxies for boundary visibility and can be affected by illumination, texture and reflections. Expert boundary ratings would be needed to determine how closely these proxies agree with human assessments. The few-shot pilot addresses training variation only for its own short schedule and the tested support draws.

## References cited in this chapter draft

[1] D. Jha et al., “Kvasir-SEG: A Segmented Polyp Dataset,” in Multimedia Modeling, 2020, pp. 451–462. doi: 10.1007/978-3-030-37734-2_37.

[2] J. Bernal et al., “WM-DOVA maps for accurate polyp highlighting in colonoscopy: Validation vs. saliency maps from physicians,” Computerized Medical Imaging and Graphics, vol. 43, pp. 99–111, 2015. doi: 10.1016/j.compmedimag.2015.02.007.

[3] A. Kirillov et al., “Segment Anything,” in Proc. IEEE/CVF International Conference on Computer Vision, 2023, pp. 4015–4026. arXiv:2304.02643.

[4] J. Ma et al., “Segment anything in medical images,” Nature Communications, vol. 15, art. 654, 2024. doi: 10.1038/s41467-024-44824-z.

[5] K. Zhang and D. Liu, “Customized Segment Anything Model for Medical Image Segmentation,” arXiv:2304.13785, 2023.

[6] J. Wu et al., “Medical SAM adapter: Adapting segment anything model for medical image segmentation,” Medical Image Analysis, vol. 102, art. 103547, 2025. doi: 10.1016/j.media.2025.103547.

[7] M. M. Rahman et al., “PP-SAM: Perturbed Prompts for Robust Adaption of Segment Anything Model for Polyp Segmentation,” in Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops, 2024, pp. 4989–4995.

[12] J. Silva, A. Histace, O. Romain, X. Dray, and B. Granado, “Toward embedded detection of polyps in WCE images for early diagnosis of colorectal cancer,” International Journal of Computer Assisted Radiology and Surgery, vol. 9, no. 2, pp. 283–293, 2014, doi: 10.1007/s11548-013-0926-3.

[13] J. Bernal, F. J. Sánchez, and F. Vilariño, “Towards automatic polyp detection with a polyp appearance model,” Pattern Recognition, vol. 45, no. 9, pp. 3166–3182, 2012, doi: 10.1016/j.patcog.2012.03.002.

[14] D.-P. Fan et al., “PraNet benchmark test dataset,” PraNet project repository. [Online]. Available: https://github.com/DengPingFan/PraNet#31-trainingtesting
