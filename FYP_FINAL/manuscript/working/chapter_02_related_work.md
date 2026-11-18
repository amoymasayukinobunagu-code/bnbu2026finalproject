# 2 Related Work

## 2.1 Endoscopic polyp segmentation

Supervised polyp segmentation methods learn to map endoscopic images to binary masks. PraNet combines features into an initial polyp prediction and uses reverse attention to refine regions that the initial prediction leaves unresolved [8]. UACANet explicitly models uncertain regions in an intermediate saliency map and uses this information when refining its predictions [9]. These studies illustrate two reasons to examine more than average overlap. Fine structures may be lost when features are combined across scales, and a plausible foreground region may still have an inaccurate contour.

Cross-dataset performance is also relevant. The PolypGen data descriptor documents images and sequences from multiple centres and frames generalisability as a distinct evaluation problem [10]. In this thesis, Kvasir-SEG is the development dataset and CVC-ClinicDB provides the principal cross-dataset comparison [1, 2]. Relative foreground area captures the amount of annotated polyp visible within the image. Boundary contrast provides a separate image-derived description of the transition around the reference contour. Both properties can vary between datasets, which motivates reporting subgroup results separately for each dataset.

PraNet's coarse-to-fine design offers a way to revisit ambiguous regions, while UACANet explicitly separates predictions by their estimated uncertainty. Both can run from an image alone, which makes them relevant to automatic segmentation. A reported score for either method also reflects its training partitions and test images. The present benchmark gives each configuration an annotation-derived box, so a direct comparison with an image-only score would require an evaluation that accounts for the extra location information.

## 2.2 SAM and medical-domain adaptation

SAM separates image encoding, prompt encoding and mask decoding [3]. The image encoder produces a representation of the frame. The prompt encoder represents the spatial cue. The mask decoder uses both representations to predict the segmentation. A point conveys a location, while a box conveys location and approximate extent. This distinction matters in colonoscopy because the amount of information supplied by the prompt can affect the apparent difficulty of the task.

MedSAM adapts the SAM framework using medical images and annotations [4]. It offers a relevant comparison with general-purpose SAM because the two checkpoints reflect different training domains. The comparison in this thesis evaluates both without further fitting on Kvasir-SEG. It therefore addresses the behaviour of the selected pretrained configurations on the evaluated polyp datasets. The same prompting protocol makes their outputs comparable for this task, while checkpoint and input pipeline differences remain part of the configurations being tested.

Figure 2.1 illustrates how MedSAM combines the image embedding with a bounding-box prompt to produce a mask.

![Two panels show the distribution of medical training examples by modality and the MedSAM image and prompt encoders feeding a mask decoder.](legacy_reference/figures/chapter2/fig_2.1_MedSAM.png)
Figure 2.1 MedSAM training modalities and box-prompted architecture. The CT example illustrates the published model. Source: Ma et al. [4], Fig. 2.

## 2.3 Parameter-efficient adaptation

Low-rank adaptation (LoRA) represents a weight update as the product of two smaller trainable matrices [11]. The rank determines the size of that update. SAMed applies this approach to SAM for medical image segmentation [5]. A LoRA configuration can still update components outside the layers with low-rank updates. For reproducibility, the exact set of trainable modules must therefore be stated alongside the rank.

Medical SAM Adapter introduces trainable adapter components into SAM for medical image segmentation [6]. LoRA constrains updates to selected weight matrices, while adapters add a separate trainable path within the network. Both approaches offer a way to adapt the image encoder without updating all its pretrained weights. In these implementations, other modules can also be trainable. A comparison between the complete recipes therefore measures the combined effects of initial weights, trainable modules and inference settings. Section 4.3 documents the modules updated in each configuration.

Table 2.1 Roles of the main model families in the study

| Model family | Role in this study |
|---|---|
| SAM and MedSAM | Pretrained zero-shot references under the common box-prompt protocol. |
| SAMed and MedSAM with LoRA | Low-rank polyp adaptation compared with the appropriate available zero-shot baselines. |
| Medical SAM Adapter | Polyp-trained adapter configurations with ViT-B and ViT-L backbones. |
| PP-SAM | Published polyp adaptation method used as a full-encoder fine-tuning reference. |

## 2.4 Limited annotation and prompt sensitivity

Few-shot adaptation examines what can be achieved with a small set of labelled training images. PP-SAM used perturbed bounding-box prompts during fine-tuning and evaluated its sensitivity to such perturbations at inference [7]. This strategy directly addresses variation in boxes. Its results depend on how training perturbations and evaluation boxes are generated. In this study, PP-SAM provides a full-encoder reference alongside the lightweight adaptations. The number of labelled training images governs supervision during adaptation, while the inference prompt determines the spatial information available for an individual prediction.

These choices are evaluated separately here. The annotation-budget comparison reports SAMed and PP-SAM runs trained with 1, 5, 10 and 880 images. The prompt comparisons reuse fixed checkpoints and change the prompts at inference. Outward expansion changes the amount of surrounding tissue included in the box. Translation changes the position of a box whose dimensions are held fixed. A point and a box supply different spatial cues, so the point-versus-box comparison concerns a change in prompt representation.

## 2.5 Position of this study

Prior work provides supervised systems for delineating polyps, general and medical promptable checkpoints, and several ways to adapt the checkpoints with labelled data. The present study brings selected configurations together under a common box-prompt protocol for two-dimensional colonoscopy images. It compares zero-shot and polyp-adapted performance across the designated datasets and then examines where outcomes differ with prompt conditions, foreground area and boundary contrast. These complementary comparisons connect the methods reviewed in this chapter to the experimental design in Chapter 3.
