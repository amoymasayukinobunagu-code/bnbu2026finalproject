# 1 Introduction

## 1.1 Background and motivation

Endoscopic polyp segmentation delineates the visible extent of a polyp in a colonoscopy image. The output is a mask that assigns each pixel to foreground or background. A location cue such as a point or a box can narrow the region of interest, but the model must still infer the shape and boundary of the lesion from the image. Variation in apparent size, indistinct transitions between polyps and surrounding tissue, and differences across imaging sources can make this task difficult [8, 9, 10].

The Segment Anything Model (SAM) accepts spatial prompts and predicts a corresponding mask [3]. Its use on colonoscopy images raises two practical questions. The first is how well existing checkpoints segment polyps when given an informative prompt. The second is how much the observed performance changes when the models are adapted to labelled polyp images. MedSAM provides a medical-domain reference [4], while low-rank updates and adapter modules offer different ways to fit a segmentation model to a specific task. These approaches motivate a comparison that gives the evaluated configurations the same prompt information and uses the same outcome definitions.

## 1.2 Research problem

Performance figures from separate publications can describe different tasks. A supervised image-only polyp segmenter receives the image, while a promptable model may also receive a box around the lesion. Training partitions, test datasets and metric implementations can differ as well. To assess the models under a defined condition, this study compares SAM-based configurations on shared evaluation images using boxes generated from the reference masks.

The main experiments distinguish zero-shot evaluation from adaptation on the Kvasir-SEG training split [1]. Checkpoints are evaluated on the Kvasir-SEG validation split and CVC-ClinicDB [2]. This design makes it possible to examine performance within the development dataset and on a second dataset. Additional experiments test whether results depend on the prompt setting or vary across images grouped by relative polyp area or boundary contrast.

## 1.3 Research questions

RQ1. Under the same box-prompt protocol, how do zero-shot SAM and MedSAM differ in polyp segmentation performance on the Kvasir-SEG validation split and CVC-ClinicDB?

RQ2. After adaptation on the full Kvasir-SEG training split, how do the LoRA and Adapter configurations perform relative to the zero-shot baselines on Kvasir-SEG validation and CVC-ClinicDB?

RQ3. For the evaluated models, how does segmentation performance change with prompt type, box expansion or box translation? What differences are observed across groups defined by relative polyp area and boundary contrast?

RQ1 and RQ2 define the main comparison. RQ3 is exploratory. The effect of using smaller labelled support sets is examined as a supplementary analysis of RQ2.

## 1.4 Contributions and scope

This thesis establishes a common evaluation protocol for zero-shot and polyp-adapted configurations. It records how the prompts, data partitions and metrics are constructed so that each reported result can be interpreted under its evaluation conditions.

It compares LoRA and Adapter configurations trained on Kvasir-SEG against the available zero-shot baselines. PP-SAM provides an existing fine-tuning reference. The primary comparison reports within-dataset and cross-dataset performance. A follow-up evaluates five fixed configurations on two additional datasets.

It also examines sensitivity to box size, box position and point prompts, and reports descriptive differences across image-area and boundary-contrast groups. The main experiments concern individual images with prompts derived from the annotations. Relative foreground area and contrast describe image properties, and the conclusions apply to the evaluated configurations and prompt conditions.

## 1.5 Thesis organisation

Chapter 2 reviews polyp segmentation, promptable models and adaptation methods. Chapter 3 specifies the datasets, experimental comparisons, prompting protocols and evaluation measures. Chapter 4 describes the implementation and the information required to reproduce the reported results. Chapter 5 presents and discusses the findings for each research question. Chapter 6 summarises the conclusions and outlines further work.
