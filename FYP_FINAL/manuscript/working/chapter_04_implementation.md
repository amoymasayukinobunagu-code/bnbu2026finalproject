# 4 Design and Implementation

## 4.1 Experimental pipeline

The pipeline loaded released weights for the zero-shot baselines and fitted the adapted configurations on the Kvasir-SEG training partition. Checkpoint selection used mean Dice on Kvasir-SEG validation. Both branches converged at evaluation, where the scripts passed a common image and annotation-derived box through each configuration's input transformation and decoder (Figure 4.1).

![A two-branch flowchart shows released zero-shot checkpoints and checkpoints adapted on Kvasir-SEG converging in a common evaluation pipeline. Each evaluation frame and annotation provide the image and box prompt. Native-resolution masks are scored against the reference annotations.](figures/implementation/experimental_pipeline.png)
Figure 4.1 Training and evaluation pipeline for the primary comparison. Reference boxes for inference are derived from the corresponding frame annotations.

## 4.2 Input preprocessing and prompt preparation

Evaluation scripts paired each frame with its annotation by image identifier and decoded the frame as RGB. They converted the grayscale annotation into a binary reference, treating values above 127 as foreground. The scripts constructed prompts on that original image grid and then mapped the prompt coordinates together with the image into the model's input space.

SAM, MedSAM, both LoRA configurations and PP-SAM used the SAM predictor. It resized the longest side to 1,024 pixels while preserving aspect ratio, then normalised and padded the image for the encoder. It scaled the prompt coordinates using the resized dimensions. MSA-B and MSA-L used bilinear interpolation to resize images to 512 × 512 before model preprocessing. Horizontal and vertical prompt coordinates were scaled by 512 divided by the original width and height, respectively.

## 4.3 Model adaptation and training

Table 4.1 records the full-data training runs used in the main comparison. The selected epoch corresponds to the best recorded Kvasir-SEG validation Dice for each configuration.

Table 4.1 Training settings and selected epochs for the 880-image runs

| Configuration | Initial weights and updated modules | Optimisation and checkpoint selection |
|---|---|---|
| SAMed | SAM ViT-B. Rank-four LoRA on query and value projections, plus the prompt encoder and mask decoder. | AdamW, peak learning rate 0.0005, batch size 1. Ran 160 epochs of a 200-epoch schedule. Selected epoch 35. |
| MedSAM + LoRA | MedSAM ViT-B. Same trainable modules as SAMed. | AdamW, peak learning rate 0.0005, batch size 1. Ran 160 epochs of a 200-epoch schedule. Selected epoch 50. |
| MSA-B | SAM ViT-B. Image encoder adapters and mask decoder. | Adam, LR 0.0001, batch size 4. 40 epochs, selected epoch 26. |
| MSA-L | SAM ViT-L. Image encoder adapters and mask decoder. | Adam, LR 0.0001, batch size 2. 40 epochs, selected epoch 21. |
| PP-SAM | SAM ViT-B. Full image encoder and prompt encoder. | AdamW, LR 0.0001, batch size 1. 100 epochs, selected epoch 61. |

The SAMed and MedSAM + LoRA scripts attached rank-four low-rank updates to the query and value projections in the image encoder. The pretrained image encoder weights remained frozen, and the prompt encoder and mask decoder also received gradient updates. Each run updated approximately 4.21 million parameters, or 4.49% of the instantiated model. Both scripts used tight annotation boxes during training and combined binary cross-entropy and Dice losses with weights of 0.2 and 0.8. The full-data runs used a 250-iteration learning-rate warm-up, polynomial decay, and AdamW weight decay of 0.1.

For MSA-B and MSA-L, the training loop encoded a sampled positive point without updating the prompt encoder. It used foreground-weighted binary cross-entropy at a mask output size of 256 × 256. The default evaluation used the expanded box defined in Section 3.4.

PP-SAM updated the full image encoder and prompt encoder with a frozen mask decoder. For training, it expanded each side of the tight box by an independently sampled integer between 0 and 50 pixels and clipped the coordinates to the image bounds. The training objective combined spatially weighted binary cross-entropy and IoU losses. This randomised training prompt preceded the deterministic box construction used at evaluation.

The annotation-budget experiments used separate SAMed and PP-SAM runs at 1, 5 and 10 labelled images. The archived SAMed few-shot runs used 100 epochs and a peak learning rate of 0.005. PP-SAM used its 100-epoch schedule at a learning rate of 0.0001 across budgets. The SAMed budget comparison therefore reflects the recorded recipes at each budget, including their different learning rates and schedules.

## 4.4 Inference and evaluation outputs

The SAM predictor configurations requested one candidate mask per prompt. The predictor restored the selected logits to native resolution, removed the input padding during that transformation and thresholded the restored logits at zero. The MSA evaluators requested multiple candidates and selected the mask with the highest model-predicted IoU score. They bilinearly resized that mask's logits to 512 × 512, applied a sigmoid and binarised at a probability threshold of 0.5. They then used nearest-neighbour interpolation to restore the binary mask to the original frame dimensions. Metric computation in both paths used the restored binary mask. During the dedicated point and box-translation experiments, each model reused the image embedding for the alternative prompts on the same frame.

The evaluators wrote a predicted mask and per-image metric record keyed by image identifier, and a JSON summary for each run. The thesis results use those records for overlap metrics. For surface HD95, a common post-evaluation script paired the saved predictions with reference masks, verified the pairings by reconciling Dice and IoU with the recorded scores, and recomputed the contour statistic defined in Section 3.5.1. The aggregation retained the number of images with a defined surface distance.

## 4.5 Reproducibility records

Appendix A maps results to data splits, checkpoints, scripts and outputs. REPRODUCTION.md gives the commands for the prompt toolkit, evaluation and analysis. Training logs record the initial weights, optimisation settings and selected epoch. To rerun a few-shot experiment, use its recorded support set or sampling seed. Random evaluation points are reproduced from the base seed and image identifier.

The primary SAMed runs used a cloud NVIDIA RTX 4090D. The MedSAM + LoRA and adapter runs used a local NVIDIA GeForce RTX 4070 Ti SUPER. The formal SAMed evaluation, local box series and shortened repeatability pilot have separate run records.
