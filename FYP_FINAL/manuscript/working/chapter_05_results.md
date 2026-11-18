# 5 Results and Discussion

Tables 5.1 and 5.2 report the primary comparison on Kvasir-SEG validation and CVC-ClinicDB. All configurations received an annotation-derived box expanded by 30 pixels. The tables include the three zero-shot baselines and five configurations adapted using 880 labelled training images. Dice is reported as mean ± image-level standard deviation. The other columns report image-level means. BF1 denotes boundary F1. Surface HD95 is measured in native-image pixels and averaged over defined distances.

Table 5.1 Segmentation on Kvasir-SEG validation (120 images per configuration)

| Model | Dice (mean ± SD) | IoU | Precision | Recall | BF1 | HD95 (px) |
|---|---|---|---|---|---|---|
| SAM-B | 0.7497 ± 0.2951 | 0.6680 | 0.7290 | 0.8012 | 0.4378 | 79.57 |
| SAM-L | 0.8140 ± 0.2390 | 0.7367 | 0.7873 | 0.8862 | 0.5222 | 76.12 |
| MedSAM | 0.7034 ± 0.2350 | 0.5846 | 0.6217 | 0.8394 | 0.2144 | 84.13 |
| SAMed | 0.8955 ± 0.1208 | 0.8280 | 0.8866 | 0.9334 | 0.6434 | 53.20 |
| MedSAM+LoRA | 0.9049 ± 0.0978 | 0.8375 | 0.8692 | 0.9624 | 0.5845 | 54.52 |
| MSA-B | 0.8524 ± 0.2017 | 0.7826 | 0.8864 | 0.8670 | 0.5863 | 67.95 |
| MSA-L | 0.8555 ± 0.1823 | 0.7820 | 0.9118 | 0.8567 | 0.5288 | 62.29 |
| PP-SAM | 0.9226 ± 0.0705 | 0.8630 | 0.8962 | 0.9594 | 0.6133 | 41.58 |

Table 5.2 Segmentation on CVC-ClinicDB (612 images per configuration)

| Model | Dice (mean ± SD) | IoU | Precision | Recall | BF1 | HD95 (px) |
|---|---|---|---|---|---|---|
| SAM-B | 0.5182 ± 0.3625 | 0.4300 | 0.4767 | 0.6377 | 0.2950 | 54.02 |
| SAM-L | 0.6588 ± 0.3225 | 0.5676 | 0.5994 | 0.8181 | 0.3948 | 42.56 |
| MedSAM | 0.5172 ± 0.2105 | 0.3755 | 0.4096 | 0.7672 | 0.0863 | 46.07 |
| SAMed | 0.8608 ± 0.1587 | 0.7814 | 0.8513 | 0.9125 | 0.5181 | 28.06 |
| MedSAM+LoRA | 0.8476 ± 0.1531 | 0.7596 | 0.8246 | 0.9098 | 0.4458 | 29.41 |
| MSA-B | 0.7685 ± 0.2808 | 0.6896 | 0.7998 | 0.8342 | 0.4821 | 52.27 |
| MSA-L | 0.8234 ± 0.2165 | 0.7432 | 0.8904 | 0.8170 | 0.4652 | 36.78 |
| PP-SAM | 0.8576 ± 0.1299 | 0.7699 | 0.8067 | 0.9511 | 0.4340 | 22.78 |

Surface HD95 was defined for all 120 Kvasir frames per configuration. On CVC-ClinicDB, it was defined for 611 frames each for SAM-B, SAM-L and MSA-L, 610 for SAMed, and all 612 for the remaining configurations.

## 5.1 Zero-shot performance (RQ1)

SAM-L had the highest mean Dice and IoU among the zero-shot configurations on both datasets. Its Dice was 0.8140 on Kvasir validation and 0.6588 on CVC-ClinicDB, compared with 0.7497 and 0.5182 for SAM-B. All three zero-shot configurations had lower mean Dice on CVC-ClinicDB than on Kvasir validation. The size of the difference varied by checkpoint.

MedSAM reached a mean Dice of 0.7034 on Kvasir and 0.5172 on CVC. On CVC its Dice was close to SAM-B's, but its higher recall (0.7672 versus 0.6377) came with lower precision (0.4096 versus 0.4767) and substantially lower boundary F1 (0.0863 versus 0.2950). Under the shared box protocol, medical pretraining alone did not confer a zero-shot Dice advantage for this MedSAM checkpoint.

## 5.2 Polyp-specific adaptation (RQ2)

### 5.2.1 Full-data comparison

Each full-data LoRA or Adapter configuration improved mean Dice over its corresponding zero-shot baseline on both primary datasets. SAMed improved over SAM-B by 0.1457 on Kvasir and 0.3427 on CVC. MedSAM+LoRA improved over MedSAM by 0.2014 and 0.3304, respectively. For the adapters, MSA-B reached 0.8524 on Kvasir and 0.7685 on CVC, while MSA-L reached 0.8555 and 0.8234. The adapter comparisons reflect the full configurations, including the input resolutions and mask-selection policies described in Chapter 4.

PP-SAM had the highest mean Dice on Kvasir (0.9226) and the lowest mean surface HD95 on both datasets. On CVC, SAMed and PP-SAM were close in mean Dice (0.8608 and 0.8576). SAMed had higher boundary F1 on both sets, whereas PP-SAM had lower mean surface HD95. The measures capture different aspects of the contour: boundary F1 counts contour agreement within a tolerance, while surface HD95 summarises distances near the tail of the contour-distance distribution. The small CVC Dice difference is descriptive and does not establish a stable ranking across runs.

### 5.2.2 Additional cross-dataset evaluation

Table 5.3 extends the SAM-B and MSA-B comparison to the full-data SAMed, MedSAM+LoRA and PP-SAM checkpoints. All five configurations used the same screened frames and 30-pixel annotation-derived boxes within each external dataset.

Table 5.3 Mean Dice and IoU on two additional datasets

| Model | CVC-ColonDB (342 images), Dice / IoU | ETIS-LaribPolypDB (190 images), Dice / IoU |
|---|---|---|
| SAM-B | 0.6444 / 0.5534 | 0.8613 / 0.7768 |
| MSA-B | 0.6654 / 0.5819 | 0.5575 / 0.4926 |
| SAMed | 0.7754 / 0.6928 | 0.7008 / 0.6339 |
| MedSAM+LoRA | 0.7134 / 0.6270 | 0.6582 / 0.5816 |
| PP-SAM | 0.8493 / 0.7555 | 0.8436 / 0.7483 |

On CVC-ColonDB, all four adapted configurations scored higher than zero-shot SAM-B on both metrics. PP-SAM led with a mean Dice of 0.8493, an increase of 0.2049 over SAM-B. On ETIS-LaribPolypDB, SAM-B led with 0.8613. PP-SAM was the closest adapted configuration at 0.8436, followed by SAMed at 0.7008. The mean Dice and IoU rankings agreed within each dataset. The change in ranking shows that the advantage over SAM-B depended on the external dataset. These are descriptive comparisons of the five complete configurations.

### 5.2.3 Annotation budget

Table 5.4 reports the results for the archived SAMed and PP-SAM runs at four labelled training budgets. Both methods reached their highest mean Dice using all 880 images. Across the 1-, 5- and 10-image runs, PP-SAM scored higher than SAMed on both datasets. More labels within the small budgets did not always yield a higher score. For example, on CVC, PP-SAM scored 0.8040 with five images and 0.7709 with ten. These are individual training runs for each method and budget. SAMed's full-data run also used a different learning rate and schedule, so the budget results describe complete training recipes.

Table 5.4 Mean Dice by number of labelled training images

| Training images | SAMed Kvasir | SAMed CVC | PP-SAM Kvasir | PP-SAM CVC |
|---|---|---|---|---|
| 1 | 0.7456 | 0.5759 | 0.7992 | 0.7516 |
| 5 | 0.7571 | 0.5747 | 0.8512 | 0.8040 |
| 10 | 0.7230 | 0.5435 | 0.8550 | 0.7709 |
| 880 | 0.8955 | 0.8608 | 0.9226 | 0.8576 |

The six-run SAMed pilot used a shorter schedule and the final checkpoint. Across three seeds with the same five support images, mean Dice ranged from 0.5437 to 0.6313 on Kvasir and from 0.4643 to 0.5008 on the 29-frame CVC subset. With redrawn support, the ranges were 0.6046–0.6937 and 0.4840–0.5822. Variation even with fixed support indicates sensitivity to the optimisation seed under this pilot schedule. The pilot ranges apply to its own schedule and evaluation subset.

## 5.3 Prompt sensitivity (RQ3)

### 5.3.1 Box expansion

Figure 5.1 shows mean Dice at fixed outward expansions of 0, 10, 30 and 50 pixels. In every plotted series, mean Dice at 50 pixels was below the values at 0 and 10 pixels. PP-SAM peaked at 10 pixels on both datasets. On CVC, its mean Dice fell from 0.8859 with the tight box to 0.8025 at 50 pixels, a decrease of 8.34 percentage points. On Kvasir, MSA-B declined from 0.8573 to 0.8462 over the same interval, a decrease of 1.11 points. The two values describe both the sensitivity to expansion and the accuracy retained with the wider prompt.

![Mean Dice at four box expansions. Four series are plotted on Kvasir validation and three on CVC-ClinicDB. PP-SAM has the steepest plotted drop on CVC.](figures/supplementary/prompt_expansion.png)
Figure 5.1 Mean Dice across fixed box expansions. MedSAM+LoRA is plotted for Kvasir validation, where a complete series of archived run outputs was available. The SAMed series uses its own local 30-pixel evaluation.

Figure 5.2 illustrates one failure under box expansion. With the tight box, PP-SAM follows the annotated contour closely. With 50-pixel expansion, its prediction includes a much larger region of surrounding tissue. This frame had the largest paired Dice drop among the 612 CVC frames for PP-SAM.

![Three panels show the reference contour and the PP-SAM predictions on CVC frame 559 with 0 and 50 pixels of box expansion. Dice falls from 0.906 to 0.103.](figures/results/box_expansion_example.png)
Figure 5.2 PP-SAM on CVC-ClinicDB frame 559. Cyan marks the reference contour, amber marks the prediction, and white dashed lines mark the input boxes. The frame was selected for the largest Dice decrease from the tight box to the 50-pixel box.

### 5.3.2 Point prompts and translated boxes

Table 5.5 compares the box with the three point-prompt settings. All three point settings reduced SAM-B's mean Dice on Kvasir and increased it on CVC. On CVC, the interior point with a background point raised SAM-B's Dice from 0.5182 to 0.6608. MSA-B gained Dice with point prompts on both datasets, with a smaller change on Kvasir. For example, the interior point on Kvasir increased MSA-B Dice by 0.0077, while boundary F1 fell from 0.5863 to 0.5705. In that setting, increased overlap did not imply better contour agreement.

Table 5.5 Mean Dice for the reference box and point conditions

| Dataset | Model | Box | Interior point | With background point | Random positive |
|---|---|---|---|---|---|
| Kvasir val. | SAM-B | 0.7497 | 0.6840 | 0.6935 | 0.6813 |
| Kvasir val. | MSA-B | 0.8524 | 0.8601 | 0.8592 | 0.8604 |
| CVC-ClinicDB | SAM-B | 0.5182 | 0.6305 | 0.6608 | 0.6159 |
| CVC-ClinicDB | MSA-B | 0.7685 | 0.8068 | 0.8027 | 0.8015 |

Box translation gave direction-dependent results. Relative to the reference box, SAM-B's mean Dice change on CVC ranged from −0.0683 for a left shift to +0.0562 for a right shift. For MSA-B, the four CVC changes ranged from +0.0011 to +0.0040. The corresponding Kvasir ranges were −0.0436 to +0.0031 for SAM-B and −0.0108 to +0.0057 for MSA-B. Image bounds sometimes limited the displacement. The CVC subset that received the full requested shift retained the opposite signs for SAM-B's left and right translations (−0.0677 and +0.0669). The tested shifts describe sensitivity to this specific perturbation of an annotation-derived box.

### 5.3.3 Sequence-level comparison on CVC-ClinicDB

Across the 29 equally weighted CVC sequences, the MSA-B interior point increased Dice by 0.0424 relative to the default box. The marginal 95% bootstrap interval ran from 0.0250 to 0.0622, with Holm-adjusted p = 0.00021. All three MSA-B point-versus-box comparisons had adjusted p below 0.05. The between-model difference for left-shift effects also met that threshold (MSA-B minus SAM-B = 0.0737, adjusted p = 0.0365). No other contrast among the 21 met the threshold. The sequence weighting, multiplicity correction and sampling assumptions in Section 3.6.1 apply to these exploratory results.

## 5.4 Relative polyp area and boundary contrast (RQ3)

### 5.4.1 Relative polyp area

Area-group results depended on the configuration (Figure 5.3). On CVC, SAM-B's mean Dice decreased from 0.5573 for small-area polyps to 0.2295 for large-area polyps. PP-SAM increased from 0.7970 to 0.8986 over the same groups. On Kvasir, SAM-L scored 0.8892 in the small-area group and 0.7187 in the large-area group. Using the area tertiles defined on the training data retained the low upper-group result for SAM-B on both evaluation sets.

![Two panels show mean Dice for eight configurations across small, medium and large foreground-area groups. Group counts are marked on the horizontal axes.](figures/supplementary/scale_dice.png)
Figure 5.3 Mean Dice by annotated foreground area. Small covers up to 5% of the frame, medium covers over 5% through 20%, and large covers over 20%.

To quantify relative prompt enlargement, the area of each clipped 50-pixel box was divided by the area of its tight box. Across CVC-ClinicDB, the mean ratios were 7.09 for small polyps (229 frames), 3.06 for medium polyps (321 frames) and 1.92 for large polyps (62 frames). The same pixel margin therefore produced substantially different proportional changes across groups. Comparisons of expansion sensitivity by size reflect both target area and the relative amount of extra image context supplied by the prompt.

### 5.4.2 Boundary colour contrast

On Kvasir, MSA-B's low-contrast group had a mean boundary F1 of 0.4945, compared with 0.6332 in the intermediate group and 0.5842 in the high group. On CVC, SAMed's boundary F1 rose from 0.4787 in the low group to 0.5617 in the high group. Group ordering varied by configuration and metric. For instance, CVC SAM-B had higher boundary F1 in the high-contrast group than in the low group (0.3374 versus 0.2526), while its Dice was lower (0.4135 versus 0.5633).

![Two panels show mean boundary F1 for eight configurations across low, intermediate and high boundary-colour-contrast groups.](figures/supplementary/contrast_boundary_f1.png)
Figure 5.4 Mean boundary F1 by within-dataset quartiles of boundary colour contrast. Group counts are shown in each panel.

Area and boundary contrast were associated, with Spearman correlations of 0.257 on Kvasir and 0.371 on CVC. The area-by-contrast analysis had uneven group sizes. For example, the CVC large-area group included 35 high-contrast frames and four low-contrast frames. The estimated differences describe the observed groups, including their different area distributions. The contrast proxy measures local colour separation at an annotated boundary, and expert ratings would be required to assess its agreement with perceived boundary visibility.

Figure 5.5 illustrates the default-box outputs on two medium-area, low-contrast frames. Each frame was selected as the closest to the median paired Dice change from SAM-B to MSA-B within its dataset's eligible subgroup. On the illustrated CVC frame, the SAM-B mask extends beyond the annotated contour into the surrounding tissue. The MSA-B mask has less extension into that area.

![Two rows show a Kvasir-SEG frame and a CVC-ClinicDB frame, followed by the annotated contour and the contours predicted by SAM-B and MSA-B. The panels give each predicted mask's Dice.](figures/results/default_box_examples.png)
Figure 5.5 Selected low-contrast examples under the default box. Cyan shows the annotated contour and amber shows a prediction. Frame identifiers are cju32qr9tbvsj08013pkpjenq (Kvasir, top) and 307 (CVC, bottom).

## 5.5 Inference speed

Table 5.6 reports the three configurations timed using the same batch-one procedure. MSA-B had the lowest mean latency on both sets. Its configured input was 512 × 512, while the SAM baselines used a 1,024-pixel longest side. The timings reflect the full inference paths at their configured resolutions, including preprocessing, encoding, decoding and returning the mask to the host.

Table 5.6 Mean inference latency and throughput on one RTX 4070 Ti SUPER

| Model | Kvasir validation (ms / FPS) | CVC-ClinicDB (ms / FPS) |
|---|---|---|
| SAM-B | 124.75 / 8.02 | 122.62 / 8.16 |
| SAM-L | 272.04 / 3.68 | 270.18 / 3.70 |
| MSA-B | 38.27 / 26.13 | 37.42 / 26.72 |

## 5.6 Interpretation and limitations

The full-data adapted configurations improved on their corresponding baselines in the primary comparison. The follow-up extended this comparison to five configurations and found that their ordering depended on the dataset. All four adapted configurations exceeded SAM-B on CVC-ColonDB, and SAM-B led on ETIS-LaribPolypDB. The few-shot comparisons and short repeatability pilot indicate sensitivity to the training recipe and support images. Kvasir validation also informed checkpoint selection, so CVC-ClinicDB provides the main comparison on data excluded from adaptation and selection. The CVC sequence-level tests concern prompt changes on fixed checkpoints.

The reference prompts supplied lesion location from the annotations. The resulting scores characterise segmentation given that information. In the exploratory automatic-prompt pilot, the teacher-generated box followed by SAM-B yielded Dice of 0.2644 on Kvasir and 0.1298 on CVC, with missing boxes counted as empty outputs. Appendix B reports the related pseudo-label experiment and its supervised comparator. Claims about boundary visibility are limited to the image-derived proxy, and the speed estimates apply to the tested hardware, resolutions and inference paths.
