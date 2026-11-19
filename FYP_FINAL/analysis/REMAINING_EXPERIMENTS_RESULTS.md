# Follow-up experiment results for the thesis

## Core accuracy and surface HD95

Use [the corrected main table](../manuscript/tables/table_main_boxprompt_surface_hd95.csv)
for all 14 primary model settings, including few-shot rows, and both datasets.
The 28 rows cover 10,248 predictions.
Surface HD95 is the maximum of the two directed 95th percentiles of native-pixel
boundary distance. Five empty predictions have undefined surface HD95, yielding
10,243 finite values. The new table records per-row denominators. Dice, IoU,
precision and recall retain their formal values. Keep the archive for provenance.
Additional appendix models have their original metric provenance.

## MSA-B point and box translation results

MSA-B used 120 Kvasir and 612 CVC frames with the same 7,320 prompt specifications
as SAM ViT-B. Its default-box per-image Dice and IoU matched the formal records
exactly. The MSA-B decoder selected the highest predicted quality mask, its
original decoding policy. The SAM-B experiment used single-mask decoding.

| Dataset | Model | Default box Dice | Interior positive | Positive and negative | Random positive |
|---|---|---:|---:|---:|---:|
| Kvasir validation | SAM-B | 0.7497 | 0.6840 | 0.6935 | 0.6813 |
| Kvasir validation | MSA-B | 0.8524 | 0.8601 | 0.8592 | 0.8604 |
| CVC-ClinicDB | SAM-B | 0.5182 | 0.6305 | 0.6608 | 0.6159 |
| CVC-ClinicDB | MSA-B | 0.7685 | 0.8068 | 0.8027 | 0.8015 |

On Kvasir, the interior point changed mean Dice by −0.0657 for SAM-B and
+0.0077 for MSA-B. On CVC both changes were positive. MSA-B's four all-frame
translation changes on CVC ranged from +0.0011 to +0.0040 Dice. These are
comparisons between complete configurations, including input resolution and
decoder policy. The table relies on annotation-derived prompts.

[MSA-B run record](outputs/prompt_variants_msa_20261118/run.json) ·
[All prompt conditions and paired effects](outputs/prompt_variants_msa_20261118/summary.csv)

## Sequence-level statistical comparison

The CVC dataset's local metadata maps all 612 frames to 29 sequences. The
exploratory analysis first averages random clicks per frame, then averages
paired Dice differences within each sequence. The test estimates the mean
change with equal sequence weights, which differs from the frame-weighted
numbers in the table above. It assumes symmetric, independent sequence-level
paired differences under the null. The unit is a video sequence.

Across 21 contrasts, Holm-adjusted p values were below 0.05 for MSA-B's
three point-versus-box conditions and the between-configuration difference
in left-translation changes. For MSA-B's interior point, the sequence-weighted
mean change was +0.0424 Dice, the unadjusted bootstrap 95% interval was
[0.0250, 0.0622], and adjusted p was 0.00021. Intervals are marginal.
Kvasir results remain descriptive because local sequence or patient groups were
unavailable. This analysis tests prompt changes on fixed checkpoints.

[All 21 contrasts and intervals](outputs/cvc_prompt_sequence_statistics.csv)

## Uniform inference speed

Batch one on RTX 4070 Ti SUPER, using 30 frames per dataset and three timed
passes. Timing includes preprocessing, encoding, decoding and returning the
mask to the host. Disk I/O and model loading are outside the timer. Each
configuration retained its normal inference resolution.

| Model | Kvasir mean ms / FPS | CVC mean ms / FPS |
|---|---:|---:|
| SAM-B | 124.75 / 8.02 | 122.62 / 8.16 |
| SAM-L | 272.04 / 3.68 | 270.18 / 3.70 |
| MSA-B | 38.27 / 26.13 | 37.42 / 26.72 |

FPS is 1,000 divided by mean latency in milliseconds. Report these three
settings as a separate speed comparison. Their speed reflects their full
inference configurations, including resolution.

[Speed details](outputs/core_speed_20261118/summary.csv)

## Additional external sets

The reviewed table uses unique encoded image content within each dataset and
excludes duplicate-image groups with conflicting binary masks. The audit
compared original image bytes against the local Kvasir and CVC-ClinicDB folders
and between the external datasets. Exact hashes do not detect re-encoded near
duplicates or establish image provenance in base-model pretraining.

The follow-up extended the original SAM-B and MSA-B comparison to the formal
full-data SAMed, MedSAM+LoRA and PP-SAM checkpoints. It reused the original
reviewed image identifiers and verified the image hashes, decoded binary masks
and box coordinates for all 532 frames. The adapted runs used the same
30-pixel annotation-derived boxes and the original single-mask decoder policy.
Dice and IoU matched the archived scores within 1e-6 on six primary-dataset
frames per added model.

| Model | CVC-ColonDB (342 images), Dice / IoU | ETIS-LaribPolypDB (190 images), Dice / IoU |
|---|---:|---:|
| SAM-B | 0.6444 / 0.5534 | 0.8613 / 0.7768 |
| MSA-B | 0.6654 / 0.5819 | 0.5575 / 0.4926 |
| SAMed | 0.7754 / 0.6928 | 0.7008 / 0.6339 |
| MedSAM+LoRA | 0.7134 / 0.6270 | 0.6582 / 0.5816 |
| PP-SAM | 0.8493 / 0.7555 | 0.8436 / 0.7483 |

PP-SAM led on CVC-ColonDB and SAM-B led on ETIS-LaribPolypDB. The original
SAM-B and MSA-B results were carried forward from their audited per-image
records. All models were evaluated with fixed weights.

[Five-model summary](outputs/external_box_20261118_trained/summary_reviewed.csv) ·
[Per-image scores](outputs/external_box_20261118_trained/per_image.csv) ·
[Run and checkpoint hashes](outputs/external_box_20261118_trained/run.json) ·
[Checkpoint verification](outputs/external_box_20261118_trained/checkpoint_verification.csv) ·
[Evaluation script](extend_external_evaluation.py) ·
[Duplicate and label audit](outputs/external_box_20261118/duplicate_label_audit.csv)

## SAMed five-shot training repeatability pilot

Six local rank-4 SAMed runs used 12 epochs and the final checkpoint. Each of
three optimization seeds was run on the same five-image support set and on
a separately redrawn support set. The short schedule used learning rate
0.0005 with 10 warm-up steps. CVC was evaluated on one frame per sequence.
The initial 0.005 trial returned zero Dice on all frames in its sole completed
run. This failed result motivated the lower learning rate and remains archived.

| Support | Kvasir mean Dice and run range | CVC mean Dice and run range |
|---|---:|---:|
| Fixed | 0.5940 [0.5437, 0.6313] | 0.4781 [0.4643, 0.5008] |
| Redrawn | 0.6555 [0.6046, 0.6937] | 0.5249 [0.4840, 0.5822] |

Run ranges describe three outcomes per condition. The pilot reflects
optimization and support sensitivity under its own training budget. The
100-epoch formal few-shot settings have single runs in the retained evidence.

[Run summary](outputs/samed_repeat_lr0005_20261118/summary.csv) ·
[Run settings](outputs/samed_repeat_lr0005_20261118/config.json) ·
[Initial failed trial](outputs/samed_repeat_pilot_20261118/fixed_support_1234/train.log)

## Automatic prompts and pseudo-label pilot

A lightweight pixel classifier learned from 24 Kvasir training images. Its
largest predicted component gave an image-derived box for the frozen SAM-B
checkpoint. The teacher produced a valid proposal for 109/120 Kvasir and
241/612 CVC frames. Missing proposals count as empty segmentations.

| Setting | Kvasir mean Dice | CVC mean Dice |
|---|---:|---:|
| Supervised pixel classifier | 0.2139 | 0.0785 |
| Same classifier with pseudo-labeled pixels from 120 training frames | 0.2154 | 0.0793 |
| Teacher-generated box followed by SAM-B | 0.2644 | 0.1298 |
| SAM-B with the annotation-derived reference box | 0.7497 | 0.5182 |

The automatic proposal was weak, especially on CVC. The small changes from
pseudo-labeling reflect one split and one lightweight baseline. The pilot
supports neither a performance gain from semi-supervised SAM training nor a
claim of reliable automatic localization.

[Pixel-classifier pilot](outputs/pseudo_label_pilot_20261118/pixel_pilot_summary.csv) ·
[Automatic SAM results](outputs/pseudo_label_pilot_20261118/automatic_sam_summary.csv) ·
[Training IDs and pseudo-label thresholds](outputs/pseudo_label_pilot_20261118/training_split.json)

## Expert weak-boundary review

Prepared 72 frames with up to four examples per dataset, size and contrast
combination. The local packet contains score-blinded images, a three-grade
rubric and two independent rating sheets. An expert must supply labels before
agreement and proxy validity can be measured. The current manuscript can use
the completed image-feature proxy analysis while the review is pending.

[Local review instructions](outputs/boundary_expert_review/reviewer_packet/README.md)
