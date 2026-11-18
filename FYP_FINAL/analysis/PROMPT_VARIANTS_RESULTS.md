# SAM ViT-B prompt evaluation results

Completed on 2026-11-18. The run used the existing zero-shot SAM ViT-B checkpoint
on 120 Kvasir-SEG validation images and 612 CVC-ClinicDB images. It generated
7,320 prompt evaluations. All 732 reference-box Dice and IoU scores exactly
matched the formal per-image records. The full run took 146.69 seconds.

| Prompt | Kvasir mean Dice | CVC mean Dice |
|---|---:|---:|
| Reference box, 30 px outward expansion | 0.7497 | 0.5182 |
| One interior positive point | 0.6840 | 0.6305 |
| Same interior point with a background point | 0.6935 | 0.6608 |
| One random positive point, averaged over three seeds | 0.6813 | 0.6159 |
| Box translated left | 0.7181 | 0.4499 |
| Box translated right | 0.7061 | 0.5744 |
| Box translated up | 0.7266 | 0.5053 |
| Box translated down | 0.7528 | 0.5675 |

All-image means include translations limited by the image edge. Box width and
height were fixed. Point prompts scored lower than the reference box on Kvasir
and higher on CVC. Adding the sampled negative point increased mean Dice on
both sets, while boundary F1 fell from 0.3978 to 0.3801 on Kvasir and from 0.3637
to 0.3607 on CVC. The mean within-image Dice range across the three random
clicks was 0.3015 on Kvasir and 0.2116 on CVC. The range describes the three
evaluated clicks per image.

When restricted to images receiving the full target displacement, the paired
mean Dice changes relative to each image's own reference box were as follows.

| Direction | Kvasir n | Kvasir Δ Dice | CVC n | CVC Δ Dice |
|---|---:|---:|---:|---:|
| Left | 92 | −0.0282 | 540 | −0.0677 |
| Right | 88 | −0.0334 | 552 | +0.0669 |
| Up | 84 | −0.0343 | 519 | −0.0094 |
| Down | 84 | +0.0109 | 368 | +0.0761 |

These direction-specific subsets contain different images. The experiment
evaluated annotation-derived prompts and one checkpoint. The observed gains
are specific to the tested dataset and prompt conditions.

- [Protocol and reproduction](PROMPT_VARIANTS_PROTOCOL.md)
- [Run record](outputs/prompt_variants_20261118/run.json)
- [Dice, IoU, boundary F1 and paired summary](outputs/prompt_variants_20261118/summary.csv)
- [Per-image scores, coordinates, seeds and deltas](outputs/prompt_variants_20261118/per_image.csv)
- [Random-click results by seed](outputs/prompt_variants_20261118/random_point_seed_summary.csv)
- [Within-image random-click spread](outputs/prompt_variants_20261118/random_point_image_spread.csv)
