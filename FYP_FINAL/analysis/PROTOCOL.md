# Supplementary analysis protocol

Recorded on 2026-11-18 before calculating the new subgroup outcome tables. This is a
post hoc descriptive extension of completed experiments, not a preregistered study.
No model training, checkpoint selection, or formal source results will be changed.

## Object scale

The exposure is foreground area / image area, obtained from the original-resolution
ground-truth mask thresholded at grayscale >127. Primary image-relative bins are
small <=5%, medium >5% to20%, large >20%. These are operational image-area categories,
not lesion diameter in millimetres or clinical size categories. Sensitivity uses
tertiles estimated from the 880 Kvasir training masks alone and applied to both
evaluation datasets. Do not tune cutoffs to model performance.

## Boundary visibility proxy

For each image, convert RGB to CIELAB using OpenCV float conversion. Compute the
Euclidean distance between mean Lab colours in inner and outer bands adjacent to
the ground-truth boundary. Band width is max(2, round(0.005 * image diagonal)) pixels.
Lower colour separation is an operational low-contrast proxy, not a clinician-rated
weak-boundary label. It can be affected by illumination, texture, specular highlights,
foreground size, global colour cancellation and band geometry. If either band is
empty the score is missing and its denominator is reported.

Within each dataset, split proxy values at the 25th and 75th percentiles, using
image/GT features only: low <=Q25, intermediate >Q25 to<Q75, high >=Q75.
Sensitivity uses band factors 0.0025 and0.01. Also report a grayscale gradient proxy
on the one-pixel GT boundary, and crossed area/contrast groups. Summarise rank
correlation between size and contrast. These checks expose possible dependence;
they do not establish independent effects or clinical boundary validity.

## Outcomes and provenance

For eight core methods, join official per-image metrics to image features by exact
dataset and filename stem. Fail on duplicates, missing IDs or count disagreements.
Check group-weighted means recover the official Dice and IoU summary means.
Groups use the same images across models. Report sample count, Dice, IoU, existing
recorded distance, and supplementary surface metrics where matched predictions exist.
The existing field named hd95 requires a label audit: inspected SAM/PP-SAM helpers
actually calculate maximum Hausdorff distance on foreground pixels, not a95th percentile.
Keep that field as legacy_distance and do not relabel it as validated HD95.

Recompute surface HD95 as max(P95(pred-boundary to GT-boundary nearest-neighbour
distance), P95(reverse)) in native image pixels. Boundary uses a4-connected erosion
with outside-image background. Boundary F1 uses tolerance max(1, round(0.005*diagonal)).
Empty prediction: BF1=0 and surface HD95 undefined; report empty/missing counts and
finite-HD95 denominator. Missing predictions are never silently filled.

Saved masks are thresholded >127 and must match original GT geometry. Saved JPEG
masks may differ from in-memory evaluation masks; compute and report Dice discrepancy.
Include surface metrics in primary summaries only for masks whose reconstructed Dice
differs from the official per-image value by <=1e-6. Report all exclusions explicitly.
For discrepant predictions, source provenance is unresolved; never silently assign
their surface metrics to a formal run.

## Prompt study and toolkit

Reuse the existing fixed outward0/10/30/50 pixel evaluations. This is a bounded prompt
extent sensitivity study, not random jitter, box translation, point robustness,
automatic detection or clinical prompt evaluation. Keep user-provided MedSAM CVC
0/10/50 values separate; primary CSV curves contain only available raw summary JSON.
Generate and validate reusable GT-derived box/positive-point/negative-point manifests.
Translation and point prompts are toolkit capabilities only and have no new model
performance claims. Preserve legacy_v3 coordinate endpoints for reproducibility and
provide a documented half-open alternative, including the single-pixel edge case.

## Interpretation

No p-values, multi-seed claims, patient-level independence or causal interpretation.
Kvasir scores remain model-selection validation scores. CVC contains related video
frames, so image counts are not counts of independent patients. This extension
provides operational subgroup analyses; clinical weak-boundary assessment remains
unverified. Retain negative and inconsistent trends.
