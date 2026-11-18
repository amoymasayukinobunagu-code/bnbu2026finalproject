# Follow-up experiments for the thesis

- Recompute the core comparison table from the eight reconciled model runs.
  Report surface HD95 with the finite denominator and preserve archived exports.
- Evaluate MSA-B with the same frames and prompt coordinates as SAM ViT-B.
  Retain each model's original decoding policy and reconcile the default box.
- Benchmark batch-one box inference on the same fixed 30 images per dataset
  for SAM-B, SAM-L, and MSA-B. Include host preprocessing, GPU work, and return
  of the mask. Exclude disk I/O, use five warm-ups per dataset, synchronize CUDA,
  and repeat three passes per model. The bench is scoped to these three models.
- Evaluate SAM-B and MSA-B on local ETIS-LaribPolypDB and CVC-ColonDB.
  Check exact image duplicates against Kvasir and CVC-ClinicDB. Consider these
  additional external benchmarks with no selection on their labels. Where
  pretraining provenance is unknown, report that limitation.
- Use SAMed rank-4 with five Kvasir training images for a short repeatability
  pilot. Pair each of three optimization seeds with fixed support and with a
  separately seeded support draw. Use 12 epochs and the final epoch model,
  retain a common AdamW schedule with 10 warm-up iterations, and evaluate on
  Kvasir validation and CVC once per run. Report seed-level results. This new
  budget and checkpoint selection rule differ from the formal runs.
  The initial 0.005 learning-rate pilot yielded Dice 0 for every frame in
  its first completed run. Preserve that result. A follow-up fixes the learning
  rate at 0.0005 and evaluates all Kvasir validation frames plus one middle
  frame by ID from each of the 29 CVC sequences, chosen before the reruns.
  Report the follow-up as an exploratory change after observing pilot failure.
- Use a lightweight pixel classifier for a clearly labelled automatic prompt
  and pseudo-label pilot. The teacher uses 24 labeled Kvasir training frames.
  The student uses the same labels plus high-confidence pseudo-labeled pixels
  from 120 disjoint training frames. Compare both on Kvasir validation and
  CVC. Test the teacher's annotation-free predicted proposal as a SAM-B box.
- Prepare a score-blinded weak-boundary review packet with balanced sampling
  by dataset, relative area, and contrast where possible. Expert labels and
  agreement require independent reviewers.
- CVC metadata provides video sequence IDs. For all seven nonreference prompt
  conditions, test paired sequence-mean Dice contrasts against the reference
  within each model and the difference of the two models' paired changes.
  Average random clicks within each frame first, and give each sequence equal
  weight. Use a two-sided Monte Carlo sign-flip test under symmetry of independent
  sequence differences, with Holm correction across the 21 contrasts. Give
  unadjusted percentile intervals with sequence bootstrap. The SAM-B results
  were available when the analysis was planned, so treat inference as exploratory.
  The sequence-level estimand does not imply independent patients, and Kvasir
  remains descriptive without verified groups.
