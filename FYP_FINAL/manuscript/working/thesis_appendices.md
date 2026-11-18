# Appendix A Reproduction Index

The source package includes REPRODUCTION.md, which gives commands and data assumptions for the archived experiments. The primary evaluation uses the fixed Kvasir-SEG and CVC-ClinicDB partitions and the annotation-derived box defined in Section 3.4. Checkpoint selection and training settings are described in Section 4.3. The original training and inference scripts and pretrained weights are held in the accompanying implementation workspace.

Table A.1 Inputs and outputs for the reported comparisons

| Results | Inputs and run records | Scripts or outputs |
|---|---|---|
| Primary comparison and annotation budgets, Tables 5.1, 5.2 and 5.4 | Kvasir train and validation lists, CVC-ClinicDB images, the checkpoint and run locations in NUMBERS_INDEX.md | Model evaluators in the implementation workspace. Surface metric export from build_full_hd95_table.py. |
| External evaluation, Table 5.3 | Screened identifiers and image and mask hashes in duplicate_label_audit.csv. Checkpoint hashes in the extension's run.json. | run_external_and_speed.py and extend_external_evaluation.py. Reviewed summary and per-image CSVs. |
| Prompt comparisons, Section 5.3 | Archived expansion runs and paired prompt records for SAM-B and MSA-B | evaluate_prompt_variants.py and sequence_prompt_statistics.py. Per-image and sequence-level paired effects. |
| Image subgroups and timing, Sections 5.4 and 5.5 | Archived masks and data paths. Fixed timing sample and recorded GPU environment. | complete_supplement.py and run_external_and_speed.py. Subgroup tables and timing summaries. |

The standalone prompt_toolkit.py utility generates boxes and points from binary annotations. The dedicated prompt-variant evaluator caps box translation while preserving box dimensions, as described in Section 3.4.

<!-- appendix break -->

# Appendix B Pseudo-label and Automatic-prompt Pilot

The pilot trained a supervised pixel classifier on 24 labelled Kvasir-SEG training frames. A second classifier used the same labelled pixels plus pseudo-labelled pixels from another 120 training frames. Pixels with teacher probabilities of at most 0.10 or at least 0.90 were eligible for pseudo-labelling. The pseudo-labelled pixels received half the weight of the labelled pixels during training. A separate experiment supplied the largest predicted foreground component's box to frozen SAM-B. Missing proposals yielded empty predictions.

Table B.1 Mean Dice and IoU for the exploratory pilot

| Configuration | Kvasir-SEG validation, Dice / IoU | CVC-ClinicDB, Dice / IoU |
|---|---|---|
| Supervised pixel classifier | 0.2139 / 0.1342 | 0.0785 / 0.0479 |
| Pixel classifier with pseudo-labels | 0.2154 / 0.1361 | 0.0793 / 0.0490 |
| Teacher-generated box with SAM-B | 0.2644 / 0.2028 | 0.1298 / 0.1004 |
| Annotation-derived reference box with SAM-B | 0.7497 / 0.6680 | 0.5182 / 0.4300 |

The teacher yielded a box on 109 of 120 Kvasir validation frames and 241 of 612 CVC frames. Adding pseudo-labels changed mean Dice by approximately 0.0015 and 0.0008, respectively. These results describe one training split and a lightweight pixel classifier. A separate study would be needed to estimate the effects of pseudo-labels on SAM fine-tuning.
