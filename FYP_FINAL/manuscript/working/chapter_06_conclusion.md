# 6 Conclusion and Future Work

## 6.1 Answers to the research questions

For RQ1, SAM-L led the zero-shot configurations in mean Dice on Kvasir-SEG validation (0.8140) and CVC-ClinicDB (0.6588). On CVC-ClinicDB, SAM-B and MedSAM had similar Dice. MedSAM recorded higher recall, while SAM-B achieved higher precision and boundary F1.

For RQ2, the four full-data LoRA and Adapter configurations improved mean Dice over their paired zero-shot baselines on both primary datasets. PP-SAM led on Kvasir-SEG validation (0.9226) and SAMed on CVC-ClinicDB (0.8608). In the external follow-up, all four adapted configurations exceeded SAM-B on CVC-ColonDB, where PP-SAM led at 0.8493. On ETIS-LaribPolypDB, SAM-B led at 0.8613, followed by PP-SAM at 0.8436.

For RQ3, 50-pixel expansion yielded lower mean Dice than 0 or 10 pixels in all plotted series. Across 29 CVC sequences, all three point conditions improved MSA-B's mean Dice relative to the default box (Holm-adjusted p < 0.05 for each). Frame-level point and translation effects varied by model and dataset. Area and boundary colour-contrast results also depended on the configuration and metric. Colour contrast served as an image-derived proxy for boundary visibility.

## 6.2 Contributions and scope

The study compared released and polyp-adapted checkpoints under a common 30-pixel box protocol and recomputed surface HD95 from saved masks using one definition. The main results describe single-frame segmentation given annotation-derived location cues. The formal few-shot comparisons use one training run per method and budget. The boundary analysis uses an image-derived contrast proxy.

## 6.3 Future work

Independent endoscopy expert ratings could assess the contrast proxy, with agreement measured among raters and against the proxy. Repeating the budget experiment with multiple support draws and optimisation seeds under matched schedules would help estimate the effect of label count.

A controlled training comparison could test tight boxes, fixed-pixel expansions and proportional margins using the same checkpoints, training split and optimisation schedule. A separate evaluation could pair an image-based localiser with fixed segmentation checkpoints on held-out data and report missed proposals and mask errors on prompted frames.
