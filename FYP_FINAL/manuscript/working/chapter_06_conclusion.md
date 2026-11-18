# 6 Conclusion and Future Work

## 6.1 Answers to the research questions

For RQ1, SAM-L led the zero-shot configurations in mean Dice on Kvasir-SEG validation (0.8140) and CVC-ClinicDB (0.6588). On CVC-ClinicDB, SAM-B and MedSAM had similar Dice. MedSAM recorded higher recall, while SAM-B achieved higher precision and boundary F1.

For RQ2, the four full-data LoRA and Adapter configurations improved mean Dice over their paired zero-shot baselines on both primary datasets. PP-SAM led on Kvasir-SEG validation (0.9226) and SAMed on CVC-ClinicDB (0.8608). In the external follow-up, all four adapted configurations exceeded SAM-B on CVC-ColonDB, where PP-SAM led at 0.8493. On ETIS-LaribPolypDB, SAM-B led at 0.8613, followed by PP-SAM at 0.8436.

For RQ3, 50-pixel expansion reduced mean Dice relative to 0 and 10 pixels in every verified series. Point prompts improved Dice for SAM-B and MSA-B on CVC-ClinicDB. On Kvasir, they reduced Dice for SAM-B and slightly improved it for MSA-B. Translation effects varied by direction and model. Area and contrast group rankings varied by model, dataset and metric.

## 6.2 Contributions and scope

The study compared released and polyp-adapted checkpoints under a common 30-pixel box protocol and recomputed surface HD95 from saved masks using one definition. The main results describe single-frame segmentation given annotation-derived location cues. The formal few-shot comparisons use one training run per method and budget. The boundary analysis uses an image-derived contrast proxy.

## 6.3 Future work

Independent endoscopy expert ratings can test whether the contrast proxy reflects perceived boundary visibility. Report agreement among raters and with the proxy.

Repeat the budget study with multiple support draws and optimisation seeds under matched training schedules to estimate the effect of label count more reliably.

Compare training with tight boxes, fixed-pixel expansions and margins proportional to box width and height. Hold the initial checkpoint, training split and optimisation schedule fixed, then evaluate all three strategies on the same prompt conditions.

Evaluate an image-based localiser with fixed segmentation checkpoints on held-out data. Report missed proposals and mask errors on prompted frames to identify where accuracy is lost.
