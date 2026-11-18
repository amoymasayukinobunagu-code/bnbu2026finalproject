"""Build a new thesis table from reconciled official and recomputed metrics."""
from pathlib import Path
import pandas as pd
import numpy as np

root = Path(__file__).resolve().parents[1]
out = root/'analysis/outputs'
original = pd.read_csv(root/'manuscript/tables/table_main_boxprompt.csv')
joined = pd.read_csv(out/'per_image_joined.csv')
assert joined.prediction_match.all()
results = []
for (model, dataset), group in joined.groupby(['model','dataset'], sort=False):
    matches = original[(original.split_key==dataset) &
                       np.isclose(original.dice_m, group.dice.mean(), rtol=0, atol=1e-8) &
                       np.isclose(original.iou_m, group.iou.mean(), rtol=0, atol=1e-8)]
    assert len(matches)==1, (model,dataset)
    main = matches.iloc[0]
    finite = group.surface_hd95_px.notna()
    results.append(dict(model=model, original_label=main.method, dataset=dataset, n=len(group),
                        dice=main.dice_m, dice_sd=main.dice_s, iou=main.iou_m,
                        precision=main.p_m, recall=main.r_m, boundary_f1=group.boundary_f1.mean(),
                        surface_hd95_px=group.surface_hd95_px.mean(), surface_hd95_n=int(finite.sum()),
                        undefined_surface_hd95_n=int((~finite).sum()), official_source=main.dest_hint))
assert len(results)==16
pd.DataFrame(results).to_csv(root/'manuscript/tables/table_core_boxprompt_surface_hd95.csv', index=False)
