"""Add verified few-shot masks to the corrected primary comparison table."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from complete_supplement import ROOT, ORIG, index, dataset_paths, loadmask, surface_scores

tables=ROOT/'manuscript/tables'
original=pd.read_csv(tables/'table_main_boxprompt.csv').dropna(subset=['dest_hint'])
core=pd.read_csv(tables/'table_core_boxprompt_surface_hd95.csv').set_index(['official_source','dataset'])
details=[]
rows=[]
gt_index={}
for ds in original.split_key.unique():
    gt_index[ds]=index(dataset_paths(ds)[1])
for row in original.itertuples():
    source,ds=row.dest_hint,row.split_key
    if (source,ds) in core.index:
        data=core.loc[(source,ds)].to_dict()
    else:
        suffix=source.split('/',1)[1]
        items=json.loads((ROOT/'numbers'/source/'per_image.json').read_text())[ds]
        official=items if isinstance(items,list) else items['per_image']
        pred=index(ORIG/suffix/ds/'preds')
        assert len(official)==int(row.n) and len(pred)==int(row.n)
        rec=[]
        for item in official:
            stem=Path(item['name']).stem
            gt=loadmask(gt_index[ds][stem])
            saved=loadmask(pred[stem])
            assert gt.shape==saved.shape and gt.any()
            inter=(saved & gt).sum()
            dice=float(2*inter/(saved.sum()+gt.sum()))
            iou=float(inter/(saved | gt).sum())
            assert abs(dice-item['dice'])<=1e-6 and abs(iou-item['iou'])<=1e-6,(source,ds,stem)
            rec.append(dict(source=source,dataset=ds,id=stem,dice=dice,iou=iou,
                            **surface_scores(saved,gt)))
        group=pd.DataFrame(rec)
        details.extend(rec)
        assert np.isclose(row.dice_m,group.dice.mean(),atol=1e-7,rtol=0)
        data=dict(model=row.method,original_label=row.method,n=len(group),dice=row.dice_m,
                 dice_sd=row.dice_s,iou=row.iou_m,precision=row.p_m,recall=row.r_m,
                 boundary_f1=group.boundary_f1.mean(),surface_hd95_px=group.surface_hd95_px.mean(),
                 surface_hd95_n=int(group.surface_hd95_px.notna().sum()),
                 undefined_surface_hd95_n=int(group.surface_hd95_px.isna().sum()))
    rows.append(dict(data,dataset=ds,official_source=source))
result=pd.DataFrame(rows)
assert len(result)==28 and len(details)==4392
result.to_csv(tables/'table_main_boxprompt_surface_hd95.csv',index=False)
pd.DataFrame(details).to_csv(ROOT/'analysis/outputs/few_shot_surface_per_image.csv',index=False)
print('All primary comparison rows:',len(result),'total',result.n.sum(),
      'finite surface HD95',result.surface_hd95_n.sum(),flush=True)
