"""Training-only pseudo-label pilot and optional evaluation of automatic SAM prompts."""
import argparse
from pathlib import Path
import json

import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from complete_supplement import ROOT, dataset_paths, image, index, loadmask

SIZE = 96
SEED = 20261118


def resize(rgb):
    return cv2.resize(rgb, (SIZE,SIZE), interpolation=cv2.INTER_AREA)


def features(rgb):
    lab = cv2.cvtColor(resize(rgb), cv2.COLOR_RGB2LAB).astype(np.float32)/255
    y,x=np.indices(lab.shape[:2],dtype=np.float32)/(SIZE-1)
    return np.dstack([lab,cv2.GaussianBlur(lab,(0,0),3),x,y]).reshape(-1,8)


def ids(split):
    path=ROOT/'splits/Kvasir-SEG/Data-split'/f'{split}.txt'
    return sorted(Path(x).stem for x in path.read_text().splitlines() if x.strip())


def fit(x,y,weights=None):
    return HistGradientBoostingClassifier(max_iter=55,max_leaf_nodes=15,
        min_samples_leaf=30,random_state=SEED).fit(x,y,sample_weight=weights)


def prompt_from_prediction(pred, shape):
    # Drop isolated background speckles and keep the largest candidate.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(pred.astype(np.uint8))
    if count<=1:
        return None
    component=1+np.argmax(stats[1:,cv2.CC_STAT_AREA])
    if stats[component,cv2.CC_STAT_AREA]<12:
        return None
    x,y,w,h,_=map(int,stats[component])
    H,W=shape
    return [max(0,int(x*W/SIZE)-10),max(0,int(y*H/SIZE)-10),
            min(W,int(np.ceil((x+w)*W/SIZE))+10),min(H,int(np.ceil((y+h)*H/SIZE))+10)]


def scores(pred,gt):
    intersection=(pred & gt).sum()
    return dict(dice=float(2*intersection/(pred.sum()+gt.sum())),iou=float(intersection/(pred | gt).sum()))


def run_cpu(out):
    rng=np.random.default_rng(SEED)
    training=rng.permutation(ids('train'))
    labeled,unlabeled=training[:24],training[24:144]
    assert not set(training)&set(ids('val'))
    image_paths,mask_paths=map(index,dataset_paths('Kvasir-SEG-val'))
    xs,ys=[],[]
    for stem in labeled:
        feat=features(image(image_paths[stem]))
        mask=cv2.resize(loadmask(mask_paths[stem]).astype(np.uint8),(SIZE,SIZE),
                        interpolation=cv2.INTER_NEAREST).ravel()
        selected=rng.choice(len(mask),1500,replace=False)
        xs.append(feat[selected]);ys.append(mask[selected])
    x=np.concatenate(xs);y=np.concatenate(ys)
    teacher=fit(x,y)
    pseudo_x,pseudo_y,pseudo_audit=[],[],[]
    for stem in unlabeled:
        feat=features(image(image_paths[stem]))
        prob=teacher.predict_proba(feat)[:,1]
        selected=[]
        for high_confidence in [np.flatnonzero(prob>=.9),np.flatnonzero(prob<=.1)]:
            selected.extend(rng.choice(high_confidence,min(200,len(high_confidence)),replace=False))
        pseudo_x.append(feat[selected]);pseudo_y.append(prob[selected]>=.5)
        pseudo_audit.append(dict(id=stem,positive=int((prob[selected]>=.5).sum()),
                                 negative=int((prob[selected]<.5).sum())))
    px=np.concatenate(pseudo_x);py=np.concatenate(pseudo_y)
    student=fit(np.concatenate([x,px]),np.concatenate([y,py]),
                np.r_[np.ones(len(y)),np.full(len(py),.5)])
    rows=[]
    for ds in ['Kvasir-SEG-val','CVC-ClinicDB']:
        images,masks=map(index,dataset_paths(ds))
        for stem in ids('val') if ds.startswith('Kvasir') else sorted(masks):
            rgb,gt=image(images[stem]),loadmask(masks[stem])
            feat=features(rgb)
            for name,model in [('supervised_teacher',teacher),('pseudo_label_student',student)]:
                pred=model.predict(feat).reshape(SIZE,SIZE).astype(np.uint8)
                native=cv2.resize(pred,(gt.shape[1],gt.shape[0]),interpolation=cv2.INTER_NEAREST)>0
                rows.append(dict(dataset=ds,id=stem,condition=name,
                    box_json=json.dumps(prompt_from_prediction(pred,rgb.shape[:2])),
                    **scores(native,gt)))
    (out/'training_split.json').write_text(json.dumps(dict(seed=SEED,labeled=labeled.tolist(),
        pseudo_unlabeled=unlabeled.tolist(),pixels_labeled=len(y),pixels_pseudo=len(py),
        pseudo_thresholds=[.1,.9],pseudo_weight=.5),indent=2))
    pd.DataFrame(pseudo_audit).to_csv(out/'pseudo_label_audit.csv',index=False)
    save(rows,out,'pixel_pilot')


def save(rows,out,stem):
    data=pd.DataFrame(rows)
    data.to_csv(out/f'{stem}_per_image.csv',index=False)
    summary=data.groupby(['dataset','condition']).agg(n=('dice','size'),dice=('dice','mean'),
                                                       iou=('iou','mean')).reset_index()
    summary.to_csv(out/f'{stem}_summary.csv',index=False)
    print(summary.to_string(index=False),flush=True)


def run_sam(out):
    import torch
    from run_external_and_speed import predictor, forward
    data=pd.read_csv(out/'pixel_pilot_per_image.csv',dtype={'id':str},keep_default_na=False)
    model=predictor('SAM-B')
    rows=[]
    with torch.inference_mode():
        for ds,g in data[data.condition=='supervised_teacher'].groupby('dataset'):
            images,masks=map(index,dataset_paths(ds))
            for row in g.itertuples():
                box=json.loads(row.box_json)
                gt=loadmask(masks[row.id])
                # A missing proposal is an empty prediction in the all-frame analysis.
                pred=forward(model,image(images[row.id]),box) if box is not None else np.zeros_like(gt)
                rows.append(dict(dataset=ds,id=row.id,condition='sam_b_automatic_box',
                                 box_json=row.box_json,proposal_found=box is not None,**scores(pred,gt)))
    save(rows,out,'automatic_sam')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--auto-sam',action='store_true')
    args=parser.parse_args()
    if args.auto_sam:
        run_sam(args.out)
    else:
        args.out.mkdir(parents=True,exist_ok=False)
        run_cpu(args.out)
