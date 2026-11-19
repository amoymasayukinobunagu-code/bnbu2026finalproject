"""Repeat the local SAMed training implementation under a short fixed budget."""
import argparse
import contextlib
import json
import os
from pathlib import Path
import random
import sys

import cv2
import numpy as np
import pandas as pd
import torch

from complete_supplement import DATA, ROOT, image, loadmask, dataset_paths, index
from evaluate_prompt_variants import file_hash
from prompt_toolkit import box_from_mask

EPOCHS = 12
SEEDS = (1234, 2345, 3456)
FIXED = (
    'cju893jmdompz0817xn3g1w4h', 'cju5b9oyda4yr0850g9viziyv',
    'cju2rmd2rsw9g09888hh1efu0', 'cju5i39mreass0817au8p22zy',
    'cju3xuj20ivgp0818mij8bjrd',
)


def indexed_split(split):
    names=(ROOT/'splits/Kvasir-SEG/Data-split'/f'{split}.txt').read_text().splitlines()
    a,b=map(index,dataset_paths('Kvasir-SEG-val'))
    return {stem:(a[stem],b[stem]) for stem in sorted(Path(s).stem for s in names if s.strip())}


@torch.inference_mode()
def evaluate(model, predictor_cls, datasets):
    model.eval()
    predictor=predictor_cls(model)
    rows=[]
    for ds,pairs in datasets.items():
        for stem,(ip,mp) in pairs.items():
            gt=loadmask(mp)
            predictor.set_image(image(ip))
            mask=predictor.predict(box=np.array(box_from_mask(gt)),multimask_output=False)[0][0]
            inter=(mask & gt).sum()
            rows.append(dict(dataset=ds,id=stem,dice=2*inter/(mask.sum()+gt.sum()),
                             iou=inter/(mask | gt).sum()))
    return pd.DataFrame(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--lr',type=float,default=.005)
    parser.add_argument('--cvc-sequence-sample',action='store_true')
    args=parser.parse_args()
    out=args.out.resolve()
    out.mkdir(parents=True,exist_ok=True)
    code=Path('E:/Project/FYP/code/samed_baseline')
    os.environ['FYP_DATA_ROOT']=str(DATA)
    os.environ['FYP_OUT_DIR']=str(out/'training_logs')
    sys.path.insert(0,str(code))
    import train_samed as trainer

    train=indexed_split('train')
    val=indexed_split('val')
    a,b=map(index,dataset_paths('CVC-ClinicDB'))
    assert set(a)==set(b) and len(val)==120 and len(a)==612 and not set(train)&set(val)
    cvc_ids=sorted(a)
    cvc_label='CVC-ClinicDB'
    if args.cvc_sequence_sample:
        groups=pd.read_csv(DATA/'CVC-ClinicDB/metadata.csv')
        assert groups.frame_id.astype(str).is_unique and set(groups.frame_id.astype(str))==set(a)
        cvc_ids=[str(ordered.iloc[len(ordered)//2].frame_id)
                 for _,group in groups.groupby('sequence_id')
                 for ordered in [group.sort_values('frame_id')]]
        cvc_label='CVC-ClinicDB-one-frame-per-sequence'
    datasets={'Kvasir-SEG-val':val,cvc_label:{s:(a[s],b[s]) for s in cvc_ids}}
    trainer.load_kvasir_train_paths=lambda:list(train.values())
    trainer.load_kvasir_val_pairs=lambda:list(val.values())
    trainer._imread=lambda p,mode=cv2.IMREAD_COLOR:cv2.imdecode(np.fromfile(str(p),np.uint8),mode)
    trainer.WARMUP_PERIOD=10
    trainer.LR=args.lr
    torch.set_num_threads(4)
    config=dict(epochs=EPOCHS,seeds=SEEDS,shots=5,warmup=10,lr=trainer.LR,
                trainer_sha256=file_hash(code/'train_samed.py'),base_sha256=file_hash(trainer.SAM_CKPT),
                selection='final epoch',torch=torch.__version__,gpu=torch.cuda.get_device_name(),
                cvc_ids=cvc_ids)
    (out/'config.json').write_text(json.dumps(config,indent=2))
    num_checks=sum(i==0 or (i+1)%5==0 or i>=EPOCHS-5 for i in range(EPOCHS))
    for case in ['fixed_support','resampled_support']:
        for seed in SEEDS:
            name=f'{case}_{seed}'
            path=out/name
            path.mkdir(exist_ok=True)
            if (path/'per_image.csv').exists():
                continue
            support=FIXED if case=='fixed_support' else random.Random(seed+10000).sample(sorted(train),5)
            trainer.SEED=seed
            trainer.resolve_train_sample=lambda *a,**k:[train[s] for s in support]
            for setter in [random.seed,np.random.seed,torch.manual_seed,torch.cuda.manual_seed_all]:
                setter(seed)
            (path/'support.json').write_text(json.dumps(dict(seed=seed,support=list(support)),indent=2))
            evaluated=[]
            validation_calls=0

            def final_validation(model,pairs,**kwargs):
                nonlocal validation_calls
                validation_calls+=1
                if validation_calls!=num_checks:
                    return 0.,0,{}
                table=evaluate(model,trainer.SamPredictor,datasets)
                evaluated.append(table)
                kv=table[table.dataset=='Kvasir-SEG-val']
                return float(kv.dice.mean()),len(kv),{'iou':float(kv.iou.mean())}

            trainer.validate=final_validation
            print('Starting',name,flush=True)
            with (path/'train.log').open('w',encoding='utf-8') as log, contextlib.redirect_stdout(log):
                trainer.train_samed(rank=4,epochs=EPOCHS,stop_epoch=EPOCHS,shots=5,save_dir=str(path))
            assert validation_calls==num_checks and len(evaluated)==1 and len(evaluated[0])==120+len(cvc_ids)
            evaluated[0].assign(condition=case,seed=seed).to_csv(path/'per_image.csv',index=False)
            print('Completed',name,flush=True)
            torch.cuda.empty_cache()
    rows=pd.concat([pd.read_csv(p) for p in sorted(out.glob('*/per_image.csv'))])
    assert len(rows)==6*(120+len(cvc_ids))
    result=rows.groupby(['condition','seed','dataset']).agg(
        n=('dice','size'),dice=('dice','mean'),iou=('iou','mean')).reset_index()
    result.to_csv(out/'summary.csv',index=False)
    print(result.to_string(index=False),flush=True)


if __name__=='__main__':
    main()
