"""Exploratory CVC cluster analysis of matched prompt changes."""
import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
DATA=Path('E:/Project/FYP/datasets_link')
OUT=ROOT/'analysis/outputs'
DS='CVC-ClinicDB'
DRAWS=100_000
SEED=20261118


def sequence_changes(path, metadata):
    d=pd.read_csv(path,dtype={'id':str})
    d=d[d.dataset==DS]
    assert len(d)==6120
    frame=d.groupby(['id','condition']).delta_dice.mean().reset_index()
    matched=frame.merge(metadata,left_on='id',right_on='frame_id',validate='many_to_one')
    assert len(matched)==len(frame) and matched.id.nunique()==612
    seq=matched.groupby(['sequence_id','condition']).delta_dice.mean().unstack()
    assert not seq.isna().any().any() and (seq.box_30==0).all()
    return seq.drop(columns='box_30')


def main():
    mapping=DATA/'CVC-ClinicDB/metadata.csv'
    meta=pd.read_csv(mapping,dtype={'frame_id':str})[['frame_id','sequence_id']]
    assert len(meta)==612 and meta.frame_id.is_unique
    sam=sequence_changes(OUT/'prompt_variants_20261118/per_image.csv',meta)
    msa=sequence_changes(OUT/'prompt_variants_msa_20261118/per_image.csv',meta)
    assert sam.index.equals(msa.index) and sam.columns.equals(msa.columns)
    rng=np.random.default_rng(SEED)
    signs=rng.choice(np.array([-1,1],dtype=np.int8),size=(DRAWS,len(sam)))
    draws=rng.integers(len(sam),size=(DRAWS,len(sam)))
    rows=[]
    for label,table in [('SAM-B',sam),('MSA-B',msa),('MSA-B minus SAM-B',msa-sam)]:
        for condition,series in table.items():
            v=series.to_numpy()
            observed=v.mean()
            p=(1+(np.abs(signs@v/len(v))>=abs(observed)-1e-12).sum())/(DRAWS+1)
            lo,hi=np.quantile(v[draws].mean(axis=1),[.025,.975])
            rows.append(dict(contrast=label,condition=condition,sequence_n=len(v),
                             mean_delta_dice=observed,ci95_lo=lo,ci95_hi=hi,p_raw=p))
    result=pd.DataFrame(rows).sort_values('p_raw')
    result['p_holm']=np.minimum(1,np.maximum.accumulate(
        result.p_raw.to_numpy()*np.arange(len(result),0,-1)))
    result.to_csv(OUT/'cvc_prompt_sequence_statistics.csv',index=False)
    meta.groupby('sequence_id').size().rename('frame_n').to_csv(OUT/'cvc_sequence_counts.csv')
    (OUT/'cvc_prompt_sequence_statistics_run.json').write_text(json.dumps(dict(
        seed=SEED,draws=DRAWS,metadata_sha256=hashlib.sha256(mapping.read_bytes()).hexdigest(),sequences=len(sam),
        frames=len(meta),contrasts=len(result)),indent=2))
    print(result.to_string(index=False))


if __name__=='__main__':
    main()
