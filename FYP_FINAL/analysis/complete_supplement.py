"""Reproducible post hoc feature extraction and subgroup analysis, read-only inputs."""
from pathlib import Path
from collections import defaultdict
import csv
import hashlib
import json
import math
import platform
import sys
import time
import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
DATA=Path('E:/Project/FYP/datasets_link')
ORIG=Path('E:/Project/FYP/results')
OUT=ROOT/'analysis/outputs'
OUT.mkdir(parents=True,exist_ok=True)
FIG=ROOT/'manuscript/figures/supplementary'
FIG.mkdir(parents=True,exist_ok=True)
MODELS={
 'SAM-B':('sam_v3','sam_v3'),
 'SAM-L':('sam_v3/vit_l','sam_v3/vit_l'),
 'MedSAM':('medsam_v3','medsam_v3'),
 'SAMed':('samed_baseline_cloud/samed_baseline/r4_full_best','samed_baseline_cloud/samed_baseline/r4_full_best'),
 'MedSAM+LoRA':('medsam_peft/r4_full_best','medsam_peft/r4_full_best'),
 'MSA-B':('msa_polyp_baseline_v3','msa_polyp_train/eval_v3'),
 'MSA-L':('msa_vitl_polyp_baseline_v3','msa_vitl_polyp_train/eval_v3'),
 'PP-SAM':('pp_sam_baseline_v3/eval_best_880shot','pp_sam_baseline_v3/eval_best_880shot')}
STRUCT=generate_binary_structure(2,1)
HASHES=[]

def tracked(p):
    p=Path(p)
    HASHES.append(dict(path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    return p

def loadmask(p):
    a=cv2.imdecode(np.fromfile(str(p),np.uint8),cv2.IMREAD_GRAYSCALE)
    if a is None: raise ValueError(f'Unreadable mask {p}')
    return a>127

def image(p):
    a=cv2.imdecode(np.fromfile(str(p),np.uint8),cv2.IMREAD_COLOR)
    if a is None:raise ValueError(f'Unreadable image {p}')
    return cv2.cvtColor(a,cv2.COLOR_BGR2RGB)

def index(folder):
    d={}
    for p in Path(folder).iterdir():
        if p.suffix.lower() not in ['.png','.jpg','.jpeg','.bmp','.tif','.tiff']:continue
        if p.stem in d:raise ValueError(f'Duplicate {p.stem}')
        d[p.stem]=p
    return d

def dataset_paths(ds):
    if ds=='Kvasir-SEG-val':return DATA/'Kvasir-SEG/images',DATA/'Kvasir-SEG/masks'
    return DATA/'CVC-ClinicDB/PNG/Original',DATA/'CVC-ClinicDB/PNG/Ground Truth'

def area_bin(a):return 'Small' if a<=.05 else ('Medium' if a<=.20 else 'Large')

def boundary(m):return m & ~binary_erosion(m,structure=STRUCT,border_value=0)

def surface_scores(pred,gt):
    if not pred.any():return dict(surface_hd95_px=np.nan,boundary_f1=0.,empty_prediction=True)
    a=np.argwhere(boundary(pred));b=np.argwhere(boundary(gt))
    d1=cKDTree(b).query(a)[0];d2=cKDTree(a).query(b)[0]
    tol=max(1,round(.005*math.hypot(*gt.shape)))
    pr=(d1<=tol).mean();rc=(d2<=tol).mean()
    return dict(surface_hd95_px=max(np.percentile(d1,95),np.percentile(d2,95)),
                boundary_f1=2*pr*rc/(pr+rc) if pr+rc else 0.,empty_prediction=False)

def features(ds,stems,ims,masks):
    result=[]
    for i,stem in enumerate(stems):
        mp=tracked(masks[stem]);ip=tracked(ims[stem]);m=loadmask(mp);rgb=image(ip)
        assert m.shape==rgb.shape[:2] and m.any(),stem
        h,w=m.shape;lab=cv2.cvtColor(rgb.astype(np.float32)/255.,cv2.COLOR_RGB2LAB)
        din=distance_transform_edt(np.pad(m,1))[1:-1,1:-1]
        dout=distance_transform_edt(~m)
        row=dict(dataset=ds,id=stem,image=str(ip),mask=str(mp),height=h,width=w,area_ratio=float(m.mean()))
        row['size_group']=area_bin(row['area_ratio'])
        for label,f in [('narrow',.0025),('main',.005),('wide',.01)]:
            r=max(2,round(f*math.hypot(h,w)))
            inner=m&(din<=r);outer=(~m)&(dout<=r)
            row[f'ring_{label}_px']=r
            row[f'contrast_{label}']=float(np.linalg.norm(lab[inner].mean(0)-lab[outer].mean(0))) if inner.any() and outer.any() else np.nan
        gray=cv2.cvtColor(rgb.astype(np.float32)/255,cv2.COLOR_RGB2GRAY)
        gx=cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3)/8
        gy=cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3)/8
        row['boundary_gradient']=float(np.median(np.hypot(gx,gy)[boundary(m)]))
        result.append(row)
        if (i+1)%150==0:print('features',ds,i+1,flush=True)
    return result

def save(df,name):df.to_csv(OUT/name,index=False,float_format='%.10g')

def grouped(df,cols):
    rows=[]
    for key,g in df.groupby(cols,observed=True,dropna=False):
        if not isinstance(key,tuple):key=(key,)
        row=dict(zip(cols,key));row['n']=len(g)
        for field in ['dice','iou','legacy_distance','surface_hd95_px','boundary_f1']:
            row[field+'_mean']=g[field].mean();row[field+'_n']=int(g[field].notna().sum())
        row['dice_std']=g.dice.std(ddof=0)
        row['prediction_match_n']=int(g.prediction_match.sum())
        row['area_ratio_mean']=g.area_ratio.mean();row['contrast_main_mean']=g.contrast_main.mean()
        rows.append(row)
    return pd.DataFrame(rows)

def plot_groups(table,col,levels,name,title,y='dice_mean'):
    fig,axes=plt.subplots(1,2,figsize=(11,4.6),sharey=True)
    colors=plt.get_cmap('tab10').colors
    markers=['o','s','^','D','v','P','X','*']
    for ax,ds in zip(axes,['Kvasir-SEG-val','CVC-ClinicDB']):
        for i,model in enumerate(MODELS):
            g=table[(table.dataset==ds)&(table.model==model)].set_index(col).reindex(levels)
            ax.plot(range(len(levels)),g[y],label=model,color=colors[i],marker=markers[i],linewidth=1.2)
        counts=table[(table.dataset==ds)&(table.model=='SAM-B')].set_index(col).reindex(levels)['n']
        ax.set_xticks(range(len(levels)),[f'{a}\n(n={int(n) if pd.notna(n) else 0})' for a,n in zip(levels,counts)])
        ax.set_title(ds);ax.set_ylim(0,1);ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('Mean Dice' if y=='dice_mean' else 'Mean boundary F1')
    fig.suptitle(title,fontsize=12)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=4,frameon=False)
    fig.tight_layout(rect=(0,.13,1,.93))
    for ext in ['png','svg']:fig.savefig(FIG/f'{name}.{ext}',dpi=220)
    plt.close(fig)

def main():
    start=time.time()
    train_path=tracked(ROOT/'splits/Kvasir-SEG/Data-split/train.txt')
    val_path=tracked(ROOT/'splits/Kvasir-SEG/Data-split/val.txt')
    train=[Path(s.strip()).stem for s in train_path.read_text().splitlines() if s.strip()]
    val=[Path(s.strip()).stem for s in val_path.read_text().splitlines() if s.strip()]
    assert len(train)==880 and len(val)==120 and not set(train)&set(val)
    ki,km=dataset_paths('Kvasir-SEG-val');maskidx=index(km)
    train_area=[]
    for s in train:train_area.append(loadmask(tracked(maskidx[s])).mean())
    tertiles=np.quantile(train_area,[1/3,2/3])
    fs=[]
    for ds in ['Kvasir-SEG-val','CVC-ClinicDB']:
        a,b=dataset_paths(ds);ims=index(a);masks=index(b)
        stems=sorted(val if ds=='Kvasir-SEG-val' else masks)
        assert len(stems)==(120 if ds=='Kvasir-SEG-val' else 612)
        fs+=features(ds,stems,ims,masks)
    f=pd.DataFrame(fs)
    cuts={};correlations=[]
    for ds,g in f.groupby('dataset'):
        cuts[ds]={}
        for field in ['contrast_main','contrast_narrow','contrast_wide','boundary_gradient']:
            lo,hi=g[field].quantile([.25,.75])
            cuts[ds][field]=dict(q25=float(lo),q75=float(hi))
            labels=np.where(g[field]<=lo,'Low',np.where(g[field]>=hi,'High','Intermediate'))
            f.loc[g.index,field+'_group']=np.where(g[field].isna(),'Missing',labels)
        rho,p=spearmanr(g.area_ratio,g.contrast_main,nan_policy='omit')
        correlations.append(dict(dataset=ds,n=len(g),area_contrast_spearman=float(rho)))
    f['train_tertile_group']=np.where(f.area_ratio<=tertiles[0],'Lower third',np.where(f.area_ratio<=tertiles[1],'Middle third','Upper third'))
    save(f,'image_features.csv')
    rows=[];audits=[]
    for model,(formal,predroot) in MODELS.items():
        fp=tracked(ROOT/'numbers/official_core'/formal/'per_image.json')
        sp=tracked(ROOT/'numbers/official_core'/formal/'summary.json')
        raw=json.loads(fp.read_text(encoding='utf-8-sig'));summ=json.loads(sp.read_text(encoding='utf-8-sig'))
        for ds,g in f.groupby('dataset'):
            rec=raw[ds] if isinstance(raw[ds],list) else raw[ds]['per_image']
            names=[Path(x['name']).stem for x in rec]
            assert len(names)==len(set(names)) and set(names)==set(g.id),(model,ds)
            lookup={Path(x['name']).stem:x for x in rec}
            pred_dir=ORIG/predroot/ds/'preds'
            preds=index(pred_dir) if pred_dir.exists() else {}
            for feature in g.to_dict('records'):
                r=lookup[feature['id']]
                row=dict(feature,model=model,dice=r['dice'],iou=r['iou'],legacy_distance=r.get('hd95',np.nan),
                         prediction_match=False,prediction_status='missing',reconstructed_dice=np.nan,
                         dice_discrepancy=np.nan,surface_hd95_px=np.nan,boundary_f1=np.nan,empty_prediction=False)
                pp=preds.get(feature['id'])
                if pp:
                    pm=loadmask(tracked(pp));gt=loadmask(feature['mask'])
                    if pm.shape!=gt.shape:
                        row['prediction_status']='shape_mismatch'
                    else:
                        denom=pm.sum()+gt.sum();d=2*np.logical_and(pm,gt).sum()/denom if denom else 1
                        row['reconstructed_dice']=float(d);row['dice_discrepancy']=abs(d-r['dice'])
                        row['prediction_match']=bool(row['dice_discrepancy']<=1e-6)
                        row['prediction_status']='matched' if row['prediction_match'] else 'dice_mismatch'
                        if row['prediction_match']:row.update(surface_scores(pm,gt))
                rows.append(row)
            a=np.mean([r['dice'] for r in rec]);b=np.mean([r['iou'] for r in rec])
            assert abs(a-summ[ds]['dice_mean'])<1e-7 and abs(b-summ[ds]['iou_mean'])<1e-7,(model,ds)
            audits.append(dict(model=model,dataset=ds,n=len(rec),dice_mean=a,iou_mean=b,summary_match=True))
        print('metrics',model,flush=True)
    joined=pd.DataFrame(rows);save(joined,'per_image_joined.csv')
    groups={
        'scale_groups.csv':['dataset','model','size_group'],
        'scale_training_tertiles.csv':['dataset','model','train_tertile_group'],
        'boundary_contrast_groups.csv':['dataset','model','contrast_main_group'],
        'boundary_narrow_sensitivity.csv':['dataset','model','contrast_narrow_group'],
        'boundary_wide_sensitivity.csv':['dataset','model','contrast_wide_group'],
        'boundary_gradient_sensitivity.csv':['dataset','model','boundary_gradient_group'],
        'scale_contrast_crossed.csv':['dataset','model','size_group','contrast_main_group']}
    tables={}
    for name,cols in groups.items():
        t=grouped(joined,cols);save(t,name);tables[name]=t
        for (ds,model),g in t.groupby(['dataset','model']):
            full=joined[(joined.dataset==ds)&(joined.model==model)]
            assert abs(np.average(g.dice_mean,weights=g.n)-full.dice.mean())<1e-10
    plot_groups(tables['scale_groups.csv'],'size_group',['Small','Medium','Large'],'scale_dice','Segmentation by relative foreground area')
    plot_groups(tables['boundary_contrast_groups.csv'],'contrast_main_group',['Low','Intermediate','High'],'contrast_dice','Segmentation by boundary colour contrast proxy')
    plot_groups(tables['boundary_contrast_groups.csv'],'contrast_main_group',['Low','Intermediate','High'],'contrast_boundary_f1','Boundary F1 by boundary colour contrast proxy','boundary_f1_mean')
    prompt=[]
    for p in sorted((ROOT/'numbers/ablation_box_expand').rglob('summary.json')):
        key=next((x for x in ['medsam_peft','pp_sam_baseline','samed_baseline','msa_polyp_train'] if x in p.parts),None)
        if not key:continue
        parts=[x for x in p.parts if x.startswith('box') or x.startswith('kvasir_box')]
        if not parts:continue
        expansion=int(parts[-1].split('box')[-1]);d=json.loads(tracked(p).read_text())
        for ds,v in d.items():
            if isinstance(v,dict) and 'dice_mean' in v:
                prompt.append(dict(model={'medsam_peft':'MedSAM+LoRA','pp_sam_baseline':'PP-SAM','samed_baseline':'SAMed','msa_polyp_train':'MSA-B'}[key],dataset=ds,expansion_px=expansion,dice=v['dice_mean'],iou=v['iou_mean'],source=str(p.relative_to(ROOT))))
    prompt=pd.DataFrame(prompt);assert not prompt.duplicated(['dataset','model','expansion_px']).any()
    save(prompt,'prompt_expansion_curves.csv')
    fig,axs=plt.subplots(1,2,figsize=(10,4),sharey=True)
    for ax,ds in zip(axs,['Kvasir-SEG-val','CVC-ClinicDB']):
        for i,m in enumerate(['MSA-B','MedSAM+LoRA','PP-SAM','SAMed']):
            g=prompt[(prompt.dataset==ds)&(prompt.model==m)].sort_values('expansion_px')
            if len(g):ax.plot(g.expansion_px,g.dice,label=m,marker=['o','s','^','D'][i],color=plt.get_cmap('tab10').colors[i])
        ax.set_title(ds);ax.set_xticks([0,10,30,50]);ax.set_xlabel('Fixed outward expansion (pixels)');ax.set_ylim(0,1);ax.grid(alpha=.2)
    axs[0].set_ylabel('Mean Dice');axs[0].legend(frameon=False,fontsize=8)
    fig.suptitle('Prompt extent sensitivity from available raw JSON only')
    fig.tight_layout()
    for ext in ['png','svg']:fig.savefig(FIG/f'prompt_expansion.{ext}',dpi=220)
    plt.close(fig)
    fig,axs=plt.subplots(2,4,figsize=(12,6.5))
    examples=[]
    for r,ds in enumerate(['Kvasir-SEG-val','CVC-ClinicDB']):
        g=f[(f.dataset==ds)&(f.size_group=='Medium')].sort_values(['contrast_main','id'])
        selected=pd.concat([g.head(2),g.tail(2)])
        for col,(_,row) in enumerate(selected.iterrows()):
            ax=axs[r,col];rgb=image(row.image);m=loadmask(row['mask'])
            ax.imshow(rgb);ax.contour(m,levels=[.5],colors=['cyan'],linewidths=.6)
            ax.set_title(f"{'Kvasir' if r==0 else 'CVC'} | {'Lower' if col<2 else 'Higher'} contrast\nArea {row.area_ratio:.1%}, proxy {row.contrast_main:.2f}",fontsize=9)
            ax.axis('off');examples.append(dict(dataset=ds,id=row.id,area_ratio=row.area_ratio,contrast_main=row.contrast_main,selection='two lowest/two highest contrast within medium-area group'))
    fig.suptitle('Feature-selected examples with GT contours; no model-based selection',fontsize=12)
    fig.tight_layout()
    for ext in ['png','svg']:fig.savefig(FIG/f'boundary_proxy_examples.{ext}',dpi=220)
    plt.close(fig);save(pd.DataFrame(examples),'boundary_proxy_example_ids.csv')
    pa=joined.groupby(['dataset','model','prediction_status']).size().rename('n').reset_index();save(pa,'prediction_audit.csv')
    save(pd.DataFrame(audits),'summary_reconciliation.csv');save(pd.DataFrame(correlations),'feature_dependence.csv')
    info=dict(date='2026-11-18',analysis='post_hoc_descriptive',python=sys.version,platform=platform.platform(),
        train_tertile_cutoffs=tertiles.tolist(),boundary_cutoffs=cuts,elapsed_seconds=time.time()-start,
        official_per_image_rows=len(joined),features=len(f),models=list(MODELS),
        prediction_matching_tolerance=1e-6,unique_source_files=len({x['path'] for x in HASHES}),
        source_hashes=list({x['path']:x for x in HASHES}.values()))
    (OUT/'provenance.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    print(pa.to_string(index=False),flush=True)
    print('Finished in',round(time.time()-start,1),'seconds',flush=True)


if __name__=='__main__':main()
