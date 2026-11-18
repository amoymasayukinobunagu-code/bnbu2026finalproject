"""Reproduce external box evaluation and batch-one inference timing."""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from segment_anything import SamPredictor, sam_model_registry

from evaluate_prompt_variants import MsaPredictor, file_hash
from prompt_toolkit import box_from_mask
from complete_supplement import DATA, ROOT, dataset_paths, image, index, loadmask

WEIGHTS = Path('E:/Project/FYP/checkpoints')
SEED = 20261118


def predictor(name):
    if name == 'MSA-B':
        os.environ['FYP_MSA_REPO'] = 'E:/Project/FYP/msa_repo'
        os.environ['FYP_MSA_IMAGE_SIZE'] = '512'
        return MsaPredictor()
    kind, weight = ('vit_b','sam_vit_b.pth') if name=='SAM-B' else ('vit_l','sam_vit_l_0b3195.pth')
    return SamPredictor(sam_model_registry[kind](checkpoint=str(WEIGHTS/weight)).cuda().eval())


def forward(model, rgb, box):
    model.set_image(rgb)
    return model.predict(box=np.array(box),multimask_output=False)[0][0]


def prepared(im, mask):
    gt = loadmask(mask)
    rgb = image(im)
    assert gt.any() and rgb.shape[:2]==gt.shape, im
    return rgb, gt, box_from_mask(gt)


def external_pairs():
    known = set()
    for ds in ['Kvasir-SEG-val','CVC-ClinicDB']:
        known.update(file_hash(p) for p in index(dataset_paths(ds)[0]).values())
    for ds in ['ETIS-LaribPolypDB','CVC-ColonDB']:
        base = DATA/'data/TestDataset'/ds
        images, masks = index(base/'images'), index(base/'masks')
        assert images.keys()==masks.keys(), ds
        seen = set()
        for stem, im in images.items():
            digest = file_hash(im)
            status = 'overlaps_development_data' if digest in known else (
                'repeated_external_image' if digest in seen else 'included')
            seen.add(digest)
            yield ds, stem, im, masks[stem], digest, status


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('speed','external'),required=True)
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    (args.out/'run.json').write_text(json.dumps(dict(mode=args.mode,device=torch.cuda.get_device_name(),
        torch=torch.__version__,cuda=torch.version.cuda,batch_size=1,seed=SEED,
        warmups_per_dataset=5,timed_passes=3,sampled_images_per_dataset=30,
        timing_excludes='image and mask disk reads; model loading; prompt generation',
        timing_includes='model-specific input preprocessing, encoding, prompt decoding, mask returned on CPU',
        checkpoint_sha256={path.name:file_hash(path) for path in (
            WEIGHTS/'sam_vit_b.pth', WEIGHTS/'sam_vit_l_0b3195.pth',
            WEIGHTS/'medical_sam_adapter/polyp/best_dice_checkpoint.pth')}),indent=2))
    results=[]
    if args.mode=='speed':
        known = {}
        for ds in ['Kvasir-SEG-val','CVC-ClinicDB']:
            images,masks = map(index,dataset_paths(ds))
            if ds.startswith('Kvasir'):
                split = ROOT/'splits/Kvasir-SEG/Data-split/val.txt'
                ids = sorted(Path(x).stem for x in split.read_text().splitlines() if x.strip())
            else:
                ids = sorted(masks)
            sample = np.random.default_rng(SEED).choice(ids,30,replace=False)
            known[ds] = [(str(stem),*prepared(images[stem],masks[stem])[::2]) for stem in sample]
        for name in ['SAM-B','SAM-L','MSA-B']:
            pred = predictor(name)
            for ds,frames in known.items():
                for _,rgb,box in frames[:5]:
                    forward(pred,rgb,box)
                for rep in range(1,4):
                    for stem,rgb,box in frames:
                        torch.cuda.synchronize()
                        start=time.perf_counter()
                        forward(pred,rgb,box)
                        torch.cuda.synchronize()
                        results.append(dict(model=name,dataset=ds,id=stem,pass_no=rep,
                                            latency_ms=(time.perf_counter()-start)*1000))
            del pred
            torch.cuda.empty_cache()
            print(name, 'timed',flush=True)
        data = pd.DataFrame(results)
        agg = data.groupby(['model','dataset']).latency_ms.agg(
            n='size',mean_ms='mean',median_ms='median',p95_ms=lambda v:v.quantile(.95)).reset_index()
        agg['fps_from_mean_latency']=1000/agg.mean_ms
    else:
        inputs = list(external_pairs())
        pd.DataFrame(inputs,columns=['dataset','id','image','mask','sha256','status']).to_csv(
            args.out/'input_audit.csv',index=False)
        for name in ['SAM-B','MSA-B']:
            pred = predictor(name)
            for ds,stem,im,mask,_,status in inputs:
                if status!='included':
                    continue
                rgb,gt,box=prepared(im,mask)
                result=forward(pred,rgb,box)
                intersection=(result & gt).sum()
                results.append(dict(model=name,dataset=ds,id=stem,
                                    dice=2*intersection/(result.sum()+gt.sum()),
                                    iou=intersection/(result | gt).sum(),box=box))
            del pred
            torch.cuda.empty_cache()
            print(name, 'external evaluation complete',flush=True)
        data=pd.DataFrame(results)
        agg=data.groupby(['model','dataset']).agg(n=('dice','size'),dice=('dice','mean'),
                                                iou=('iou','mean')).reset_index()
    data.to_csv(args.out/'per_image.csv',index=False)
    agg.to_csv(args.out/'summary.csv',index=False)
    print(agg.to_string(index=False),flush=True)


if __name__=='__main__':
    main()
