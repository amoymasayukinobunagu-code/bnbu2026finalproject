"""Paired SAM ViT-B evaluation of annotation-derived points and translated boxes."""
import argparse
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import cv2
from segment_anything import SamPredictor, sam_model_registry

from complete_supplement import ROOT, dataset_paths, image, index, loadmask, surface_scores

sys.path.insert(0, str(ROOT / 'tools'))
from prompt_toolkit import generate_prompts


DATASETS = {'Kvasir-SEG-val': ('prompts_kvasir_val.json', 120),
            'CVC-ClinicDB': ('prompts_cvc.json', 612)}
SEEDS = (24, 42, 1234)
BASELINE = 'box_30'
METRICS = ('dice', 'iou', 'boundary_f1')
MSA_CODE = Path('E:/Project/FYP/code/medical_sam_adapter')
MSA_WEIGHT = Path('E:/Project/FYP/checkpoints/medical_sam_adapter/polyp/best_dice_checkpoint.pth')


class MsaPredictor:
    """Use the formal MSA-B resize, decoder selection and mask threshold."""
    def __init__(self):
        sys.path.insert(0, str(MSA_CODE))
        import eval_polyp_v3 as implementation
        self.model = implementation.build_model()
        self.size = implementation.IMAGE_SIZE
        self.device = implementation.base.DEVICE

    def set_image(self, rgb):
        self.h, self.w = rgb.shape[:2]
        resized = cv2.resize(rgb, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        tensor = torch.as_tensor(resized, dtype=torch.float, device=self.device).permute(2,0,1)[None]
        self.embedding = self.model.image_encoder(self.model.preprocess(tensor))

    def predict(self, multimask_output=False, box=None, point_coords=None, point_labels=None):
        scale = np.array([self.size / self.w, self.size / self.h])
        def coordinates(x):
            return torch.as_tensor(x * scale, dtype=torch.float32, device=self.device)
        boxes = coordinates(box.reshape(2,2)).reshape(1,4) if box is not None else None
        points = (coordinates(point_coords)[None], torch.as_tensor(
            point_labels, dtype=torch.int64, device=self.device)[None]) if point_coords is not None else None
        sparse, dense = self.model.prompt_encoder(points=points, boxes=boxes, masks=None)
        masks, quality = self.model.mask_decoder(
            image_embeddings=self.embedding, image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense, multimask_output=True)
        selected = int(quality[0].argmax())
        resized = torch.nn.functional.interpolate(masks[:,selected:selected+1], size=(self.size,self.size),
                                                  mode='bilinear',align_corners=False)
        binary = (torch.sigmoid(resized).squeeze().cpu().numpy() > 0.5).astype(np.uint8)
        return np.array([cv2.resize(binary,(self.w,self.h),interpolation=cv2.INTER_NEAREST).astype(bool)]), None, None


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def prompts_for_image(gt, stem):
    # Preserve box extent by capping translation to available room on each axis.
    h, w = gt.shape
    salt = int(hashlib.sha256(stem.encode()).hexdigest()[:8], 16)
    seed = (SEEDS[0] + salt) % 2**32
    generated = generate_prompts(gt, seed)
    base = generated['boxes']['30']
    yield BASELINE, {'box': base}, {}
    for direction, dx, dy in [('left', -round(.05*w), 0), ('right', round(.05*w), 0),
                              ('up', 0, -round(.05*h)), ('down', 0, round(.05*h))]:
        ax = min(max(dx, -base[0]), w-base[2])
        ay = min(max(dy, -base[1]), h-base[3])
        box = [base[0]+ax, base[1]+ay, base[2]+ax, base[3]+ay]
        yield 'shift_'+direction, {'box': box}, {
            'requested_dx': dx, 'requested_dy': dy, 'actual_dx': ax, 'actual_dy': ay,
            'shift_capped': (dx,dy)!=(ax,ay), 'shift_nonzero': bool(ax or ay)}
    pts = {p['kind']:p['xy'] for p in generated['points']}
    centre = pts['interior_positive']
    yield 'interior_positive', {'point_coords':[centre], 'point_labels':[1]}, {}
    yield 'interior_positive_plus_negative', {
        'point_coords':[centre, pts['random_negative']], 'point_labels':[1,0]}, {
        'seed': SEEDS[0], 'image_seed': seed}
    fg = np.argwhere(gt)
    for base_seed in SEEDS:
        image_seed = (base_seed + salt) % 2**32
        y,x = fg[np.random.default_rng(image_seed).integers(len(fg))]
        yield 'random_positive', {'point_coords':[[int(x),int(y)]], 'point_labels':[1]}, {
            'seed':base_seed, 'image_seed':image_seed}


def summarize(rows, out):
    frame = pd.DataFrame(rows)
    base = frame[frame.condition==BASELINE].set_index(['dataset','id'])
    for metric in METRICS:
        frame['delta_'+metric] = frame[metric] - base[metric].reindex(
            pd.MultiIndex.from_frame(frame[['dataset','id']])).to_numpy()
    frame.to_csv(out/'per_image.csv',index=False)
    summaries = []
    random = frame[frame.condition=='random_positive'].copy()
    for (dataset, condition), group in frame.groupby(['dataset','condition'], sort=False):
        subsets = [('all_images',group)]
        if condition.startswith('shift_'):
            subsets.append(('full_requested_shift',group[group.shift_capped.eq(False)]))
        for scope, subset in subsets:
            means = subset.groupby('id')[list(METRICS)+['delta_'+m for m in METRICS]].mean()
            row = dict(dataset=dataset,condition=condition,scope=scope,n_images=len(means),
                       n_evaluations=len(subset))
            for metric in means:
                row[metric+'_mean'] = means[metric].mean()
                row[metric+'_image_sd'] = means[metric].std(ddof=0)
            if condition.startswith('shift_'):
                row.update(n_zero_shift=int(subset.shift_nonzero.eq(False).sum()),
                           n_capped=int(subset.shift_capped.eq(True).sum()))
            summaries.append(row)
    pd.DataFrame(summaries).to_csv(out/'summary.csv',index=False)
    random.groupby(['dataset','seed'])[list(METRICS)+['delta_'+m for m in METRICS]].agg(
        ['size','mean']).to_csv(out/'random_point_seed_summary.csv')
    spread = random.groupby(['dataset','id'])[list(METRICS)].agg(lambda s:s.max()-s.min())
    spread.groupby('dataset').agg(['mean','median','max']).to_csv(out/'random_point_image_spread.csv')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--limit', type=int, help='Smoke-test image count per dataset')
    parser.add_argument('--model', choices=('sam-b','msa-b'), default='sam-b')
    parser.add_argument('--checkpoint', type=Path, default=Path('E:/Project/FYP/checkpoints/sam_vit_b.pth'))
    args = parser.parse_args()
    start = time.monotonic()
    out = args.out
    out.mkdir(parents=True, exist_ok=False)
    if args.model=='msa-b':
        args.checkpoint = MSA_WEIGHT
    formal = 'sam_v3' if args.model=='sam-b' else 'msa_polyp_baseline_v3'
    official_path = ROOT/'numbers/official_core'/formal/'per_image.json'
    official = json.loads(official_path.read_text())
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    metadata = dict(status='running', started_at=datetime.now().astimezone().isoformat(),
                    checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=file_hash(args.checkpoint),
                    official_per_image_sha256=file_hash(official_path), random_seeds=list(SEEDS),
                    script_sha256=file_hash(__file__), toolkit_sha256=file_hash(ROOT/'tools/prompt_toolkit.py'),
                    metric_script_sha256=file_hash(ROOT/'analysis/complete_supplement.py'),
                    inference_runtime=dict(device=device, torch=str(torch.__version__),
                                           gpu=torch.cuda.get_device_name(0) if device=='cuda' else None),
                    limit=args.limit, model=args.model, mask_selection=(
                        'single output' if args.model=='sam-b' else 'highest predicted IoU among decoder candidates'),
                    translation='round(5% of width or height), capped to available space, extent preserved',
                    boundary_f1_tolerance='max(1, round(0.005 * native image diagonal))')
    (out/'run.json').write_text(json.dumps(metadata,indent=2))
    predictor = (SamPredictor(sam_model_registry['vit_b'](checkpoint=str(args.checkpoint)).to(device).eval())
                 if args.model=='sam-b' else MsaPredictor())
    rows = []
    max_errors = {}
    with (out/'predictions.jsonl').open('w',encoding='utf-8') as log, torch.inference_mode():
        for ds,(manifest,n_expected) in DATASETS.items():
            ip,mp = dataset_paths(ds)
            images,masks = index(ip),index(mp)
            manifest_file = ROOT/'analysis/outputs'/manifest
            metadata.setdefault('input_manifest_hashes',{})[manifest] = file_hash(manifest_file)
            records = json.loads(manifest_file.read_text())['records']
            official_rows = official[ds] if isinstance(official[ds],list) else official[ds]['per_image']
            reference = {Path(r['name']).stem:r for r in official_rows}
            assert len(records)==n_expected and {r['id'] for r in records}==reference.keys(),ds
            assert n_expected==len(reference),ds
            selected = records[:args.limit]
            max_errors[ds] = dict(dice=0.,iou=0.,n=len(selected))
            for i, record in enumerate(selected,1):
                stem = record['id']
                gt = loadmask(masks[stem])
                rgb = image(images[stem])
                assert gt.any() and not gt.all() and gt.shape==rgb.shape[:2],(ds,stem)
                assert file_hash(masks[stem])==record['sha256'],(ds,stem)
                predictor.set_image(rgb)
                image_rows=[]
                for condition,prompt,details in prompts_for_image(gt,stem):
                    if condition==BASELINE:
                        assert prompt['box']==record['boxes']['30'],(ds,stem)
                    arrays = {k:np.asarray(v) for k,v in prompt.items()}
                    pred,_,_ = predictor.predict(multimask_output=False,**arrays)
                    mask = pred[0]
                    inter = (mask & gt).sum()
                    metrics = dict(dice=float(2*inter/(mask.sum()+gt.sum())),
                                   iou=float(inter/(mask | gt).sum()),
                                   boundary_f1=surface_scores(mask,gt)['boundary_f1'])
                    if condition==BASELINE:
                        for name in ['dice','iou']:
                            err=abs(metrics[name]-reference[stem][name])
                            max_errors[ds][name]=max(max_errors[ds][name],err)
                    image_rows.append(dict(dataset=ds,id=stem,condition=condition,
                                           prompt_json=json.dumps(prompt),**details,**metrics))
                for row in image_rows:
                    log.write(json.dumps(row)+'\n')
                log.flush()
                rows.extend(image_rows)
                if i%50==0 or i==len(selected):
                    print(f'{ds}: {i}/{len(selected)}, baseline max error {max_errors[ds]}',flush=True)
    summarize(rows,out)
    metadata.update(status='complete',max_baseline_errors=max_errors,n_rows=len(rows),
                    elapsed_seconds=round(time.monotonic()-start,2))
    (out/'run.json').write_text(json.dumps(metadata,indent=2))


if __name__=='__main__':
    main()
