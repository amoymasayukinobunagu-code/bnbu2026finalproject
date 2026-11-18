"""Prepare blinded, local review files for independent boundary raters."""
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from complete_supplement import ROOT

OUT=ROOT/'analysis/outputs/boundary_expert_review'
OUT.mkdir(exist_ok=False)
pack=OUT/'reviewer_packet'
pack.mkdir()
data=pd.read_csv(ROOT/'analysis/outputs/image_features.csv')
rng=np.random.default_rng(20261118)
selected=[]
for _,group in data.groupby(['dataset','size_group','contrast_main_group']):
    selected.extend(rng.choice(group.index,min(4,len(group)),replace=False))
sample=data.loc[rng.permutation(selected)].copy()
sample['review_id']=[f'B{i:03d}' for i in range(1,len(sample)+1)]
sample[['review_id','dataset','id','size_group','contrast_main_group']].to_csv(
    OUT/'coordinator_key.csv',index=False)
for row in sample.itertuples():
    with Image.open(row.image) as src:
        src.convert('RGB').save(pack/f'{row.review_id}.png')
for reviewer in (1,2):
    ratings=sample[['review_id']].copy()
    for field in ['boundary_grade','uncertain','notes']:
        ratings[field]=''
    ratings.to_csv(pack/f'reviewer_{reviewer}_ratings.csv',index=False)
(pack/'README.md').write_text('''# Independent boundary review

Review the supplied frames independently at native size and without model scores.
Enter one grade per frame: 0 if the visible polyp perimeter is predominantly
clear, 1 if a substantial section is hard to trace, 2 if most of the perimeter
is hard to trace. If the lesion cannot be identified or image quality prevents
judgment, mark uncertain and document the reason. Use the notes field to flag
frames with multiple lesions or ambiguity. Do not consult the other reviewer's
ratings. The coordinator retains dataset, area and contrast labels in a separate
key. Leave doubtful grades blank. These categories are preliminary and should
be agreed by the reviewers before scoring. Raw frames are local review material.
''',encoding='utf-8')
print(f'Prepared {len(sample)} blinded frames and two blank ratings sheets')
