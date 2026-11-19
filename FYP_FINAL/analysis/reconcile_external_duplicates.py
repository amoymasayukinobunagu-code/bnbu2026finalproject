"""Reconcile duplicate external image content and flag conflicting binary masks."""
import hashlib
import pandas as pd
from complete_supplement import ROOT, loadmask

out=ROOT/'analysis/outputs/external_box_20261118'
audit=pd.read_csv(out/'input_audit.csv',dtype={'id':str})
audit['binary_mask_sha256']=audit['mask'].map(
    lambda p:hashlib.sha256(loadmask(p).tobytes()).hexdigest())
group=audit.groupby(['dataset','sha256']).binary_mask_sha256.transform('nunique')
audit['conflicting_mask_for_duplicate_image']=group.gt(1)
audit['include_in_reviewed_summary']=audit.status.eq('included') & group.eq(1)
audit.to_csv(out/'duplicate_label_audit.csv',index=False)
scores=pd.read_csv(out/'per_image.csv',dtype={'id':str})
checked=scores.merge(audit[['dataset','id','include_in_reviewed_summary']],
                     on=['dataset','id'],validate='many_to_one')
assert len(checked)==len(scores)
primary=checked[checked.include_in_reviewed_summary]
agg=primary.groupby(['model','dataset']).agg(n=('dice','size'),dice=('dice','mean'),
                                            iou=('iou','mean')).reset_index()
agg.to_csv(out/'summary_reviewed.csv',index=False)
print('Cross-external byte-identical images:',
      len(set(audit[audit.dataset=='CVC-ColonDB'].sha256)&set(audit[audit.dataset=='ETIS-LaribPolypDB'].sha256)))
print(agg.to_string(index=False))
