import json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from genimage_detector.data import load_supplement,write_json,write_csv,file_hash
from genimage_detector.engine import evaluate


class SupplementTests(unittest.TestCase):
 def setup_manifest(self,root):
  directory=root/'manifest';directory.mkdir()
  rows={}
  for name,n in [('midjourney_test',1),('external_real_test',2)]:
   rows[name]=[dict(path=f'{name}/{i}.jpg',label=int(name=='midjourney_test'),generator='Midjourney' if name=='midjourney_test' else 'Real',split=name,source_split='val',sha256=f'sha-{name}-{i}',pixel_sha256=f'pixel-{name}-{i}',original_id=f'id-{name}-{i}',group_id=f'group-{name}-{i}') for i in range(n)]
   write_csv(directory/(name+'.csv'),rows[name],tuple(rows[name][0]))
  metadata=dict(schema_version=1,kind='frozen_v3_supplement',data_root=str(root),base_manifest_sha256={'train':'base'},counts={n:len(r) for n,r in rows.items()},manifest_sha256={n:file_hash(directory/(n+'.csv')) for n in rows})
  write_json(directory/'supplement_config.json',metadata)
  return directory,rows,metadata

 def test_frozen_base_and_file_tampering_are_rejected(self):
  with tempfile.TemporaryDirectory() as tmp:
   d,rows,meta=self.setup_manifest(Path(tmp))
   load_supplement(d,{}, {'train':'base'})
   with self.assertRaisesRegex(ValueError,'different base'):load_supplement(d,{}, {'train':'different'})
   with (d/'midjourney_test.csv').open('a') as f:f.write('\n')
   with self.assertRaisesRegex(ValueError,'changed'):load_supplement(d,{}, {'train':'base'})

 def test_overlap_and_unsafe_paths_are_rejected(self):
  with tempfile.TemporaryDirectory() as tmp:
   d,rows,meta=self.setup_manifest(Path(tmp))
   with self.assertRaisesRegex(ValueError,'leakage'):
    load_supplement(d,{'train':[rows['midjourney_test'][0]]},{'train':'base'})
   rows['midjourney_test'][0]['path']='../outside.jpg'
   write_csv(d/'midjourney_test.csv',rows['midjourney_test'],tuple(rows['midjourney_test'][0]))
   meta['manifest_sha256']['midjourney_test']=file_hash(d/'midjourney_test.csv');write_json(d/'supplement_config.json',meta)
   with self.assertRaisesRegex(ValueError,'Unsafe'):load_supplement(d,{}, {'train':'base'})

 def test_real_only_evaluation_emits_fpr_without_nan_or_threshold_refit(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);d,rows,meta=self.setup_manifest(root);checkpoint=root/'checkpoint.pt';checkpoint.write_bytes(b'fixture')
   args=SimpleNamespace(conditions=['clean'],seed=42,device='cpu',manifest_dir='fixture',checkpoint=str(checkpoint),splits=['external_real_test'],supplement_dir=str(d),max_samples=None,log_root=str(root/'logs'),name='evaluation',data_root=str(root),batch_size=2,workers=0,amp=False)
   predictions=dict(labels=[0,0],probabilities=[.1,.9],paths=[r['path'] for r in rows['external_real_test']],generators=['Real','Real'])
   with patch('genimage_detector.engine.load_manifests',return_value=({}, {'train':'base'})),patch('genimage_detector.engine.load_checkpoint',return_value=(None,dict(manifest_sha256={'train':'base'},threshold=.7,image_size=64,smoke_test=False))),patch('genimage_detector.engine.snapshot'),patch('genimage_detector.engine.loader',return_value=None),patch('genimage_detector.engine.infer',return_value=predictions):
    evaluate(args)
   report=json.loads((root/'logs/evaluation/metrics.json').read_text());metric=report['results']['external_real_test']['clean']['overall']
   self.assertEqual(metric['real_fpr'],.5);self.assertEqual(metric['threshold'],.7);self.assertIsNone(metric['auroc'])

if __name__=='__main__':unittest.main()
