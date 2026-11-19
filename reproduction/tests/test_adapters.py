import importlib.util,json,sys,tempfile,unittest
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
class ManifestAdapterTests(unittest.TestCase):
 def test_native_dual_loader_consumes_frozen_train_validation_only(self):
  p=Path('/mnt/ssd/data/cache/arin5201/reproduction/sources/clip_dual_aigi/dataset.py')
  spec=importlib.util.spec_from_file_location('native_dual_dataset',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
  cfg=yaml.safe_load((ROOT/'reproduction/configs/U_dual_adapter_S0_s42.yaml').read_text())
  with tempfile.TemporaryDirectory(dir='/mnt/ssd/data/cache/arin5201/reproduction') as tmp:
   cfg['manifest_dir']=tmp;result=m.prepare_manifests(cfg)
   self.assertEqual(result['train_n'],4000);self.assertEqual(result['val_n'],10000);self.assertEqual(result['available_test'],[])
   self.assertEqual({p.name for p in Path(tmp).iterdir()},{'train.csv','val.csv'})
   ds=m.ImageCSV(Path(tmp)/'val.csv',m.eval_transform());x,y=ds[0];self.assertEqual(tuple(x.shape),(3,224,224))
 def test_missing_final_source_raises(self):
  p=Path('/mnt/ssd/data/cache/arin5201/reproduction/sources/clip_dual_aigi/dataset.py')
  spec=importlib.util.spec_from_file_location('native_dual_dataset2',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
  cfg=yaml.safe_load((ROOT/'reproduction/configs/U_dual_adapter_S0_s42.yaml').read_text())
  with tempfile.TemporaryDirectory(dir='/mnt/ssd/data/cache/arin5201/reproduction') as tmp:
   cfg.update(manifest_dir=tmp,evaluate_final=True,test_generators=['Wukong'])
   with self.assertRaises((FileNotFoundError,ValueError)):m.prepare_manifests(cfg)
if __name__=='__main__':unittest.main()
