import csv,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import numpy as np
from reproduction.p1.run_method import metrics_and_scores
class PredictionTests(unittest.TestCase):
 def test_csv_uses_float64_fpr_boundary(self):
  rows=[dict(path=str(i),original_id=str(i),generator='Real' if i<100 else 'SD14',label=int(i>=100)) for i in range(200)]
  scores=np.array([.4]*100+[.5]*100,dtype=np.float32)
  with tempfile.TemporaryDirectory(dir='/mnt/ssd/data/cache/arin5201/reproduction') as tmp:
   metrics=metrics_and_scores(Path(tmp),rows,scores)
   with (Path(tmp)/'validation_scores.csv').open() as f:out=list(csv.DictReader(f))
   self.assertEqual(metrics['FPR01']['real_fpr'],0.)
   self.assertTrue(all(int(r['prediction_FPR01'])==0 for r in out[:100]))
if __name__=='__main__':unittest.main()
