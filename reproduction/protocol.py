"""Shared frozen-manifest boundary; importing this module does not train/evaluate."""
import csv, hashlib, json
from pathlib import Path

def read_manifest(path):
 path=Path(path);protocol=json.loads((path.parent/'protocol.json').read_text())
 expected=protocol['manifest_sha256'].get(path.stem)
 if not expected or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
  raise ValueError(f'Unregistered or modified manifest: {path}')
 with path.open() as f:rows=list(csv.DictReader(f))
 if not rows:raise ValueError('Empty manifest')
 for r in rows:
  r['label']=int(r['label'])
  if not Path(r['path']).is_file():raise FileNotFoundError(r['path'])
 return rows

def assert_disjoint(*splits):
 for field in ['original_id','group_id','sha256','pixel_sha256']:
  seen=set()
  for rows in splits:
   keys={r[field] for r in rows}
   if '' in keys or len(keys)!=len(rows):raise ValueError(f'Duplicate or empty {field} within split')
   if seen&keys:raise ValueError(f'Leakage by {field}')
   seen.update(keys)

def derived_row(parent,path,**metadata):
 forbidden={'original_id','group_id','split','source_split'}&metadata.keys()
 if forbidden:raise ValueError(f'Cannot override identity/split: {forbidden}')
 return dict(parent,**metadata,path=str(path),parent_sha256=parent['sha256'])

class ManifestImages:
 def __init__(self,path,transform):self.rows=read_manifest(path);self.transform=transform
 def __len__(self):return len(self.rows)
 def __getitem__(self,index):
  from PIL import Image,ImageOps
  r=self.rows[index]
  with Image.open(r['path']) as im:im.load();image=ImageOps.exif_transpose(im).convert('RGB')
  return self.transform(image),r['label']
