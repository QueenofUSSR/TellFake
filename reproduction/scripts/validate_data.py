"""Independent P0 gate over frozen artifacts; never run model inference."""
import csv,hashlib,json,sqlite3,sys
from collections import Counter,defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from reproduction.protocol import read_manifest,assert_disjoint
M=ROOT/'reproduction/manifests/v1_20261114'
def validate():
 p=json.loads((M/'protocol.json').read_text());splits={n:read_manifest(M/(n+'.csv')) for n in p['manifest_sha256']}
 independent={n:rows for n,rows in splits.items() if n not in ['train_S0','train_S1','train_S2']}
 assert_disjoint(*independent.values())
 for name,n in [('train_S0',4000),('train_S1',16000),('train_S2',64000),('train_S3',128000)]:
  rs=splits[name];assert len(rs)==n
  assert Counter(r['generator'] for r in rs)=={'Real':n//2,'SD14':n//4,'BigGAN':n//4}
  assert all(r['source_split']=='train' for r in rs)
 for a,b in [('train_S0','train_S1'),('train_S1','train_S2'),('train_S2','train_S3')]:
  assert {r['sha256'] for r in splits[a]}<{r['sha256'] for r in splits[b]}
 assert len(splits['val'])==10000 and len(splits['dev'])==8000
 for n,rs in splits.items():
  if n.startswith('test_'):assert all(r['source_split']=='val' for r in rs)
  else:assert all(r['source_split']=='train' for r in rs)
 historic=defaultdict(set);old_dhash=set()
 for name,expected in p['historical_manifest_sha256'].items():
  f=ROOT/name;assert hashlib.sha256(f.read_bytes()).hexdigest()==expected
  with f.open() as stream:
   for r in csv.DictReader(stream):
    for field in ['sha256','pixel_sha256','original_id','group_id']:
     if r.get(field):historic[field].add(r[field])
    if r.get('dhash'):old_dhash.add(int(r['dhash'],16))
 rows=[r for rs in independent.values() for r in rs]
 for field,old in historic.items():assert not {r[field] for r in rows}&old,field
 db=sqlite3.connect('file:'+p['audit_db']+'?mode=ro',uri=True)
 cache={path:(sha,size,mt) for path,sha,size,mt in db.execute('SELECT path,sha,size,mtime FROM decoded')}
 for r in rows:
  f=Path(r['path']);stat=f.stat();h,size,mt=cache[str(f)]
  assert (r['sha256'],stat.st_size,stat.st_mtime_ns)==(h,size,mt),'File changed since full hash/decode'
 # Include legacy v2 dHashes obtained from its actual standard_v1 source paths.
 for (j,) in db.execute('SELECT row_json FROM decoded'):
  r=json.loads(j)
  if r.get('historical') and r.get('dhash'):old_dhash.add(int(r['dhash'],16))
 # Independent cross-component check, including historic near-duplicates.
 bands=[defaultdict(set) for _ in range(5)];allhash=set()
 def add(h):
  allhash.add(h)
  for b in range(5):bands[b][(h>>(b*13))&8191].add(h)
 for h in old_dhash:add(h)
 for r in rows:
  h=int(r['dhash'],16);near=set()
  for b in range(5):near.update(bands[b][(h>>(b*13))&8191])
  assert not any((h^v).bit_count()<=4 for v in near),'Near duplicate survived exclusion'
  add(h)
 result=dict(status='passed',protocol_sha256=hashlib.sha256((M/'protocol.json').read_bytes()).hexdigest(),unique_selected=len(rows),counts={n:len(rs) for n,rs in splits.items()},checks=['frozen manifest SHA256','nested S0/S1/S2/S3','balanced real/fake and seen sources','source official train/val boundaries','identity/group/file/pixel disjointness','historical manifest exclusions','independent exhaustive dHash <=4 exclusion','post-audit size/mtime unchanged'],limitations=['dHash is not crop-invariant or a semantic duplicate guarantee','Reserve outside candidate set not fully pixel-audited this round','Confirmation sets remain in previously studied domains'])
 (ROOT/'reproduction/data_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':validate()
