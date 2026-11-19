"""Record available provenance and its limits; do not fabricate archive hashes."""
import csv,hashlib,json,sqlite3
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'reproduction';LOG=Path('/mnt/ssd/data/logs/arin5201');reserve=LOG/'reserve_20261114'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
evidence=[]
for p in [reserve/'complete.json',reserve/'retained.csv',reserve/'SD14_bad.json',reserve/'ADM_bad.json',reserve/'ADM_archives_removed.json',LOG/'downloads_v3/sd14_mirror_manifest.json',LOG/'downloads_v3/sd14_extracted_archive_index.json',LOG/'reserve_finalize.log',ROOT/'scripts/finalize_dataset_reserve.py']:
 if p.exists():evidence.append(dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size))
db=sqlite3.connect('file:'+str(reserve/'retention.sqlite')+'?mode=ro',uri=True)
raw=Counter()
for source,path,label in db.execute('SELECT source,path,label FROM files'):
 parts=Path(path).parts;split='train' if 'train' in parts else 'val' if 'val' in parts else 'unknown';raw[f'{source}/{split}/{label}']+=1
ancillary={name:dict(status='not_found_in_checked_roots',checked_roots=['/mnt/ssd/data/datasets','/mnt/ssd/data/downloads'],search='directory names, maxdepth 3; not a whole-machine absence proof',required_for=uses) for name,uses in [('RAISE','D3j/D3c training'),('Mirage','original out-of-domain evaluation'),('EvalGEN','original evaluation'),('Wukong','original eight-source GenImage mean')]}
data=dict(status='local_sources_traceable_with_disclosed_limits',historical_inventory_counts_before_retention=dict(raw),retained_inventory='data_inventory.json',evidence=evidence,SD14=dict(path='/mnt/ssd/data/datasets/GenImage/official_v3/SD14/imagenet_ai_0419_sdv4',mirror_revision=json.loads((LOG/'downloads_v3/sd14_mirror_manifest.json').read_text())['revision'],quarantined=3,provenance='Mirror manifest, archive member index, extraction/decoding and retention receipts retained; no new full archive validation claimed.'),ADM=dict(path='/mnt/ssd/data/datasets/GenImage/official_v3/ADM/imagenet_ai_0508_adm',quarantined=6,provenance='LAN archive transfer, canonical layout, CRC-checked zip extraction and decode/retention records. Original volumes already removed before this task; no surviving external archive SHA256 verification receipt was found. Local selected file SHA256 is frozen; do not claim external byte-for-byte authenticity.'),auxiliary=ancillary,legacy='v2 uses standard_v1; v3 uses official_v3. No version directories merged. All previously used v2/v3/supplement manifests excluded from new selections.',licenses='GenImage underlying real/generated source terms are not replaced by code licenses. This task records local provenance; no redistribution or license grant is implied.')
(R/'data_provenance.json').write_text(json.dumps(data,indent=2)+'\n');print('Provenance and missing-data register recorded')
