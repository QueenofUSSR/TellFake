"""Create deterministic, reviewable patches; never edit third_party working trees."""
import difflib,hashlib,json,subprocess,tarfile,io
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'reproduction';BASE=Path('/mnt/ssd/data/cache/arin5201/reproduction/sources')
locks=json.loads((ROOT/'third_party/sources.lock.json').read_text())+json.loads((ROOT/'third_party/external_baselines.lock.json').read_text())
changes={}
def modify(repo,path,fn):
 p=Path(next(r['local_path'] for r in locks if r['name']==repo))/path
 old=p.read_text();new=fn(old)
 if old==new:raise ValueError('No change '+path)
 changes.setdefault(repo,{})[path]=(old,new)
def replace(old,a,b):
 if a not in old:raise ValueError('Patch anchor missing '+a[:100])
 return old.replace(a,b)

def dual_data(s):
 start=s.index('def build_rows(');end=s.index('\ndef train_transform(',start)
 return s[:start]+'''def build_rows(data_root, generator, split):
    """Strict legacy reader; missing labels/splits are errors, never fallback."""
    rows=[]
    for label,folder in [(0,'nature'),(1,'ai')]:
        paths=list_images(Path(data_root)/generator/split/folder)
        if not paths: raise FileNotFoundError(f'Missing {generator}/{split}/{folder}')
        rows.extend(dict(path=str(p.resolve()),label=label,generator=generator,split=split) for p in paths)
    return rows


def prepare_manifests(cfg):
    """Consume externally audited manifests only; never regenerate a data split."""
    from reproduction.protocol import read_manifest, assert_disjoint
    source=Path(cfg['frozen_manifest_dir']);out=Path(cfg['manifest_dir'])
    if out.resolve()==source.resolve(): raise ValueError('Do not overwrite frozen manifests')
    train=read_manifest(source/(cfg['train_manifest']+'.csv'))
    val=read_manifest(source/(cfg.get('val_manifest','val')+'.csv'))
    assert_disjoint(train,val)
    if any(r['source_split']!='train' for r in train+val): raise ValueError('Model selection requires official train')
    # Final test loading is explicitly opt-in, separate from model selection.
    tests={}
    if cfg.get('evaluate_final',False):
        real=read_manifest(source/'test_real.csv')
        for g in cfg['test_generators']:
            tests[g]=real+read_manifest(source/('test_'+g+'.csv'))
            assert_disjoint(train,val,tests[g])
    out.mkdir(parents=True,exist_ok=True)
    def write(name,rows):
        with (out/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=['path','label','generator','split'],extrasaction='ignore');w.writeheader();w.writerows(rows)
    write('train.csv',train);write('val.csv',val)
    for g,rows in tests.items():write('test_'+safe_name(g)+'.csv',rows)
    return dict(train_n=len(train),val_n=len(val),available_test=list(tests),missing_test=[])

''' +s[end:]
modify('clip_dual_aigi','dataset.py',dual_data)
def dual_train(s):
 s=replace(s,'loss = loss_fn(model(images), labels) / accum','window_size = min(accum, len(train_loader) - ((step - 1) // accum) * accum)\n                loss = loss_fn(model(images), labels) / window_size')
 s=replace(s,'running += loss.item() * accum * labels.size(0)','running += loss.item() * window_size * labels.size(0)')
 s=replace(s,'    model.load_state_dict(ckpt[\'trainable\'], strict=False)','    model.load_state_dict(ckpt[\'trainable\'], strict=False)\n    if not cfg.get("evaluate_final", False):\n        return {"method": method, "best_epoch": best_epoch, "final_evaluation": "not_requested"}')
 return s
modify('clip_dual_aigi','train.py',dual_train)
def iapl_main(s):
 s=replace(s,"    parser.add_argument('--seed'","    parser.add_argument('--frozen-manifest-dir', required=True)\n    parser.add_argument('--train-manifest', default='train_S0')\n    parser.add_argument('--val-manifest', default='val')\n    parser.add_argument('--clip-path', required=True)\n    parser.add_argument('--seed'")
 s=replace(s,'    dataset_vals, selected_subsets = dataset_creator.build_dataset("test", selected_subsets=args.test_selected_subsets)', '''    from reproduction.protocol import ManifestImages, assert_disjoint
    if args.eval:
        raise ValueError('P0/P1 launcher does not expose final evaluation; use frozen evaluation protocol in P3')
    manifest_root = Path(args.frozen_manifest_dir)
    validation = ManifestImages(manifest_root/(args.val_manifest+'.csv'), dataset_creator.transforms['val'])
    training = ManifestImages(manifest_root/(args.train_manifest+'.csv'), dataset_creator.transforms['train'])
    assert_disjoint(training.rows, validation.rows)
    if any(r['source_split'] != 'train' for r in training.rows+validation.rows):
        raise ValueError('Training/model selection must use official train only')
    dataset_vals, selected_subsets = [validation], ['validation']''')
 s=replace(s,'    dataset_train = dataset_creator.build_dataset("train", selected_subsets=args.train_selected_subsets)','    dataset_train = training')
 # Disable the unvalidated native TTA route until the frozen scorer exists.
 s=replace(s,'        testtime_main(args)','        raise RuntimeError("TTA requires the P1 reset/coverage validation and P3 frozen scorer")')
 return s
modify('IAPL','main.py',iapl_main)
modify('IAPL','models/clip_models.py',lambda s:replace(s,"load_clip_to_cpu('/Path/to/ViT-L-14.pt',","load_clip_to_cpu(args.clip_path,"))
# SD2.1 stochastic posterior sampling gets per-parent RNG; retries preserve the draw.
for path in ['code/data/build_sets.py','code/data/build_d3.py']:
 def vae(s):
  s=replace(s,"'sd2-community/stable-diffusion-2-1', subfolder='vae'","os.environ['REPRO_VAE_PATH'], subfolder='vae'")
  s=replace(s,'        with torch.no_grad():','        sample_seed = int.from_bytes(__import__("hashlib").sha256((str(os.environ.get("SEED", "42"))+"|"+r["path"]).encode()).digest()[:8], "big") % (2**63)\n        with torch.no_grad():')
  s=replace(s,'.latent_dist.sample()', '.latent_dist.sample(generator=torch.Generator(device="cuda").manual_seed(sample_seed))')
  return s
 modify('ARIN5201-AIGI-Detection',path,vae)
# Enforce local immutable CLIP path rather than fetching a moving main revision.
modify('ARIN5201-AIGI-Detection','code/exp/train_ft.py',lambda s:replace(s,"CLIPVisionModel.from_pretrained('openai/clip-vit-large-patch14')","CLIPVisionModel.from_pretrained(os.environ['REPRO_CLIP_PATH'], local_files_only=True)"))
register=[]
for r in locks:
 target=BASE/r['name'];target.mkdir(parents=True,exist_ok=True)
 # archive only tracked source; no .git, no upstream mutations
 data=subprocess.check_output(['git','-C',r['local_path'],'archive',r['commit']])
 with tarfile.open(fileobj=io.BytesIO(data)) as t:t.extractall(target,filter='data')
 patch=''.join(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='a/'+path,tofile='b/'+path)) for path,(old,new) in changes.get(r['name'],{}).items())
 record=dict(name=r['name'],upstream_commit=r['commit'],source=str(target))
 if patch:
  p=OUT/'patches'/(r['name']+'.patch');p.write_text(patch)
  subprocess.run(['git','apply','--check',str(p)],cwd=target,check=True)
  subprocess.run(['git','apply',str(p)],cwd=target,check=True)
  record.update(patch=str(p.relative_to(ROOT)),patch_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
 register.append(record)
(OUT/'patches/registry.json').write_text(json.dumps(register,indent=2)+'\n')
print('Materialized',len(register),'fixed snapshots; patched',len(changes))
