"""Read-only model evaluation on registered seen validation; never fit thresholds."""
import csv, io, json, pickle, time, traceback, os, fcntl
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageFilter
from reproduction.proposal import models
from reproduction.proposal.run import data_loader, sha, write
from reproduction.protocol import read_manifest
from genimage_detector.metrics import grouped_metrics

HERE=Path(__file__).resolve().parent
CONDITIONS=['jpeg90','jpeg70','jpeg50','blur0.5','blur1','blur2','downsample0.75','downsample0.5','downsample0.25']

def corrupt(im, condition):
    if condition=='clean':return im
    if condition.startswith('jpeg'):
        with io.BytesIO() as b:
            im.save(b,'JPEG',quality=int(condition[4:]),subsampling=2);b.seek(0)
            with Image.open(b) as decoded:return decoded.convert('RGB')
    if condition.startswith('blur'):return im.filter(ImageFilter.GaussianBlur(float(condition[4:])))
    if condition.startswith('downsample'):
        w,h=im.size;s=float(condition[10:])
        return im.resize((max(1,round(w*s)),max(1,round(h*s))),Image.Resampling.BICUBIC).resize((w,h),Image.Resampling.BICUBIC)
    raise ValueError(condition)

class Transform:
    def __init__(self,method,meta,condition):
        self.base=models.Transform(method,False,meta);self.condition=condition
    def __call__(self,im,seed):return self.base(corrupt(im,self.condition),seed)

def verify_operations():
    import random
    im=Image.fromarray(np.random.default_rng(42).integers(0,256,(127,301,3),dtype=np.uint8))
    for op,condition in enumerate(['clean']+CONDITIONS):
        seed=next(s for s in range(10000) if random.Random(s+2000).randrange(10)==op)
        np.testing.assert_array_equal(np.asarray(corrupt(im,condition)),np.asarray(models.degrade(im,seed)))

def predict(model,clf,meta,rows,method,condition):
    probabilities=[]
    dl=data_loader(rows,Transform(method,meta,condition),42,batch=32 if clf is not None else 16)
    with torch.inference_mode():
        for i,(x,_,_) in enumerate(dl):
            x=x.cuda(non_blocking=True)
            if clf is not None:
                features=torch.nn.functional.normalize(model(pixel_values=x).image_embeds,dim=-1).float().cpu().numpy()
                p=clf.predict_proba(features)[:,1]
            else:
                with torch.autocast('cuda',dtype=torch.bfloat16,enabled=meta['amp']):p=models.scores(model(x),meta['kind']).sigmoid().cpu().numpy()
            assert np.isfinite(p).all();probabilities.extend(p.tolist())
            if len(rows)>64 and i%200==0:print('EVAL',method,condition,min((i+1)*dl.batch_size,len(rows)),'/',len(rows),flush=True)
    return np.array(probabilities,dtype=np.float64)

def summarize(path,rows,thresholds,condition):
    with path.open() as f: saved=list(csv.DictReader(f))
    assert len(saved)==len(rows)
    assert [s['original_id'] for s in saved]==[r['original_id'] for r in rows]
    for s,r in zip(saved,rows):
        assert s['source']==r['generator'] and int(s['label'])==r['label'] and s['condition']==condition
        for k,t in thresholds.items():assert int(s['prediction_'+k])==int(float(s['score'])>=t)
    p=np.array([float(s['score']) for s in saved]);assert np.isfinite(p).all()
    return {k:grouped_metrics([r['label'] for r in rows],p,[r['generator'] for r in rows],t) for k,t in thresholds.items()}

def main():
    cfg=json.loads((HERE/'validation_degradations64k.json').read_text());out=Path(cfg['output_root']);out.mkdir(parents=True,exist_ok=True)
    lock=open(HERE/'queue.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for p,h in cfg['implementation_sha256'].items():assert sha(p)==h,p
    assert sha(cfg['validation_manifest'])==cfg['validation_sha256']
    rows=read_manifest(cfg['validation_manifest'])
    assert len(rows)==10000 and len({r['original_id'] for r in rows})==10000
    assert {r['generator'] for r in rows}=={'Real','SD14','BigGAN'}
    assert all(r['split']=='val' and r['source_split']=='train' for r in rows)
    verify_operations()
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    cfg_hash=sha(HERE/'validation_degradations64k.json')
    write(out/'status.json',dict(status='running',pid=os.getpid(),test_access=False))
    try:
        for spec in cfg['models']:
            method=spec['method'];dest=out/method;dest.mkdir(exist_ok=True)
            for key in ['checkpoint','thresholds','clean_scores']:assert sha(spec[key])==spec[key+'_sha256']
            thresholds=json.loads(Path(spec['thresholds']).read_text())['thresholds']
            model,meta=models.build(method);model=model.cuda().eval();clf=None
            if meta['kind']=='features':clf=pickle.loads(Path(spec['checkpoint']).read_bytes())
            else:
                st=torch.load(spec['checkpoint'],map_location='cpu',weights_only=False);model.load_state_dict(st['model'],strict=True);del st
            # Clean score replay checks the model, geometry, precision and classifier path.
            with open(spec['clean_scores']) as f:clean=list(csv.DictReader(f))
            assert [r['original_id'] for r in clean]==[r['original_id'] for r in rows]
            p=predict(model,clf,meta,rows[:64],method,'clean')
            np.testing.assert_allclose(p,[float(r['score']) for r in clean[:64]],rtol=0,atol=1e-7)
            write(dest/'preflight.json',dict(clean_replay_max_abs_error=float(np.max(np.abs(p-np.array([float(r['score']) for r in clean[:64]])))),degradation_matches_training_implementation=True))
            for condition in cfg['conditions']:
                path=dest/(condition+'.csv');receipt=dest/(condition+'.json')
                if receipt.exists():
                    old=json.loads(receipt.read_text());assert old['config_sha256']==cfg_hash and old['prediction_sha256']==sha(path)
                    summarize(path,rows,thresholds,condition);print('SKIP',method,condition,flush=True);continue
                started=time.time();p=predict(model,clf,meta,rows,method,condition)
                tmp=path.with_suffix('.tmp')
                with tmp.open('w',newline='') as f:
                    w=csv.DictWriter(f,fieldnames=['original_id','source','condition','label','score']+['prediction_'+k for k in thresholds]);w.writeheader()
                    for r,v in zip(rows,p):w.writerow(dict(original_id=r['original_id'],source=r['generator'],condition=condition,label=r['label'],score=float(v),**{'prediction_'+k:int(v>=t) for k,t in thresholds.items()}))
                tmp.replace(path)
                metrics=summarize(path,rows,thresholds,condition)
                write(receipt,dict(status='complete',config_sha256=cfg_hash,method=method,condition=condition,metrics=metrics,seconds=time.time()-started,prediction_sha256=sha(path),test_access=False))
                print('COMPLETE',method,condition,metrics['p05']['overall'],flush=True)
            del model,clf;torch.cuda.empty_cache()
        write(out/'status.json',dict(status='complete',conditions_per_model=9,models=3,predictions=270000,test_access=False))
    except BaseException:
        write(out/'status.json',dict(status='failed',traceback=traceback.format_exc(),test_access=False));raise
if __name__=='__main__':main()
