"""Sequential S1 training/validation only; restart from atomic batch checkpoints."""
import argparse, csv, hashlib, json, os, pickle, random, time, traceback, warnings
from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from reproduction.protocol import read_manifest, assert_disjoint
from reproduction.proposal import models as core
from reproduction.p1 import models_extended as extra
from reproduction.p1.run_method import rng_state, restore_rng, sha, write
from genimage_detector.metrics import binary_metrics, grouped_metrics, choose_threshold, choose_fpr_threshold

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
METHODS = list(core.CORE_METHODS) + ['resnet50', 'deit_s', 'swin_t', 'cnnspot']

def atomic_save(obj, path):
    tmp = path.with_suffix('.tmp')
    torch.save(obj, tmp)
    tmp.replace(path)

def data_loader(rows, tf, seed, epoch=0, schedule=None, batch=16):
    ds = core.Images(rows, tf, seed, epoch)
    return torch.utils.data.DataLoader(ds, batch_size=batch, sampler=schedule,
        num_workers=4, pin_memory=True, generator=torch.Generator().manual_seed(seed + epoch))

def schedule_for(n, seed, epoch, replacement):
    gen = torch.Generator().manual_seed(seed + epoch)
    return (torch.multinomial(torch.ones(n), n, replacement=True, generator=gen)
            if replacement else torch.randperm(n, generator=gen)).tolist()

def evaluate(model, dl, meta):
    model.eval(); probabilities=[]; total=0.; count=0
    with torch.no_grad():
        for x, y, _ in dl:
            x,y=x.cuda(),y.cuda()
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=meta['amp']):
                output=model(x)
                loss=torch.nn.functional.cross_entropy(output.float(),y)
                p=core.scores(output,meta['kind']).sigmoid()
            assert torch.isfinite(p).all() and torch.isfinite(loss)
            probabilities.extend(p.cpu().tolist()); total+=float(loss)*len(y); count+=len(y)
    return np.array(probabilities), total/count

def report(rows, p):
    return grouped_metrics([r['label'] for r in rows], p, [r['generator'] for r in rows], .5)

def save_scores(path, rows, p, thresholds):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['original_id','source','condition','label','score']+['prediction_'+k for k in thresholds]);w.writeheader()
        for r,v in zip(rows,p):
            w.writerow(dict(original_id=r['original_id'],source=r['generator'],condition='clean',label=r['label'],score=float(v),**{'prediction_'+k:int(float(v)>=t) for k,t in thresholds.items()}))

def convergence(curve):
    if len(curve)<3:return 'insufficient_history'
    last=curve[-3:]; a=[r['validation']['overall']['auroc'] for r in last]; v=[r['validation_loss'] for r in last]; t=[r['train_loss'] for r in last]
    if a[2]>a[1]>a[0] and v[2]<v[1]<v[0]:return 'still_improving'
    if t[2]<t[1]<t[0] and v[2]>v[1]>v[0] and a[2]<a[1]<a[0]:return 'suspected_overfitting'
    if max(a)-min(a)<=.002 and max(v)-min(v)<=.01:return 'approximate_plateau'
    return 'inconclusive_or_oscillating'

def run(method, cfg, smoke=False):
    seed=cfg['seed']; out=Path(cfg['output_root'])/method; ckpt=Path(cfg['checkpoint_root'])/method
    out.mkdir(parents=True,exist_ok=True);ckpt.mkdir(parents=True,exist_ok=True)
    fingerprint=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
    if (out/'result.json').exists():
        old=json.loads((out/'result.json').read_text());assert old['config_sha256']==fingerprint
        for name,h in old['artifacts'].items():assert sha(name)==h
        print('SKIP verified complete',method,flush=True);return
    for source,h in cfg['implementation_sha256'].items():assert sha(source)==h,source
    for split in ['train','validation']:assert sha(cfg[split+'_manifest'])==cfg[split+'_sha256']
    started=time.time();write(out/'status.json',dict(status='running',pid=os.getpid(),started=started,method=method,test_access=False))
    try:
        train=read_manifest(cfg['train_manifest']);val=read_manifest(cfg['validation_manifest']);assert_disjoint(train,val)
        assert len(train)==16000 and len(val)==10000
        assert {r['generator'] for r in train+val}<={'Real','SD14','BigGAN'}
        assert all(r['source_split']=='train' for r in train+val)
        if smoke:
            train=sum(([r for r in train if r['label']==label][:32] for label in [0,1]),[])
            val=sum(([r for r in val if r['label']==label][:16] for label in [0,1]),[])
        torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);np.random.seed(seed);random.seed(seed)
        binding=core if method in core.CORE_METHODS else extra
        model,meta=binding.build(method);model=model.cuda()
        write(out/'config.json',dict(config=cfg,method=method,model={k:v for k,v in meta.items() if k!='native'},preprocess=core.PREPROCESS_VERSION if binding is core else 'p1_native_baseline',sampling='shuffle_without_replacement' if binding is core else 'balanced_with_replacement'))
        write(out/'trainable_parameters.json',{n:list(p.shape) for n,p in model.named_parameters() if p.requires_grad})
        print('INITIALIZED',method,'trainable',sum(p.numel() for p in model.parameters() if p.requires_grad),flush=True)
        trtf=binding.Transform(method,True,meta);vatf=binding.Transform(method,False,meta)
        if meta['kind']=='features':
            assert not any(p.requires_grad for p in model.parameters());model.eval()
            def extract(rows,tf):
                result=[]
                with torch.no_grad():
                    for i,(x,_,_) in enumerate(data_loader(rows,tf,seed,batch=32)):
                        f=model(pixel_values=x.cuda()).image_embeds
                        f=torch.nn.functional.normalize(f,dim=-1).float();assert torch.isfinite(f).all();result.append(f.cpu().numpy())
                        if i%100==0:print('EXTRACT',method,i,len(rows),flush=True)
                return np.concatenate(result)
            X=extract(train,trtf);V=extract(val,vatf)
            # Independent forward path on the new common geometry, no legacy resize processor.
            x,_,_=next(iter(data_loader(val,vatf,seed,batch=32)))
            with torch.no_grad():
                z=model.visual_projection(model.vision_model(pixel_values=x.cuda()).pooler_output)
                z=torch.nn.functional.normalize(z,dim=-1).cpu().numpy()
            np.testing.assert_allclose(z,V[:len(z)],rtol=2e-4,atol=2e-5)
            cache=Path(cfg['cache_root'])/fingerprint/method;cache.mkdir(parents=True,exist_ok=True)
            np.savez(cache/'features.npz',train=X,val=V,train_labels=[r['label'] for r in train],val_labels=[r['label'] for r in val])
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter('always');clf=LogisticRegression(C=1,class_weight='balanced',max_iter=3000).fit(X,[r['label'] for r in train])
            p=clf.predict_proba(V)[:,1];path=ckpt/'classifier.pkl';path.write_bytes(pickle.dumps(clf))
            np.testing.assert_array_equal(p,pickle.loads(path.read_bytes()).predict_proba(V)[:,1])
            detail=dict(iterations=clf.n_iter_.tolist(),convergence_warnings=[str(w.message) for w in captured if issubclass(w.category,ConvergenceWarning)],max_iter_reached=bool(max(clf.n_iter_)>=3000),feature_cache=str(cache),native_common_geometry_equivalent=True)
        else:
            params=[p for p in model.parameters() if p.requires_grad]
            if method.startswith('effort'):assert all(any(k in n for k in ['U_residual','S_residual','V_residual','head.']) for n,p in model.named_parameters() if p.requires_grad)
            opt=getattr(torch.optim,meta['optimizer'])(params,lr=meta['lr'],weight_decay=meta['weight_decay'])
            scheduler=torch.optim.lr_scheduler.StepLR(opt,step_size=10,gamma=.1)
            scaler=torch.amp.GradScaler('cuda',enabled=False)
            curve=[];epoch=0;offset=0;total_loss=0.;presented=0;updates=0;best=-1.;best_epoch=None
            def snapshot():
                return dict(model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),rng=rng_state(),epoch=epoch,offset=offset,schedule=schedule_for(len(train),seed,epoch,binding is extra),total_loss=total_loss,presented=presented,updates=updates,curve=curve,best=best,best_epoch=best_epoch,config_sha256=fingerprint)
            def restore(st):
                assert st['config_sha256']==fingerprint;model.load_state_dict(st['model']);opt.load_state_dict(st['optimizer']);scheduler.load_state_dict(st['scheduler']);scaler.load_state_dict(st['scaler']);restore_rng(st['rng'])
            if (ckpt/'last.pt').exists():
                st=torch.load(ckpt/'last.pt',map_location='cpu',weights_only=False);restore(st)
                epoch,offset,total_loss,presented,updates,curve,best,best_epoch=[st[k] for k in ['epoch','offset','total_loss','presented','updates','curve','best','best_epoch']];del st
            else:
                # Real disk replay includes optimizer, RNG and mutable model buffers.
                x,y,_=next(iter(data_loader(train,trtf,seed)));x,y=x[:2].cuda(),y[:2].cuda()
                atomic_save(snapshot(),ckpt/'preflight.pt')
                def step():
                    model.train();opt.zero_grad(set_to_none=True)
                    with torch.autocast('cuda',dtype=torch.bfloat16,enabled=meta['amp']):loss=binding.loss(model,x,y,meta['kind'])
                    assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(params,float('inf'),error_if_nonfinite=True);opt.step();return float(loss.detach())
                l=step();expected={n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}
                restore(torch.load(ckpt/'preflight.pt',map_location='cpu',weights_only=False));assert step()==l
                for n,p0 in model.named_parameters():
                    if p0.requires_grad:torch.testing.assert_close(p0.cpu(),expected[n],rtol=1e-5,atol=1e-7)
                restore(torch.load(ckpt/'preflight.pt',map_location='cpu',weights_only=False));del expected
                (ckpt/'preflight.pt').unlink();write(out/'preflight.json',dict(disk_optimizer_rng_replay=True,finite_loss_gradients=True))
            epochs=2 if smoke else cfg['epochs']
            while epoch<epochs:
                schedule=schedule_for(len(train),seed,epoch,binding is extra)
                model.train();epoch_start=time.time()
                for x,y,idx in data_loader(train,trtf,seed,epoch,schedule[offset:]):
                    x,y=x.cuda(),y.cuda();opt.zero_grad(set_to_none=True)
                    with torch.autocast('cuda',dtype=torch.bfloat16,enabled=meta['amp']):loss=binding.loss(model,x,y,meta['kind'])
                    assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(params,float('inf'),error_if_nonfinite=True);opt.step()
                    total_loss+=float(loss.detach())*len(y);presented+=len(y);offset+=len(y);updates+=1
                    if updates%100==0:print('TRAIN',method,'epoch',epoch+1,'samples',offset,'loss',total_loss/presented,flush=True)
                    if updates%250==0 or (smoke and offset==32):
                        atomic_save(snapshot(),ckpt/'last.pt')
                        if smoke:restore(torch.load(ckpt/'last.pt',map_location='cpu',weights_only=False))
                p,vl=evaluate(model,data_loader(val,vatf,seed),meta)
                entry=dict(epoch=epoch+1,train_loss=total_loss/presented,validation_loss=vl,validation=report(val,p),learning_rate=opt.param_groups[0]['lr'],updates=updates,epoch_seconds=time.time()-epoch_start)
                curve.append(entry);score=entry['validation']['overall']['auroc']
                if score>best:
                    best=score;best_epoch=epoch+1;atomic_save(snapshot(),ckpt/'best.pt')
                    np.save(out/'best_validation_probabilities.npy',p)
                save_scores(out/f'validation_epoch_{epoch+1}.csv',val,p,{'p05':.5})
                print('EPOCH',method,json.dumps(entry),flush=True)
                scheduler.step();epoch+=1;offset=0;total_loss=0.;presented=0
                atomic_save(snapshot(),ckpt/'last.pt');write(out/'curve.json',curve)
                # Reload the actual epoch boundary to exercise continuation state.
                restore(torch.load(ckpt/'last.pt',map_location='cpu',weights_only=False))
            st=torch.load(ckpt/'best.pt',map_location='cpu',weights_only=False);model.load_state_dict(st['model']);del st
            p,_=evaluate(model,data_loader(val,vatf,seed),meta)
            np.testing.assert_array_equal(p,np.load(out/'best_validation_probabilities.npy'))
            path=ckpt/'best.pt';detail=dict(best_epoch=best_epoch,updates=updates,convergence=convergence(curve),checkpoint_predictions_equal=True,epoch_boundary_reload=True)
        labels=[r['label'] for r in val]
        thresholds={'p05':.5,'max_BA':choose_threshold(labels,p),'FPR01':choose_fpr_threshold(labels,p)}
        write(out/'thresholds.json',dict(scope='validation_only',thresholds=thresholds))
        save_scores(out/'validation_scores.csv',val,p,thresholds)
        # Independently round-trip predictions before completion.
        with (out/'validation_scores.csv').open() as f: rows=list(csv.DictReader(f))
        q=np.array([float(r['score']) for r in rows]);np.testing.assert_array_equal(q,p)
        assert len(rows)==len(val) and len({r['original_id'] for r in rows})==len(val)
        for k,t in thresholds.items():assert all(int(r['prediction_'+k])==int(float(r['score'])>=t) for r in rows)
        result=dict(status='complete',method=method,seed=seed,config_sha256=fingerprint,test_access=False,train_originals=len(train),validation_originals=len(val),wall_seconds=time.time()-started,metrics={k:binary_metrics(labels,q,t) for k,t in thresholds.items()},by_source=report(val,q),detail=detail,checkpoint=str(path),artifacts={str(p):sha(p) for p in [path,out/'validation_scores.csv',out/'thresholds.json']})
        write(out/'result.json',result);write(out/'status.json',dict(status='complete',method=method));print('COMPLETE',method,flush=True)
    except BaseException:
        write(out/'status.json',dict(status='failed',method=method,traceback=traceback.format_exc()));raise

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',default=str(HERE/'round1.json'));parser.add_argument('--method',choices=METHODS);parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    cfg=json.loads(Path(args.config).read_text())
    if args.smoke:
        for k in ['output_root','checkpoint_root','cache_root']:cfg[k]+='_smoke'
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    import fcntl
    lock=open(HERE/'queue.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for method in ([args.method] if args.method else cfg['methods']):run(method,cfg,args.smoke)
if __name__=='__main__':main()
