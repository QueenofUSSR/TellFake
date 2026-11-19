"""Post-P3 exploratory comparison on a hash-selected, paired image panel."""
import csv
import hashlib
import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from reproduction.p1 import models as native
from reproduction.proposal.evaluate_degradations_repeats import corrupt, CONDITIONS
from reproduction.proposal.p3 import panel
from reproduction.proposal.run import data_loader, sha, write
from reproduction.protocol import read_manifest, assert_disjoint
from genimage_detector.metrics import grouped_metrics

OUT = Path('/mnt/ssd/data/logs/arin5201/reproduction/demo_comparison_v1')
ROOT = Path(__file__).resolve().parents[2]
SRC = native.SRC
P3 = OUT.parent / 'proposal_p3_20261119'
CONDITIONS = ['clean', *CONDITIONS]
TRAIN = ROOT/'reproduction/manifests/R_BigGAN/train_S0.csv'
DUAL = SRC/'clip_dual_aigi/checkpoints/dual_adapter/best.pt'


def sample():
    all_rows = panel()
    rows = []
    for source in ['Real', 'SD14', 'BigGAN', 'ADM', 'GLIDE']:
        ordered = sorted((r for r in all_rows if r['generator'] == source),
                         key=lambda r: hashlib.sha256(('demo-v1|' + r['original_id']).encode()).digest())
        rows.extend(ordered[:600 if source == 'Real' else 300])
    assert len(rows) == 1800
    assert_disjoint(rows, read_manifest(TRAIN))
    return rows


class Transform:
    def __init__(self, base, condition, processor=False):
        self.base, self.condition, self.processor = base, condition, processor

    def __call__(self, image, seed):
        image = corrupt(image, self.condition)
        if self.processor:
            return self.base(images=image, return_tensors='pt')['pixel_values'][0]
        return self.base(image)


def build(name):
    if name == 'dual_adapter':
        model, _ = native.build(name)
        state = torch.load(DUAL, map_location='cpu', weights_only=True)
        weights = state['trainable']
        assert set(weights) == {k for k,v in model.named_parameters() if v.requires_grad}
        model.load_state_dict(weights, strict=False)
        dataset = native.module('dual_eval_data', SRC/'clip_dual_aigi/dataset.py')
        return model.cuda().eval(), dataset.eval_transform(), None
    path = native.WEIGHTS['openai/clip-vit-base-patch16']
    model = CLIPVisionModelWithProjection.from_pretrained(path, local_files_only=True).cuda().eval()
    proc = CLIPImageProcessor.from_pretrained(path, local_files_only=True)
    classifier_file = OUT/'thomas_classifier.pkl'
    if not classifier_file.exists():
        upstream = native.module('thomas_fit', SRC/'ARIN5201_CV_Project/scripts/train_baseline.py')
        train = [(r['path'],r['label'],r['generator']) for r in read_manifest(TRAIN)]
        args = SimpleNamespace(preprocess='resize',train_jpeg=(70,100),train_jpeg_p=.5,seed=0,batch=64)
        features, labels, _ = upstream.extract(train,'/',model,proc,args,'cuda',True)
        clf = LogisticRegression(C=1.,class_weight='balanced',max_iter=3000).fit(features,labels)
        tmp = classifier_file.with_suffix('.tmp')
        tmp.write_bytes(pickle.dumps(clf))
        tmp.replace(classifier_file)
    return model, proc, pickle.loads(classifier_file.read_bytes())


def predict(loaded, rows, condition):
    model, base, clf = loaded
    loader = data_loader(rows, Transform(base,condition,clf is not None),42,batch=32)
    scores = []
    with torch.inference_mode():
        for images, _, _ in loader:
            images = images.cuda()
            if clf is None:
                with torch.autocast('cuda'):
                    logits = model(images)
                p = logits.float().sigmoid().cpu().numpy()
            else:
                f = torch.nn.functional.normalize(model(pixel_values=images).image_embeds,dim=-1)
                p = clf.predict_proba(f.float().cpu().numpy())[:,1]
            scores.extend(p.tolist())
    return scores


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    rows = sample()
    write(OUT/'protocol.json',dict(ids=[r['original_id'] for r in rows],conditions=CONDITIONS,
          local_seed=42,threshold=.5,thomas_train_sha256=sha(TRAIN),dual_sha256=sha(DUAL),
          weighting='Real, SD14+BigGAN, ADM+GLIDE equal; within fake groups equal. Report clean and nine-corruption mean separately.'))
    for name in ['frozen_L14_resize','effort_degradation','thomas_b16_jpeg','dual_adapter']:
        local = name in ('frozen_L14_resize','effort_degradation')
        loaded = None
        for c in CONDITIONS:
            folder = OUT/name
            folder.mkdir(exist_ok=True)
            path = folder/(c+'.csv')
            if not path.exists():
                if local:
                    origin = P3/(name+'_seed42')/(c+'.csv')
                    assert sha(origin) == json.loads(origin.with_suffix('.json').read_text())['prediction_sha256']
                    with origin.open() as f:
                        saved = {r['original_id']:float(r['score']) for r in csv.DictReader(f)}
                    scores = [saved[r['original_id']] for r in rows]
                else:
                    if loaded is None:
                        loaded = build(name)
                    scores = predict(loaded,rows,c)
                tmp = path.with_suffix('.tmp')
                with tmp.open('w',newline='') as f:
                    w = csv.writer(f)
                    w.writerow(['original_id','source','label','score'])
                    w.writerows((r['original_id'],r['generator'],r['label'],p) for r,p in zip(rows,scores,strict=True))
                tmp.replace(path)
            with path.open() as f:
                scored = list(csv.DictReader(f))
            assert [(r['original_id'],r['generator'],r['label']) for r in rows] == [(r['original_id'],r['source'],int(r['label'])) for r in scored]
            metrics = grouped_metrics([r['label'] for r in rows],[float(r['score']) for r in scored],[r['generator'] for r in rows],.5)
            write(path.with_suffix('.json'),dict(metrics=metrics,prediction_sha256=sha(path)))
            print('DONE',name,c,flush=True)
        del loaded
        torch.cuda.empty_cache()


if __name__ == '__main__':
    torch.set_num_threads(4)
    main()
