"""Register planned experiments without scheduling or starting training."""
import hashlib,itertools,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'reproduction'
p=json.loads((R/'manifests/v1_20261114/protocol.json').read_text());patches={x['name']:x for x in json.loads((R/'patches/registry.json').read_text())}
methods=[('frozen_L14','ARIN5201_CV_Project'),('effort','ARIN5201-AIGI-Detection'),('effort_degradation','ARIN5201-AIGI-Detection'),('lora','ARIN5201-AIGI-Detection'),('dual_adapter','clip_dual_aigi'),('dual_ssf','clip_dual_aigi'),('frozen_only','clip_dual_aigi'),('adapter_only','clip_dual_aigi'),('ssf_only','clip_dual_aigi'),('IAPL_full_TTA_OIS','IAPL'),('IAPL_no_TTA','IAPL'),('effort_npr','ARIN5201-AIGI-Detection'),('npr_only','ARIN5201-AIGI-Detection'),('effort_rgb','ARIN5201-AIGI-Detection'),('effort_npr_noaux','ARIN5201-AIGI-Detection'),('resnet50','ARIN5201-AIGI-Detection'),('deit_s','ARIN5201-AIGI-Detection'),('swin_t','ARIN5201-AIGI-Detection'),('cnnspot','ARIN5201-AIGI-Detection')]
rows=[]
for method,repo in methods:
 for size in ['S0','S1','S2','S3']:
  for seed in ([42] if size=='S0' else [42,43,44]):
   row=dict(experiment_id=f'U_{method}_{size}_s{seed}',track='U',method=method,upstream=patches[repo],scale=size,seed=seed,train_manifest='train_'+size,data_sha256=p['manifest_sha256']['train_'+size],val_manifest='val',dev_manifest='dev',environment='core-4.46.3' if repo!='IAPL' else 'iapl-4.46.3',status='pending_P1' if size=='S0' else 'pending_S0_budget_and_recipe_freeze' if size!='S3' else 'provisional_requires_pretest_panel_freeze',training_budget={'epochs':None,'updates':None,'effective_batch':None},estimated_gpu_hours=None,actual_gpu_hours=None,dependencies=['P0 acceptance','method interface tests','native preprocessing preservation'],evaluate_final=False)
   if repo=='IAPL':row['dependencies']+=['per-image TTA parameter/optimizer reset test','single GPU coverage','inference cost measurement']
   rows.append(row)
for model,prep,jpeg,source in itertools.product(['B16'],['resize','crop128'],[False,True],['BigGAN','BigGAN+ADM']):
 rows.append(dict(experiment_id=f'R_Thomas_{model}_{prep}_jpeg{int(jpeg)}_{source}',track='R',method='frozen_features_logistic',model=model,preprocessing=prep,jpeg=jpeg,train_source=source,upstream=patches['ARIN5201_CV_Project'],status='pending_P1',dependencies=['R source-balanced manifests','equal total original count','GLIDE-only reporting'],estimated_gpu_hours=None))
rows.append(dict(experiment_id='R_Thomas_B32_resize',track='R',method='frozen_B32',upstream=patches['ARIN5201_CV_Project'],status='pending_P1',estimated_gpu_hours=None))
for method in ['D0','D1','D2','D3j','D3c','chroma_data_x_online_2x2']:
 rows.append(dict(experiment_id='R_mixinran_'+method,track='R',method=method,upstream=patches['ARIN5201-AIGI-Detection'],status='blocked_missing_auxiliary_data' if method in ['D3j','D3c','chroma_data_x_online_2x2'] else 'pending_data_constructor_validation',dependencies=['source/pair provenance','derived manifest','SD2.1 VAE revision and per-parent seed'],estimated_gpu_hours=None))
for method in ['dual_adapter','dual_ssf']:
 rows.append(dict(experiment_id='R_xhgSh_'+method,track='R',method=method,upstream=patches['clip_dual_aigi'],status='partial_data_missing_Wukong',train_sources=['SD14'],expected_test_sources=['SD14','SD15','BigGAN','ADM','GLIDE','VQDM','Midjourney','Wukong'],missing_sources=['Wukong'],dependencies=['R_SD14 corrected validation','preserve eight-source denominator; no full GenImage mAcc until Wukong available'],estimated_gpu_hours=None))
rows.append(dict(experiment_id='R_IAPL_native',track='R',method='IAPL_full_TTA_OIS',upstream=patches['IAPL'],status='pending_P1_with_missing_Wukong',native_epochs=1,dependencies=['native transform preservation','single GPU/reset tests','Wukong for full eight-source result'],estimated_gpu_hours=None))
rows.append(dict(experiment_id='R_GAPL_optional',track='R',status='optional_not_queued',upstream=patches['GAPL'],dependencies=['P1 budget decision','Community Forensics source and leakage review'],estimated_gpu_hours=None))
(R/'experiment_matrix.json').write_text(json.dumps({'status':'planning_only_no_training_started','budget_note':'P0 does not invent cost estimates: 100 warm-up-excluded batches and validation measurement required in P1 before formal queue freeze. S3 is provisional. Fixed-update comparisons require P1 measured recipe.','rows':rows},indent=2)+'\n')
print('Registered',len(rows),'planned rows')
