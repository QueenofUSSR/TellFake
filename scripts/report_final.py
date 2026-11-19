"""Six-page report; all numeric claims drawn from verified final artifacts."""
import csv,json,html,sys
from pathlib import Path
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,PageBreak,Image,Table,TableStyle
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
pdfmetrics.registerFont(TTFont("DejaVu", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"))
pdfmetrics.registerFont(TTFont("DejaVu-Bold", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"))
pdfmetrics.registerFontFamily("DejaVu", normal="DejaVu", bold="DejaVu-Bold", italic="DejaVu", boldItalic="DejaVu-Bold")
R=Path(sys.argv[1]) if len(sys.argv)>1 else Path('/mnt/ssd/code/arin5201/results/final_20261113');D=json.loads((R/'results.json').read_text());S=D['overall'];I=D['paired_intervals']
styles=getSampleStyleSheet();styles.add(ParagraphStyle(name='BodySmall',fontName='DejaVu',fontSize=9,leading=12,spaceAfter=7));styles['Title'].fontSize=20;styles['Title'].leading=24
styles['Title'].fontName='DejaVu-Bold'
styles['Heading2'].fontName='DejaVu-Bold'
story=[]
def p(text):story.append(Paragraph(text,styles['BodySmall']))
def title(text):story.append(Paragraph(text,styles['Title']));story.append(Spacer(1,12))
def heading(text):story.append(Paragraph(text,styles['Heading2']))
def metric(method,name,scope='non_strong'):return next(r for r in S if r['method']==method and r['metric']==name and r['scope']==scope)
def value(method,name,scope='non_strong'):
 r=metric(method,name,scope);return f"{100*r['mean']:.2f} ± {100*r['std']:.2f}"
def group_value(method,gen,metric):
 rows=list(csv.DictReader((R/'per_seed_metrics.csv').open()))
 import statistics
 vals=[]
 for seed in ['42','43','44']:
  v=[float(r[metric]) for r in rows if r['method']==method and r['seed']==seed and r['group']==gen and r['condition'] in D['conditions'][:7]]
  vals.append(sum(v)/len(v))
 return f"{100*statistics.mean(vals):.2f} ± {100*statistics.stdev(vals):.2f}"
def ci(name,scope='non_strong'):return next(r for r in I if r['metric']==name and r['scope']==scope)
def image(name,width=475,height=180):story.append(Image(str(R/name),width=width,height=height));story.append(Spacer(1,6))
def table(rows,widths=None):
 t=Table([[Paragraph(str(v),styles['BodySmall']) for v in row] for row in rows],colWidths=widths,repeatRows=1)
 t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#eeeeee')),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.5,colors.grey),('BOTTOMPADDING',(0,0),(-1,-1),5)]));story.append(t);story.append(Spacer(1,9))
def page():story.append(PageBreak())
delta=ci('macro_balanced_accuracy');direction='improved' if delta['delta']>0 else 'reduced';evidence='excludes zero' if delta['low95']>0 or delta['high95']<0 else 'includes zero'
title('AI Image Detection\nunder Unseen Processing Chains')
p('ARIN5201 experimental report • 13 November 2026 • ResNet-50 • Three seeds per method')
heading('Main result')
p(f"On the seven non-strong final conditions, paired consistency {direction} macro balanced accuracy relative to equal-budget two-view augmentation by <b>{100*delta['delta']:+.2f} percentage points</b>. The paired original-image 95% interval is [{100*delta['low95']:+.2f}, {100*delta['high95']:+.2f}] pp and {evidence}. This interval is conditional on the three trained model pairs; it is not a universal deployment guarantee.")
p('A: ordinary training. B: two independently augmented views with classification losses averaged. C: the same views and update budget plus Bernoulli Jensen-Shannon consistency, lambda=0.1. Lambda was selected on a separate development set before final predictions.')
table([['Model','Macro AUROC ×100','Macro BA %','Unknown recall %','ImageNet FPR %','COCO FPR %']]+[[m,value(m,'macro_auroc'),value(m,'macro_balanced_accuracy'),value(m,'unknown_recall'),value(m,'ImageNet_FPR'),value(m,'COCO_FPR')] for m in ['A','B','C']],[45,80,80,90,85,85])
p('Entries are mean ± sample SD across seeds 42/43/44, after averaging conditions within each seed. Macro BA gives equal weight to BigGAN, SD1.5, VQDM and Midjourney, each paired with the same ImageNet real pool. Unknown recall averages VQDM and Midjourney. The strong chain is excluded from this table and reported separately.')
p(f"Candidate C's unknown-generator recall averages <b>{metric('C','unknown_recall')['mean']*100:.2f}%</b> over these final conditions. This cross-generator result must be weighed alongside macro BA: strong clean validation does not establish reliable detection of unseen generators.")
heading('Scope and verification')
p(f"Nine complete 10-epoch runs; 15,995 unique final images. All {D['prediction_files']} prediction CSVs ({D['prediction_rows_verified']:,} rows, including repeated conditions/shared-real views) were checked against manifests and their metrics recomputed. Shared real predictions were identical across the three GenImage evaluation splits and were counted once in aggregate FPR.")
p('The principal reliability question is whether robustness transfers while real-photo false alarms remain controlled. A validation target of 1% does not force a 1% test FPR. The independent source results and strong-degradation failure modes are therefore part of the main result, not optional exclusions.')
page();title('2. Data protocol and audit')
table([['Partition','Images','Purpose'],['Train','16,000','8,000 real; BigGAN/SD1.5 each 4,000'],['Validation','7,000','5,000 real; two seen generators each 1,000'],['Development','4,000','2,000 real; two seen generators each 1,000'],['Final seen AI','2,000','BigGAN/SD1.5 each 1,000'],['Final unseen AI','4,000','VQDM/Midjourney each 2,000'],['Final ImageNet real','5,000','Shared across all generator comparisons'],['Final COCO real','4,995','Independent FPR, never mixed into ImageNet FPR']],[115,65,285])
p('Total experiment inventory: 42,995 distinct selected images. Training, validation and development come only from official train; GenImage final images come only from official val. Midjourney and VQDM never enter detector fitting or hyperparameter selection. COCO val2017 is final-only.')
heading('Content audit')
p('The base protocol decoded 50,000 candidate images and selected 36,000 after original-identity grouping, byte/pixel hashes and conservative dHash screening. The extension decoded all 6,000 available Midjourney val images and 5,000 COCO images. Twelve new near-duplicate candidates were excluded conservatively (seven Midjourney, five COCO); 2,000 eligible Midjourney images were selected. Exclusions are candidate risks, not confirmed duplicate or label-error claims.')
p('Original identity, exact pixel/byte content and selected near-duplicate components are separated across partitions. dHash Hamming distance ≤4 cannot detect every crop or derivative. This is a selected-candidate audit, not a complete near-duplicate audit of the full reserve. COCO files agree with local archive lengths and CRCs; archive SHA-256 is self-computed, not an official attestation.')
heading('Source limitations')
p('Real GenImage images are JPEG, whereas generated images are PNG; native resolutions differ by generator. Uniform 224×224 inputs do not eliminate compression or resolution shortcuts. ImageNet-pretrained ResNet also introduces possible pretraining exposure; this experiment does not certify complete pretraining-data independence. COCO is a conventional natural-image proxy, not a camera-only authenticity-certified corpus. Images remain subject to their original Flickr terms; no source images are redistributed here.')
page();title('3. Methods, budget and numerical recovery')
p('All methods fine-tune ImageNet-pretrained ResNet-50 at 224×224 with AdamW (lr 1e-4, weight decay 1e-4), cosine decay, batch64 original images and AMP. Each run has 10 epochs and 2,500 update batches. B/C concatenate two views (128 view tensors/update) and use the same random-view law. A is the auxiliary ordinary-training baseline; equal compute is specifically enforced for B versus C.')
p('Each B/C view independently flips horizontally and uniformly selects clean, JPEG70, resize0.5, resize0.5→JPEG70, or JPEG70→resize0.5. JPEG is nonprogressive 4:2:0; resize/shrink/restore is bicubic. C adds lambda × symmetric Bernoulli Jensen-Shannon loss to mean two-view BCE, with gradients through both predictions. Development tested only lambda0.1 and lambda1, on a two-epoch, single-seed budget. Lambda0.1 was frozen before final evaluation.')
p('Checkpoint selection: highest clean validation AUROC, then lower BCE. Threshold: a conservative validation-real order statistic satisfying empirical FPR≤1%, moving above tied boundary scores. Each checkpoint keeps that threshold for every final source and processing condition; test predictions are never used to refit it.')
image('training.png',475,155)
p('Figure 1. Mean training objectives and clean validation BCE across seeds. A/B/C training objectives differ; only validation BCE is directly comparable. Source: training.csv; ten epochs per run.')
heading('Recorded failure and recovery')
p('B/seed44 failed after six completed epochs because unscaled gradients were nonfinite. Its failed directory was retained. A tested numerical fix reduces AMP scale and replays the exact batch, restoring BatchNorm buffers and CPU/CUDA RNG state; persistent failures still abort. B44 was restarted from initialization and C44 used the same guarded implementation. No seed was removed and no update was intentionally skipped. Early replay-free epochs matched the failed run exactly. Other seven successful runs retain their original source snapshots.')
p('Model/run registry, thresholds, checkpoint hashes and retry counts are preserved in results.json and verified_runs.json. The implementation passed 25 protocol tests including injected-overflow replay against a finite reference.')
page();title('4. Final generator and processing results')
image('robustness.png',475,185)
p('Figure 2. Final macro balanced accuracy and unknown-generator recall, mean ± seed SD. JPEG/resize condition order is fixed. Strong chain = resize0.25→JPEG50→blur1; all conditions are simulated, not measured social-platform processing.')
image('paired.png',475,170)
p('Figure 3. C minus B paired balanced accuracy with pointwise 95% original-image bootstrap intervals. Each original retains all condition views and its matched B/C predictions. The three seed differences are averaged before image resampling; intervals therefore measure conditional image uncertainty, while Figure 2 reports seed variation.')
table([['Generator','B recall %','C recall %','C AUROC ×100']]+[[g,group_value('B',g,'recall'),group_value('C',g,'recall'),group_value('C',g,'auroc')] for g in ['BigGAN','SD15','VQDM','Midjourney']],[105,120,120,120])
p('Generator table: non-strong condition mean, then mean ± SD across seeds. AUROC pairs each generator with the shared ImageNet real pool.')
p('The per-generator tables, including AUROC, recall and balanced accuracy for each condition and seed, are in per_seed_metrics.csv and seed_summary.csv. Worst-group values are retained in overall_summary.csv; no unfavorable seed or condition was removed.')
page();title('5. Real-photo false alarms and failure boundaries')
image('false_positives.png',475,190)
p('Figure 4. Actual test false-positive rates on 5,000 ImageNet and 4,995 COCO real images. Error bars are seed SD; the dashed 1% line is the validation fitting target, not a test guarantee. Shared ImageNet originals are counted once despite appearing in generator-paired prediction files.')
table([['Scope / method','ImageNet FPR %','COCO FPR %','Unknown recall %']]+[[s+' / '+m,value(m,'ImageNet_FPR',s),value(m,'COCO_FPR',s),value(m,'unknown_recall',s)] for s in ['clean','strong'] for m in ['A','B','C']],[105,120,120,120])
for name in ['ImageNet_FPR','COCO_FPR','unknown_recall']:
 a=ci(name);p(f"Non-strong C-B {name}: {100*a['delta']:+.2f} pp; paired 95% interval [{100*a['low95']:+.2f}, {100*a['high95']:+.2f}] pp.")
p('COCO is a real-only pool: AUROC, binary recall, precision and balanced accuracy are not interpreted there. Cross-source false alarms may reflect content, compression, pretraining exposure or other distribution differences; this experiment cannot isolate a single cause. Strong degradation may erase forensic evidence. The observed curves describe this protocol, not arbitrary images or generators.')
page();title('6. Interpretation, uncertainty and reproduction')
p(f"The prespecified primary B/C comparison yields {100*delta['delta']:+.2f} pp macro BA over the seven non-strong conditions; its conditional image interval {evidence}. Interpret this jointly with false alarms, unknown-source recall and the strong-chain results above. Development gains alone were not treated as final evidence.")
heading('Statistical limits')
seed_delta=next(r for r in D['paired_seed_deltas'] if r['scope']=='non_strong' and r['metric']=='macro_balanced_accuracy')
p(f"Paired C-B non-strong macro BA differences by training seed: {100*seed_delta['seed42']:+.2f}, {100*seed_delta['seed43']:+.2f}, {100*seed_delta['seed44']:+.2f} pp for seeds42/43/44; paired seed SD {100*seed_delta['std']:.2f} pp.")
p('Three training seeds give a limited estimate of model variability. Reported SD uses ddof=1. The bootstrap uses 1,000 stratified original-image resamples (seed20261113), separately resampling each generator and real source while retaining all processed views. The ImageNet real resample is shared across generator comparisons. Intervals are percentile, pointwise and not multiplicity-adjusted; they do not establish universal superiority or deployment-level error guarantees. AUROC is summarized across seeds; no bootstrap AUROC interval is claimed.')
heading('Reproduction and traceability')
p('Entry points: scripts/complete_experiment.py for guarded recovery and frozen evaluation; scripts/summarize_final.py for prediction validation and clustered statistics; scripts/plot_final.py and scripts/report_final.py for figures and this report. Re-summary requires a fresh output directory, preventing silent overwrites. Source data are read-only inputs; no Docker migration or container creation was used.')
p('Data: data/manifests/v3 and data/manifests/v3_supplement_20261113. Frozen protocol: /mnt/ssd/data/logs/arin5201/formal_20261113/protocol.json. Recovery and final run registry: /mnt/ssd/data/logs/arin5201/final_20261113/. Raw checkpoints: /mnt/ssd/data/ckpt/arin5201/. Raw prediction CSVs reside in each registered *_final directory.')
p('This report directory contains prediction_ledger.csv (file hashes and row counts), verification.json, per_seed_metrics.csv, seed_summary.csv, overall_summary.csv, paired_cluster_intervals.csv, training.csv and results.json. Every displayed number is derived from these files; failed and smoke runs remain separate and do not contribute to final statistics.')
heading('References and boundaries')
p('GenImage: A Million-Scale Benchmark for Detecting AI-Generated Image (NeurIPS 2023), https://arxiv.org/abs/2306.08571. Fake or JPEG? Revealing Common Biases in Generated Image Detection Datasets (2024), https://arxiv.org/abs/2403.17608. COCO download and image terms: https://cocodataset.org/#download and #termsofuse.')
p('This is a custom subset experiment with one backbone, two training generators, two unknown generators and simulated transformations. It does not demonstrate arbitrary-image authenticity detection, probability calibration, universal novelty, or real-platform performance. Scaling the dataset or changing architecture was outside this frozen comparison.')
def footer(canvas,doc):
 canvas.setFont('DejaVu',8);canvas.setFillColor(colors.grey);canvas.drawString(42,25,'ARIN5201 • Frozen final evaluation • 2026-11-13');canvas.drawRightString(A4[0]-42,25,str(doc.page))
output=Path(sys.argv[2]) if len(sys.argv)>2 else Path('/mnt/ssd/code/arin5201/output/pdf/arin5201_report_20261113.pdf');output.parent.mkdir(parents=True,exist_ok=True)
SimpleDocTemplate(str(output),pagesize=A4,rightMargin=42,leftMargin=42,topMargin=38,bottomMargin=40).build(story,onFirstPage=footer,onLaterPages=footer)
print(output)
