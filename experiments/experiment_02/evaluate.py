"""Evaluate experiment 2's best model on October-December and print the results.

This is the evaluation code used on 2026-11-08. Only this description and the
project import path were added when saving the original in-memory command.
It does not train the model or write result files.

Run from the project root with the training environment activated:
    python -B experiments/experiment_02/evaluate.py
"""

import ast
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch

sys.path.insert(0, str(Path.cwd()))
from deeplob import DeepLOB

# Reuse the existing Dataset and metric functions in memory, without importing plotting.
needed={'DailyWindowDataset','load_daily_dataset','classification_metrics','run_epoch'}
source=ast.parse(Path('train_deeplob.py').read_text())
functions=ast.Module(body=[node for node in source.body if isinstance(node,(ast.ClassDef,ast.FunctionDef)) and node.name in needed],type_ignores=[])
exec(compile(functions,'train_deeplob.py','exec'),globals())

torch.set_num_threads(1)
assert torch.backends.mps.is_available(), 'Apple GPU must be available for this evaluation.'
device=torch.device('mps')
path=Path('experiments/experiment_02/best_model.pt')
original_digest=hashlib.sha256(path.read_bytes()).hexdigest()
checkpoint=torch.load(path,map_location='cpu',weights_only=True)
config=checkpoint['config']
model=DeepLOB(n_levels=config['n_levels'],batch_norm=config.get('batch_norm',False),
              dropout=config.get('dropout',0.0),initialization=config.get('initialization','pytorch')).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
criterion=torch.nn.CrossEntropyLoss()
print(f"Checkpoint epoch={checkpoint['epoch']}; device={device}; batch_size={config['batch_size']}; evaluation only; no files written.",flush=True)

first_position=max(config['window_length']-1,config['horizon'])
training_counts=np.zeros(3,dtype=np.int64)
for date in config['data_dates']['train']:
    with np.load(Path(config['label_dir'])/f'{date}.npz',allow_pickle=False) as arrays:
        labels=arrays['labels']
    targets=labels[first_position:len(labels)-config['horizon']]
    training_counts+=np.bincount(targets,minlength=3)
majority_class=int(training_counts.argmax())
print('Training class counts:',training_counts.tolist(),'; majority class:',majority_class,flush=True)


def summary(metrics, *, name, days, seconds):
    return {'period':name,'days':days,'samples':int(metrics['support'].sum()),
            'loss':float(metrics['loss']),'accuracy':metrics['accuracy'],'macro_f1':metrics['macro_f1'],
            'precision':metrics['precision'].tolist(),'recall':metrics['recall'].tolist(),
            'f1':metrics['f1'].tolist(),'support':metrics['support'].tolist(),
            'predicted_counts':metrics['confusion'].sum(axis=0).tolist(),
            'confusion':metrics['confusion'].tolist(),'seconds':seconds}


combined=np.zeros((3,3),dtype=np.int64)
weighted_loss=0.0
monthly=[]
started=perf_counter()
for month in ('201710','201711','201712'):
    dates=[date for date in config['data_dates']['test'] if date.startswith(month)]
    dataset=load_daily_dataset(config['feature_dir'],config['label_dir'],dates,
                               window_length=config['window_length'],horizon=config['horizon'])
    loader=torch.utils.data.DataLoader(dataset,batch_size=config['batch_size'],shuffle=False,
                                      num_workers=0,drop_last=False)
    print(f"Starting {month}: {len(dates)} trading days; {len(dataset):,} samples; {len(loader):,} batches.",flush=True)
    month_started=perf_counter()
    metrics=run_epoch(model,loader,criterion,device,log_every=5000)
    assert int(metrics['support'].sum())==len(dataset)
    result=summary(metrics,name=month,days=len(dates),seconds=perf_counter()-month_started)
    monthly.append(result)
    print('MONTH_RESULT '+json.dumps(result),flush=True)
    combined+=metrics['confusion']
    weighted_loss+=metrics['loss']*len(dataset)
    del dataset,loader

assert int(combined.sum())==config['sample_counts']['test']
metrics=classification_metrics(combined)
metrics['loss']=weighted_loss/int(combined.sum())
overall=summary(metrics,name='201710-201712',days=len(config['data_dates']['test']),seconds=perf_counter()-started)
baseline_confusion=np.zeros((3,3),dtype=np.int64)
baseline_confusion[:,majority_class]=metrics['support']
baseline=classification_metrics(baseline_confusion)
print('TEST_RESULT '+json.dumps(overall),flush=True)
print('MAJORITY_BASELINE '+json.dumps({'class':majority_class,'accuracy':baseline['accuracy'],
                                      'macro_f1':baseline['macro_f1']}),flush=True)
assert hashlib.sha256(path.read_bytes()).hexdigest()==original_digest
print('Evaluation complete. Checkpoint unchanged; no output files created.',flush=True)
