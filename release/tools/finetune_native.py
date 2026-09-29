#!/usr/bin/env python3
"""Offline finite fine-tuning of a verified owned native export.

No provider calls, original experiment continuation, dataset downloads or exact
resume. The input is never written. Run in a fresh process; install the native
package's requirements first. A completed pass is not a model-quality claim.
"""
from __future__ import annotations
import argparse
from contextlib import nullcontext
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time

INFER_SHA='6bc6fa6ef1973c69f6092689f1e6d023e80b7da7c6ee23c4c00e6f1e7832afd7'
MAX_DATA_BYTES=32*1024**2
MAX_LINE_BYTES=1024**2
MAX_EXAMPLES=10000
MAX_POSITIONS=5_000_000
FORMAT='udgam-local-native-finetune-v1'


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def regular(path):
    path=Path(path).absolute()
    if any(p.is_symlink() for p in (path,*path.parents)) or not path.is_file() or path.stat().st_nlink!=1:
        raise ValueError('Only regular non-linked files are accepted')
    return path


def write_json(path,value):
    with Path(path).open('x',encoding='utf-8') as f:
        json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())


def inference():
    directory=Path(__file__).resolve().parent
    path=directory/'infer_dense.py' if (directory/'infer_dense.py').is_file() else directory/'infer.py'
    if digest(regular(path))!=INFER_SHA:raise ValueError('Reviewed offline inference helper is required beside this script')
    spec=importlib.util.spec_from_file_location('_owned_native_inference',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def strict_object(pairs):
    result={}
    for key,value in pairs:
        if key in result:raise ValueError('Duplicate JSON field')
        result[key]=value
    return result


def examples(path,tokenizer,context):
    """Read and validate the entire finite source before any optimizer update."""
    path=regular(path)
    if not 0<path.stat().st_size<=MAX_DATA_BYTES:raise ValueError('JSONL must be nonempty and at most32MiB')
    before=digest(path);rows=[];positions=targets=0
    with path.open('rb') as stream:
        while True:
            raw=stream.readline(MAX_LINE_BYTES+1)
            if not raw:break
            if len(raw)>MAX_LINE_BYTES:raise ValueError('JSONL line exceeds1MiB')
            if not raw.strip():raise ValueError('Blank JSONL examples are not accepted')
            try:row=json.loads(raw,object_pairs_hook=strict_object)
            except (ValueError,UnicodeError):raise ValueError('Invalid JSONL example') from None
            if (not isinstance(row,dict) or set(row)-{'prompt','response','system'} or
                not {'prompt','response'}<=set(row) or any(not isinstance(v,str) for v in row.values()) or
                not row['prompt'].strip() or not row['response'].strip()):
                raise ValueError('Each row needs nonempty prompt/response strings and optional system only')
            prefix=tokenizer.conversation([{'role':'user','content':row['prompt']}],row.get('system',''))
            response=tokenizer.encode(row['response'])+[tokenizer.specials['<|assistant_end|>']]
            ids=prefix+response
            if len(ids)>context:raise ValueError('An example exceeds the model context; truncation is forbidden')
            rows.append((ids,[False]*len(prefix)+[True]*len(response)))
            positions+=len(ids);targets+=len(response)
            if len(rows)>MAX_EXAMPLES or positions>MAX_POSITIONS:raise ValueError('Finite dataset limit exceeded')
    if not rows or digest(path)!=before:raise ValueError('JSONL is empty or changed during preparation')
    return rows,{'sha256':before,'bytes':path.stat().st_size,'examples':len(rows),'processed_positions':positions,
                 'supervised_targets':targets,'raw_text_copied':False,'order':'original_JSONL_order'}


def batch(rows,begin,count,torch,device):
    if not 0<=begin<len(rows) or not 0<count<=len(rows)-begin:raise ValueError('Finite batch exhausted; no wrapping')
    selected=rows[begin:begin+count];width=max(len(ids)-1 for ids,_ in selected)
    x=torch.zeros((count,width),dtype=torch.long,device=device)
    y=torch.full_like(x,-1);targets=positions=0
    for i,(ids,mask) in enumerate(selected):
        length=len(ids)-1;x[i,:length]=torch.tensor(ids[:-1],device=device)
        t=torch.tensor(ids[1:],device=device);m=torch.tensor(mask[1:],dtype=torch.bool,device=device)
        y[i,:length]=torch.where(m,t,-1);targets+=sum(mask[1:]);positions+=len(ids)
    if not targets:raise ValueError('No supervised assistant targets')
    return x,y,{'examples':count,'processed_positions':positions,'supervised_targets':targets,'padded_input_positions':x.numel()}


def tensor_hash(model,torch):
    h=hashlib.sha256()
    for name,tensor in sorted(model.state_dict().items()):
        value=tensor.detach().cpu().contiguous()
        h.update(name.encode());h.update(str(value.dtype).encode());h.update(str(tuple(value.shape)).encode())
        h.update(value.view(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def rate(step,steps,peak):
    warm=min(max(1,math.ceil(.03*steps)),max(1,steps-1))
    if steps==1:return peak
    if step<warm:return peak*(step+1)/warm
    return peak*.1+peak*.9*.5*(1+math.cos(math.pi*(step-warm+1)/(steps-warm)))


def validate_args(a):
    for name,low,high in [('max_steps',1,100000),('batch_size',1,32),('cpu_threads',1,16),('seed',0,2**32-1)]:
        value=getattr(a,name)
        if type(value) is not int or not low<=value<=high:raise ValueError('Invalid bounded '+name)
    if type(a.max_seconds) not in (int,float) or not math.isfinite(a.max_seconds) or not 0<a.max_seconds<=86400:raise ValueError('Finite max-seconds in(0,86400] required')
    if not math.isfinite(a.learning_rate) or not 0<a.learning_rate<=.01:raise ValueError('Learning rate outside(0,.01]')
    if len(a.expected_parent_sha256)!=64 or any(c not in '0123456789abcdef' for c in a.expected_parent_sha256):raise ValueError('Exact parent export manifest SHA256 required')
    parent=Path(a.model).absolute();out=Path(a.output).absolute();data=Path(a.data).absolute()
    if any(p.is_symlink() for p in (parent,out,*parent.parents,*out.parents)):raise ValueError('Symlink model/output paths are unsupported')
    if not parent.is_dir() or not out.parent.is_dir():raise ValueError('Existing model and output parent directories required')
    if out.exists():raise FileExistsError('Output must be a new directory')
    if parent==out or parent in out.parents or out in parent.parents or out==data or out in data.parents:raise ValueError('Output must be separate from model and JSONL input')
    return parent,out,data


def run(a):
    started=time.monotonic();parent,out,data=validate_args(a);deadline=started+a.max_seconds
    native=inference();manifest_path=regular(parent/'export-manifest.json')
    if digest(manifest_path)!=a.expected_parent_sha256:raise ValueError('Parent export manifest SHA differs')
    parent_manifest=native.verify_export(parent)
    for name in parent_manifest['files']:regular(native.artifact_path(parent,name))
    if parent_manifest['files'].get('infer.py',{}).get('sha256')!=INFER_SHA:raise ValueError('Parent uses a different inference implementation')
    import torch
    torch.set_num_threads(a.cpu_threads);random.seed(a.seed);torch.manual_seed(a.seed)
    model,tokenizer,torch,dtype,runtime=native.load(parent,a.device)
    rows,data_info=examples(data,tokenizer,model.config.sequence_len)
    parent_tensors=tensor_hash(model,torch)
    # Detached copies ensure no mapped input storage is ever a training target.
    copied={k:v.detach().clone().to(dtype=torch.float32) for k,v in model.state_dict().items()}
    model.load_state_dict(copied,strict=True,assign=True);del copied
    model.requires_grad_(True);model.train()
    device=next(model.parameters()).device.type
    optimizer=torch.optim.AdamW(model.parameters(),lr=a.learning_rate,betas=(.9,.95),eps=1e-8,weight_decay=.01,foreach=False,fused=False)
    planned=math.ceil(len(rows)/a.batch_size);history=[];cursor=0;totals={'examples':0,'processed_positions':0,'supervised_targets':0,'padded_input_positions':0}
    stop='finite_pass_complete'
    for step in range(planned):
        if step>=a.max_steps:stop='step_limit';break
        if time.monotonic()>=deadline:stop='time_limit';break
        x,y,counts=batch(rows,cursor,min(a.batch_size,len(rows)-cursor),torch,device)
        lr=rate(step,planned,a.learning_rate)
        for group in optimizer.param_groups:group['lr']=lr
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type='cuda',dtype=torch.bfloat16) if device=='cuda' else nullcontext():loss=model(x,y)
        if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite fine-tuning loss')
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
        if device=='cuda':torch.cuda.synchronize()
        cursor+=counts['examples']
        for key in totals:totals[key]+=counts[key]
        history.append({'step':step+1,'loss_nats_per_supervised_target':float(loss.detach()),'learning_rate':lr,**counts})
    if not history:raise TimeoutError('No optimizer update fit the requested time bound; no export written')
    complete=cursor==len(rows)
    if complete:stop='finite_pass_complete'
    for value in model.state_dict().values():
        if not bool(torch.isfinite(value).all()):raise FloatingPointError('Nonfinite adapted model tensor')
    final_tensors=tensor_hash(model,torch)
    # Hash-bound parent and input are checked again before publication.
    if digest(manifest_path)!=a.expected_parent_sha256 or native.verify_export(parent)!=parent_manifest or digest(data)!=data_info['sha256']:
        raise ValueError('Parent export or local data changed; no output published')
    if shutil.disk_usage(out.parent).free<sum(i['bytes'] for n,i in parent_manifest['files'].items() if n!='model.pt')+sum(p.numel()*4 for p in model.parameters())+1024**3:
        raise ValueError('Output requires full artifact space plus1GiB reserve')
    out.mkdir(exist_ok=False)
    record={'format':FORMAT,'status':'finite_pass_complete' if complete else 'finite_pass_incomplete','finite_pass_complete':complete,
        'stop_reason':stop,'updates':len(history),'planned_updates':planned,'cursor':cursor,'dataset':data_info,'consumed':totals,
        'max_steps':a.max_steps,'max_seconds':a.max_seconds,'batch_size':a.batch_size,'seed':a.seed,'learning_rate':a.learning_rate,
        'optimizer':{'name':'AdamW','betas':[.9,.95],'eps':1e-8,'weight_decay':.01,'foreach':False,'fused':False,'gradient_clip_norm':1.,'schedule':'3percent_warmup_then_cosine_to10percent'},
        'loss':'assistant_content_and_end_only','one_finite_pass':True,'wrapping':False,'truncation':False,'raw_data_copied':False,
        'parent_export_manifest_sha256':a.expected_parent_sha256,'parent_model_file_sha256':parent_manifest['files']['model.pt']['sha256'],
        'parent_tensor_sha256':parent_tensors,'adapted_tensor_sha256':final_tensors,'tokenizer_sha256':parent_manifest['files']['tokenizer.json']['sha256'],
        'source_sha256':digest(Path(__file__)),'inference_helper_sha256':INFER_SHA,'source_checkpoint_sha256':None,'source_checkpoint_kind':'no_checkpoint_created',
        'exact_resume_supported':False,'model_quality_assessed':False,'original_benchmark_results_apply':False,
        'runtime':{'torch':str(torch.__version__),'device':device,'parameter_dtype':'float32','attention':'offline_sdpa','compile':False},
        'seconds_before_export':time.monotonic()-started,'time_bound_scope':'checked before each optimizer update; loading, an in-flight update, final verification and export are not hard-interruptible',
        'loss_history':history}
    try:
        # Preserve all source/attribution/license assets; replace only the new
        # model and this tool's generated records, keeping prior records below.
        reserved={'model.pt','training-record.json','FINE_TUNING_NOTICE.md','finetune.py'}
        history_root=out/'provenance'/'parents'/a.expected_parent_sha256
        history_root.mkdir(parents=True)
        shutil.copyfile(manifest_path,history_root/'export-manifest.json')
        for name in parent_manifest['files']:
            source=native.artifact_path(parent,name)
            if name=='model.pt':continue
            target=(history_root/name if name in reserved else out/name)
            target.parent.mkdir(parents=True,exist_ok=True)
            with source.open('rb') as f,target.open('xb') as g:shutil.copyfileobj(f,g)
            if target.stat().st_size!=parent_manifest['files'][name]['bytes'] or digest(target)!=parent_manifest['files'][name]['sha256']:
                raise ValueError('Parent asset changed during copy; incomplete output retained')
        torch.save({'format':native.FORMAT,'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},'model_config':json.loads((out/'model-config.json').read_text()),'source_revision':native.REVISION,'source_checkpoint_sha256':None},out/'model.pt')
        record['new_model_file_sha256']=digest(out/'model.pt');write_json(out/'training-record.json',record)
        shutil.copyfile(Path(__file__),out/'finetune.py')
        (out/'FINE_TUNING_NOTICE.md').write_text('This is a new local fine-tune, not the original evaluated model. Consult training-record.json for finite completion or interruption. No quality or benchmark result is inherited. Source checkpoint: none was created; the model file and tensors have new explicit hashes. Exact training resume is not supported. Raw input JSONL is not copied. You remain responsible for your training-data rights and resulting model terms; copied parent/code licenses and source notices remain applicable and do not newly license your data. The time limit is checked before each update; loading, an in-flight update and final verification/export may overrun it.\n',encoding='utf-8')
        files={p.relative_to(out).as_posix():{'sha256':digest(p),'bytes':p.stat().st_size} for p in out.rglob('*') if p.is_file()}
        result={'format':native.FORMAT,'source_checkpoint_sha256':None,'source_checkpoint_kind':'no_checkpoint_created',
            'total_parameters':sum(p.numel() for p in model.parameters()),'parameter_tensor_bytes':sum(p.numel()*p.element_size() for p in model.parameters()),
            'dtype_policy':'float32 adapted parameters','files':files,'optimizer_included':False,'hf_compatible':False,
            'model_stage':'local_finetune_complete' if complete else 'local_finetune_incomplete','finite_pass_complete':complete,
            'model_quality_assessed':False,'original_benchmark_results_apply':False,'new_model_file_sha256':record['new_model_file_sha256'],
            'lineage':{'phase':'local_native_finetune','parent_export_manifest_sha256':a.expected_parent_sha256,
                'parent_model_file_sha256':record['parent_model_file_sha256'],'parent_source_checkpoint_sha256':parent_manifest.get('source_checkpoint_sha256'),
                'parent_lineage':parent_manifest.get('lineage',{}),'adapted_tensor_sha256':final_tensors,'dataset_sha256':data_info['sha256'],
                'training_record_sha256':files['training-record.json']['sha256']},
            'license':'parent_and_code_terms_preserved; new_data_and_adaptation_rights_not_assessed',
            'license_scope':'FINE_TUNING_NOTICE.md; inherited LICENSE_SCOPE.md and attribution retained',
            'inference_artifact_bytes_before_this_manifest':sum(i['bytes'] for i in files.values()),'runtime_dependencies':'requirements.txt'}
        write_json(out/'export-manifest.json',result);native.verify_export(out)
    except BaseException:
        # This owned partial directory is retained; never call it complete or
        # overwrite it on a later run. Input and unrelated files are untouched.
        raise
    return {'status':record['status'],'finite_pass_complete':complete,'updates':len(history),'cursor':cursor,'source_checkpoint_sha256':None,
            'new_model_file_sha256':record['new_model_file_sha256'],'export_manifest_sha256':digest(out/'export-manifest.json'),'model_quality_assessed':False}


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True);p.add_argument('--expected-parent-sha256',required=True)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-steps',type=int,required=True);p.add_argument('--max-seconds',type=float,required=True)
    p.add_argument('--batch-size',type=int,default=4);p.add_argument('--learning-rate',type=float,default=1e-5)
    p.add_argument('--seed',type=int,default=20260924);p.add_argument('--cpu-threads',type=int,default=4)
    p.add_argument('--device',choices=('cpu','cuda'),default='cpu');return p


if __name__=='__main__':
    try:print(json.dumps(run(parser().parse_args()),sort_keys=True))
    except Exception as error:
        print(json.dumps({'status':'failed_no_quality_claim','error_type':type(error).__name__,'message':str(error)}),file=sys.stderr)
        raise SystemExit(1)
