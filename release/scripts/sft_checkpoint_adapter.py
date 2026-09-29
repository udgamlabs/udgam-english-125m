"""Read/export the owned finite SFT format without changing completion scoring.

This adapter does not train, allocate provider resources or authorize spending.
The selected complete SFT pass and its original owned main parent are verified.
"""
from __future__ import annotations
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import train_sft as sft
import train_dense as base
import train_main as main_training

FORMAT = sft.FORMAT
MODEL_STAGE = 'supervised_instruction_tuning'


def tensor_hash(tensors):
    return base.model_hash(SimpleNamespace(state_dict=lambda:tensors))


def read(path,pilot_digest,torch,*,sft_data,main_parent,main_stage,pilot_data,tiny=False):
    if any(value is None for value in (sft_data,main_parent,main_stage,pilot_data)):
        raise ValueError('SFT evaluation/export requires SFT data, owned main parent, current main stage and owned pilot data')
    payload=sft.inspect_checkpoint(Path(path),torch)
    state=payload['state'];data=sft.FiniteExamples(sft_data,tiny=tiny)
    saved_plan=state['plan'];plan=sft.plan(data,saved_plan['batch_size'],saved_plan['seed'])
    if saved_plan!=plan:raise ValueError('SFT immutable finite plan/data mismatch')
    if state['step']!=plan['steps'] or state['cursor']!=data.count:
        raise ValueError('Only a completed single finite SFT pass is eligible for this export/evaluation adapter')
    if state['counts']!=data.counts_at(data.count):raise ValueError('SFT finite processed/supervised cursor mismatch')
    if state['lineage']['pilot_manifest_sha256']!=pilot_digest:raise ValueError('SFT owned pilot data identity mismatch')
    parent,lineage=sft.validate_parent(main_parent,state['lineage']['main_checkpoint_sha256'],pilot_data,main_stage,data,torch,tiny=tiny)
    if parent['model_config']!=payload['model_config']:raise ValueError('SFT model configuration differs from its owned parent')
    del parent
    stable=lambda value:{k:v for k,v in value.items() if k!='main_last_validated_manifest_sha256'}
    if stable(lineage)!=stable(state['lineage']):raise ValueError('SFT original parent lineage mismatch')
    if any(value.dtype!=torch.float32 for value in payload['model'].values()):raise ValueError('SFT exact FP32 master tensors required')
    history=state['loss_history']
    if len(history)!=plan['steps']:raise ValueError('SFT finite step history is incomplete')
    for index,item in enumerate(history):
        begin=index*plan['batch_size'];end=min(begin+plan['batch_size'],data.count)
        a,b=data.counts_at(begin),data.counts_at(end)
        if (item['step']!=index+1 or item['examples']!=end-begin or
            item['processed_positions']!=b['processed_positions']-a['processed_positions'] or
            item['supervised_targets']!=b['supervised_targets']-a['supervised_targets'] or
            item['lr']!=sft.learning_rate(index,plan)):
            raise ValueError('SFT finite step history/cursor differs')
        base.finite_number(item['loss_nats_per_supervised_target'],positive=False)
    for group in payload['optimizer']['param_groups']:
        if (tuple(group['betas'])!=tuple(plan['betas']) or group['weight_decay']!=plan['weight_decay'] or
            group['eps']!=plan['eps'] or group['foreach']!=plan['foreach'] or group['fused']!=plan['fused'] or
            group['lr']!=sft.learning_rate(plan['steps']-1,plan)):
            raise ValueError('SFT completed optimizer settings differ')
    verification={'sft_manifest_sha256':data.digest,'tokenizer_sha256':data.manifest['hashes']['tokenizer.json'],
        'main_parent_sha256':state['lineage']['main_checkpoint_sha256'],
        'main_stage_manifest_sha256':lineage['main_last_validated_manifest_sha256'],
        'model_tensor_sha256':tensor_hash(payload['model']), 'sft_pass_complete':True,
        'source_manifest_sha256':data.manifest['source']['receipt_sha256']}
    return payload,verification


def lineage(payload):
    state=payload['state'];parent=state['lineage'];counts=state['counts']
    pretraining_targets=parent['pilot_target_positions']+parent['main_target_positions']
    return {'phase':'sft','step':state['step'],'initial_weights_sha256':parent['initial_weights_sha256'],
        'main_checkpoint_sha256':parent['main_checkpoint_sha256'],'pilot_checkpoint_sha256':parent['pilot_checkpoint_sha256'],
        'pilot_target_positions':parent['pilot_target_positions'],'main_target_positions':parent['main_target_positions'],
        'pretraining_target_positions':pretraining_targets,'sft_processed_positions':counts['processed_positions'],
        'sft_supervised_targets':counts['supervised_targets'],
        'token_offset':pretraining_targets+counts['supervised_targets'],
        'token_offset_policy':'pretraining targets plus supervised SFT targets; masked SFT prompt positions excluded',
        'sft_pass_complete':state['step']==state['plan']['steps'] and state['cursor']==state['plan']['examples'],
        'sft_manifest_sha256':state['plan']['source_manifest_sha256'],'main_horizon_complete':parent['main_horizon_complete'],
        'initial_sft_weights_sha256':state['initial_sft_weights_sha256'],'sft_runtime_identity':state['runtime_identity']}


def export(model,pilot_data,sft_data,destination,checkpoint_hash,torch,*,verified,lineage_record):
    # Import lazily so the shared evaluator can dispatch to this adapter.
    import evaluate_dense as evaluation
    if verified.get('sft_pass_complete') is not True or lineage_record.get('sft_pass_complete') is not True:
        raise ValueError('A completed verified SFT pass is required')
    sft_data=Path(sft_data);destination=Path(destination)
    current=sft.FiniteExamples(sft_data,tiny=model.config.vocab_size<=1024)
    if current.digest!=verified['sft_manifest_sha256'] or base.sha256(Path(pilot_data)/'tokenizer.json')!=verified['tokenizer_sha256']:
        raise ValueError('SFT export data/tokenizer lineage changed')
    if base.model_hash(model)!=verified['model_tensor_sha256']:
        raise ValueError('SFT export tensors differ from the verified owned checkpoint')
    result=evaluation.export_weights(model,Path(pilot_data),destination,checkpoint_hash,torch,
        lineage=lineage_record,model_stage=MODEL_STAGE)
    notice_dir=destination/'attribution/sft';shutil.copytree(sft_data/'source',notice_dir)
    shutil.copyfile(sft_data/'manifest.json',notice_dir/'prepared-manifest.json')
    result.pop('inference_artifact_bytes',None)
    result['sft_source_notices']='attribution/sft'
    result['sft_manifest_sha256']=current.digest
    result['verified_sft_checkpoint_tensors_sha256']=verified['model_tensor_sha256']
    result['sft_pass_complete']=True
    result['files']={str(p.relative_to(destination)):{'sha256':base.sha256(p),'bytes':p.stat().st_size}
        for p in destination.rglob('*') if p.is_file() and p!=destination/'export-manifest.json'}
    result['inference_artifact_bytes_before_this_manifest']=sum(info['bytes'] for info in result['files'].values())
    base.write_json(destination/'export-manifest.json',result)
    result['inference_artifact_bytes']=sum(p.stat().st_size for p in destination.rglob('*') if p.is_file())
    return result


def main():
    """CPU-only export; no scoring, model generation, training or cloud calls."""
    import argparse
    import sys
    import types
    import evaluate_dense as evaluation
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','main-parent','main-stage','pilot-data','sft-data','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--tiny-cpu-test',action='store_true')
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Use a fresh immutable export directory')
    # The export constructs tensors only on CPU. Suppress upstream automatic
    # kernel discovery even on a computer that happens to have a CUDA device.
    blocker=types.ModuleType('kernels')
    blocker.has_kernel=lambda *args,**kwargs:False
    def offline(*args,**kwargs):raise RuntimeError('CPU export never downloads kernels')
    blocker.get_kernel=offline;previous=sys.modules.get('kernels');sys.modules['kernels']=blocker
    try:torch,GPT,GPTConfig,*_=base.runtime(tiny=True,attention='sdpa',optimizer_compile=False)
    finally:
        if previous is None:sys.modules.pop('kernels',None)
        else:sys.modules['kernels']=previous
    _,pilot_digest=base.verify_prepared(a.pilot_data,production=not a.tiny_cpu_test)
    payload,verified=read(a.checkpoint,pilot_digest,torch,sft_data=a.sft_data,main_parent=a.main_parent,
        main_stage=a.main_stage,pilot_data=a.pilot_data,tiny=a.tiny_cpu_test)
    model=evaluation.model_from_payload(payload,GPT,GPTConfig,torch,'cpu')
    record=lineage(payload);del payload
    result=export(model,a.pilot_data,a.sft_data,a.output,base.sha256(a.checkpoint),torch,
                  verified=verified,lineage_record=record)
    print(json.dumps({'model_stage':result['model_stage'],'parameters':result['total_parameters'],
        'inference_artifact_bytes':result['inference_artifact_bytes'],'optimizer_included':False}))


if __name__=='__main__':main()
