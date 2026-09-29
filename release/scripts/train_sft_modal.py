#!/usr/bin/env python3
"""One finite owned SFT pass under a genuine Modal credit-only contract.

Training loop is ported verbatim from the reviewed GPU entry except its Modal
identity binding. Model, masking, optimizer, schedule and checkpoint format use
unchanged helpers. This file performs no provider operation or automatic retry.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import random
import shutil
import time

import train_dense as base
import train_sft as core
import train_sft_gpu as reviewed
import modal_final_contract as boundary
from train_sft_gpu import (build_model, ForwardPrecision, optimizer_for, compatibility_smoke,
    verify_gpu_runtime, require_time, require_completed_main)

FORMAT=core.FORMAT
PROFILE='modal-conversation-v1'


def source_hashes():
    return {**core.code_hashes(), **{n:base.sha256(Path(__file__).with_name(n)) for n in
        ('train_sft_gpu.py','train_sft_modal.py','modal_final_contract.py')}}


def parser():
    p=reviewed.parser()
    p.description=__doc__
    p.add_argument('--execution-contract',type=Path)
    p.add_argument('--package-manifest',type=Path)
    p.add_argument('--main-result',type=Path)
    return p


def validate_options(args):
    # Reuse all original numerical and finite-supply gates. Only the old
    # provider-evidence presence check is replaced, without forging its fields.
    if args.guard_evidence is not None:raise ValueError('Runpod evidence is not valid for Modal SFT')
    if not args.tiny_cpu_test and (args.execution_contract is None or args.package_manifest is None):
        raise ValueError('Explicit Modal execution and sealed input package required')
    copied=boundary.proxy(args,guard_evidence=args.execution_contract)
    reviewed.validate_options(copied)
    if args.export_reserve_seconds!=600:raise ValueError('Modal SFT backup reserve is exactly600 seconds')
    if not args.tiny_cpu_test and (args.batch_size!=4 or args.save_every!=200):
        raise ValueError('Production finite SFT uses the reviewed batch4/checkpoint200 plan')


def validate_execution(args):
    evidence,stop=boundary.validate(args.execution_contract,profile=PROFILE,source_hashes=source_hashes())
    if evidence['approved_jobs']!=['sft-finite-pass']:
        raise ValueError('Exactly one declared finite SFT pass required')
    if base.sha256(args.package_manifest)!=evidence['package_manifest_sha256']:
        raise ValueError('Sealed SFT input package changed')
    result_path=args.main_result or args.main_checkpoint.parent/'result.json'
    paths={'main_checkpoint':args.main_checkpoint,'main_result':result_path,
        'main_stage_manifest':args.main_stage/'manifest.json','pilot_manifest':args.pilot_data/'manifest.json',
        'sft_manifest':args.sft_data/'manifest.json'}
    if args.resume_sft:paths['resume_sft']=args.resume_sft
    boundary.bind_artifacts(evidence,paths)
    if args.expected_main_sha256!=evidence['artifacts']['main_checkpoint']:
        raise ValueError('Explicit main checkpoint identity differs')
    result=boundary.read(result_path)
    if (result.get('main_horizon_complete') is not True or result.get('candidate_valid') is not True or
        result.get('checkpoint_reload_verified') is not True or result.get('step')!=61035 or
        result.get('horizon_steps')!=61035 or result.get('checkpoint_sha256')!=args.expected_main_sha256):
        raise ValueError('Fully completed61035-step owned main endpoint required')
    return evidence,stop


def runtime_identity(torch,np,backend,device):
    value=reviewed.runtime_identity(torch,np,backend,device)
    value.update(entrypoint_sha256=base.sha256(Path(__file__)),modal_source_hashes=source_hashes())
    return value


def save_with_recovery(path,model,optimizer,state,torch):
    """Independent regular prior pair; safe for nlink=1 transfer after a crash."""
    path=Path(path);side=path.with_suffix('.sha256.json')
    recovery=path.with_name('checkpoint-recovery.pt');receipt_path=recovery.with_suffix('.sha256.json')
    if path.exists():
        if any(p.is_symlink() or not p.is_file() for p in (path,side)):
            raise ValueError('Invalid previous owned checkpoint pair')
        receipt=boundary.read(side)
        if receipt.get('format')!=FORMAT or receipt.get('sha256')!=base.sha256(path):
            raise ValueError('Previous SFT checkpoint pair unverified')
        temp=path.parent/'.checkpoint-recovery.pt.pending'
        temp_side=path.parent/'.checkpoint-recovery.sha256.json.pending'
        if any(p.exists() or p.is_symlink() for p in (temp,temp_side)):
            raise FileExistsError('Unexpected recovery temporary')
        try:
            shutil.copyfile(path,temp,follow_symlinks=False)
            with temp.open('rb') as handle:os.fsync(handle.fileno())
            with temp_side.open('x') as handle:
                handle.write(boundary.canonical(receipt)+'\n');handle.flush();os.fsync(handle.fileno())
            os.replace(temp,recovery);os.replace(temp_side,receipt_path)
            descriptor=os.open(path.parent,os.O_RDONLY)
            try:os.fsync(descriptor)
            finally:os.close(descriptor)
        finally:temp.unlink(missing_ok=True);temp_side.unlink(missing_ok=True)
    core.save(path,model,optimizer,state,torch)
    for item in (path,side):
        with item.open('rb') as handle:os.fsync(handle.fileno())
    descriptor=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(descriptor)
    finally:os.close(descriptor)
    return recovery if recovery.exists() else None


def run(args):
    validate_options(args)
    evidence=None;stop_at=float('inf')
    if not args.tiny_cpu_test:
        evidence,stop_at=validate_execution(args)
    # A backward wall-clock jump cannot extend the initial work window.
    monotonic_deadline=time.perf_counter()+max(0,stop_at-time.time())
    if base.sha256(args.sft_data/'manifest.json')!=args.expected_sft_manifest_sha256:
        raise ValueError('Prepared SFT manifest differs from the explicitly reviewed bundle')
    data=core.FiniteExamples(args.sft_data,tiny=args.tiny_cpu_test)
    config=core.plan(data,args.batch_size,args.seed)
    torch,GPT,GPTConfig,_,_,backend=base.runtime(tiny=args.tiny_cpu_test,
        attention='sdpa' if args.tiny_cpu_test else 'fa3',optimizer_compile=False)
    import numpy as np
    device='cpu' if args.tiny_cpu_test else 'cuda'
    if not args.tiny_cpu_test:verify_gpu_runtime(torch,np.__version__,backend)
    parent,lineage=core.validate_parent(args.main_checkpoint,args.expected_main_sha256,args.pilot_data,
        args.main_stage,data,torch,tiny=args.tiny_cpu_test)
    require_completed_main(parent,lineage)
    identity=runtime_identity(torch,np,backend,device)
    if not args.tiny_cpu_test:
        directory=args.output.parent
        while not directory.exists():directory=directory.parent
        if shutil.disk_usage(directory).free<8*2**30:raise ValueError('SFT needs 8GiB free for checkpoints and backup reserve')
    require_time(stop_at,monotonic_deadline)
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    smoke=compatibility_smoke(torch,GPT,GPTConfig,device,tiny=args.tiny_cpu_test)
    require_time(stop_at,monotonic_deadline)
    model=build_model(parent,GPT,GPTConfig,torch,device);del parent
    optimizer=optimizer_for(model,config,torch)
    state={'step':0,'cursor':0,'counts':data.counts_at(0),'plan':config,'lineage':lineage,'runtime_identity':identity,
        'optimizer_transition':'fresh_AdamW_once_after_owned_main; no main optimizer moments reused',
        'initial_sft_weights_sha256':base.model_hash(model),'loss_history':[],'cumulative_compute_seconds':0.,'sessions':[]}
    if args.resume_sft:
        loaded=core.inspect_checkpoint(args.resume_sft,torch)
        state=core.restore(loaded,model,optimizer,data,config,lineage,identity,torch);del loaded
    if state['cursor']==data.count:raise ValueError('The finite SFT pass is already complete; repetition is forbidden')
    require_time(stop_at,monotonic_deadline)
    args.output.mkdir(parents=True)
    guard_binding=None if evidence is None else {'profile':PROFILE,
        'execution_id':evidence['execution_id'],'app_id':evidence['app_id'],
        'function_call_id':evidence['function_call_id'],'task_id':evidence['task_id'],
        'execution_contract_sha256':boundary.digest(evidence),
        'function_started_at':evidence['function_started_at'],
        'absolute_deadline_at':evidence['absolute_deadline_at'],'work_cutoff_unix':stop_at}
    contract={'format':FORMAT,'plan':config,'lineage':lineage,'runtime_identity':identity,
        'adapter_code_hashes':core.code_hashes(),'guard':guard_binding,'compatibility_smoke':smoke,
        'session_compute_seconds':args.session_compute_seconds,'export_reserve_seconds':args.export_reserve_seconds,
        'save_every_updates':args.save_every,'processed_token_cap':5_000_000,'passes':1,
        'wrapping':False,'truncation':False,'entrypoint':'train_sft_modal.py','modal_source_hashes':source_hashes(),
        'gpu_execution':not args.tiny_cpu_test,'heldout_or_confirmatory_scoring':False}
    base.write_json(args.output/'run-contract.json',contract)
    checkpoint=args.output/'checkpoint.pt'
    session={'start_step':state['step'],'start_cursor':state['cursor'],'compute_seconds':smoke['compute_seconds'],
        'synthetic_smoke_seconds':smoke['compute_seconds'],'optimizer_compute_seconds':0.,'guard':guard_binding,
        'save_every_updates':args.save_every,'periodic_checkpoint_count':0}
    state['cumulative_compute_seconds']+=smoke['compute_seconds']
    save_with_recovery(checkpoint,model,optimizer,state,torch)
    execute=ForwardPrecision(model,torch,device)
    stop_reason='finite_pass_complete';largest_step_seconds=0.
    if device=='cuda':torch.cuda.reset_peak_memory_stats()
    while state['step']<config['steps']:
        completed=state['step']-session['start_step']
        if args.stop_after_session_steps is not None and completed>=args.stop_after_session_steps:
            stop_reason='synthetic_interruption';break
        # Synchronous CUDA work cannot be preempted by a Python timer. Avoid
        # starting near either bound and reserve twice the largest observed
        # update (at least five seconds) before scheduling another update.
        margin=max(.001 if args.tiny_cpu_test else 5.,2*largest_step_seconds)
        if session['compute_seconds']+margin>=args.session_compute_seconds:
            stop_reason='session_compute_limit';break
        if time.time()+margin>=stop_at or time.perf_counter()+margin>=monotonic_deadline:
            stop_reason='provider_export_reserve';break
        started=time.perf_counter()
        next_cursor,item=core.train_step(execute,optimizer,data,state['cursor'],state['step'],config,torch,device)
        if device=='cuda':torch.cuda.synchronize()
        elapsed=time.perf_counter()-started;largest_step_seconds=max(largest_step_seconds,elapsed)
        state['step']+=1;state['cursor']=next_cursor;state['counts']=data.counts_at(next_cursor)
        state['loss_history'].append(item);state['cumulative_compute_seconds']+=elapsed
        session['compute_seconds']+=elapsed;session['optimizer_compute_seconds']+=elapsed
        if (completed+1)%args.save_every==0 and state['step']<config['steps']:
            save_with_recovery(checkpoint,model,optimizer,state,torch);session['periodic_checkpoint_count']+=1
    session.update(end_step=state['step'],end_cursor=state['cursor'],stop_reason=stop_reason)
    initial_counts=data.counts_at(session['start_cursor'])
    session['processed_positions']=state['counts']['processed_positions']-initial_counts['processed_positions']
    session['supervised_targets']=state['counts']['supervised_targets']-initial_counts['supervised_targets']
    session['padded_compute_positions']=sum(item['padded_compute_positions'] for item in state['loss_history'][session['start_step']:])
    session['supervised_targets_per_optimizer_second']=(session['supervised_targets']/session['optimizer_compute_seconds']
        if session['optimizer_compute_seconds'] else None)
    state['sessions'].append(session);recovery=save_with_recovery(checkpoint,model,optimizer,state,torch)
    before=base.model_hash(model);loaded=core.inspect_checkpoint(checkpoint,torch)
    core.restore(loaded,model,optimizer,data,config,lineage,identity,torch);del loaded
    if before!=base.model_hash(model):raise AssertionError('SFT checkpoint cold tensor reload mismatch')
    result={'format':FORMAT,'status':'finite_sft_complete' if state['cursor']==data.count else 'finite_sft_paused',
        'validation_scope':'tiny_cpu_only' if args.tiny_cpu_test else 'owned_model_cuda_sft',
        'step':state['step'],'cursor':state['cursor'],'finite_pass_complete':state['cursor']==data.count,
        'counts':state['counts'],'session':session,'compatibility_smoke':smoke,'guard':guard_binding,
        'checkpoint_sha256':base.sha256(checkpoint),'checkpoint_bytes':checkpoint.stat().st_size,
        'previous_checkpoint_recovery_path':str(recovery) if recovery else None,
        'checkpoint_reload_verified':True,'final_weights_sha256':before,
        'initial_weights_sha256':lineage['initial_weights_sha256'],'main_checkpoint_sha256':lineage['main_checkpoint_sha256'],
        'sft_manifest_sha256':data.digest,'optimizer_transition':state['optimizer_transition'],
        'heldout_or_confirmatory_scoring':False,'execution_profile':PROFILE,
        'modal_source_hashes':source_hashes(),'modal_execution_contract_sha256':boundary.digest(evidence) if evidence else None,
        'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated() if device=='cuda' else None,
        'cuda_peak_reserved_bytes':torch.cuda.max_memory_reserved() if device=='cuda' else None}
    base.write_json(args.output/'result.json',result)
    return result


if __name__=='__main__':print(json.dumps(run(parser().parse_args()),indent=2))
