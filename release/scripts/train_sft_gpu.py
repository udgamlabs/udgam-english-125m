#!/usr/bin/env python3
"""Separately guarded finite SFT entrypoint; no automatic resource operations.

Requires an explicitly hashed, fully completed owned main parent and finite
human-source bundle. The old CPU core/main/pilot entrypoints stay unchanged.
A tiny CPU mode exercises this entrypoint without claiming CUDA compatibility.
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import time

import train_dense as base
import train_sft as core

FORMAT = core.FORMAT
PROFILE = "conversation-v1"
MAX_COMPUTE_SECONDS = 900
MIN_EXPORT_RESERVE = 600
MAX_EXPORT_RESERVE = 1200
CHECKPOINT_EVERY = 200


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(",", ":"),allow_nan=False)


def validate_conversation_guard(path, *, now=None, export_reserve_seconds=600):
    """Verify public local-controller evidence; it does not itself stop a Pod."""
    now = time.time() if now is None else now
    base.finite_number(now)
    base.finite_number(export_reserve_seconds)
    if not MIN_EXPORT_RESERVE <= export_reserve_seconds <= MAX_EXPORT_RESERVE:
        raise ValueError("Conversation export reserve must be 600..1200 seconds")
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 256*1024:
        raise ValueError("Invalid conversation evidence file")
    value = json.loads(Path(path).read_text())
    intent = value["intent"]; reservation = value["budget_reservation"]; budget = value["budget_snapshot"]
    required_intent = {"version", "pod_name", "armed_at", "deadline_at", "max_seconds",
        "max_gpu_rate", "max_extra_rate", "max_dollars", "poll_seconds", "stop_reserve_seconds",
        "budget_reserve", "expected_gpu_count", "profile", "reservation_id", "budget_path"}
    if set(intent) != required_intent or value.get("profile") != "conversation-v1" or intent["profile"] != "conversation-v1":
        raise ValueError("Explicit conversation-v1 guard profile required")
    checksum = hashlib.sha256(canonical(intent).encode()).hexdigest()
    if value.get("guard_intent_sha256") != checksum or reservation.get("intent_sha256") != checksum:
        raise ValueError("Conversation guard intent/reservation hash mismatch")
    rid = value.get("reservation_id", "")
    if not isinstance(rid, str) or not re.fullmatch(r"[a-f0-9]{32}", rid) or intent["reservation_id"] != rid:
        raise ValueError("Conversation reservation identity mismatch")
    pod = value.get("pod_id")
    if not isinstance(pod, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", pod):
        raise ValueError("Missing bound provider pod")
    if (type(intent["version"]) is not int or intent["version"] != 1 or
        type(intent["expected_gpu_count"]) is not int or intent["expected_gpu_count"] != 1 or
        value.get("pod_name") != intent["pod_name"] or reservation.get("pod_name") != intent["pod_name"] or
        not isinstance(intent["pod_name"], str) or not re.fullmatch(r"udgam-conversation-[a-f0-9]{32}", intent["pod_name"])):
        raise ValueError("Conversation lifecycle pod identity mismatch")
    if not isinstance(intent["budget_path"], str) or not Path(intent["budget_path"]).is_absolute() or Path(intent["budget_path"]).name != "budget.json":
        raise ValueError("Conversation canonical budget identity missing")
    if (reservation.get("version") != 1 or reservation.get("profile") != "conversation-v1" or
        reservation.get("stage") != "conversation_training" or reservation.get("status") != "active" or
        reservation.get("pod_id") != pod or reservation.get("amount_usd") != 3 or
        not isinstance(reservation.get("state_path"), str) or not Path(reservation["state_path"]).is_absolute() or
        not re.fullmatch(r"conversation-guard-[A-Za-z0-9_-]+\.json", Path(reservation["state_path"]).name)):
        raise ValueError("Conversation budget reservation not active for this pod")
    observed = base.utc_timestamp(value["observed_at_utc"])
    if not 0 <= now - observed <= 120 or not 0 <= now - base.utc_timestamp(budget["observed_at_utc"]) <= 120:
        raise ValueError("Conversation billing/budget snapshot stale or in future")
    rate = base.finite_number(value["hourly_total_usd"])
    ceiling = base.finite_number(value["cash_ceiling_usd"])
    gpu_rate = base.finite_number(intent["max_gpu_rate"])
    extra_rate = base.finite_number(intent["max_extra_rate"], positive=False)
    seconds = base.finite_number(intent["max_seconds"])
    armed = base.finite_number(intent["armed_at"])
    deadline = base.utc_timestamp(value["provider_stop_deadline_utc"])
    if (ceiling != 3 or intent["max_dollars"] != 3 or not 60 < seconds <= 2700 or
        armed > now or abs(base.finite_number(intent["deadline_at"]) - armed - seconds) > .001 or
        abs(deadline - intent["deadline_at"]) > .001 or rate < gpu_rate + extra_rate):
        raise ValueError("Conversation lifecycle exceeds its declared time/rate/cash bounds")
    if intent["poll_seconds"] != 5 or intent["stop_reserve_seconds"] != 60 or intent["budget_reserve"] != .25:
        raise ValueError("Conversation controller safety reserves changed")
    guard = value["provider_stop_guard"]
    heartbeat = base.utc_timestamp(guard["heartbeat_at_utc"])
    if (guard.get("verified") is not True or guard.get("pod_id") != pod or
        base.utc_timestamp(guard["deadline_utc"]) != deadline or not 0 <= now - heartbeat <= 60):
        raise ValueError("Conversation provider guard identity/heartbeat mismatch")
    if (budget.get("experiment_ceiling_usd") != 99 or budget.get("stage") != "conversation_training" or
        budget.get("stage_ceiling_usd") != 8 or budget.get("auto_top_up_authorized") is not False):
        raise ValueError("Conversation project/stage budget authorization mismatch")
    for key in ("consumed_credit_usd", "unsettled_charges_usd", "stage_consumed_credit_usd", "new_cash_deposits_usd",
                "observed_prepaid_credit_usd", "reserved_total_usd", "reserved_stage_usd"):
        base.finite_number(budget[key], positive=False)
    if (budget["new_cash_deposits_usd"] > 99 or budget["reserved_total_usd"] != 3 or budget["reserved_stage_usd"] != 3 or
        budget["consumed_credit_usd"] + budget["unsettled_charges_usd"] + budget["reserved_total_usd"] > 99 + 1e-9 or
        budget["stage_consumed_credit_usd"] + budget["unsettled_charges_usd"] + budget["reserved_stage_usd"] > 8 + 1e-9):
        raise ValueError("Conversation reservation exceeds project/stage capacity")
    if budget["unsettled_charges_usd"] + budget["reserved_total_usd"] > budget["observed_prepaid_credit_usd"] + 1e-9:
        raise ValueError("Conversation reservation exceeds prepaid credit; no automatic top-up")
    cost = base.finite_number(value["conservative_elapsed_cost_usd"], positive=False)
    # The guard stops before the raw deadline or cash ceiling, whichever comes
    # first. Preserve export time BEFORE that earlier stop, including heartbeat
    # age and conservative monotonic elapsed accounting from the controller.
    elapsed = max(now - armed, cost * 3600 / rate + now - heartbeat)
    cash_remaining_seconds = (ceiling - intent["budget_reserve"]) * 3600 / rate - elapsed - intent["stop_reserve_seconds"]
    remaining = min(deadline - now - intent["stop_reserve_seconds"], seconds - elapsed - intent["stop_reserve_seconds"], cash_remaining_seconds)
    if remaining <= export_reserve_seconds + 30:
        raise ValueError("Insufficient conversation lifecycle time/cash for safe export")
    return value, now + remaining - export_reserve_seconds


def parser():
    p = core.parser()
    p.description = __doc__
    p.set_defaults(session_compute_seconds=MAX_COMPUTE_SECONDS)
    p.add_argument('--guard-evidence',type=Path)
    p.add_argument('--expected-sft-manifest-sha256',required=True)
    p.add_argument('--export-reserve-seconds',type=float,default=MIN_EXPORT_RESERVE)
    p.add_argument('--save-every',type=int,default=CHECKPOINT_EVERY)
    return p


def validate_options(args):
    for number in (args.session_compute_seconds,args.export_reserve_seconds):base.finite_number(number)
    if not MIN_EXPORT_RESERVE<=args.export_reserve_seconds<=MAX_EXPORT_RESERVE:
        raise ValueError('Conversation export reserve must be 600..1200 seconds')
    if not 0<args.session_compute_seconds<=(60 if args.tiny_cpu_test else MAX_COMPUTE_SECONDS):
        raise ValueError('SFT compute bound exceeded')
    if type(args.batch_size) is not int or not 1<=args.batch_size<=4:
        raise ValueError('Reviewed SFT batch size must be 1..4')
    if type(args.save_every) is not int or not 1<=args.save_every<=1000:
        raise ValueError('Invalid periodic SFT checkpoint cadence')
    if type(args.seed) is not int or not 0<=args.seed<2**32:raise ValueError('Invalid SFT seed')
    if not re.fullmatch(r'[a-f0-9]{64}',args.expected_sft_manifest_sha256):
        raise ValueError('Explicit prepared SFT manifest hash required')
    if args.stop_after_session_steps is not None and (not args.tiny_cpu_test or
            type(args.stop_after_session_steps) is not int or not 0<=args.stop_after_session_steps<=64):
        raise ValueError('Artificial interruption is allowed only for tiny CPU tests')
    if args.output.exists():raise FileExistsError('Use a fresh immutable SFT session directory')
    if not args.tiny_cpu_test and args.guard_evidence is None:
        raise ValueError('Fresh conversation-v1 guard evidence required')


def require_completed_main(parent,lineage):
    state=parent['state'];horizon=state['horizon_config']
    if (lineage.get('main_horizon_complete') is not True or
        type(horizon['max_steps']) is not int or horizon['max_steps']<1 or
        type(state['step']) is not int or state['step']!=horizon['max_steps']):
        raise ValueError('SFT requires the fully completed owned main horizon')
    expected=state['step']*horizon['batch_rows']*horizon['accumulation_steps']*parent['model_config']['sequence_len']
    if state['main_target_positions']!=expected or lineage['main_target_positions']!=expected:
        raise ValueError('Completed owned main target count mismatch')


def verify_gpu_runtime(torch,numpy_version,backend):
    if str(torch.__version__).split('+')[0]!='2.9.1' or torch.version.cuda!='12.8' or numpy_version!='2.3.5':
        raise ValueError('Reviewed SFT runtime requires torch2.9.1/cu128 and numpy2.3.5')
    if (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported() or
        torch.cuda.device_count()!=1 or tuple(torch.cuda.get_device_capability())!=(9,0) or
        'H100' not in torch.cuda.get_device_name() or backend!='fa3'):
        raise ValueError('Reviewed SFT runtime requires one BF16 H100 with pinned FA3')


def build_model(parent,GPT,GPTConfig,torch,device):
    """Promote only owned tensors; regenerate non-persistent rotary buffers."""
    with torch.device('meta'):model=GPT(GPTConfig(**parent['model_config']))
    model.load_state_dict(parent['model'],strict=True,assign=True)
    # Materialize meta buffers before .to(), then restore compute dtype after
    # promoting parameters to FP32. BF16 rotary buffers are required by GPT.
    def rotary(target):
        return model._precompute_rotary_embeddings(model.rotary_seq_len,
            model.config.n_embd//model.config.n_head,device=target)
    model.cos,model.sin=rotary('cpu')
    model.to(device=device,dtype=torch.float32)
    model.cos,model.sin=rotary(device)
    if any(p.dtype!=torch.float32 for p in model.parameters()):raise ValueError('SFT requires FP32 master parameters')
    return model


class ForwardPrecision:
    """Autocast only forward; backward and AdamW run outside its context."""
    def __init__(self,model,torch,device):self.model,self.torch,self.device=model,torch,device
    def train(self):return self.model.train()
    def parameters(self):return self.model.parameters()
    def __call__(self,*args,**kwargs):
        context=self.torch.autocast(device_type='cuda',dtype=self.torch.bfloat16) if self.device=='cuda' else nullcontext()
        with context:return self.model(*args,**kwargs)


def optimizer_for(model,config,torch):
    return torch.optim.AdamW(model.parameters(),lr=config['peak_lr'],betas=tuple(config['betas']),eps=config['eps'],
        weight_decay=config['weight_decay'],foreach=config['foreach'],fused=config['fused'])


def compatibility_smoke(torch,GPT,GPTConfig,device,*,tiny=False):
    """Two synthetic updates, separate from SFT supply and owned model weights."""
    class Synthetic:
        count=4;digest='synthetic-compatibility-only'
        def batch(self,cursor,count,device,torch):
            x=(torch.arange(count*16,device=device).reshape(count,16)+cursor)%256
            y=(x+1)%256;y[:,:8]=-1
            return x,y,{'examples':count,'processed_positions':count*17,'supervised_targets':count*8,
                       'padded_compute_positions':count*16}
    started=time.perf_counter();saved_rng=base.rng_state(torch)
    try:
        # GPU smoke uses the production head dimension (128) and grouped query
        # attention. CPU fixture stays below the existing reduced-model limits.
        config=dict(sequence_len=16,vocab_size=320,n_layer=2,n_head=2,n_kv_head=1,
                    n_embd=64 if tiny else 256,window_pattern='SSSL')
        with torch.device(device):model=GPT(GPTConfig(**config))
        model.init_weights();model.to(dtype=torch.float32)
        model.cos,model.sin=model._precompute_rotary_embeddings(model.rotary_seq_len,
            model.config.n_embd//model.config.n_head,device=device)
        supply=Synthetic();schedule=core.plan(supply,2,0);optimizer=optimizer_for(model,schedule,torch)
        execute=ForwardPrecision(model,torch,device);cursor=0;losses=[]
        for step in range(2):
            cursor,item=core.train_step(execute,optimizer,supply,cursor,step,schedule,torch,device)
            losses.append(item['loss_nats_per_supervised_target'])
        if any(p.dtype!=torch.float32 or not bool(torch.isfinite(p).all()) for p in model.parameters()):
            raise FloatingPointError('Compatibility smoke produced invalid FP32 masters')
        if device=='cuda':torch.cuda.synchronize()
        result={'status':'passed','device':device,'synthetic_only':True,'updates':2,
            'cuda_compatibility_verified':device=='cuda','model_config':config,'losses':losses,
            'compute_seconds':time.perf_counter()-started}
        del model,optimizer,execute
        if device=='cuda':torch.cuda.empty_cache()
        return result
    finally:base.restore_rng(saved_rng,torch)


def runtime_identity(torch,np,backend,device):
    return {'torch':str(torch.__version__),'numpy':np.__version__,'device':device,
        'compute_dtype':'torch.bfloat16' if device=='cuda' else 'torch.float32','parameter_dtype':'torch.float32',
        'attention':backend,'fa3_revision':base.FA3_REVISION if device=='cuda' else None,
        'cuda':torch.version.cuda if device=='cuda' else None,
        'gpu_name':torch.cuda.get_device_name() if device=='cuda' else None,
        'model_compile':False,'optimizer':'AdamW_nonfused_nonforeach',
        'entrypoint_sha256':base.sha256(Path(__file__))}


def require_time(stop_at,monotonic_deadline):
    if time.time()>=stop_at or time.perf_counter()>=monotonic_deadline:
        raise TimeoutError('Conversation work cutoff reached; export reserve preserved')


def save_with_recovery(path,model,optimizer,state,torch):
    """Keep one verified prior pair across the core's model/sidecar replacement."""
    path=Path(path);side=path.with_suffix('.sha256.json')
    recovery=path.with_name(path.stem+'-recovery.pt');recovery_side=recovery.with_suffix('.sha256.json')
    if path.exists():
        if path.is_symlink() or side.is_symlink() or not path.is_file() or not side.is_file():
            raise ValueError('Invalid owned SFT checkpoint pair for recovery')
        receipt=json.loads(side.read_text())
        if receipt.get('format')!=FORMAT or receipt.get('sha256')!=base.sha256(path):
            raise ValueError('Previous SFT checkpoint pair is not verified')
        temporary=path.parent/'.checkpoint-recovery.pt.pending'
        temporary_side=path.parent/'.checkpoint-recovery.sha256.json.pending'
        linked=False;written=False
        try:
            # The core replaces the model inode; it does not overwrite it.
            # Copy the small sidecar because the core does overwrite that file.
            os.link(path,temporary,follow_symlinks=False);linked=True
            with temporary_side.open('x') as handle:
                written=True;handle.write(canonical(receipt)+'\n');handle.flush();os.fsync(handle.fileno())
            os.replace(temporary,recovery);linked=False
            os.replace(temporary_side,recovery_side);written=False
            directory=os.open(path.parent,os.O_RDONLY)
            try:os.fsync(directory)
            finally:os.close(directory)
        finally:
            if linked:temporary.unlink(missing_ok=True)
            if written:temporary_side.unlink(missing_ok=True)
    core.save(path,model,optimizer,state,torch)
    for artifact in (path,side):
        with artifact.open('rb') as handle:os.fsync(handle.fileno())
    directory=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(directory)
    finally:os.close(directory)
    return recovery if recovery.exists() else None


def run(args):
    validate_options(args)
    evidence=None;stop_at=float('inf')
    if not args.tiny_cpu_test:
        evidence,stop_at=validate_conversation_guard(args.guard_evidence,export_reserve_seconds=args.export_reserve_seconds)
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
    guard_binding=None if evidence is None else {'profile':PROFILE,'pod_id':evidence['pod_id'],
        'reservation_id':evidence['reservation_id'],'intent_sha256':evidence['guard_intent_sha256'],
        'guard_evidence_sha256':base.sha256(args.guard_evidence),'armed_at':evidence['intent']['armed_at'],
        'provider_stop_deadline_utc':evidence['provider_stop_deadline_utc'],'work_cutoff_unix':stop_at}
    contract={'format':FORMAT,'plan':config,'lineage':lineage,'runtime_identity':identity,
        'adapter_code_hashes':core.code_hashes(),'guard':guard_binding,'compatibility_smoke':smoke,
        'session_compute_seconds':args.session_compute_seconds,'export_reserve_seconds':args.export_reserve_seconds,
        'save_every_updates':args.save_every,'processed_token_cap':5_000_000,'passes':1,
        'wrapping':False,'truncation':False,'entrypoint':'train_sft_gpu.py',
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
        'heldout_or_confirmatory_scoring':False,
        'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated() if device=='cuda' else None,
        'cuda_peak_reserved_bytes':torch.cuda.max_memory_reserved() if device=='cuda' else None}
    base.write_json(args.output/'result.json',result)
    return result


if __name__=='__main__':print(json.dumps(run(parser().parse_args()),indent=2))
