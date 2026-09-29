#!/usr/bin/env python3
"""One finite owned-model instruction pass. GPU launch is deliberately disabled.

The pure masking/schedule/step/checkpoint core is testable on reduced CPU models.
A separately reviewed conversation-stage guard must be integrated before a
production entrypoint is enabled. Pilot/main budget profiles cannot authorize it.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import random
import time

import prepare_sft as data_builder
import train_dense as base
import train_main as main_training

FORMAT = 'udgam-owned-human-sft-v1'
sha256 = base.sha256


def code_hashes():
    names=('train_sft.py','prepare_sft.py','infer_dense.py','train_main.py','train_dense.py',
           'prepare_dense_tokens.py','prepare_production.py','prepare_english.py')
    return {name:sha256(Path(__file__).with_name(name)) for name in names}


class FiniteExamples:
    def __init__(self,directory,*,tiny=False):
        import numpy as np
        self.directory=Path(directory);self.manifest=data_builder.verify(self.directory)
        self.digest=sha256(self.directory/'manifest.json')
        m=self.manifest
        if m.get('status')!='prepared_for_supply_review_not_training_authorization' or m.get('profile')!=('tiny_cpu_test' if tiny else 'production'):
            raise ValueError('Prepared SFT profile/status mismatch')
        if m['source']!={'repo':data_builder.REPO,'revision':data_builder.REVISION,'license':'apache-2.0',
                        'receipt_sha256':sha256(self.directory/'source/acquisition-receipt.json')}:
            raise ValueError('Pinned human source receipt identity mismatch')
        if m['counts']['processed_tokens_all_splits']>5_000_000 or m['processed_token_cap']>5_000_000:
            raise ValueError('SFT hard finite cap exceeded')
        required={f'{split}.{suffix}' for split in ('train','dev','final') for suffix in
                  ('tokens.npy','loss-mask.npy','offsets.npy','provenance.jsonl.gz')}
        required|={'tokenizer.json','source/acquisition-receipt.json','source/OpenAssistant--oasst2.README.md',
                   'source/OpenAssistant--oasst2.json','source/SOURCE_NOTICE.md','tree-decisions.json'}
        if not required<=set(m['hashes']):raise ValueError('SFT artifact hash inventory is incomplete')
        expected_sources={'prepare_sft.py','infer_dense.py','prepare_dense_tokens.py','prepare_production.py','prepare_english.py'}
        if set(m['source_code_sha256'])!=expected_sources:raise ValueError('SFT preparation source inventory is incomplete')
        for name,digest in m['source_code_sha256'].items():
            if name not in code_hashes() or code_hashes()[name]!=digest:
                raise ValueError('SFT preparation/template source changed')
        self.tokens=np.load(self.directory/'train.tokens.npy',mmap_mode='r',allow_pickle=False)
        self.mask=np.load(self.directory/'train.loss-mask.npy',mmap_mode='r',allow_pickle=False)
        self.offsets=np.load(self.directory/'train.offsets.npy',mmap_mode='r',allow_pickle=False)
        self.count=len(self.offsets)-1
        if not self.count:raise ValueError('No eligible finite SFT training supply')
        if tiny and (self.count>64 or len(self.tokens)>16384 or m['sequence_len']>64 or m['vocab_size']>1024):
            raise ValueError('CPU flag permits only tiny SFT data')
        if any(int(b-a)<3 for a,b in zip(self.offsets[:-1],self.offsets[1:])):
            raise ValueError('SFT examples need a valid causal prefix')
        self.target_prefix=np.concatenate(([0],np.cumsum(np.add.reduceat(self.mask.astype(np.int64),self.offsets[:-1]))))

    def counts_at(self,cursor):
        if type(cursor) is not int or not 0<=cursor<=self.count:raise ValueError('Invalid finite SFT cursor')
        return {'processed_positions':int(self.offsets[cursor]),'supervised_targets':int(self.target_prefix[cursor])}

    def batch(self,cursor,count,device,torch):
        import numpy as np
        if type(cursor) is not int or type(count) is not int or cursor<0 or count<1 or cursor+count>self.count:
            raise ValueError('Finite SFT supply exhausted; wrapping is forbidden')
        bounds=[(int(self.offsets[i]),int(self.offsets[i+1])) for i in range(cursor,cursor+count)]
        width=max(hi-lo-1 for lo,hi in bounds)
        x=torch.zeros((count,width),dtype=torch.long,device=device)
        y=torch.full((count,width),-1,dtype=torch.long,device=device)
        targets=0;positions=0
        for index,(lo,hi) in enumerate(bounds):
            ids=torch.from_numpy(np.array(self.tokens[lo:hi],dtype=np.int64)).to(device)
            mask=torch.from_numpy(np.array(self.mask[lo+1:hi],dtype=np.bool_)).to(device)
            length=hi-lo-1;x[index,:length]=ids[:-1]
            y[index,:length]=torch.where(mask,ids[1:],-1)
            targets+=int(mask.sum());positions+=hi-lo
        if not targets:raise ValueError('Batch has no assistant targets')
        return x,y,{'examples':count,'processed_positions':positions,'supervised_targets':targets,
                    'padded_compute_positions':x.numel()}


def plan(data,batch_size,seed):
    if type(batch_size) is not int or not 1<=batch_size<=64:raise ValueError('Invalid SFT batch size')
    steps=math.ceil(data.count/batch_size)
    return {'passes':1,'examples':data.count,'batch_size':batch_size,'steps':steps,
        'warmup_steps':min(max(1,math.ceil(.03*steps)),max(1,steps-1)),
        'warmup_fraction':.03,'peak_lr':1e-4,'final_lr':1e-5,'betas':[.9,.95],
        'weight_decay':.01,'eps':1e-8,'optimizer':'AdamW','foreach':False,'fused':False,
        'parameter_storage':'float32_master_exactly_promoted_from_owned_parent',
        'order':'immutable_prepared_train_storage_order','loss':'mean_over_assistant_content_and_end_targets',
        'seed':seed,'source_manifest_sha256':data.digest}


def learning_rate(step,config):
    n=config['steps'];warm=config['warmup_steps'];peak=config['peak_lr'];end=config['final_lr']
    if type(step) is not int or not 0<=step<n:raise ValueError('SFT schedule exhausted')
    if n==1:return peak # A one-update synthetic pass has no separate annealing update.
    if step<warm:return peak*(step+1)/warm
    progress=(step-warm+1)/(n-warm)
    return end+(peak-end)*.5*(1+math.cos(math.pi*progress))


def train_step(model,optimizer,data,cursor,step,config,torch,device='cpu'):
    count=min(config['batch_size'],data.count-cursor)
    x,y,counts=data.batch(cursor,count,device,torch)
    lr=learning_rate(step,config)
    for group in optimizer.param_groups:group['lr']=lr
    model.train();optimizer.zero_grad(set_to_none=True)
    # A batch row is one complete example; causal padding cannot affect earlier
    # supervised positions. There is no concatenation or cross-example attention.
    loss_sum=model(x,targets=y,loss_reduction='sum')
    loss=loss_sum/counts['supervised_targets']
    if not bool(torch.isfinite(loss)):raise FloatingPointError('Non-finite SFT loss')
    loss.backward()
    if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
        optimizer.zero_grad(set_to_none=True);raise FloatingPointError('Non-finite SFT gradient')
    optimizer.step();optimizer.zero_grad(set_to_none=True)
    return cursor+count,{'step':step+1,'loss_nats_per_supervised_target':float(loss.detach()),'lr':lr,**counts}


def validate_parent(path,expected_hash,pilot_data,main_stage,sft_data,torch,*,tiny=False):
    if len(expected_hash)!=64 or any(c not in '0123456789abcdef' for c in expected_hash) or sha256(path)!=expected_hash:
        raise ValueError('Explicit owned main parent checkpoint hash mismatch')
    payload,side=main_training.inspect_owned(path,main=True,torch=torch)
    prepared,digest=base.verify_prepared(pilot_data,production=not tiny)
    state=payload['state'];config=payload['model_config']
    base.validate_config({**config,'source_revision':base.REVISION,'initialization':'random_weights_only'},tiny=tiny)
    if state['step']<1 or state['pilot_data_manifest_sha256']!=digest or side['pilot_data_manifest_sha256']!=digest:
        raise ValueError('A trained owned main parent with matching pilot lineage is required')
    if (sft_data.manifest['lineage']['pilot_manifest_sha256']!=digest or
        sft_data.manifest['lineage']['pilot_tokenizer_sha256']!=prepared['hashes']['tokenizer.json'] or
        sft_data.manifest['lineage']['pilot_validation_array_sha256']!=prepared['hashes']['val.npy'] or
        sft_data.manifest['hashes']['tokenizer.json']!=prepared['hashes']['tokenizer.json']):
        raise ValueError('SFT data/tokenizer is not from the owned main lineage')
    if config['vocab_size']!=sft_data.manifest['vocab_size'] or config['sequence_len']!=sft_data.manifest['sequence_len']:
        raise ValueError('Owned model and SFT context/vocabulary differ')
    stream=main_training.FiniteShards(main_stage,pilot_data,prepared,digest,tiny=tiny,
        contract=state['stage_contract'],accepted_shards=state['accepted_shards'],cursor=state['cursor'])
    expected_rows=state['step']*state['horizon_config']['batch_rows']*state['horizon_config']['accumulation_steps']
    if state['cursor']['total_rows']!=expected_rows or state['main_target_positions']!=expected_rows*config['sequence_len']:
        raise ValueError('Owned main parent step/cursor mismatch')
    return payload,{'main_checkpoint_sha256':expected_hash,'initial_weights_sha256':state['initial_weights_sha256'],
        'pilot_checkpoint_sha256':state['pilot_checkpoint_sha256'],'pilot_manifest_sha256':digest,
        'main_target_positions':state['main_target_positions'],'pilot_target_positions':state['pilot_target_positions'],
        'main_horizon_complete':state['step']==state['horizon_config']['max_steps'],
        'main_runtime_identity':state['runtime_identity'],
        'main_stage_contract':stream.contract,'main_last_validated_manifest_sha256':stream.manifest_hash}


def save(path,model,optimizer,state,torch):
    payload={'format':FORMAT,'model':model.state_dict(),'optimizer':optimizer.state_dict(),'model_config':asdict(model.config),
        'state':state,'rng':base.rng_state(torch),'source_revision':base.REVISION,'source_hashes':base.source_hashes(),
        'adapter_code_hashes':code_hashes()}
    temporary=path.with_suffix('.tmp');torch.save(payload,temporary);os.replace(temporary,path)
    base.write_json(path.with_suffix('.sha256.json'),{'format':FORMAT,'sha256':sha256(path),
        'initial_weights_sha256':state['lineage']['initial_weights_sha256'],
        'main_checkpoint_sha256':state['lineage']['main_checkpoint_sha256'],'sft_manifest_sha256':state['plan']['source_manifest_sha256']})


def inspect_checkpoint(path,torch):
    path=Path(path);side=json.loads(path.with_suffix('.sha256.json').read_text())
    if side.get('format')!=FORMAT or side.get('sha256')!=sha256(path):raise ValueError('SFT checkpoint integrity/format mismatch')
    payload=torch.load(path,weights_only=True,map_location='cpu')
    if (payload.get('format')!=FORMAT or payload.get('source_revision')!=base.REVISION or
        payload.get('source_hashes')!=base.source_hashes() or payload.get('adapter_code_hashes')!=code_hashes()):
        raise ValueError('SFT source/runtime implementation identity mismatch')
    state=payload['state']
    for key,expected in [('initial_weights_sha256',state['lineage']['initial_weights_sha256']),
            ('main_checkpoint_sha256',state['lineage']['main_checkpoint_sha256']),('sft_manifest_sha256',state['plan']['source_manifest_sha256'])]:
        if side.get(key)!=expected:raise ValueError('SFT checkpoint lineage sidecar mismatch')
    return payload


def restore(payload,model,optimizer,data,config,lineage,runtime,torch):
    state=payload['state']
    if payload['model_config']!=asdict(model.config) or state['plan']!=config:raise ValueError('SFT immutable model/data/schedule changed')
    # A harmless append to the main stage may change its current manifest; the
    # immutable accepted parent contract and selected main checkpoint must match.
    stable=lambda d:{k:v for k,v in d.items() if k!='main_last_validated_manifest_sha256'}
    if stable(state['lineage'])!=stable(lineage):raise ValueError('SFT parent lineage changed')
    if state['runtime_identity']!=runtime:raise ValueError('SFT runtime identity changed')
    if type(state['step']) is not int or not 0<=state['step']<=config['steps'] or state['cursor']!=min(state['step']*config['batch_size'],data.count):
        raise ValueError('SFT optimizer-step/example cursor mismatch')
    if state['counts']!=data.counts_at(state['cursor']):raise ValueError('SFT processed/supervised cursor mismatch')
    if any(t.dtype!=torch.float32 for t in payload['model'].values()):raise ValueError('SFT master-parameter dtype changed')
    last_lr=learning_rate(state['step']-1,config) if state['step'] else config['peak_lr']
    for group in payload['optimizer']['param_groups']:
        if (tuple(group['betas'])!=tuple(config['betas']) or group['weight_decay']!=config['weight_decay'] or
            group['eps']!=config['eps'] or group['foreach']!=config['foreach'] or group['fused']!=config['fused'] or
            group['lr']!=last_lr):raise ValueError('SFT optimizer configuration changed')
    model.load_state_dict(payload['model'],strict=True);optimizer.load_state_dict(payload['optimizer'])
    base.restore_rng(payload['rng'],torch)
    return state


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--main-checkpoint',type=Path,required=True)
    p.add_argument('--expected-main-sha256',required=True)
    p.add_argument('--main-stage',type=Path,required=True)
    p.add_argument('--pilot-data',type=Path,required=True)
    p.add_argument('--sft-data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--resume-sft',type=Path)
    p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--seed',type=int,default=20260922)
    p.add_argument('--tiny-cpu-test',action='store_true')
    p.add_argument('--session-compute-seconds',type=float,default=60)
    p.add_argument('--stop-after-session-steps',type=int)
    return p


def run(args):
    # This gate precedes all file, model, optimizer and device work. No main/pilot
    # evidence can broaden it; conversation-stage integration needs new review.
    if not args.tiny_cpu_test:
        raise RuntimeError('Production/GPU SFT is disabled pending separately reviewed conversation-stage guard integration')
    if args.output.exists():raise FileExistsError('Use a fresh immutable SFT session directory')
    base.finite_number(args.session_compute_seconds)
    if args.session_compute_seconds>60 or not 1<=args.batch_size<=4 or (args.stop_after_session_steps is not None and not 0<=args.stop_after_session_steps<=64):
        raise ValueError('Only a bounded tiny CPU SFT test is enabled')
    data=FiniteExamples(args.sft_data,tiny=True);config=plan(data,args.batch_size,args.seed)
    torch,GPT,GPTConfig,_,dtype,backend=base.runtime(tiny=True,attention='sdpa',optimizer_compile=False)
    import numpy as np
    parent,lineage=validate_parent(args.main_checkpoint,args.expected_main_sha256,args.pilot_data,args.main_stage,data,torch,tiny=True)
    runtime={'torch':str(torch.__version__),'numpy':np.__version__,'device':'cpu','compute_dtype':str(dtype),
        'parameter_dtype':'torch.float32','attention':backend,'model_compile':False,'optimizer':'AdamW_nonfused_nonforeach'}
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    with torch.device('meta'):model=GPT(GPTConfig(**parent['model_config']))
    model.load_state_dict(parent['model'],strict=True,assign=True)
    model.cos,model.sin=model._precompute_rotary_embeddings(model.rotary_seq_len,model.config.n_embd//model.config.n_head,device='cpu')
    model.to(device='cpu',dtype=torch.float32)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['peak_lr'],betas=tuple(config['betas']),eps=config['eps'],
        weight_decay=config['weight_decay'],foreach=config['foreach'],fused=config['fused'])
    state={'step':0,'cursor':0,'counts':data.counts_at(0),'plan':config,'lineage':lineage,'runtime_identity':runtime,
        'optimizer_transition':'fresh_AdamW_once_after_owned_main; no main optimizer moments reused',
        'initial_sft_weights_sha256':base.model_hash(model),'loss_history':[],'cumulative_compute_seconds':0.,'sessions':[]}
    del parent
    if args.resume_sft:
        loaded=inspect_checkpoint(args.resume_sft,torch)
        state=restore(loaded,model,optimizer,data,config,lineage,runtime,torch);del loaded
    if state['cursor']==data.count:raise ValueError('The finite SFT pass is already complete; repetition is forbidden')
    args.output.mkdir(parents=True)
    base.write_json(args.output/'run-contract.json',{'format':FORMAT,'plan':config,'lineage':lineage,'runtime_identity':runtime,
        'adapter_code_hashes':code_hashes(),'production_enabled':False,'guard_integration':'not_implemented',
        'session_compute_seconds':args.session_compute_seconds})
    checkpoint=args.output/'checkpoint.pt';save(checkpoint,model,optimizer,state,torch)
    session={'start_step':state['step'],'start_cursor':state['cursor'],'compute_seconds':0.};stop_reason='finite_pass_complete'
    while state['step']<config['steps']:
        completed=state['step']-session['start_step']
        if args.stop_after_session_steps is not None and completed>=args.stop_after_session_steps:stop_reason='synthetic_interruption';break
        if session['compute_seconds']>=args.session_compute_seconds:stop_reason='session_compute_limit';break
        start=time.perf_counter()
        next_cursor,item=train_step(model,optimizer,data,state['cursor'],state['step'],config,torch)
        elapsed=time.perf_counter()-start
        state['step']+=1;state['cursor']=next_cursor;state['counts']=data.counts_at(next_cursor)
        state['loss_history'].append(item);state['cumulative_compute_seconds']+=elapsed;session['compute_seconds']+=elapsed
        save(checkpoint,model,optimizer,state,torch)
    session.update(end_step=state['step'],end_cursor=state['cursor'],stop_reason=stop_reason)
    state['sessions'].append(session);save(checkpoint,model,optimizer,state,torch)
    before=base.model_hash(model);loaded=inspect_checkpoint(checkpoint,torch)
    restore(loaded,model,optimizer,data,config,lineage,runtime,torch)
    if before!=base.model_hash(model):raise AssertionError('SFT checkpoint cold tensor reload mismatch')
    result={'format':FORMAT,'status':'tiny_cpu_core_validation_only','step':state['step'],'cursor':state['cursor'],
        'finite_pass_complete':state['cursor']==data.count,'counts':state['counts'],'session':session,
        'checkpoint_sha256':sha256(checkpoint),'checkpoint_reload_verified':True,'final_weights_sha256':before,
        'initial_weights_sha256':lineage['initial_weights_sha256'],'main_checkpoint_sha256':lineage['main_checkpoint_sha256'],
        'optimizer_transition':state['optimizer_transition'],'production_enabled':False,'heldout_or_confirmatory_scoring':False}
    base.write_json(args.output/'result.json',result);return result


if __name__=='__main__':print(json.dumps(run(parser().parse_args()),indent=2))
