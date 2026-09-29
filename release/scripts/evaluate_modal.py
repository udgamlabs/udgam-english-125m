#!/usr/bin/env python3
"""Modal boundary for unchanged, frozen completion scoring and native exports.

No provider calls, downloads or model selection. A sealed job selects a complete
owned endpoint or one pinned baseline. Original evaluator modules are loaded in
isolation; only their provider validator and clock are replaced. Their prompts,
token boundaries, metrics, cache validation and quality floors are unchanged.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
from types import SimpleNamespace

import modal_final_contract as boundary
import evaluate_dense as original
import evaluate_baselines as baselines
from prepare_dense_tokens import sha256, write_json

PROFILE = 'modal-evaluation-v1'
MANDATORY_MODELS = ('owned-base','owned-sft','sarvam-1','smollm2-135m','smollm2-360m')
LATENCY_ALLOWANCE_SECONDS = 120
TIMING_METHOD = 'single-session-fixed-plus25x-scoring-upper-bound-2x-v1'


def source_hashes():
    import train_sft
    names = ('evaluate_modal.py','modal_final_contract.py','evaluate_dense.py','evaluate_baselines.py',
             'sft_checkpoint_adapter.py','instruction_checks.py')
    return {**train_sft.code_hashes(), **{n:sha256(Path(__file__).with_name(n)) for n in names}}


def validate_job(job):
    required={'id','kind','model','suite','max_seconds','batch_size','paths','resume_evaluation','latency','export_weights'}
    if set(job)!=required or job['kind'] not in ('owned','baseline','instructions'):
        raise ValueError('Exact sealed evaluation job required')
    if job['model'] not in MANDATORY_MODELS or job['suite'] not in ('development','final'):
        raise ValueError('Declared model/suite required')
    if (job['kind']=='baseline') != (job['model'] in baselines.MODELS):
        raise ValueError('Baseline and owned model roles cannot be exchanged')
    original.training.finite_number(job['max_seconds'])
    if job['max_seconds']>600 or type(job['batch_size']) is not int or not 1<=job['batch_size']<=8:
        raise ValueError('Original evaluation time/batch bound exceeded')
    for key in ('resume_evaluation','latency','export_weights'):
        if type(job[key]) is not bool:raise ValueError('Explicit evaluation flags required')
    if job['kind']!='owned' and job['export_weights']:
        raise ValueError('Baseline weights are not released by this adapter')
    if job['kind']=='instructions' and (job['resume_evaluation'] or job['latency']):
        raise ValueError('Instruction diagnostics use one fixed finite generation pass')
    if not isinstance(job['paths'],dict) or any(not isinstance(v,str) or not v for v in job['paths'].values()):
        raise ValueError('Explicit local artifact paths required')
    common={'main_checkpoint','main_result','main_stage','pilot_data','output'}
    needed = common | ({'snapshot','benchmarks','development','task_definitions'} if job['kind']=='baseline' else
        {'checkpoint','benchmarks','development','task_definitions'} if job['kind']=='owned' else
        {'checkpoint','inference','instruction_suite'})
    if job['model']=='owned-sft':needed|={'sft_data'}
    if job['suite']=='final' and job['kind']!='instructions':needed|={'development_evidence'}
    if set(job['paths'])!=needed:raise ValueError('Job artifact inventory differs from its declared role')
    return {k:Path(v) for k,v in job['paths'].items()}


def endpoint(contract,paths):
    boundary.bind_artifacts(contract,{'main_checkpoint':paths['main_checkpoint'],'main_result':paths['main_result'],
        'main_stage_manifest':paths['main_stage']/'manifest.json','pilot_manifest':paths['pilot_data']/'manifest.json'})
    result=boundary.read(paths['main_result'])
    if (result.get('main_horizon_complete') is not True or result.get('candidate_valid') is not True or
        result.get('checkpoint_reload_verified') is not True or result.get('step')!=61035 or
        result.get('horizon_steps')!=61035 or result.get('checkpoint_sha256')!=contract['artifacts']['main_checkpoint']):
        raise ValueError('Completed owned61035-step main endpoint required before final-stage evaluation')


def require_complete_payload(payload):
    if payload['format']==original.main_training.FORMAT:
        s=payload['state']
        if (s['step']!=61035 or s['horizon_config']['max_steps']!=61035 or
            s['main_target_positions']!=1_999_994_880 or s['cursor']['total_rows']!=976560):
            raise ValueError('Partial main checkpoint is ineligible for final evaluation')
    elif payload['format']==original.SFT_CHECKPOINT_FORMAT:
        if payload['state']['lineage'].get('main_horizon_complete') is not True:
            raise ValueError('SFT parent main horizon incomplete')
    else:raise ValueError('Pilot checkpoint is ineligible for final evaluation')


def isolated_evaluators(execution_path,clock=None):
    clock=clock or boundary.MonotonicWall()
    shared=boundary.isolated(Path(original.__file__),'_owned_modal_final_evaluator')
    execution_digest=boundary.digest(boundary.read(execution_path)) if Path(execution_path).exists() else None
    def guard(path,*,export_reserve_seconds):
        if Path(path)!=Path(execution_path) or export_reserve_seconds!=120:
            raise ValueError('Unexpected evaluator guard boundary')
        result=boundary.validate(path,profile=PROFILE,source_hashes=source_hashes(),now=clock.time())
        if boundary.digest(result[0])!=execution_digest:
            raise ValueError('Execution contract changed after adapter initialization')
        return result
    def runtime(**kwargs):
        result=original.training.runtime(**kwargs)
        if kwargs.get('tiny'):raise ValueError('Production Modal evaluation cannot use tiny CPU mode')
        torch=result[0]
        if (str(torch.__version__).split('+')[0]!='2.9.1' or torch.version.cuda!='12.8' or
            torch.cuda.device_count()!=1 or tuple(torch.cuda.get_device_capability())!=(9,0) or
            'H100' not in torch.cuda.get_device_name() or result[-1]!='fa3'):
            raise ValueError('Owned completion scoring requires the reviewed H100/FA3 runtime')
        return result
    shared.training=boundary.proxy(original.training,validate_guard=guard,runtime=runtime)
    shared.time=clock
    read=shared.read_owned_checkpoint
    def read_complete(*args,**kwargs):
        payload=read(*args,**kwargs);require_complete_payload(payload);return payload
    shared.read_owned_checkpoint=read_complete
    baseline=boundary.isolated(Path(baselines.__file__),'_baseline_modal_final_evaluator')
    baseline.shared=shared;baseline.time=clock
    latency=baseline.forced_latency
    def timed_latency(*args,**kwargs):
        kwargs['clock']=clock
        return latency(*args,**kwargs)
    baseline.forced_latency=timed_latency
    load=baseline.load_model
    def load_h100(*args,**kwargs):
        torch=kwargs['torch']
        if (torch.version.cuda!='12.8' or torch.cuda.device_count()!=1 or
            tuple(torch.cuda.get_device_capability())!=(9,0) or 'H100' not in torch.cuda.get_device_name()):
            raise ValueError('Comparable baseline H100/cu128 runtime required')
        return load(*args,**kwargs)
    baseline.load_model=load_h100
    return shared,baseline


def development_timing(job,paths):
    """Verify the complete dev cache before using timing, never its accuracy."""
    directory=paths['development_evidence']
    summary=boundary.read(directory/'summary.json');contract=boundary.read(directory/'evaluation-contract.json')
    _,suite=original.load_development_smoke(paths['development'])
    requests=original.make_requests(suite)
    items,_=original.cached_items(directory/'items.jsonl',requests,contract)
    checkpoint_hash=(baselines.verify_snapshot(job['model'],paths['snapshot']) if job['kind']=='baseline' else None)
    identity=original.digest(original.canonical(checkpoint_hash)) if checkpoint_hash else sha256(paths['checkpoint'])
    if (len(items)!=100 or summary.get('complete') is not True or summary.get('contract')!=contract or
        summary.get('items_sha256')!=sha256(directory/'items.jsonl') or not summary.get('cost_history_complete') or
        contract.get('checkpoint_sha256')!=identity or contract.get('policy')!=original.POLICY or
        contract.get('suite_kind')!='development_smoke_not_confirmatory' or
        contract.get('benchmark_manifest_sha256')!=sha256(paths['development']/'manifest.json') or
        contract.get('task_source_manifest_sha256')!=sha256(paths['task_definitions']/'manifest.json') or
        contract.get('requests_sha256')!=sha256(directory/'requests.jsonl') or
        contract.get('requests_sha256')!=original.digest(''.join(original.canonical(r)+'\n' for r in requests))):
        raise ValueError('Complete same-model/policy/source dev100 timing evidence required')
    for item,request in zip(items,requests,strict=True):
        if (any(item.get(k)!=request[k] for k in ('task','gold','primary_metric','document_sha256')) or
            any(type(item.get(k)) is not int or item[k] not in (0,1) for k in ('acc','acc_norm','primary_correct')) or
            type(item.get('supported')) is not bool or (not item['supported'] and item['primary_correct'])):
            raise ValueError('Invalid dev scoring record')
    if summary.get('development_macro_accuracy')!=original.summarize(items,requests)['Q']:
        raise ValueError('Dev summary does not match its scoring cache')
    elapsed=original.training.finite_number(summary['total_recorded_process_seconds'])
    evaluated=original.training.finite_number(summary['evaluation_seconds'])
    cold=original.training.finite_number(summary['cold_load_seconds'],positive=False)
    session_key=('elapsed_seconds_including_verification_load_scoring_latency' if job['kind']=='baseline'
                 else 'elapsed_seconds_including_export_and_latency')
    sessions=summary.get('sessions')
    if (summary.get('reused_cached_items')!=0 or not isinstance(sessions,list) or len(sessions)!=1 or
        sessions[0].get('new_items')!=100 or sessions[0].get(session_key)!=elapsed or
        not 0<=cold<evaluated<=elapsed or 'latency' in summary or 'export' in summary):
        raise ValueError('Pristine single-session dev100 timing split required')
    # The frozen evaluators record cold-load and pre-export evaluation clocks.
    # Their difference includes scoring plus request/cache/hash bookkeeping;
    # scaling this upper bound is conservative, without scaling the fixed load.
    scalable=evaluated-cold
    fixed=elapsed-scalable
    latency_allowance=LATENCY_ALLOWANCE_SECONDS if job['latency'] else 0
    estimate=2*(fixed+25*scalable)+latency_allowance
    if estimate>job['max_seconds']:
        raise ValueError('Dev timing with2x headroom exceeds declared final window; revise budget, never shrink suite')
    return {'timing_method':TIMING_METHOD,'development_seconds':elapsed,
            'fixed_overhead_seconds':fixed,'scoring_and_bookkeeping_upper_bound_seconds':scalable,
            'estimated_final_seconds_with_2x_headroom':estimate,
            'latency_allowance_seconds':latency_allowance,
            'summary_sha256':sha256(directory/'summary.json'),
            'development_evidence_sha256':{name:sha256(directory/name) for name in
                ('summary.json','evaluation-contract.json','items.jsonl','requests.jsonl')},
            'quality_used_for_selection':False}


def argv_for(job,paths,execution):
    result=[]
    if job['kind']=='baseline':result=['evaluate','--model',job['model'],'--snapshot',str(paths['snapshot'])]
    else:
        result=['--checkpoint',str(paths['checkpoint']),'--main-stage',str(paths['main_stage']),
                '--data',str(paths['pilot_data'])]
        if job['model']=='owned-sft':result+=['--sft-data',str(paths['sft_data']),'--main-parent',str(paths['main_checkpoint'])]
    for flag,key in [('--output','output'),('--benchmarks','benchmarks'),('--task-definitions','task_definitions')]:
        result+=[flag,str(paths[key])]
    result+=['--guard-evidence',str(execution),'--max-eval-seconds',str(job['max_seconds']),'--batch-size',str(job['batch_size'])]
    if job['suite']=='development':result+=['--development-smoke',str(paths['development'])]
    elif job['kind']=='baseline':result+=['--development-evidence',str(paths['development_evidence']),'--development-data',str(paths['development'])]
    for key in ('resume_evaluation','latency','export_weights'):
        if job[key]:result+=['--'+key.replace('_','-')]
    return result


def instructions(job,paths,contract,deadline):
    import instruction_checks as checks
    import infer_dense as native
    expected=contract['artifacts']['instruction_suite']
    _,rows=checks.load(paths['instruction_suite'],expected)
    manifest=native.verify_export(paths['inference'])
    checkpoint=sha256(paths['checkpoint'])
    lineage=manifest['lineage']
    complete_parent=(lineage.get('main_horizon_complete') is True if job['model']=='owned-sft' else
                     lineage.get('phase')=='main' and lineage.get('step')==61035 and
                     lineage.get('main_target_positions')==1_999_994_880)
    if manifest['source_checkpoint_sha256']!=checkpoint or not complete_parent:
        raise ValueError('Native inference artifact differs from completed owned endpoint')
    if job['model']=='owned-sft' and manifest['lineage'].get('sft_pass_complete') is not True:
        raise ValueError('Completed finite SFT export required')
    # Export is produced during the same bounded invocation, so its final file
    # hash cannot be known at launch. Derive trust from the selected immutable
    # owned checkpoint instead of merely trusting a self-written export label.
    verify_native_derivation(paths,manifest)
    if paths['output'].exists():raise FileExistsError('Fresh immutable instruction response output required')
    paths['output'].mkdir(parents=True)
    clock=boundary.MonotonicWall();started=clock.time()
    loaded=native.load(paths['inference'],device='cuda')
    model,tokenizer,torch,dtype=loaded[:4]
    if (str(torch.__version__).split('+')[0]!='2.9.1' or torch.version.cuda!='12.8' or
        torch.cuda.device_count()!=1 or tuple(torch.cuda.get_device_capability())!=(9,0) or
        'H100' not in torch.cuda.get_device_name()):
        raise ValueError('Native instruction check requires the same reviewed H100 runtime')
    responses=[];measurements=[]
    with (paths['output']/'responses.jsonl').open('x') as handle:
        for row in rows:
            if row['split']!=job['suite']:continue
            if clock.time()>=min(deadline,started+job['max_seconds']):break
            tokens=tokenizer.conversation(row['messages'])
            result=native.generate(model,tokenizer,tokens,torch,dtype,**checks.POLICY['generation'])
            response={'id':row['id'],'prompt_sha256':row['prompt_sha256'],'suite_sha256':expected,
                'checkpoint_sha256':checkpoint,'generation':checks.POLICY['generation'],
                'prompt_format':'native_chat','external_tools':False,'text':result['text']}
            handle.write(checks.canonical(response)+'\n');handle.flush();responses.append(response)
            measurements.append({'id':row['id'],**{k:v for k,v in result.items() if k not in ('text','token_ids')}})
    summary=checks.score(rows,responses,split=job['suite'],suite_sha256=expected,checkpoint_sha256=checkpoint)
    summary.update(complete=summary['responses_complete'],measurements=measurements,native_load_metadata=loaded[4],
        cold_load_scope='optimizer-free native inference export',
        model_role=job['model'],no_external_baseline_chat_comparison=True)
    write_json(paths['output']/'summary.json',summary)
    return summary


def verify_native_derivation(paths,manifest):
    import torch
    for relative in manifest['files']:
        prefix='vendor/nanochat-'+original.REVISION+'/'
        if relative.startswith(prefix) and relative.endswith('.py'):
            if sha256(paths['inference']/relative)!=sha256(original.VENDOR/relative[len(prefix):]):
                raise ValueError('Native exported implementation differs from pinned source')
    if sha256(paths['inference']/'tokenizer.json')!=sha256(paths['pilot_data']/'tokenizer.json'):
        raise ValueError('Native exported tokenizer differs from owned tokenizer')
    owned=torch.load(paths['checkpoint'],weights_only=True,map_location='cpu',mmap=True)
    exported=torch.load(paths['inference']/'model.pt',weights_only=True,map_location='cpu',mmap=True)
    if owned['model_config']!=exported['model_config']:
        raise ValueError('Native export model configuration changed')
    tensor_hash=lambda p:original.training.model_hash(SimpleNamespace(state_dict=lambda:p['model']))
    if tensor_hash(owned)!=tensor_hash(exported):
        raise ValueError('Native export tensors differ from selected owned checkpoint')


def run(execution,package,job_path):
    contract,deadline=boundary.validate(execution,profile=PROFILE,source_hashes=source_hashes())
    if sha256(package)!=contract['package_manifest_sha256']:raise ValueError('Sealed evaluation package changed')
    job=boundary.read(job_path);paths=validate_job(job)
    if job['id'] not in contract['approved_jobs']:raise ValueError('Undeclared evaluation job')
    boundary.bind_artifacts(contract,{'job-'+job['id']:job_path});endpoint(contract,paths)
    selected={}
    if job['kind']=='baseline':selected[job['model']+'_snapshot']=paths['snapshot']/baselines.MANIFEST_NAME
    else:
        selected[job['model']+'_checkpoint']=paths['checkpoint']
        if job['model']=='owned-base' and sha256(paths['checkpoint'])!=contract['artifacts']['main_checkpoint']:
            raise ValueError('Selected owned base is not the completed parent')
        if job['model']=='owned-sft':selected['sft_manifest']=paths['sft_data']/'manifest.json'
    if job['kind']=='instructions':
        selected.update(instruction_suite=paths['instruction_suite']/'manifest.json')
    else:
        selected.update(benchmark_manifest=paths['benchmarks']/'manifest.json',
                        development_manifest=paths['development']/'manifest.json',
                        task_definitions=paths['task_definitions']/'manifest.json')
    boundary.bind_artifacts(contract,selected)
    timing=None
    if job['suite']=='final' and job['kind']!='instructions':timing=development_timing(job,paths)
    if job['kind']=='instructions':summary=instructions(job,paths,contract,deadline)
    else:
        owned,baseline=isolated_evaluators(execution)
        summary=(owned if job['kind']=='owned' else baseline).main(argv_for(job,paths,execution))
    receipt={'profile':PROFILE,'execution_contract_sha256':boundary.digest(contract),'job_sha256':sha256(job_path),
        'job_id':job['id'],'model':job['model'],'suite':job['suite'],'source_hashes':source_hashes(),
        'complete':summary['complete'] and (not job['latency'] or summary.get('latency',{}).get('complete') is True),
        'completion_scoring_complete':summary['complete'],
        'requested_latency_complete':summary.get('latency',{}).get('complete') if job['latency'] else None,
        'summary_sha256':sha256(paths['output']/'summary.json'),
        'development_timing':timing,'mandatory_model_set':list(MANDATORY_MODELS),
        'cash_cap_enforced_by_this_script':False,'no_quality_claim_from_incomplete_work':True}
    write_json(paths['output']/'modal-evaluation-receipt.json',receipt)
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execution-contract',type=Path,required=True)
    p.add_argument('--package-manifest',type=Path,required=True)
    p.add_argument('--job',type=Path,required=True)
    a=p.parse_args();print(json.dumps(run(a.execution_contract,a.package_manifest,a.job),indent=2))
