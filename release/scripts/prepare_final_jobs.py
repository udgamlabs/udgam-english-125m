#!/usr/bin/env python3
"""Offline allowlists, immutable job plans and joint timing gate for final stages.

No provider/network/model operation. Main/SFT endpoint hashes are resolved only
after those files actually exist. Baseline snapshots are private research inputs,
never copied into a native owned-model release. Workers use a fixed POSIX root.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path, PurePosixPath

import evaluate_modal as evaluator
import evaluate_baselines as baselines
import modal_final_contract as boundary
import train_sft_modal as sft
import package_main
from prepare_dense_tokens import sha256, write_json

FORMAT='udgam-final-jobs-v1'
REMOTE_ROOT='/workspace/udgam'
JOB_DIR='jobs/final-001'
EVAL_OUTPUT='runs/eval-modal-001'
SFT_OUTPUT='runs/sft-modal-001'
SFT_DATA='data/english-sft-001'
PILOT_DATA='data/dense-tokens-production-001'
MAIN_STAGE='data/english-main-001'
BENCHMARKS='data/benchmark-exclusions-002'
DEVELOPMENT='data/benchmark-smoke-001'
INSTRUCTIONS='data/instruction-checks-001'
DEFINITIONS='research/harness-task-definitions'
PROTOCOL='research/evaluation_protocol.md'
BASELINES={model:'baselines/'+model for model in baselines.MODELS}


def relative(value):
    if (not isinstance(value,str) or not value or '\\' in value or
        PurePosixPath(value).is_absolute() or PurePosixPath(value).as_posix()!=value or
        any(p.startswith('.') or p in ('','..') for p in PurePosixPath(value).parts)):
        raise ValueError('Canonical public relative artifact path required')
    return value


def metadata(root,name):
    path=package_main.safe_path(root,relative(name))
    if path.stat().st_size>8*1024**2:raise ValueError('Oversized public preparation metadata')
    return json.loads(path.read_text())


def remote(value):return REMOTE_ROOT+'/'+relative(value)


def checkpoint_path(value):
    relative(value)
    if not value.startswith('runs/') or not value.endswith('/checkpoint.pt'):
        raise ValueError('Owned run checkpoint path required')
    return value


def make_jobs(main_checkpoint,*,sft_checkpoint=SFT_OUTPUT+'/checkpoint.pt'):
    checkpoint_path(main_checkpoint);checkpoint_path(sft_checkpoint)
    common={k:remote(v) for k,v in {'main_checkpoint':main_checkpoint,
        'main_result':str(PurePosixPath(main_checkpoint).parent/'result.json'),'main_stage':MAIN_STAGE,
        'pilot_data':PILOT_DATA}.items()}
    jobs=[]
    for suite in ('development','final'):
        for model in evaluator.MANDATORY_MODELS:
            kind='baseline' if model in baselines.MODELS else 'owned'
            identifier=model+'-'+suite
            paths={**common,'output':remote(EVAL_OUTPUT+'/'+identifier),'benchmarks':remote(BENCHMARKS),
                'development':remote(DEVELOPMENT),'task_definitions':remote(DEFINITIONS)}
            if kind=='baseline':paths['snapshot']=remote(BASELINES[model])
            else:paths['checkpoint']=remote(main_checkpoint if model=='owned-base' else sft_checkpoint)
            if model=='owned-sft':paths['sft_data']=remote(SFT_DATA)
            if suite=='final':paths['development_evidence']=remote(EVAL_OUTPUT+'/'+model+'-development')
            jobs.append(dict(id=identifier,kind=kind,model=model,suite=suite,max_seconds=600,batch_size=4,
                paths=paths,resume_evaluation=False,latency=suite=='final',export_weights=kind=='owned' and suite=='final'))
    for model in ('owned-base','owned-sft'):
        identifier=model+'-instructions-final'
        paths={**common,'checkpoint':remote(main_checkpoint if model=='owned-base' else sft_checkpoint),
            'output':remote(EVAL_OUTPUT+'/'+identifier),'instruction_suite':remote(INSTRUCTIONS),
            'inference':remote(EVAL_OUTPUT+'/'+model+'-final/inference')}
        if model=='owned-sft':paths['sft_data']=remote(SFT_DATA)
        jobs.append(dict(id=identifier,kind='instructions',model=model,suite='final',max_seconds=60,batch_size=1,
            paths=paths,resume_evaluation=False,latency=False,export_weights=False))
    for job in jobs:evaluator.validate_job(job)
    return jobs


def write_plan(root,main_checkpoint,*,sft_checkpoint=SFT_OUTPUT+'/checkpoint.pt'):
    root=Path(root);directory=root/JOB_DIR
    if directory.exists():raise FileExistsError('Immutable final job plan already exists')
    jobs=make_jobs(main_checkpoint,sft_checkpoint=sft_checkpoint);directory.mkdir(parents=True)
    entries=[]
    for job in jobs:
        path=directory/(job['id']+'.json');write_json(path,job)
        entries.append({'id':job['id'],'path':str(path.relative_to(root)),'sha256':sha256(path)})
    plan={'format':FORMAT,'runtime_root':REMOTE_ROOT,'main_checkpoint':main_checkpoint,'sft_checkpoint':sft_checkpoint,
        'helper_sha256':sha256(Path(__file__)),'protocol_sha256':sha256(package_main.safe_path(root,PROTOCOL)),
        'mandatory_models':list(evaluator.MANDATORY_MODELS),'jobs':entries,'development_jobs':5,'final_completion_jobs':5,
        'instruction_jobs':2,'fixed_extra_work_reserve_seconds':300,
        'development_timing_method':evaluator.TIMING_METHOD,
        'final_latency_allowance_seconds_per_model':evaluator.LATENCY_ALLOWANCE_SECONDS,
        'joint_gate_required_before_any_confirmatory_job':True,
        'baseline_weight_policy':'private pinned research evaluation only; excluded from owned release',
        'automatic_retry':False,'output_prefix':EVAL_OUTPUT}
    write_json(directory/'plan.json',plan)
    (directory/'plan.sha256').write_text(sha256(directory/'plan.json')+'\n')
    return plan


def read_plan(root):
    root=Path(root);path=package_main.safe_path(root,JOB_DIR+'/plan.json')
    plan=boundary.read(path)
    if (sha256(path)!=package_main.safe_path(root,JOB_DIR+'/plan.sha256').read_text().strip() or
        plan.get('format')!=FORMAT or plan.get('helper_sha256')!=sha256(Path(__file__)) or
        plan.get('runtime_root')!=REMOTE_ROOT or plan.get('mandatory_models')!=list(evaluator.MANDATORY_MODELS) or
        plan.get('protocol_sha256')!=sha256(package_main.safe_path(root,PROTOCOL)) or
        plan.get('fixed_extra_work_reserve_seconds')!=300 or plan.get('automatic_retry') is not False or
        plan.get('development_timing_method')!=evaluator.TIMING_METHOD or
        plan.get('final_latency_allowance_seconds_per_model')!=evaluator.LATENCY_ALLOWANCE_SECONDS or
        plan.get('joint_gate_required_before_any_confirmatory_job') is not True):
        raise ValueError('Final plan integrity, source, policy or mandatory scope changed')
    jobs=make_jobs(plan['main_checkpoint'],sft_checkpoint=plan['sft_checkpoint'])
    expected=[]
    for job in jobs:
        name=JOB_DIR+'/'+job['id']+'.json';path=package_main.safe_path(root,name)
        if boundary.read(path)!=job:raise ValueError('Fixed job definition changed')
        expected.append({'id':job['id'],'path':name,'sha256':sha256(path)})
    if plan['jobs']!=expected or (plan.get('development_jobs'),plan.get('final_completion_jobs'),plan.get('instruction_jobs'))!=(5,5,2):
        raise ValueError('Job order/inventory changed')
    return plan,jobs


def local_paths(root,job):
    result={}
    for key,value in job['paths'].items():
        prefix=REMOTE_ROOT+'/'
        if not value.startswith(prefix):raise ValueError('Unexpected job runtime root')
        result[key]=Path(root)/relative(value[len(prefix):])
    return result


def sft_artifacts(root,main_checkpoint,*,resume_sft=None):
    checkpoint_path(main_checkpoint)
    paths={'main_checkpoint':main_checkpoint,'main_result':str(PurePosixPath(main_checkpoint).parent/'result.json'),
        'main_stage_manifest':MAIN_STAGE+'/manifest.json','pilot_manifest':PILOT_DATA+'/manifest.json',
        'sft_manifest':SFT_DATA+'/manifest.json'}
    if resume_sft:paths['resume_sft']=checkpoint_path(resume_sft)
    return {k:sha256(package_main.safe_path(root,v)) for k,v in paths.items()}


def evaluation_artifacts(root):
    plan,jobs=read_plan(root)
    values=sft_artifacts(root,plan['main_checkpoint'])
    values.update({'owned-base_checkpoint':values['main_checkpoint'],
        'owned-sft_checkpoint':sha256(package_main.safe_path(root,plan['sft_checkpoint']))})
    selected={'benchmark_manifest':BENCHMARKS+'/manifest.json','development_manifest':DEVELOPMENT+'/manifest.json',
        'task_definitions':DEFINITIONS+'/manifest.json','instruction_suite':INSTRUCTIONS+'/manifest.json'}
    selected.update({'job-'+item['id']:item['path'] for item in plan['jobs']})
    for model,directory in BASELINES.items():
        # This checks the complete safetensors/tokenizer/license snapshot against
        # pinned public content hashes; a user-supplied revision label is insufficient.
        snapshot=baselines.verify_snapshot(model,Path(root)/directory)
        for name in snapshot['files']:package_main.safe_path(root,directory+'/'+name)
        selected[model+'_snapshot']=directory+'/'+baselines.MANIFEST_NAME
    values.update({k:sha256(package_main.safe_path(root,v)) for k,v in selected.items()})
    evaluator.endpoint({'artifacts':values},local_paths(root,jobs[0]))
    return values


def joint_timing_gate(root,*,remaining_work_seconds):
    """All five dev runs must finish before any final scoring is authorized."""
    boundary.finite_number(remaining_work_seconds)
    plan,jobs=read_plan(root);estimates={}
    for job in jobs[5:10]:
        estimates[job['model']]=evaluator.development_timing(job,local_paths(root,job))
    required=sum(v['estimated_final_seconds_with_2x_headroom'] for v in estimates.values())+plan['fixed_extra_work_reserve_seconds']
    if required>remaining_work_seconds:
        raise ValueError('All mandatory final jobs plus fixed export/instruction reserve do not fit; no final scoring authorized')
    return {'format':'udgam-joint-final-timing-gate-v1','plan_sha256':sha256(Path(root)/JOB_DIR/'plan.json'),
        'estimates':estimates,'required_work_seconds_with_headroom':required,
        'remaining_work_seconds_at_gate':remaining_work_seconds,'fixed_extra_work_reserve_seconds':300,
        'all_mandatory_models_included':True,'quality_used_for_selection':False,
        'scope':'planning estimate, not guaranteed completion; hard common deadline still applies'}


def auxiliary_inventory(root,stage):
    """Exact nonsecret local additions; no whole-workspace recursive inventory."""
    if stage not in ('sft','evaluation'):raise ValueError('Unknown auxiliary stage')
    root=Path(root)
    names={*sft.source_hashes(),*evaluator.source_hashes(),'prepare_final_jobs.py','prepare_baselines.py'}
    paths={'scripts/'+name for name in names}
    paths|={'LICENSE','LICENSE_SCOPE.md','research/data_sources.json',PROTOCOL,
            'configs/gpu-requirements.txt','configs/baseline-eval-requirements.txt'}
    paths|={'configs/inference/'+n for n in ('README.md','requirements.txt','LICENSE.inference')}
    manifest=metadata(root,SFT_DATA+'/manifest.json')
    paths|={SFT_DATA+'/'+relative(n) for n in (*manifest['hashes'],'manifest.json','manifest.sha256')}
    # The data builder's existing verifier validates masks/splits/provenance; this
    # inventory only hashes its explicitly declared regular artifacts.
    for name,checksum in manifest['hashes'].items():
        if sha256(package_main.safe_path(root,SFT_DATA+'/'+name))!=checksum:raise ValueError('SFT auxiliary artifact changed')
    if stage=='evaluation':
        for folder in (BENCHMARKS,DEVELOPMENT):
            m=metadata(root,folder+'/manifest.json');paths.add(folder+'/manifest.json')
            if (root/folder/'manifest.sha256').is_file():paths.add(folder+'/manifest.sha256')
            for task,info in m['tasks'].items():
                if task not in evaluator.original.TASKS:raise ValueError('Unknown benchmark task')
                name=Path(info.get('file',info.get('local_file',''))).name
                full=folder+'/'+relative(name)
                if sha256(package_main.safe_path(root,full))!=info['sha256']:raise ValueError('Frozen benchmark artifact changed')
                paths.add(full)
        source=metadata(root,DEFINITIONS+'/manifest.json');paths.add(DEFINITIONS+'/manifest.json')
        for name,checksum in source['files'].items():
            full=DEFINITIONS+'/'+relative(name)
            if sha256(package_main.safe_path(root,full))!=checksum:raise ValueError('Frozen task definition changed')
            paths.add(full)
        paths|={INSTRUCTIONS+'/'+n for n in ('manifest.json','manifest.sha256','items.jsonl')}
        for model in baselines.MODELS:
            repo,_=baselines.MODELS[model];prefix='research/baseline-pins/'+repo.replace('/','--')
            paths|={prefix+'--blobs.json',prefix+'--config.json'}
        paths|={'research/baseline-pins/sarvamai--sarvam-1--LICENSE.md','research/baseline-pins/manifest.json'}
    files=[]
    for name in sorted(paths):
        path=package_main.safe_path(root,name)
        files.append({'path':name,'bytes':path.stat().st_size,'sha256':sha256(path)})
    return {'format':'udgam-final-auxiliary-inventory-v1','stage':stage,'helper_sha256':sha256(Path(__file__)),
        'files':files,'total_bytes':sum(f['bytes'] for f in files),'includes_model_weights':False,
        'includes_credentials':False,'baseline_weights_excluded_from_owned_release':True,
        'base_dependency':'unchanged sealed main input vendor/pilot data/current immutable stage and selected completed checkpoint are separate components'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('inventory','jobs'))
    p.add_argument('--root',type=Path,required=True);p.add_argument('--stage',choices=('sft','evaluation'))
    p.add_argument('--main-checkpoint');p.add_argument('--output',type=Path)
    a=p.parse_args()
    if a.command=='jobs':
        if not a.main_checkpoint:p.error('--main-checkpoint required')
        value=write_plan(a.root,a.main_checkpoint)
    else:
        if not a.stage or not a.output:p.error('--stage and --output required')
        if a.output.exists():raise FileExistsError('Inventory output already exists')
        value=auxiliary_inventory(a.root,a.stage);write_json(a.output,value)
    print(json.dumps({'format':value['format'],'files':len(value.get('files',[])),
        'jobs':len(value.get('jobs',[])),'total_bytes':value.get('total_bytes')},indent=2))
