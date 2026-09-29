"""Public, provider-specific final-stage contract; never a provider controller.

The trusted launcher supplies a durable single-use claim, enforces the platform
timeout, kills/reaps child processes and commits backups. This JSON cannot prove
those actions or enforce billing by itself. No credentials or provider calls.
"""
from __future__ import annotations
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import time
from types import SimpleNamespace

from prepare_dense_tokens import sha256
from train_dense import finite_number

FORMAT = 'udgam-modal-final-stage-v1'
RATE_FLOOR = .00118492
PROFILES = {
    'modal-conversation-v1': {'stage': 'conversation_training', 'ceiling': 4, 'timeout': 1800},
    'modal-evaluation-v1': {'stage': 'evaluation_export', 'ceiling': 5, 'timeout': 3000},
}
RESERVE_SECONDS = 600
OVERHEAD_POLICY_V2 = 'udgam-final-overhead-v2'
OVERHEAD_V2_CEILINGS = {'modal-conversation-v1': 2.90, 'modal-evaluation-v1': 4.35}


def budget_policy(profile, overhead_policy=None):
    """Version only the unused financial cushion; legacy contracts stay valid."""
    if profile not in PROFILES: raise ValueError('Unknown final-stage profile')
    policy = dict(PROFILES[profile], nonexecution_reserve_usd=1)
    if overhead_policy is not None:
        if overhead_policy != OVERHEAD_POLICY_V2: raise ValueError('Unknown nonexecution reserve policy')
        policy.update(ceiling=OVERHEAD_V2_CEILINGS[profile], nonexecution_reserve_usd=.40)
    return policy


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def read(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 256 * 1024:
        raise ValueError('Bounded regular execution contract required')
    def pairs(values):
        result = {}
        for key, value in values:
            if key in result: raise ValueError('Duplicate contract key')
            result[key] = value
        return result
    return json.loads(path.read_text(), object_pairs_hook=pairs)


def hashes(value):
    if not isinstance(value, dict) or not value or len(value) > 128:
        raise ValueError('Nonempty bounded artifact/source hash mapping required')
    for key, checksum in value.items():
        if (not isinstance(key, str) or not re.fullmatch(r'[A-Za-z0-9_.-]{1,128}', key) or
            not isinstance(checksum, str) or not re.fullmatch(r'[a-f0-9]{64}', checksum)):
            raise ValueError('Invalid artifact/source identity')


def validate(path, *, profile, source_hashes, now=None, environ=None):
    now = time.time() if now is None else now
    environ = os.environ if environ is None else environ
    finite_number(now)
    if profile not in PROFILES: raise ValueError('Unknown final-stage profile')
    v = read(path)
    versioned = 'nonexecution_reserve_policy' in v
    if versioned and v['nonexecution_reserve_policy'] != OVERHEAD_POLICY_V2:
        raise ValueError('Exact versioned nonexecution reserve policy required')
    policy = budget_policy(profile, v.get('nonexecution_reserve_policy'))
    fields = {'format','profile','stage','execution_id','app_id','function_call_id','task_id','attempt_claimed',
        'function_started_at','absolute_deadline_at','timeout_seconds','startup_timeout_seconds','retries',
        'max_containers','single_use_containers','gpu_type','gpu_count','cpu_cores','memory_mib',
        'max_session_usd','cash_ceiling_usd','rate_usd_per_second','hourly_total_usd','nonexecution_reserve_usd',
        'credit_only','user_cash_authorized','automatic_top_up','project_ceiling_usd',
        'project_consumed_and_reserved_before_usd','workspace_gross_ceiling_usd',
        'workspace_gross_consumed_and_reserved_before_usd','stage_ceiling_usd',
        'stage_gross_consumed_and_reserved_before_usd','credit_available_before_usd','credit_reserved_other_usd',
        'credit_observed_at','package_manifest_sha256','source_hashes','artifacts','finite_input','producer_enabled',
        'export_reserve_seconds','approved_jobs'}
    if versioned: fields.add('nonexecution_reserve_policy')
    if set(v) != fields or v['format'] != FORMAT or v['profile'] != profile or v['stage'] != policy['stage']:
        raise ValueError('Exact final-stage Modal contract required')
    for key in ('execution_id','app_id','function_call_id','task_id'):
        if not isinstance(v[key], str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,128}', v[key]):
            raise ValueError('Invalid Modal execution identity')
        env_key = 'UDGAM_MODAL_' + key.upper()
        if v[key] != environ.get(env_key): raise ValueError('Modal runtime identity mismatch')
    for key, expected in {'attempt_claimed':True,'single_use_containers':True,'credit_only':True,
        'user_cash_authorized':False,'automatic_top_up':False,'finite_input':True,'producer_enabled':False}.items():
        if v[key] is not expected: raise ValueError('Single-attempt/credit/finite policy mismatch')
    for key, expected in {'retries':0,'max_containers':1,'gpu_count':1,'cpu_cores':4,'memory_mib':16384,
                          'export_reserve_seconds':RESERVE_SECONDS}.items():
        if type(v[key]) is not int or v[key] != expected: raise ValueError('Resource/retry/backup policy mismatch')
    if v['gpu_type'] != 'H100!': raise ValueError('Exact H100 required')
    start, deadline, timeout, startup = [finite_number(v[k]) for k in
        ('function_started_at','absolute_deadline_at','timeout_seconds','startup_timeout_seconds')]
    if (not 0 < timeout <= policy['timeout'] or not 0 < startup <= 300 or start > now or
        not start < deadline <= start + timeout or now + RESERVE_SECONDS + 30 >= deadline):
        raise ValueError('Original deadline expired/extended or insufficient backup reserve')
    observed = finite_number(v['credit_observed_at'])
    if not 0 <= start - observed <= 900: raise ValueError('Prelaunch credit observation missing/stale')
    rate = finite_number(v['rate_usd_per_second']); hourly = finite_number(v['hourly_total_usd'])
    reserve = finite_number(v['nonexecution_reserve_usd']); ceiling = finite_number(v['max_session_usd'])
    nonnegative = ('project_consumed_and_reserved_before_usd','workspace_gross_consumed_and_reserved_before_usd',
        'stage_gross_consumed_and_reserved_before_usd','credit_reserved_other_usd','cash_ceiling_usd')
    for key in nonnegative: finite_number(v[key], positive=False)
    credit = finite_number(v['credit_available_before_usd'])
    if (v['cash_ceiling_usd'] != 0 or v['project_ceiling_usd'] != 99 or
        v['workspace_gross_ceiling_usd'] != 25 or v['stage_ceiling_usd'] != policy['ceiling'] or
        rate < RATE_FLOOR or abs(hourly - rate * 3600) > 1e-9 or
        (reserve != .40 if versioned else reserve < 1) or
        rate * (timeout + startup) + reserve > ceiling + 1e-9 or
        v['project_consumed_and_reserved_before_usd'] + ceiling > 99 or
        v['workspace_gross_consumed_and_reserved_before_usd'] + ceiling > 25 or
        v['stage_gross_consumed_and_reserved_before_usd'] + ceiling > policy['ceiling'] or
        credit < ceiling + v['credit_reserved_other_usd']):
        raise ValueError('Insufficient credit-only project/workspace/stage reservation')
    hashes(v['source_hashes']); hashes(v['artifacts'])
    hashes({'package': v['package_manifest_sha256']})
    if v['source_hashes'] != source_hashes: raise ValueError('Reviewed adapter/core source changed')
    jobs = v['approved_jobs']
    if (not isinstance(jobs, list) or not 1 <= len(jobs) <= 16 or len(jobs) != len(set(jobs)) or
        any(not isinstance(j, str) or not re.fullmatch(r'[a-z0-9_-]{1,64}', j) for j in jobs)):
        raise ValueError('Explicit unique bounded job list required')
    return v, deadline - RESERVE_SECONDS


def bind_artifacts(contract, paths):
    for key, path in paths.items():
        if key not in contract['artifacts'] or sha256(Path(path)) != contract['artifacts'][key]:
            raise ValueError('Selected artifact differs from execution: ' + key)


class MonotonicWall:
    def __init__(self, clock=time):
        self.clock = clock; self.wall = clock.time(); self.mono = clock.monotonic()
    def time(self): return max(self.clock.time(), self.wall + self.clock.monotonic() - self.mono)
    def __getattr__(self, key): return getattr(self.clock, key)


def isolated(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def proxy(module, **overrides):
    return SimpleNamespace(**{**vars(module), **overrides})
