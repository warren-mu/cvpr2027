"""Review-only submit template; execution is explicit and restricted to one PBS job.

Default mode only prints the original submitter argv. Root chooses the bounds;
they are abort limits, not measured authentication or training runtimes.
No parallel source workers, qsub call, native-science bypass flags, or changes
to the original training method are introduced by this one-off orchestration.

After population and holdout succeed, NEVER rerun the combined sequence merely
because training failed. Preview a direct retry using --direct-arm one-update,
--train-name <new engineering name>, --qualified-result-json <original ref>,
--holdout-reference-json <original ref>, and --train-seconds <explicit bound>.
The same preview accepts T0/T4/T2 after the complete engineering update has been
reviewed. It only reads original receipt/manifest metadata and prints argv; it
does not rebuild population, freeze, submit, or load cached RGB/mask payloads.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

NAME = 'smoke_e2e_pilot_certificates_update_01'
TRAIN_NAME = 'smoke_e2e_pilot_engineering_T4_01'
MODEL = {'file': 'results/smoke_e2e_eval_03.json',
    'sha256': 'd9658c976c4912fb6e07e6ed5debba591614a44f28664b4da9924faa9b43e4d9'}
GATE_SECONDS, CLEANUP_GRACE_SECONDS, RECORDING_RESERVE_SECONDS = 180, 4, 60


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'))


def logged(name, command):
    return ['python3', '-u', 'scripts/run_logged.py', '--name', name, '--seed', '0', '--', *command]


def file_reference(path):
    path = Path(path)
    return {'file': path.as_posix(), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def frozen_reference_from_receipt(path):
    # Preserve save_json's original absolute file string. The strict freeze
    # loader below authenticates the receipt and compares that exact identity;
    # constructing a new relative reference here would change its identity.
    receipt = json.loads(Path(path).read_bytes())
    meta, metrics = receipt['_meta'], receipt['metrics']
    if not (meta.get('end') and meta.get('exit_code') == 0 and metrics.get('passed') is True
            and metrics.get('producer_exit_code') == 0 and metrics.get('errors') == []):
        raise ValueError('Require the successful original freeze receipt before training')
    reference = metrics['frozen_holdout']
    if not (isinstance(reference, dict) and isinstance(reference.get('file'), str)
            and re.fullmatch(r'[0-9a-f]{64}', str(reference.get('sha256')))):
        raise ValueError('Original freeze receipt has no exact file/SHA reference')
    return dict(reference)


def run_checked(command):
    print('ORIGINAL_STAGE_COMMAND_JSON: ' + encoded(command), flush=True)
    code = subprocess.run(command).returncode
    if code:
        raise SystemExit(code)  # Preserve original gate86, timeout and science failures.


def training_argv(name, arm, manifest_ref, holdout_ref, seconds, *, engineering=False):
    pattern = r'smoke_e2e_pilot_engineering_T4_[A-Za-z0-9_]+' if engineering else r'smoke_e2e_pilot_[A-Za-z0-9_]+'
    if not re.fullmatch(pattern, name) or arm not in ('T0', 'T4', 'T2') or (engineering and arm != 'T4'):
        raise ValueError('Use the original arm and distinct engineering/formal run namespaces')
    if not engineering and name.startswith('smoke_e2e_pilot_engineering_'):
        raise ValueError('Formal runs must not use the engineering namespace')
    command = ['python3', '-u', '-m', 'src.tools.smoke_e2e_pilot', 'run', '--name', name, '--arm', arm,
        '--manifest', encoded(manifest_ref), '--holdout-reference', encoded(holdout_ref), '--model-source', encoded(MODEL),
        '--pilot-spec', 'configs/smoke_e2e_pilot.json', '--timeout-seconds', str(seconds),
        '--gate-timeout-seconds', str(GATE_SECONDS)]
    # Preserve the original driver/default template: critic_offload=False for
    # engineering and every formal arm. There is no approved true setting here.
    if engineering: command += ['--engineering-smoke', 't4_one_update_engineering_v1']
    return command


def submit_preview(name, payload, seconds):
    if not 0 < seconds <= 4 * 3600:
        raise ValueError('Original T01 submitter per-job stop bound is at most4h')
    wall = f'{seconds // 3600:02}:{seconds % 3600 // 60:02}:{seconds % 60:02}'
    submit = ['python3', 'scripts/pbs/submit_phase.py', '--name', name, '--task', 'T01', '--budget-group', 'E2Epilot',
        '--gpus', '1', '--walltime', wall, '--', *payload]
    diagnostic = 'diag_submit_' + name
    command = logged(diagnostic, ['python3', '-m', 'src.tools.record_command', '--name', diagnostic, '--', *submit])
    return {'submit_argv': command, 'wall_seconds_stoploss': seconds, 'walltime': wall,
        'allocation_gpu_hours_upper_bound_before_actual_rate': seconds / 3600,
        'critic_offload': False, 'same_gpu_job': True,
        'budget': 'Original submitter fresh E2Epilot/shared checks; actual charged rate/accounting unchanged',
        'interpretation': 'PBS abort bound, not a speed forecast or guaranteed cleanup/write completion; original stage timers do not bound final metadata I/O',
        'not_executed': True}


def direct_preview(args):
    sys.path.insert(0, str(Path.cwd()))
    from src.tools import smoke_e2e_pilot_certificates as certificates
    from src.tools import train_holdout_freeze as holdout
    if not args.train_name or not args.qualified_result_json or not args.holdout_reference_json:
        raise ValueError('Direct training requires a unique name and exact successful population/holdout references')
    result_ref = json.loads(args.qualified_result_json)
    holdout_ref = json.loads(args.holdout_reference_json)
    manifest, proof, _, manifest_ref = certificates.authenticate_certificate_qualification(result_ref)
    frozen, verified_holdout_ref = holdout.load_frozen_holdout(holdout_ref)
    holdout.assert_pilot_split(manifest['entries'], manifest['split'], frozen)
    if Path('results', args.train_name + '.json').exists() or Path('logs/runs', args.train_name).exists():
        raise ValueError('Training evidence already exists; retain it and select a new training run name')
    engineering = args.direct_arm == 'one-update'
    arm = 'T4' if engineering else args.direct_arm
    train = training_argv(args.train_name, arm, manifest_ref, verified_holdout_ref, args.train_seconds,
                          engineering=engineering)
    result = submit_preview(args.train_name, train,
        args.train_seconds + CLEANUP_GRACE_SECONDS + RECORDING_RESERVE_SECONDS)
    result.update(training_argv=train, qualification_result=result_ref, manifest_reference=manifest_ref,
        holdout_reference=verified_holdout_ref, common_pool_hash=proof['common_pool_hash'],
        population_rebuilt=False, holdout_refrozen=False,
        extra_source_reauthentication=manifest['qualification']['extra_source_reauthentication'],
        formal_launch_condition='Root reviews one complete T4 engineering update before T0/T4, then T2; preview is not launch authorization')
    return result


def run_in_pbs(args):
    sys.path.insert(0, str(Path.cwd()))
    from src.tools import smoke_e2e_pilot_certificates as certificates
    from src.tools import smoke_e2e_pilot_dataset as dataset
    from src.tools import train_holdout_freeze as holdout
    if not re.fullmatch(r'\d+\.hopper-m-\d+', os.environ.get('PBS_JOBID', '')) or not os.environ.get('CVPR_SCRATCH'):
        raise ValueError('This sequence may execute only in its original bounded PBS allocation')
    # A job that starts before the authorized time must fail before rendering.
    policy = certificates.policy()
    certificates.activate(policy['cache_sources'])
    run_checked(logged(NAME + '_gate', ['python3', '-u', '-m', 'src.tools.render_gate', '--name', NAME + '_gate',
        '--gpus', '1', '--resolution', '255', '--segmentation', '--timeout-seconds', str(GATE_SECONDS)]))
    population = ['python3', '-u', '-m', 'src.tools.smoke_e2e_pilot_dataset', '--name', certificates.NAME]
    for ref in policy['cache_sources']: population += ['--cache-source', encoded(ref)]
    population += ['--split-policy', dataset.SCENE_SPLIT, '--certificates-after-02', '--out', certificates.MANIFEST_FILE,
        '--timeout-seconds', str(args.population_seconds)]
    run_checked(logged(certificates.NAME, population))
    result_ref = file_reference(certificates.RESULT_FILE)
    # Original logged metadata receipt validation, not another cached/native scan.
    manifest, proof, _, manifest_ref = certificates.authenticate_certificate_qualification(result_ref)
    freeze = logged('train_holdout_freeze', ['python3', '-u', '-m', 'src.tools.train_holdout_freeze',
        '--name', 'train_holdout_freeze', '--qualification-result', encoded(result_ref), '--certificates-after-02'])
    # The freeze producer is metadata-only and spawns no renderer/model/children.
    # GNU timeout bounds this original logged stage, preserving any failure.
    run_checked(['timeout', '--signal=TERM', '--kill-after=20s', str(args.freeze_seconds) + 's', *freeze])
    holdout_ref = frozen_reference_from_receipt(Path('results') / (holdout.NAME + '.json'))
    frozen, holdout_ref = holdout.load_frozen_holdout(holdout_ref)
    holdout.assert_pilot_split(manifest['entries'], manifest['split'], frozen)
    train = training_argv(TRAIN_NAME, 'T4', manifest_ref, holdout_ref, args.train_seconds, engineering=True)
    print('FROZEN_TRAINING_INPUTS_JSON: ' + encoded({'manifest': manifest_ref, 'holdout': holdout_ref,
        'qualification_result': result_ref, 'common_pool_hash': proof['common_pool_hash'],
        'extra_source_reauthentication': manifest['qualification']['extra_source_reauthentication'],
        'critic_offload': False}), flush=True)
    # This original driver runs its own gate and reloads certified cached bytes.
    run_checked(logged(TRAIN_NAME, train))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--population-seconds', type=int)
    parser.add_argument('--train-seconds', type=int, required=True)
    parser.add_argument('--freeze-seconds', type=int, default=120)
    parser.add_argument('--run-in-pbs', action='store_true')
    parser.add_argument('--direct-arm', choices=('one-update', 'T0', 'T4', 'T2'))
    parser.add_argument('--train-name')
    parser.add_argument('--qualified-result-json')
    parser.add_argument('--holdout-reference-json')
    args = parser.parse_args()
    if args.direct_arm:
        if args.run_in_pbs or args.population_seconds is not None or not 300 <= args.train_seconds <= 14336:
            raise ValueError('Direct mode is metadata-only preview; pass only a training bound plus immutable inputs')
        print(json.dumps(direct_preview(args), sort_keys=True, indent=2)); return
    if not (args.population_seconds is not None and 60 <= args.population_seconds <= 1800
            and 300 <= args.train_seconds <= 3600 and 30 <= args.freeze_seconds <= 180):
        raise ValueError('Explicit bounded stages are required; these are stop-loss limits, not speed forecasts')
    if args.run_in_pbs:
        run_in_pbs(args)
        return
    body = Path(__file__).read_text()
    stage = ['python3', '-c', body, '--population-seconds', str(args.population_seconds),
        '--train-seconds', str(args.train_seconds), '--freeze-seconds', str(args.freeze_seconds), '--run-in-pbs']
    # The first gate's180 timer stops BEFORE stop_workers'4s cleanup. The
    # training driver's own gate is within its total timer; its final owned
    # worker cleanup has another4s grace. Result/log I/O has no strict wall
    # guarantee;60s is a recording reserve and PBS remains the final stop bound.
    seconds = GATE_SECONDS + CLEANUP_GRACE_SECONDS + args.population_seconds + args.freeze_seconds + 20 \
        + args.train_seconds + CLEANUP_GRACE_SECONDS + RECORDING_RESERVE_SECONDS
    result = submit_preview(NAME, ['python3', '-m', 'src.tools.record_command', '--name', NAME, '--', *stage], seconds)
    result.update(earliest_activation='2026-11-10T07:00:00Z', bounds={
        'initial_gate_main_seconds': GATE_SECONDS, 'initial_gate_cleanup_grace_seconds': CLEANUP_GRACE_SECONDS,
        'population_main_seconds': args.population_seconds, 'freeze_timeout_seconds': args.freeze_seconds,
        'freeze_kill_grace_seconds': 20, 'training_main_including_own_gate_seconds': args.train_seconds,
        'training_cleanup_grace_seconds': CLEANUP_GRACE_SECONDS, 'recording_reserve_seconds': RECORDING_RESERVE_SECONDS},
        retry_instruction='If population/freeze succeeded, use direct-arm with their original exact refs; never repeat this combined build')
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == '__main__': main()
