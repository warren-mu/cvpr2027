"""New settled-baseline entry with clone candidates and selective raw retention.

First-round and clone-only entry points stay unchanged. No disk evidence is
deleted. Raw buffers are materialized for qualified pairs, the first three
examples of each complete rejection reason, and a terminal black-frame event.
"""
import copy
import io
import json
from pathlib import Path
import time

from . import pair_v2_clone_shard as clone
from . import pair_v2_shard as base
from . import pair_v2_simulator as engine
from . import pair_v2_policy as original_policy
from . import pair_v2_stable_policy as stable_policy
from .pair_v2_clone import clone_replay_worker
from .pair_v2_evidence import EvidenceSession, RetentionPolicy, compact_attempt
from .pair_v2_stable import StableCloneSimulator, PREFLIGHT_VERSION, PREFLIGHT_POLICY
from .pair_v2_stable import augment_physical_result
from .thor_common import Worker

_CLONE_ENABLE = clone.enable_config
_CLONE_GENERATOR = clone.generator_worker
_SESSION = None
_REPLAY_RAW_COMPLETE = None

RETENTION_POLICY = {
    'version': 'memory_verdict_retention_v1',
    'common_preflight_clone_baseline': 'complete raw including reusable native placement queries saved immediately',
    'attempt_raw': 'memory until verdict; qualified pair or first three occurrences of any complete rejection reason',
    'other_rejection': 'compact JSON key measurements plus one <=128x128 JPEG',
    'black_frame': 'terminal event always materializes complete available raw before exit86',
    'process_loss': 'unreturned child memory marked incomplete; never fabricated or counted as complete',
    'disk_deletion': False,
    'source_image_transfer': 'private child config bytes only; excluded from result config and FINAL metrics',
}


def enable_config(config):
    _CLONE_ENABLE(config)
    config['preflight_version'] = PREFLIGHT_VERSION
    config['preflight_policy'] = copy.deepcopy(PREFLIGHT_POLICY)
    config['evidence_version'] = stable_policy.EVIDENCE_VERSION
    config['evidence_policy'] = copy.deepcopy(stable_policy.EVIDENCE_POLICY)
    config['raw_evidence_policy'] = copy.deepcopy(RETENTION_POLICY)
    return config


class StablePreparationSimulator(clone.PreparationSimulator, StableCloneSimulator):
    """Preparation.super resolves to the NEW stable preflight through this MRO."""
    pass


def replay_worker(config, connection):
    # The bytes map is consumed privately and cannot leak into saved public
    # config. The original replay operates unchanged through a read-only view.
    config = copy.deepcopy(config)
    imported = config.pop('_private_source_image_bytes', {})
    session = EvidenceSession(config['runtime']['scratch'], buffer_all=True, imports=imported)
    old_store = engine.EvidenceStore
    result = None
    try:
        engine.EvidenceStore = session.store
        with session.reading():
            function = clone_replay_worker if config['snapshot'].get('clone_context') is not None else engine.replay_worker
            result = function(config, connection)
        if result.get('exit_code') == 86:
            result['raw_retention'] = session.finish(True)
            result['_private_evidence_bundle'] = None
        else:
            result['_private_evidence_bundle'] = session.bundle()
        result['buffered_raw_complete'] = True
        return result
    finally:
        engine.EvidenceStore = old_store


def wait_replay(config, snapshot, actions, target, identities, directory):
    global _REPLAY_RAW_COMPLETE
    if _SESSION is None or not _SESSION.active:
        raise RuntimeError('Fresh witness requires an active attempt evidence session')
    private = {}
    image_ref = (snapshot.get('capture_reference') or {}).get('files', {}).get('rgb')
    if image_ref:
        data = _SESSION.data(image_ref['file'])
        if data is not None:
            private[image_ref['file']] = data
    child_config = {**config, 'enable_candidate_clone': True, 'snapshot': snapshot, 'actions': actions,
        'target_name': target, **identities,
        'instruction': f"Select the {identities['target_type']} closest to the {identities['anchor_type']}.",
        'raw_dir': str(directory), '_private_source_image_bytes': private,
        # The child can catch its own deadline and return buffers before the
        # unchanged parent cleanup reserve. This is an infrastructure reserve.
        'deadline': config['deadline'] - 20}
    if child_config['deadline'] <= time.time():
        raise TimeoutError('Insufficient reserved time for fresh witness')
    child = Worker(replay_worker, child_config)
    result = None
    try:
        while result is None:
            child.receive()
            if child.result is not None:
                result = child.result
                bundle = result.pop('_private_evidence_bundle', None)
                if bundle is not None:
                    _SESSION.absorb(bundle)
                if result.get('buffered_raw_complete') is not True:
                    _REPLAY_RAW_COMPLETE = False
                    result.update(success=False, passed=False,
                                  evidence_error='Fresh witness did not return complete buffered raw evidence')
            elif child.error:
                _REPLAY_RAW_COMPLETE = False
                result = {'status': 'worker_failed', 'error': child.error, 'success': False,
                          'buffered_raw_complete': False}
            elif time.time() >= config['deadline'] - 10:
                _REPLAY_RAW_COMPLETE = False
                raise TimeoutError('Witness replay deadline before buffered result returned')
            else:
                time.sleep(.05)
    except BaseException:
        _REPLAY_RAW_COMPLETE = False
        raise
    finally:
        try:
            cleanup = child.stop()
        except BaseException as error:
            if clone._REPLAY_CLEANUP is not None:
                clone._REPLAY_CLEANUP.append(f'witness cleanup raised {type(error).__name__}: {error}')
            raise
        if clone._REPLAY_CLEANUP is not None:
            clone._REPLAY_CLEANUP.extend(cleanup)
    result.setdefault('cleanup_errors', []).extend(cleanup)
    if cleanup:
        result['success'] = False
    return result


def thumbnail(sim, directory):
    from PIL import Image
    capture = sim.last_capture
    if capture is None or capture.get('rgb') is None:
        return {'available': False, 'reason': 'no captured RGB before failure'}
    image = Image.fromarray(capture['rgb'])
    image.thumbnail((128, 128))
    data = io.BytesIO()
    image.save(data, format='JPEG', quality=70)
    path = Path(directory) / 'thumbnail.jpg'
    with path.open('xb') as stream:
        stream.write(data.getvalue())
    return {'available': True, 'file': str(path), 'sha256': base.sha(path),
            'width': image.width, 'height': image.height, 'source': 'last actual captured RGB; review only'}


def generator_worker(config, connection):
    global _SESSION, _REPLAY_RAW_COMPLETE
    if (config.get('preflight_version') != PREFLIGHT_VERSION or config.get('enable_candidate_clone') is not True
            or config.get('evidence_version') != stable_policy.EVIDENCE_VERSION):
        raise ValueError('Stable generator requires explicit preflight/evidence versions and clone opt-in')
    session = EvidenceSession(config['runtime']['scratch'])
    retention = RetentionPolicy()
    old = (engine.EvidenceStore, clone.PreparationSimulator, clone.wait_replay,
           base.build_pair, base.write_json, _SESSION, _REPLAY_RAW_COMPLETE, base.physical_measurement,
           base.choose_look, original_policy.classify_pair)
    retention_rows = []

    def build(sim, child_config, baseline, identities, proposal, view, pair_id, directory, reachable, attempt_row=None):
        global _REPLAY_RAW_COMPLETE
        row = {} if attempt_row is None else attempt_row
        error = None
        session.begin()
        _REPLAY_RAW_COMPLETE = True
        try:
            with session.reading():
                return old[3](sim, child_config, baseline, identities, proposal, view,
                              pair_id, directory, reachable, attempt_row=row)
        except BaseException as caught:
            error = caught
            raise
        finally:
            black = isinstance(error, engine.BlackFrame)
            keep, decision = retention.decide(row, error, black=black)
            row['raw_complete'] = bool(_REPLAY_RAW_COMPLETE)
            try:
                stored = session.finish(keep)
            except BaseException as storage_error:
                row.update(raw_complete=False, evidence_retention={**decision, 'full_raw_retained': False,
                    'storage_error': f'{type(storage_error).__name__}: {storage_error}',
                    'partial_disk_evidence_retained': True})
                # No more candidates after a partial evidence materialization.
                # Already written files stay; remaining bytes are memory only.
                session.pending.clear()
                session.active = False
                raise SystemExit('Evidence materialization failed; retain partial files') from storage_error
            row['evidence_retention'] = {**decision, **stored,
                'raw_complete': row['raw_complete'], 'omitted_payloads_existed_only_in_memory': not keep}
            row['thumbnail'] = thumbnail(sim, directory)
            retention_rows.append({'pair_id': pair_id, **row['evidence_retention']})

    def written(path, value):
        if Path(path).name == 'attempt.json' and value.get('status') != 'passed':
            if (value.get('evidence_retention') or {}).get('full_raw_retained') is False:
                value = compact_attempt(value)
        return old[4](path, value)

    def physical_measurement(result):
        return augment_physical_result(old[7](result), result['physical'])

    try:
        _SESSION = session
        engine.EvidenceStore = session.store
        clone.PreparationSimulator = StablePreparationSimulator
        clone.wait_replay = wait_replay
        base.build_pair, base.write_json = build, written
        base.physical_measurement = physical_measurement
        base.choose_look = stable_policy.choose_look
        original_policy.classify_pair = stable_policy.classify_pair
        result = _CLONE_GENERATOR(config, connection)
        result['preflight_version'] = PREFLIGHT_VERSION
        preflight_path = Path(config['raw_evidence_dir']) / 'construction' / 'scene_preflight.json'
        if preflight_path.is_file():
            preflight = json.loads(preflight_path.read_text())
            if preflight.get('preflight_version') != PREFLIGHT_VERSION:
                raise ValueError('Saved preflight differs from the explicitly selected stable version')
            ref = {'file': str(preflight_path), 'sha256': base.sha(preflight_path)}
            result['scene_preflight_reference'] = ref
            result['stable_preflight'] = {key: preflight.get(key) for key in ('preflight_version', 'passed', 'reasons', 'error')}
            if preflight.get('passed') is not True and not result.get('black_frame_detected'):
                result['scene_excluded'] = {'scene': config['scene'], 'reason': 'settled_baseline_window_preflight_failed',
                    'reasons': preflight.get('reasons'), 'error': preflight.get('error'), 'evidence': ref,
                    'preflight_version': PREFLIGHT_VERSION}
                result['scene_exclusion_count'] = 1
        result['evidence_version'] = stable_policy.EVIDENCE_VERSION
        result['evidence_policy'] = copy.deepcopy(stable_policy.EVIDENCE_POLICY)
        result['raw_evidence_policy'] = copy.deepcopy(RETENTION_POLICY)
        result['evidence_retention'] = {'attempts': retention_rows,
            'complete_rejection_reason_counts': dict(retention.reason_counts),
            'physical_rejection_distribution': dict(retention.physical_distribution),
            'physical_check_unavailable_rejections': retention.physical_check_unavailable_attempts,
            'distribution_denominator': 'attempts; each category counted at most once per attempt; categories may overlap',
            'disk_files_deleted': 0}
        return result
    finally:
        (engine.EvidenceStore, clone.PreparationSimulator, clone.wait_replay,
         base.build_pair, base.write_json, _SESSION, _REPLAY_RAW_COMPLETE, base.physical_measurement,
         base.choose_look, original_policy.classify_pair) = old


def main():
    original_generator, original_enable = clone.generator_worker, clone.enable_config
    original_writer = base.write_result

    def writer(name, metrics, config):
        enable_config(config)
        metrics['preflight_version'] = PREFLIGHT_VERSION
        metrics['evidence_version'] = stable_policy.EVIDENCE_VERSION
        return original_writer(name, metrics, config)

    try:
        clone.generator_worker, clone.enable_config = generator_worker, enable_config
        base.write_result = writer
        return clone.main()
    finally:
        clone.generator_worker, clone.enable_config = original_generator, original_enable
        base.write_result = original_writer


if __name__ == '__main__':
    raise SystemExit(main())
