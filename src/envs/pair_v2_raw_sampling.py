"""Future-only rejected-attempt retention; scientific decisions are unchanged."""
import copy
import hashlib
import json
import os
import re
from pathlib import Path
import stat
from contextlib import contextmanager
from . import pair_v2_search_shard as search

VERSION = 'a0076849_rejected_raw_seed0_tenth_v1'
SEED = 0
FIELDS = ('raw_sampling_version', 'raw_sampling_policy', 'raw_sampling_seed')
POLICY = {
    'version': VERSION, 'seed': SEED, 'rejected_probability': {'numerator': 1, 'denominator': 10},
    'draw': 'SHA256 of canonical seed,scene,zero-based all-build attempt ordinal; integer below floor(2**256/10)',
    'accepted': 'all original raw bytes required before admission; capacity/IO failures cannot pass',
    'rejected': 'confirmed scientific rejection without exception/error only; full counts/reasons and compact records always; selected raw materialized after verdict only',
    'non_scientific_failure': 'all raw requested for exceptions, explicit errors, failed/error/unknown status or unclassified rejection; original capacity/IO guards unchanged',
    'exceptions': 'original bounded native black terminal proof and shared Controller/native-query identity evidence retained',
    'omitted': 'never materialized; no disk deletion; dangling raw references removed from compact rejection',
    'scope': 'storage only; no change to scene/candidate/search RNG/threshold or independent scientific verification',
}

def fields():
    return dict(zip(FIELDS, (VERSION, copy.deepcopy(POLICY), SEED)))

def validate_fields(value):
    if type(value.get('raw_sampling_seed')) is not int or any(value.get(key) != expected for key, expected in fields().items()):
        raise ValueError('Missing/altered fixed rejected-raw sampling declaration')

def draw(scene, ordinal):
    if not isinstance(scene, str) or not scene or type(ordinal) is not int or ordinal < 0:
        raise ValueError('Rejected sampling requires actual scene and attempt ordinal')
    key = {'seed': SEED, 'scene': scene, 'attempt_ordinal': ordinal}
    digest = hashlib.sha256(json.dumps(key, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    return {'key': key, 'sha256': digest, 'selected': int(digest, 16) < (1 << 256) // 10}

class SampledRetention(search.SearchRetention):
    def __init__(self, scene):
        super().__init__(); self.scene = scene; self.ordinal = 0; self.last_keep = None

    def decide(self, row, error=None, *, black=False):
        # Keep original complete reason/category accounting and terminal guard.
        _, prior = super().decide(row, error, black=black)
        sample = draw(self.scene, self.ordinal); self.ordinal += 1
        exceptional = error is not None or bool(row.get('error') or row.get('errors') or row.get('cleanup_errors'))
        passed = row.get('status') == 'passed' and not exceptional
        scientific_rejection = (row.get('status') == 'rejected' and not exceptional
                                and bool(prior['rejection_reasons']) and not black)
        keep = bool(passed or black or not scientific_rejection or sample['selected'])
        self.last_keep = keep
        return keep, {**prior, **fields(), 'raw_sample': sample,
            'raw_sampling_exception_raised': error is not None,
            'reason': 'black_frame_stop' if black else 'qualified_pair' if passed else
                'non_scientific_or_unclassified_failure' if not scientific_rejection else
                'seeded_rejected_raw_sample' if keep else 'compact_rejection',
            'sampling_applied_to_rejection': scientific_rejection}

def compact_without_missing_references(value):
    """Use real os.stat, not EvidenceSession's in-memory Path.stat view."""
    if isinstance(value, dict):
        filename = value.get('file', value.get('path'))
        if isinstance(filename, str) and isinstance(value.get('sha256'), str):
            try:
                materialized = stat.S_ISREG(os.stat(filename).st_mode)
            except FileNotFoundError:
                materialized = False
            if not materialized:
                return {'raw_payload_omitted': True, 'unmaterialized_sha256': value['sha256'],
                        **({'unmaterialized_bytes': value['bytes']} if 'bytes' in value else {})}
        return {key: compact_without_missing_references(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [compact_without_missing_references(item) for item in value]
    return copy.deepcopy(value)

def validate_decision(row, *, raw_path=None):
    record = row.get('evidence_retention')
    if not isinstance(record, dict):
        raise ValueError('Sampled source attempt lacks retention decision')
    validate_fields(record)
    sample = record.get('raw_sample') or {}
    expected = draw(row.get('scene'), (sample.get('key') or {}).get('attempt_ordinal'))
    if sample != expected:
        raise ValueError('Raw sample is not the declared deterministic draw')
    if raw_path is not None:
        trial = re.fullmatch(re.escape(row.get('pair_id', '')) + r'_trial([0-9]{6,})', Path(raw_path).parent.name)
        if trial is None or int(trial.group(1)) != expected['key']['attempt_ordinal']:
            raise ValueError('Raw sampling ordinal differs from its actual immutable trial path')
    if row.get('status') == 'passed':
        if (record.get('reason') != 'qualified_pair' or record.get('retention_requested') is not True
                or record.get('full_raw_retained') is not True or record.get('evidence_complete_for_acceptance') is not True):
            raise ValueError('Accepted pair lost mandatory original raw evidence')
    elif record.get('sampling_applied_to_rejection') is True:
        if (row.get('status') != 'rejected' or record.get('raw_sampling_exception_raised') is not False
                or row.get('error') or row.get('errors') or row.get('cleanup_errors')
                or not record.get('rejection_reasons')):
            raise ValueError('Only a confirmed scientific rejection may claim raw sampling')
        if record.get('retention_requested') is not expected['selected']:
            raise ValueError('Rejected raw retention disagrees with fixed draw')
    elif (record.get('sampling_applied_to_rejection') is not False
          or record.get('retention_requested') is not True):
        raise ValueError('Non-scientific/unknown failure must request complete raw retention')

@contextmanager
def installed(config):
    enabled = config is not None and config.get(FIELDS[0]) is not None
    if not enabled:
        yield; return
    validate_fields(config)
    original, thumbnail = search.SearchRetention, search.stable.thumbnail
    state = {}
    def factory():
        if state:
            raise RuntimeError('One sampling sequence is required per scene worker')
        state['policy'] = SampledRetention(config['scene'])
        return state['policy']
    def kept_thumbnail(*args, **kwargs):
        decision = state.get('policy')
        if decision is None or decision.last_keep is None:
            raise RuntimeError('Thumbnail requested before its retention verdict')
        return thumbnail(*args, **kwargs) if decision.last_keep else None
    search.SearchRetention, search.stable.thumbnail = factory, kept_thumbnail
    try:
        yield
    finally:
        search.SearchRetention, search.stable.thumbnail = original, thumbnail
