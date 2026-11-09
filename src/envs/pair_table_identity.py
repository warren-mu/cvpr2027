"""Dataset namespace; original pair IDs, label seeds and native proof stay unchanged."""
import hashlib
import json
import re

VERSION = 'pair_source_dataset_key_v1'


def identity_fields(original_pair_id, pair_reference):
    if (not isinstance(original_pair_id, str) or not original_pair_id
            or not isinstance(pair_reference, dict)
            or not isinstance(pair_reference.get('file'), str) or not pair_reference['file']
            or re.fullmatch(r'[0-9a-f]{64}', str(pair_reference.get('sha256'))) is None):
        raise ValueError('Dataset identity requires the original pair ID and exact source file/SHA')
    origin = {'original_pair_id': original_pair_id,
              'pair_reference': {key: pair_reference[key] for key in ('file', 'sha256')}}
    content = json.dumps(origin, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return {'dataset_identity_version': VERSION, **origin,
            'dataset_key': 'pair_' + hashlib.sha256(content).hexdigest()}


def population_key(value):
    key = value.get('dataset_key') or value.get('pair_id')
    if not isinstance(key, str) or not key:
        raise ValueError('Missing explicit dataset or original pair identity')
    return key


def resolve_table(tables, key):
    """Unique old-ID callers remain valid; ambiguous IDs never select a source."""
    rows = list(tables)
    exact = [table for table in rows if table.dataset_key == key]
    matches = exact or [table for table in rows if table.pair_id == key]
    if len(matches) != 1:
        raise ValueError('Dataset key is missing or original pair ID is ambiguous')
    return matches[0]
