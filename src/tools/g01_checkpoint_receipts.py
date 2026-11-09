"""Reuse published independent construction checks for quota counting only.

This validates the immutable checkpoint/source/pair receipts. It does not rerun
the native evidence tree, establish T02 acceptance, or authorize training. T02
continues to use its unchanged full source loader and native checks.
"""
import copy
from pathlib import Path

from src.envs import pair_table_source as sources
from src.envs import pair_v2_configuration as configurations
from src.envs import pair_v2_g01_policy as policy
from src.tools import merge_pair_v2 as merge
from src.tools.t01_partition import audited_result

VERSION = 'g01_published_independent_checkpoint_receipt_v1'
SCOPE = ('quota/configuration counting only: authenticated successful original independent construction receipt; '
         'native scientific checks are not repeated; no T02, human-review, or training permission inferred')


def bound_source(first, second):
    fields = ('file', 'sha256', 'metrics_sha256', 'git_commit')
    return (isinstance(first, dict) and isinstance(second, dict)
            and all(first.get(key) is not None and first.get(key) == second.get(key) for key in fields))


def load_checkpoint(expected, *, root=None):
    """Read each original scene result once; every pair keeps its original SHA.

    The call is intentionally stateless. No process-wide cache, success fallback,
    replacement source, or synthetic scientific result is created.
    """
    root = Path(merge.ROOT if root is None else root).resolve()
    container, actual_ref, entries = sources._container(root / expected['file'])
    if (any(actual_ref.get(key) != expected.get(key) for key in ('file', 'sha256'))
            or sources._checkpoint_module(container['_meta']['command']) != sources.G01_CHECKPOINT_MODULE):
        raise ValueError('Quota receipt requires the exact successful independent G01 checkpoint')
    metrics, config = container['metrics'], container['config']
    if (metrics.get('kind') != 't01v2_review_checkpoint' or metrics.get('errors') != []
            or metrics.get('status') != 'checkpoint_ready_for_human_review'
            or metrics.get('published_pair_count') != len(entries)
            or metrics.get('independently_verified_pair_count') != len(entries)):
        raise ValueError('Independent checkpoint receipt is failed or not completely published')
    source_refs, runs = container.get('sources'), metrics.get('source_runs')
    requested = config.get('requested_sources')
    if not isinstance(source_refs, list) or not isinstance(runs, list) or not isinstance(requested, list):
        raise ValueError('Checkpoint lacks its complete original source ledger')
    arguments = merge.producer_arguments(container['_meta']['command'], sources.G01_CHECKPOINT_MODULE)
    if arguments != ['--name', container['_meta']['name'], '--source', *requested]:
        raise ValueError('Checkpoint command differs from its explicit complete source cutoff')
    paths = [(root / name).resolve() for name in requested]
    if len(paths) != len(set(paths)) or not paths:
        raise ValueError('Checkpoint original source list is empty or duplicated')
    loaded, ledger = {}, {}
    for ref in source_refs:
        path = (root / ref['file']).resolve()
        if path in loaded or path not in paths:
            raise ValueError('Checkpoint source receipt is duplicate or outside requested cutoff')
        data, checked = audited_result(path)
        if not bound_source(ref, checked) or not data['_meta'].get('end'):
            raise ValueError('Original scene result changed since independent publication')
        if type(data['_meta'].get('exit_code')) is not int:
            raise ValueError('Original scene source has no final exit code')
        cfg = data['config']
        if cfg.get('producer_module') != policy.PRODUCER or cfg.get('g01_partition') != 'test':
            raise ValueError('Quota receipt source is not the explicit test G01 producer')
        arguments = merge.producer_arguments(data['_meta'].get('command'), policy.PRODUCER)
        if (merge.option(arguments, '--name') != cfg['name']
                or merge.option(arguments, '--scene') != cfg['scene']
                or merge.option(arguments, '--timeout-seconds') != '1200'):
            raise ValueError('Original source command differs from its published scene/name/grant')
        shard_ref = {'file': ref['shard_config_file'], 'sha256': ref['shard_config_sha256']}
        shard_path = merge.reference_path(shard_ref, root=root)
        if ((root / merge.option(arguments, '--shard-config')).resolve() != shard_path
                or merge.read_reference(shard_ref, root=root) != cfg.get('shard')):
            raise ValueError('Original source command/shard bytes differ from its immutable receipt')
        for filename, digest in (('log_file', 'log_sha256'), ('meta_file', 'meta_sha256')):
            merge.reference_path({'file': ref[filename], 'sha256': ref[digest]}, root=root, within=root / 'logs/runs')
        # Pure immutable policy/assignment checks only, not journal/science replay.
        policy.validate_source(cfg, data['metrics'])
        loaded[path] = (data, checked)
    if set(loaded) != set(paths):
        raise ValueError('Checkpoint omitted an originally requested source')
    for run in runs:
        ref = run.get('source') or {}
        path = (root / ref.get('file', '')).resolve()
        if path in ledger or path not in loaded or not bound_source(ref, loaded[path][1]):
            raise ValueError('Checkpoint source-run ledger differs from its authenticated sources')
        ledger[path] = run
    if set(ledger) != set(loaded):
        raise ValueError('Checkpoint has an incomplete source-run ledger')

    def read(ref, within):
        return merge.read_reference(ref, root=root, within=within)

    packages, identities = {}, set()
    counts = {kind: 0 for kind in merge.KINDS}
    reviews = metrics.get('review_files')
    if not isinstance(reviews, list) or len(reviews) != len(entries):
        raise ValueError('Checkpoint did not publish one original review image per candidate')
    for entry in entries:
        source_path = (root / entry['source_file']).resolve()
        if source_path not in loaded:
            raise ValueError('Candidate names a source absent from the complete checkpoint')
        data, source_ref = loaded[source_path]
        run, cfg = ledger[source_path], data['config']
        if (entry.get('source_sha256') != source_ref['sha256']
                or entry.get('source_metrics_sha256') != source_ref['metrics_sha256']
                or run.get('eligible') is not True or run.get('ledger_only') is not False
                or run.get('cleanup_errors') != [] or data['metrics'].get('cleanup_errors') != []):
            raise ValueError('Candidate lacks its eligible authenticated original source receipt')
        pair_ref = entry['pair_reference']
        pair_path = merge.reference_path(pair_ref, root=root, within=root / 'data/world_pairs/v2')
        expected_pair = root / 'data/world_pairs/v2' / cfg['name'] / entry['pair_id'] / 'pair.json'
        if pair_path != expected_pair or str(pair_path) in packages:
            raise ValueError('Pair receipt is duplicate or outside its exact original source namespace')
        row = read(pair_ref, pair_path.parent)
        if (any(row.get(key) != entry.get(key) for key in ('pair_id', 'scene', 'kind', 'target_w0', 'target_w1'))
                or row.get('status') != 'passed' or row.get('kind') not in counts
                or row.get('producer_module') != policy.PRODUCER or row.get('g01_partition') != 'test'
                or row.get('scene') != cfg['scene']):
            raise ValueError('Pair bytes differ from their independently published identity/class')
        checked = entry.get('independent_verification')
        if (not isinstance(checked, dict) or checked.get('g01_partition') != 'test'
                or checked.get('validity', {}).get('acceptance_passed') is not True
                or checked['validity'].get('kind') != row['kind']
                or checked['validity'].get('reasons') != []
                or checked.get('verification_scope') != 'construction-stage and applicable supplement 4(a); P-table/BFS/no-leak/cache checks not run'):
            raise ValueError('Candidate lacks its saved original independent construction verification')
        identity = merge.proposal_identity(row)
        if (entry.get('proposal_identity') != identity or checked.get('proposal_identity') != identity
                or identity in identities):
            raise ValueError('Published proposal identity differs from original pair or is duplicated')
        identities.add(identity)
        witnesses = checked.get('witnesses')
        if (not isinstance(witnesses, list) or len(witnesses) != 2
                or any(witness.get('passed') is not True or witness.get('target_name') != row[f'target_w{i}']
                       for i, witness in enumerate(witnesses))):
            raise ValueError('Saved independent witness receipts do not bind the pair targets')
        association = entry.get('controller_preflight_association')
        if (not isinstance(association, dict)
                or association != checked.get('controller_preflight_association')
                or association.get('version') != 'g01_pair_controller_preflight_association_v1'
                or association.get('controller_id') != row.get('search_controller_id')
                or association.get('source_record_mutated') is not False
                or not bound_source(association.get('source_reference'), source_ref)
                or association.get('pair_preflight_reference') != row.get('scene_preflight_reference')
                or association.get('attempt_setup_reference') != row.get('attempt_setup_reference')):
            raise ValueError('Saved Controller/native-source verification does not bind this original pair')
        if checked.get('review_image') != row.get('review_image') or entry.get('review_image') != row.get('review_image'):
            raise ValueError('Saved independent review image differs from the original pair')
        images = [image for image in reviews if image.get('source_file') == entry['source_file']
                  and image.get('pair_id') == entry['pair_id']]
        if (len(images) != 1 or images[0].get('scene') != row['scene'] or images[0].get('kind') != row['kind']
                or images[0].get('source_image') != row['review_image']
                or images[0].get('sha256') != row['review_image']['sha256']):
            raise ValueError('Checkpoint review publication does not bind the complete candidate list')
        merge.reference_path(row['review_image'], root=root, within=pair_path.parent)
        merge.reference_path(images[0], root=root, within=root / config['review_directory'])
        portable = row.get('portable_files')
        if not isinstance(portable, list):
            raise ValueError('Pair receipt lacks its original portable manifest')
        worlds = []
        for index in (0, 1):
            matches = [ref for ref in portable if Path(ref['file']).name == f'snapshot_w{index}.json']
            if len(matches) != 1:
                raise ValueError('Pair lacks one exact portable world snapshot per branch')
            worlds.append(read(matches[0], pair_path.parent))
        # Same canonical command checks as T02. No native readback is hashed.
        sources.canonical_start(row, worlds)
        configurations.from_pair(row, worlds)
        packages[str(pair_path)] = {'pair_id': row['pair_id'], 'scene': row['scene'], 'kind': row['kind'],
            'target_names': [row['target_w0'], row['target_w1']], 'row': row, 'worlds': worlds,
            'source_config': copy.deepcopy(cfg), 'independent_verification': copy.deepcopy(checked),
            'source_provenance': {'container': copy.deepcopy(actual_ref), 'scene': copy.deepcopy(source_ref),
                                  'pair': copy.deepcopy(pair_ref)},
            'count_basis_authority': VERSION, 'receipt_scope': SCOPE,
            'native_scientific_revalidation_performed': False, 'training_authorized': False}
        counts[row['kind']] += 1
    if metrics.get('published_by_class') != counts or metrics.get('independently_verified_by_class') != counts:
        raise ValueError('Published independent counts differ from the complete authenticated candidate list')
    return container, actual_ref, entries, packages
