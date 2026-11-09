"""Fixed Look follow-up plan for E2E steps 1–3, never a training permission.

The user's prior Look follow-up instruction supplies execution scope. It does
not assert that Claude reviewed these new images. Full source/science checks
remain in load_reviewed_pair; the twenty-state exception remains smoke-only.
"""
from pathlib import Path

from src.tools.t01_partition import audited_result
from src.tools.merge_pair_v2 import scene_number

VERSION = 'smoke_e2e_look_five_v1'
CHECKPOINT = 'results/t01v2_checkpoint_flip_geometry_01.json'
CHECKPOINT_METRICS_SHA256 = 'ebc796436ade0fba6f783be468463b0a10ab41f3e0eb9c226bff28099b1fe001'
SCENE_SOURCE = 'results/pilot_pairs_v2_flip60_fp203_s15_r01_j707737_hopper_m_02_g0_floorplan203.json'
PAIR_IDS = ('floorplan203_00002', 'floorplan203_00004', 'floorplan203_00010',
            'floorplan203_00020', 'floorplan203_00024')
IMAGE_HASHES = (
    '125090db6f2340a56e11e342c1aa3d0518f49acbe8c384b1856e2a3570ccd48f',
    '4755f1ecac1f3cc342d556e24d45e3abee1648eb2f9c6af873036e5df703887e',
    '5fe0aff8eb55a1e4a0b77287badea7e67daf2c211ad868d562c41327c5d0e723',
    '4beaeaaf6a8d81d72227f8cf511a47f16074531c05b6761d941231a7e0eb1fae',
    '98805597025c9365e09d472a4d6f3e92ab2382999446194671816a353eead4c9',
)
AUTHORIZATION = {
    'status': 'prior_human_authorized_Look_followup',
    'basis': 'User instruction: Look pairs available -> rerun the E2E Look follow-up',
    'Claude_review_of_these_images': 'not_confirmed',
    'GPU_submission': 'root_controlled_after_budget_check',
    'scope': 'steps_1_to_3_only; no training-method choices or training authorization',
}
PLANNING = {
    'gpus_per_PBS_job': 1, 'total_E2E_charged_GPU_hours_limit': 4,
    'depth': 4, 'max_steps': 5, 'max_model_calls': 5,
    'cache_distinct_pair_states': 20, 'formal_T02_cache_minimum': 200,
    'PBS_submission_and_cumulative_budget_guard': 'root_owned_required_before_submission',
    'step_3': 'blocked_until_actual_steps_1_and_2_pass',
    'steps_4_and_5': 'not_authorized_by_this_plan; unresolved_Look_training_contract',
    'old_Direct_T0_is_same_pool_baseline': False,
}


def expected_pairs():
    return [{'pair_id': pair_id, 'kind': 'look_flip',
             'review_image': f'logs/review/t01v2_checkpoint_flip_geometry_01/{index:05d}_look_flip.png',
             'review_image_sha256': IMAGE_HASHES[index]}
            for index, pair_id in enumerate(PAIR_IDS)]


def review_packages(plan, reference, *, root, file_ref, load_pair):
    """Authenticate the fixed first five, then run the unchanged source loader."""
    if (plan.get('version') != VERSION or plan.get('checkpoint') != CHECKPOINT
            or plan.get('checkpoint_metrics_sha256') != CHECKPOINT_METRICS_SHA256
            or plan.get('scene_source') != SCENE_SOURCE
            or plan.get('authorization') != AUTHORIZATION or plan.get('planning') != PLANNING
            or plan.get('selection') != 'first_five_unique_Look_in_checkpoint_canonical_order'
            or plan.get('coverage') != {'included': ['look_flip'], 'not_included': ['look_stay', 'direct_flip', 'direct_stay']}
            or plan.get('pairs') != expected_pairs()):
        raise ValueError('Only the frozen first-five Look E2E follow-up plan is authorized')
    root = Path(root).resolve()
    data, checkpoint_ref = audited_result(root / CHECKPOINT)
    metrics, meta = data['metrics'], data['_meta']
    if (checkpoint_ref.get('metrics_sha256') != CHECKPOINT_METRICS_SHA256
            or (root / checkpoint_ref['file']).resolve() != root / CHECKPOINT
            or meta.get('exit_code') != 0 or not meta.get('end')
            or metrics.get('kind') != 't01v2_review_checkpoint'
            or metrics.get('status') != 'checkpoint_ready_for_human_review'
            or metrics.get('errors') != [] or metrics.get('final_pilot_acceptance') is not False):
        raise ValueError('Look plan requires its exact successfully ended independent checkpoint')
    candidates = metrics.get('candidate_pairs', [])
    ordered = sorted(candidates, key=lambda row:
                     (scene_number(row['scene']), row['journal_index'], row['source_file']))
    if (candidates != ordered or len({row['proposal_identity'] for row in candidates}) != len(candidates)):
        raise ValueError('Checkpoint is not a canonical unique independent candidate list')
    selected = [row for row in ordered if row['kind'] in ('look_flip', 'look_stay')][:5]
    if tuple(row['pair_id'] for row in selected) != PAIR_IDS:
        raise ValueError('Look follow-up is not the fixed first five; no filtering or replacements')
    packages, images, sources = [], [], [checkpoint_ref]
    for frozen, entry in zip(plan['pairs'], selected):
        if (entry['kind'] != frozen['kind'] or entry['scene'] != 'FloorPlan203'
                or (root / entry['source_file']).resolve() != root / SCENE_SOURCE):
            raise ValueError('Frozen Look pair kind/scene/source identity changed')
        reviews = [row for row in metrics.get('review_files', [])
                   if row.get('pair_id') == frozen['pair_id'] and row.get('source_file') == entry['source_file']]
        if (len(reviews) != 1 or reviews[0].get('file') != frozen['review_image']
                or reviews[0].get('kind') != frozen['kind']
                or reviews[0].get('sha256') != frozen['review_image_sha256']
                or reviews[0].get('source_image') != entry['review_image']):
            raise ValueError('Review image is not the checkpoint-published image for this pair')
        image = file_ref(root / frozen['review_image'], root / 'logs/review', 8 * 1024 * 1024)
        if image['sha256'] != frozen['review_image_sha256']:
            raise ValueError('Look review image bytes changed')
        images.append(image)
        package = load_pair(root / CHECKPOINT, frozen['pair_id'], pair_file=entry['pair_reference']['file'])
        if (package['pair_id'] != frozen['pair_id'] or package['kind'] != frozen['kind']
                or package['source_provenance']['pair'] != entry['pair_reference']
                or (root / package['source_provenance']['scene']['file']).resolve() != root / SCENE_SOURCE
                or package['independent_verification']['proposal_identity'] != entry['proposal_identity']):
            raise ValueError('Independent loader returned another Look pair or proposal identity')
        for key in ('file', 'sha256', 'metrics_sha256', 'git_commit'):
            if package['source_provenance']['container'].get(key) != checkpoint_ref.get(key):
                raise ValueError('Look package is not bound to the exact checkpoint bytes')
        packages.append(package)
        for source in package['sources']:
            if source not in sources:
                sources.append(source)
    return plan, reference, images, packages, sources
