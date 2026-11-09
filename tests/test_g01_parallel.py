"""CPU-only resource/provenance fixtures; no rendering or synthetic throughput."""
import copy
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.tools import g01_admission as admission
from src.tools import g01_parallel as parent
from src.envs import pair_v2_stable_shard as stable

GPU = 'GPU-11111111-2222-3333-4444-555555555555'


class LauncherTests(unittest.TestCase):
    def test_actual_gate_and_scene_namespace_use_original_logged_child_contract(self):
        with tempfile.TemporaryDirectory() as text:
            root = Path(text)
            process = SimpleNamespace(pid=123)
            for name, module in (
                    ('gen_train_fixture_j700000_hopper_m_02_gate', 'src.tools.render_gate'),
                    ('gen_train_fixture_c2_floorplan203', parent.policy.PRODUCER),
                    ('gen_test_fixture_j700001_hopper_m_02_gate', 'src.tools.render_gate'),
                    ('gen_test_fixture_c6_floorplan221', parent.policy.PRODUCER)):
                command = ['python3', '-u', '-m', module, '--name', name]
                with patch.object(parent, 'ROOT', root), patch.object(parent.subprocess, 'Popen', return_value=process) as popen:
                    running = parent.launch_logged(name, command, {'CUDA_VISIBLE_DEVICES': GPU}, root)
                args, kwargs = popen.call_args
                self.assertEqual(args[0], [parent.sys.executable, '-u', str(root / 'scripts/run_logged.py'),
                                          '--name', name, '--seed', '0', '--', *command])
                self.assertIs(running['process'], process)
                self.assertTrue(kwargs['start_new_session'])
                self.assertEqual(kwargs['cwd'], root)
                self.assertEqual(kwargs['env']['CVPR_RUN_NAME'], name)
                self.assertEqual(kwargs['env']['CVPR_RESULTS_PATH'], str(root / 'results' / (name + '.json')))
                self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'], GPU)
                running['stream'].close()

    def test_every_existing_evidence_path_and_invalid_name_blocks_dispatch(self):
        for suffix in ('result', 'directory', '.log', '.meta.json', '.diff', 'launcher'):
            with tempfile.TemporaryDirectory() as text:
                root = Path(text); name = 'gen_train_fixture'
                (root / 'results').mkdir(); (root / 'logs/runs').mkdir(parents=True)
                path = (root / 'results' / (name + '.json') if suffix == 'result' else
                        root / 'logs/runs' / name if suffix == 'directory' else
                        root / (name + '.launcher.log') if suffix == 'launcher' else
                        root / 'logs/runs' / (name + suffix))
                if suffix == 'directory': path.mkdir()
                else: path.write_text('preserved fixture')
                with patch.object(parent, 'ROOT', root), patch.object(parent.subprocess, 'Popen') as popen:
                    with self.assertRaises(FileExistsError): parent.launch_logged(name, ['python3'], {}, root)
                    popen.assert_not_called()
        with tempfile.TemporaryDirectory() as text, patch.object(parent.subprocess, 'Popen') as popen:
            for name in ('pilot_pairs_other', 'gen_train_bad/child', 'gen_train_'):
                with self.assertRaises(ValueError): parent.launch_logged(name, ['python3'], {}, Path(text))
            popen.assert_not_called()


def identity(pid):
    return {'pid': pid, 'start_ticks': 1000 + pid, 'pgid': pid, 'sid': pid}


def make_fixture(directory, count=6):
    index = parent.CONCURRENCIES.index(count)
    rows = parent.assignments('gen_train_fixture', index, count)
    plan, reference = admission.create_plan(directory, parent_run_name='gen_train_fixture',
        stage_index=index, concurrency=count, assignments=rows, pbs_jobid='700000.hopper-m-02',
        node='synthetic-node', gpu_uuid=GPU, cpu_affinity=list(range(12)), scenes=list(parent.SCENES))
    assignment = admission.child_assignment(plan, reference, parent.SCENES[0])
    gate = {'config': {'node': 'synthetic-node', 'allocation': {'pbs_jobid': '700000.hopper-m-02',
        'cuda_visible_devices': GPU, 'devices': [{'gpu_uuid': GPU}]}}}
    gate_record = {'source': {'file': 'results/synthetic-gate.json', 'sha256': 'a' * 64},
                   'result_path': 'results/synthetic-gate.json'}
    raw = Path(directory) / parent.SCENES[0]
    shard = parent.make_shard(assignment['child_run_name'], assignment['scene'], list(range(12)),
        gate, gate_record, raw, plan, reference)
    cfg = {'name': shard['name'], 'scene': shard['scene'], 'shard': shard}
    return plan, reference, assignment, cfg


def cleanup_proof(pid=500):
    return {'cleanup_errors': [], 'worker_exitcode': 0, 'remaining_owned_group': False,
            'worker_identity': identity(pid), 'cleanup_group_identity': identity(pid), 'unity_identity': None}


class PlanningTests(unittest.TestCase):
    def test_same_six_training_scenes_full_twelve_cpu_and_one_real_gpu_each_stage(self):
        self.assertEqual(parent.SCENES, ('FloorPlan1', 'FloorPlan7', 'FloorPlan203',
                                        'FloorPlan204', 'FloorPlan301', 'FloorPlan401'))
        for count, witnesses in ((2, 2), (4, 4), (6, 2)):
            with tempfile.TemporaryDirectory() as directory:
                plan, reference, _, cfg = make_fixture(directory, count)
                self.assertEqual(plan['witness_capacity'], witnesses)
                self.assertLessEqual(plan['max_unity_instances'], 8)
                self.assertEqual([row['scene'] for row in plan['assignments']], list(parent.SCENES))
                self.assertEqual([row['lane_index'] for row in plan['assignments']],
                                 [index % count for index in range(6)])
                self.assertEqual(cfg['shard']['logical_index'], 0)
                self.assertEqual(cfg['shard']['gpu_identity']['gpu_uuid'], GPU)
                self.assertEqual(cfg['shard']['cpu_affinity'], list(range(12)))
                self.assertEqual(admission.validate_source_admission(cfg), plan)
                self.assertEqual(Path(reference['plan_file']).stat().st_mode & 0o222, 0)

    def test_full_scene_grant_never_shortens_to_remaining_allocation(self):
        self.assertEqual(parent.required_timeout(), 7840)
        needed = parent.SCENE_SECONDS + parent.CHILD_FINALIZATION_SECONDS + parent.PARENT_CLEANUP_SECONDS
        self.assertEqual(parent.full_scene_grant(needed, 0), 1200)
        with self.assertRaises(TimeoutError): parent.full_scene_grant(needed - .01, 0)
        with self.assertRaises(ValueError): parent.assignments('gen_train_fixture', 0, 6)

    def test_plan_hash_lane_gpu_and_cpu_cannot_be_silently_reassigned(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, assignment, cfg = make_fixture(directory)
            for key, value in (('plan_sha256', '0' * 64), ('lane_index', 3), ('scene', 'FloorPlan7'),
                               ('nonce', 'different'), ('child_run_name', 'another')):
                changed = copy.deepcopy(assignment); changed[key] = value
                with self.assertRaises(ValueError): admission.load_plan(changed)
            for key, value in (('logical_index', 1), ('child_cuda_visible_devices', '0'),
                               ('cpu_affinity', list(range(2))), ('parent_cuda_visible_devices', GPU + ',' + GPU)):
                changed = copy.deepcopy(cfg); changed['shard'][key] = value
                with self.assertRaises(ValueError): admission.validate_source_admission(changed)


class LeaseTests(unittest.TestCase):
    def test_six_constructors_admit_only_two_witnesses_and_release_requires_proof(self):
        with tempfile.TemporaryDirectory() as directory:
            plan, reference, _, _ = make_fixture(directory)
            owners = {pid: identity(pid) for pid in (101, 102, 103)}
            pools = [admission.AdmissionPool(admission.child_assignment(plan, reference, scene), owners.get)
                     for scene in parent.SCENES[:3]]
            first = pools[0].acquire(owners[101], time.time() + 10)
            second = pools[1].acquire(owners[102], time.time() + 10)
            with patch.object(admission.time, 'time', side_effect=[1., 2.]), patch.object(admission.time, 'sleep'):
                with self.assertRaises(TimeoutError): pools[2].acquire(owners[103], 2.)
            self.assertEqual(set(pools[0].health()['active_leases']), {first, second})
            pools[0].release(first, cleanup_proof())
            third = pools[2].acquire(owners[103], time.time() + 10)
            self.assertEqual(set(pools[2].health()['active_leases']), {second, third})
            with self.assertRaises(admission.AdmissionStopped):
                pools[2].release(third, {**cleanup_proof(), 'remaining_owned_group': True})
            self.assertTrue(pools[2].health()['poison'])
            self.assertIn(third, pools[2].health()['active_leases'])

    def test_dead_owner_poison_retains_slot_and_disallows_reclaim(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, assignment, _ = make_fixture(directory)
            owners = {101: identity(101)}
            pool = admission.AdmissionPool(assignment, owners.get)
            token = pool.acquire(owners[101], time.time() + 10)
            owners.clear()
            state = pool.health()
            self.assertTrue(state['poison']); self.assertIn(token, state['active_leases'])
            with self.assertRaises(admission.AdmissionStopped): pool.acquire(identity(102), time.time() + 10)
            self.assertIn(token, pool.health()['active_leases'])

    def test_other_scene_cannot_release_owned_lease(self):
        with tempfile.TemporaryDirectory() as directory:
            plan, reference, assignment, _ = make_fixture(directory)
            owners = {101: identity(101)}
            first = admission.AdmissionPool(assignment, owners.get)
            other = admission.AdmissionPool(admission.child_assignment(plan, reference, parent.SCENES[1]), owners.get)
            token = first.acquire(owners[101], time.time() + 10)
            with self.assertRaises(admission.AdmissionStopped): other.release(token, cleanup_proof())
            self.assertIn(token, first.health()['active_leases'])


class BridgeTests(unittest.TestCase):
    def worker_class(self, *, cleanup_errors=()):
        class Worker:
            stops = 0
            def __init__(self, target, config):
                self.process = SimpleNamespace(pid=500, exitcode=None)
                # Reproduce the real race: the initial snapshot predates setsid.
                self.identity = {**identity(500), 'pgid': 1, 'sid': 1}
                self.unity_identity = identity(501)
            def receive(self):
                raise OSError('closed synthetic pipe')
            def stop(self):
                Worker.stops += 1
                self.process.exitcode = -15
                return list(cleanup_errors)
        return Worker

    def test_receive_failure_still_cleans_and_initial_setsid_race_does_not_fake_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, _, cfg = make_fixture(directory)
            original_worker = stable.Worker
            fake = self.worker_class()
            identities = {os.getpid(): identity(os.getpid()), 500: identity(500)}
            with patch.object(stable, 'Worker', fake), patch.object(admission, 'process_identity', side_effect=identities.get), \
                    patch.object(admission, 'owned_group_present', return_value=False):
                with admission.witness_bridge(cfg) as bridge:
                    worker = stable.Worker(None, {'deadline': time.time() + 100, 'enable_candidate_clone': True})
                    self.assertEqual(worker.stop(), [])
                    self.assertEqual(worker.stop(), [])
                report = bridge.report()
            self.assertEqual(fake.stops, 1)
            self.assertTrue(report['cleanup_confirmed'])
            self.assertEqual(report['active_leases'], [])
            self.assertEqual(report['lease_records'][0]['cleanup_group_identity'], identity(500))
            self.assertIn('closed synthetic pipe', report['lease_records'][0]['receive_error_before_cleanup'])
            admission.validate_source_admission(cfg, {'g01_witness_admission': report})
            self.assertIs(stable.Worker, original_worker)

    def test_uncertain_cleanup_poison_never_releases_lease(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, assignment, cfg = make_fixture(directory)
            fake = self.worker_class(cleanup_errors=('synthetic survivor',))
            identities = {os.getpid(): identity(os.getpid()), 500: identity(500)}
            with patch.object(stable, 'Worker', fake), patch.object(admission, 'process_identity', side_effect=identities.get), \
                    patch.object(admission, 'owned_group_present', return_value=False):
                with admission.witness_bridge(cfg) as bridge:
                    worker = stable.Worker(None, {'deadline': time.time() + 100})
                    with self.assertRaises(admission.AdmissionStopped): worker.stop()
                health = admission.AdmissionPool(assignment).health()
            self.assertTrue(health['poison']); self.assertEqual(len(health['active_leases']), 1)
            self.assertFalse(bridge.report()['cleanup_confirmed'])
            with self.assertRaises(ValueError):
                admission.validate_source_admission(cfg, {'g01_witness_admission': bridge.report()})

    def test_missing_or_unverified_cleanup_report_cannot_admit_a_source(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, assignment, cfg = make_fixture(directory)
            with self.assertRaises(ValueError): admission.validate_source_admission(cfg, {})
            report = {'version': admission.VERSION, 'assignment': assignment, 'owner': identity(101),
                'cleanup_confirmed': True, 'active_leases': [], 'errors': [],
                'lease_records': [{'token': 'fixture', 'owner': identity(101),
                    'release_confirmed': True, **cleanup_proof()}]}
            admission.validate_source_admission(cfg, {'g01_witness_admission': report})
            for key, value in (('remaining_owned_group', True), ('worker_exitcode', None),
                               ('cleanup_errors', ['uncertain']), ('cleanup_group_identity', None)):
                changed = copy.deepcopy(report); changed['lease_records'][0][key] = value
                with self.assertRaises(ValueError):
                    admission.validate_source_admission(cfg, {'g01_witness_admission': changed})


if __name__ == '__main__':
    unittest.main()
