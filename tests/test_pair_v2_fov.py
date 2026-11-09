"""Synthetic camera/config contracts; run only through committed run_logged."""
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from src.envs import pair_v2_fov as fov
from src.envs import pair_v2_marked_shard as collector
from src.tools import t01v2_marked_parallel as parent
from src.tools import t01v2_parallel as parallel
from src.tools import merge_pair_v2 as merge


class FovContrastTests(unittest.TestCase):
    def test_fixed_complete_partition_and_no_threshold_policy_change(self):
        plan = json.loads((Path(__file__).resolve().parents[1] / 'configs/t01v2_fov_contrast.json').read_text())
        self.assertEqual(plan['version'], fov.FOV_VERSION)
        self.assertEqual(plan['minimum_eligible_attempts_per_scene'], 100)
        self.assertEqual(plan['selection_order'], fov.FOV_POLICY['selection_order'])
        all_scenes = []
        for group, arm in plan['groups'].items():
            self.assertEqual(len(arm['scenes']), 20)
            self.assertEqual(fov.contrast_fields(arm['field_of_view'])['fov_contrast_group'], group)
            for scene in arm['scenes']:
                self.assertEqual(fov.scene_field_of_view(scene), arm['field_of_view'])
            all_scenes.extend(arm['scenes'])
        self.assertEqual(set(all_scenes), set(fov.FOV_POLICY['scenes']))
        self.assertEqual(len(set(all_scenes)), len(all_scenes))

    def test_parent_explicit_arms_enforce_parity_and_legacy_shape_is_unchanged(self):
        for scene, value in [('FloorPlan201', 100), ('FloorPlan202', 60)]:
            args = ['--gpus', '1', '--scenes', scene, '--field-of-view', str(value)]
            self.assertEqual(parent.validate_plan(args), fov.contrast_fields(value))
            args[-1] = str(160 - value)
            with self.assertRaises(ValueError):
                parent.validate_plan(args)
        self.assertEqual(parent.validate_plan(['--gpus', '1', '--scenes', 'FloorPlan1', 'FloorPlan2']), {})
        for value in ('90', 'nan', 'True'):
            with self.assertRaises(ValueError):
                parent.validate_plan(['--gpus', '1', '--scenes', 'FloorPlan2', '--field-of-view', value])

    def test_actual_parent_pipeline_persists_fields_and_restores_hooks(self):
        old = parallel.shard_configuration, parallel.write_result, parallel.launch_logged
        fields = fov.contrast_fields(60)
        captured = {}
        def run():
            device = {'gpu_uuid': 'synthetic', 'logical_index': 0}
            gate = {'config': {'node': 'synthetic', 'allocation': {
                'devices': [device], 'pbs_jobid': 'synthetic', 'cuda_visible_devices': 'synthetic'}}}
            shard = parallel.shard_configuration('test', 'FloorPlan202', 0, list(range(12)),
                gate, {}, 'synthetic', ['FloorPlan202'], '/synthetic', field_of_view=60)
            for key, value in fields.items():
                self.assertEqual(shard[key], value)
            parallel.write_result('test', {}, {})
            return 0
        def write(name, metrics, config):
            captured.update(metrics=metrics, config=config)
        argv = ['fixture', '--gpus', '1', '--scenes', 'FloorPlan202', '--field-of-view', '60']
        with patch.object(parent.sys, 'argv', argv), patch.object(parallel, 'main', side_effect=run), \
             patch.object(parallel, 'write_result', side_effect=write):
            self.assertEqual(parent.main(), 0)
        self.assertEqual((parallel.shard_configuration, parallel.write_result, parallel.launch_logged), old)
        for location in ('metrics', 'config'):
            for key, value in fields.items():
                self.assertEqual(captured[location][key], value)

    def test_scene_configuration_and_source_binding_reject_mismatch(self):
        fields = fov.contrast_fields(60)
        shard = {**fields, 'scene': 'FloorPlan202'}
        config = {'scene': 'FloorPlan202', 'field_of_view': 60, 'shard': copy.deepcopy(shard)}
        collector.enable_config(config)
        self.assertEqual(fov.bind_source_fields(config, shard), fields)
        self.assertEqual(merge.marked_policy_configuration(config)['candidate_policy'], collector.candidates.CANDIDATE_POLICY)
        bad = copy.deepcopy(config); bad['field_of_view'] = 100
        with self.assertRaises(ValueError):
            fov.bind_source_fields(bad, shard)
        bad = copy.deepcopy(shard); bad['fov_contrast_policy']['minimum_eligible_attempts_per_scene'] = 99
        with self.assertRaises(ValueError):
            fov.bind_source_fields(config, bad)
        with self.assertRaises(ValueError):
            fov.declared_fields({'field_of_view': 60})
        self.assertEqual(fov.bind_source_fields({'field_of_view': 100}, {}), {'field_of_view': 100})

    def test_native_camera_proof_cannot_be_relabelled(self):
        for value in (60, 100):
            fov.verify_native_fov({'fov': float(value)}, value)
            for actual in (160 - value, None, str(value), True):
                with self.assertRaises(ValueError):
                    fov.verify_native_fov({'fov': actual}, value)

    def test_shard_result_binds_source_and_planned_camera(self):
        fields = fov.contrast_fields(60)
        shard = {**fields, 'name': 'fixture', 'scene': 'FloorPlan202', 'seed': 0, 'resolution': 255,
                 'logical_index': 0, 'gpu_identity': {}, 'pbs_jobid': 'synthetic', 'node': 'synthetic',
                 'parent_cuda_visible_devices': 'synthetic', 'child_cuda_visible_devices': 'synthetic',
                 'cpu_affinity': list(range(12)), 'gate_source': {}}
        data = {'_meta': {'name': 'fixture', 'exit_code': 0},
                'metrics': {'kind': 'pair_v2_shard', 'cleanup_errors': []},
                'config': {**fields, 'scene': 'FloorPlan202', 'shard': copy.deepcopy(shard)}}
        parallel.verify_shard_result({'exit_code': 0}, data, shard)
        data['config']['field_of_view'] = 100
        with self.assertRaises(ValueError):
            parallel.verify_shard_result({'exit_code': 0}, data, shard)


if __name__ == '__main__':
    unittest.main()
