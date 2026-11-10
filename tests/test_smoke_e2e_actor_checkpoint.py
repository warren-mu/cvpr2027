"""Checkpoint I/O boundaries, not model-training or GPU equivalence tests.

The portable tests use an explicit synthetic FSDP/serialization boundary; the
additional real torch serialization test runs in the formal pinned CPU suite.
"""
import ast
from collections import OrderedDict
from contextlib import contextmanager
import importlib.util
import json
import os
from pathlib import Path
import pickle
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from src.tools import smoke_e2e_actor_checkpoint as saving


class ForbiddenState:
    def state_dict(self):
        raise AssertionError('Optimizer/critic/RNG training state must not be requested')


class SyntheticTensor:
    """Explicit no-torch test boundary; never represents a trained model."""
    def __init__(self, value, dtype='float32'):
        self.value, self.dtype = value, dtype
    def detach(self): return self
    def cpu(self): return self
    def is_floating_point(self): return self.dtype in ('float32', 'bfloat16')
    def to(self, *, device, dtype):
        assert device == 'cpu'
        return SyntheticTensor(self.value, dtype)


class SyntheticConfig:
    def __init__(self):
        self.torch_dtype = 'float32'
        self.other_config = {'fixture': ['unchanged']}

    def save_pretrained(self, path):
        (Path(path) / 'config.json').write_text(json.dumps({
            'torch_dtype': str(self.torch_dtype).removeprefix('torch.'),
            'other_config': self.other_config})+'\n')


class Model:
    def __init__(self, state):
        self.state = state
        self.calls = 0
        self._fsdp_wrapped_module = SimpleNamespace(config=SyntheticConfig())

    def state_dict(self):
        self.calls += 1
        return self.state

def worker_fixture(state=None, offload=False):
    model = Model({'fixture': SyntheticTensor([1., -2.])} if state is None else state)
    def save_processor(path):
        (Path(path) / 'tokenizer.json').write_text('{"fixture":true}\n')
    manager = SimpleNamespace(model=model, optimizer=ForbiddenState(), lr_scheduler=ForbiddenState(),
        processing_class=SimpleNamespace(save_pretrained=save_processor), rank=0, world_size=1,
        get_rng_state=lambda: (_ for _ in ()).throw(AssertionError('RNG must not be read')))
    return SimpleNamespace(_is_actor=True, world_size=1, checkpoint_manager=manager,
        actor_module_fsdp=model, _is_offload_param=offload, critic=ForbiddenState())


def backend_fixture(events, save=None):
    @contextmanager
    def state_dict_type(model, kind, config):
        assert kind == 'sharded' and config.offload_to_cpu is True
        events.append('enter_model_state')
        try:
            yield
        finally:
            events.append('exit_model_state')
    torch = SimpleNamespace(distributed=SimpleNamespace(barrier=lambda: events.append('barrier')),
        save=save or (lambda state, stream: pickle.dump(state, stream)),
        Tensor=SyntheticTensor, bfloat16='bfloat16')
    return (torch, SimpleNamespace(state_dict_type=state_dict_type),
            SimpleNamespace(SHARDED_STATE_DICT='sharded'), lambda **kwargs: SimpleNamespace(**kwargs))


class ActorCheckpointTests(unittest.TestCase):
    def test_only_final_model_and_original_hf_metadata_are_written(self):
        worker, events = worker_fixture(), []
        with tempfile.TemporaryDirectory() as directory, patch.object(saving, '_state_backend', return_value=backend_fixture(events)):
            target = Path(directory) / 'checkpoint/actor'
            record = saving.save_final_actor_model(worker, target, 50)
            paths = sorted(str(path.relative_to(target)) for path in target.rglob('*') if path.is_file())
            self.assertEqual(paths, ['huggingface/config.json', 'huggingface/tokenizer.json', 'model_world_size_1_rank_0.pt'])
            saved=pickle.loads((target / 'model_world_size_1_rank_0.pt').read_bytes())
            self.assertEqual(saved['fixture'].value, worker.actor_module_fsdp.state['fixture'].value)
            self.assertEqual(saved['fixture'].dtype, 'bfloat16')
            self.assertEqual(worker.actor_module_fsdp.state['fixture'].dtype, 'float32')
            live_config=worker.actor_module_fsdp._fsdp_wrapped_module.config
            saved_config=json.loads((target/'huggingface/config.json').read_text())
            self.assertEqual(saved_config['torch_dtype'],'bfloat16')
            self.assertEqual(live_config.torch_dtype,'float32')
            self.assertEqual(saved_config['other_config'],live_config.other_config)
            self.assertEqual(worker.actor_module_fsdp.calls, 1)
            self.assertEqual(record['policy'], saving.POLICY)
            self.assertEqual(record['global_step'], 50)
            for key in ('optimizer_saved', 'critic_saved', 'extra_state_saved',
                        'intermediate_checkpoints_saved', 'training_resume_supported'):
                self.assertIs(record[key], False)
            self.assertEqual(events, ['barrier', 'enter_model_state', 'exit_model_state', 'barrier', 'barrier'])
            self.assertEqual(worker.checkpoint_manager.previous_save_local_path, str(target))

    def test_saved_cast_preserves_live_state_nonfloat_buffers_and_metadata(self):
        state=OrderedDict(weight=SyntheticTensor([1.25]), count=SyntheticTensor([7], 'int64'))
        state._metadata={'module': {'version': 1}}
        result=saving.actor_bf16_state(state, backend_fixture([])[0])
        self.assertIsNot(result, state)
        self.assertIsNot(result['weight'], state['weight'])
        self.assertEqual(result['weight'].dtype,'bfloat16')
        self.assertEqual(state['weight'].dtype,'float32')
        self.assertEqual(result['count'].dtype,'int64')
        self.assertEqual(result['count'].value,[7])
        self.assertEqual(result._metadata,state._metadata)
        self.assertIsNot(result._metadata,state._metadata)
        with self.assertRaisesRegex(ValueError,'Unsupported single-GPU actor state'):
            saving.actor_bf16_state({'unknown_shard':object()},backend_fixture([])[0])

    def test_offloaded_parameters_restored_on_success_and_serialization_failure(self):
        for failed in (False, True):
            with self.subTest(failed=failed), tempfile.TemporaryDirectory() as directory:
                worker, events = worker_fixture(offload=True), []
                def save(state, stream):
                    if failed: raise OSError('fixture disk full')
                    pickle.dump(state, stream)
                with patch.object(saving, '_state_backend', return_value=backend_fixture(events, save)), \
                        patch.object(saving, '_offload_backend', return_value=(
                            lambda model: events.append('load'), lambda model: events.append('restore'))):
                    if failed:
                        with self.assertRaisesRegex(OSError, 'disk full'):
                            saving.save_final_actor_model(worker, Path(directory)/'actor', 1)
                        self.assertFalse(hasattr(worker.checkpoint_manager, 'previous_global_step'))
                    else:
                        saving.save_final_actor_model(worker, Path(directory)/'actor', 1)
                self.assertEqual(events[0], 'load')
                self.assertEqual(events[-1], 'restore')
                self.assertFalse(list(Path(directory).rglob('*optim*')))
                self.assertFalse(list(Path(directory).rglob('*extra_state*')))

    def test_existing_destination_is_never_overwritten_or_deleted(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)/'actor'; target.mkdir()
            sentinel = target/'model_world_size_1_rank_0.pt'; sentinel.write_bytes(b'prior bytes')
            with patch.object(saving, '_state_backend') as backend, self.assertRaises(FileExistsError):
                saving.save_final_actor_model(worker_fixture(), target, 1)
            backend.assert_not_called()
            self.assertEqual(sentinel.read_bytes(), b'prior bytes')

    def test_partial_parameter_load_failure_still_restores_offload(self):
        worker, events = worker_fixture(offload=True), []
        def partial_load(model):
            events.append('one_handle_moved')
            raise RuntimeError('fixture partial load failure')
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(saving, '_state_backend') as backend, \
                patch.object(saving, '_offload_backend', return_value=(
                    partial_load, lambda model: events.append('restore'))):
            with self.assertRaisesRegex(RuntimeError, 'partial load failure'):
                saving.save_final_actor_model(worker, Path(directory)/'actor', 1)
            backend.assert_not_called()
            self.assertEqual(events, ['one_handle_moved', 'restore'])
            self.assertFalse(hasattr(worker.checkpoint_manager, 'previous_global_step'))
            self.assertFalse(list(Path(directory).rglob('*.pt')))

    def test_wrong_role_allocation_manager_or_update_fails_before_writing(self):
        for mutation in ('critic', 'multi_gpu', 'other_model', 'bad_rank', 'bool_step', 'zero_step'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                worker, step = worker_fixture(), 1
                if mutation=='critic': worker._is_actor=False
                elif mutation=='multi_gpu': worker.world_size=2
                elif mutation=='other_model': worker.checkpoint_manager.model=object()
                elif mutation=='bad_rank': worker.checkpoint_manager.rank=1
                elif mutation=='bool_step': step=True
                elif mutation=='zero_step': step=0
                with self.assertRaises(ValueError): saving.save_final_actor_model(worker, Path(directory)/'actor', step)
                self.assertEqual(list(Path(directory).iterdir()), [])

    def test_both_actual_training_endpoints_use_one_actor_rpc_and_no_critic_save(self):
        root=Path(saving.__file__).resolve().parents[2]
        for name, method in [('smoke_e2e_training_runtime.py','smoke_fit'), ('smoke_e2e_pilot_runtime.py','pilot_fit')]:
            tree=ast.parse((root/'src/tools'/name).read_text())
            endpoint=next(node for node in ast.walk(tree) if isinstance(node,ast.FunctionDef) and node.name==method)
            calls=[node for node in ast.walk(endpoint) if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute)]
            self.assertEqual([ast.unparse(node.func) for node in calls if node.func.attr=='save_final_actor_model'],
                             ['self.actor_rollout_wg.save_final_actor_model'])
            self.assertFalse([node for node in calls if node.func.attr=='save_checkpoint'])
        tree=ast.parse((root/'src/tools/smoke_e2e_training_runtime.py').read_text())
        common=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='ObservedWorkerHooks')
        hook=next(node for node in common.body if isinstance(node,ast.FunctionDef) and node.name=='save_final_actor_model')
        self.assertEqual([ast.unparse(node) for node in hook.decorator_list], ['register(dispatch_mode=Dispatch.ONE_TO_ALL)'])

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'Real torch available in the formal remote CPU environment')
    def test_real_torch_cpu_model_state_roundtrip(self):
        import torch
        from transformers import PretrainedConfig
        state={'weight':torch.tensor([[1.,-2.],[3.,4.]],dtype=torch.float32),
               'count':torch.tensor([1,2],dtype=torch.int64)}
        events=[]
        synthetic=backend_fixture(events, torch.save)
        backend=(SimpleNamespace(distributed=synthetic[0].distributed, save=torch.save,
                                 Tensor=torch.Tensor,bfloat16=torch.bfloat16),*synthetic[1:])
        with tempfile.TemporaryDirectory() as directory, patch.object(saving,'_state_backend',return_value=backend):
            target=Path(directory)/'actor'
            worker=worker_fixture(state)
            live_config=PretrainedConfig(torch_dtype=torch.float32, fixture={'keep':True})
            worker.actor_module_fsdp._fsdp_wrapped_module.config=live_config
            saving.save_final_actor_model(worker,target,1)
            actual=torch.load(target/'model_world_size_1_rank_0.pt',map_location='cpu',weights_only=True)
            self.assertEqual(actual['weight'].dtype,torch.bfloat16)
            self.assertEqual(state['weight'].dtype,torch.float32)
            self.assertEqual(actual['count'].dtype,torch.int64)
            torch.testing.assert_close(actual['weight'],state['weight'].to(torch.bfloat16),rtol=0,atol=0)
            torch.testing.assert_close(actual['count'],state['count'],rtol=0,atol=0)
            saved_config=json.loads((target/'huggingface/config.json').read_text())
            self.assertEqual(saved_config['torch_dtype'],'bfloat16')
            self.assertEqual(live_config.torch_dtype,torch.float32)
            self.assertEqual(saved_config['fixture'],live_config.fixture)


@unittest.skipUnless(os.environ.get('CVPR_SCRATCH'), 'Requires the formal pinned Hopper CPU environment')
class PinnedCheckpointDispatch(unittest.TestCase):
    def test_actual_colocated_factory_dispatches_actor_save_to_actual_role(self):
        from src.tools.smoke_e2e_train import upstream_roots
        upstream_roots(os.environ['CVPR_SCRATCH'])
        import ray
        from src.tools import smoke_e2e_training_runtime as old
        from src.tools import smoke_e2e_pilot_runtime as pilot
        from verl.single_controller.ray.base import RayClassWithInitArgs, create_colocated_worker_cls
        calls=[]
        def actor_init(worker, config, role): worker.fixture_role=role
        def critic_init(worker, config): worker.fixture_role='critic'
        def save(worker, local_path, global_step):
            calls.append((worker.fixture_role, local_path, global_step))
            return {'fixture_role':worker.fixture_role,'step':global_step}
        with patch.object(old.Worker,'__init__',return_value=None), \
                patch.object(old.ActorRolloutRefWorker,'__init__',actor_init), \
                patch.object(old.CriticWorker,'__init__',critic_init), \
                patch.object(saving,'save_final_actor_model',save):
            roles={
                'actor_rollout':RayClassWithInitArgs(ray.remote(pilot.PilotActorWorker),config='actor',role='actor_rollout'),
                'critic':RayClassWithInitArgs(ray.remote(old.ObservedCriticWorker),config='critic'),
                'ref':RayClassWithInitArgs(ray.remote(pilot.PilotActorWorker),config='reference',role='ref')}
            concrete=create_colocated_worker_cls(roles).cls.__ray_actor_class__
            workers=object.__new__(concrete);concrete.__init__(workers)
            actual=workers.actor_rollout_save_final_actor_model('/unused/final/actor',1)
        self.assertEqual(actual,{'fixture_role':'actor_rollout','step':1})
        self.assertEqual(calls,[('actor_rollout','/unused/final/actor',1)])
        self.assertFalse(ray.is_initialized())


if __name__=='__main__':
    unittest.main()
