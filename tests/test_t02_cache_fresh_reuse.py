"""Fresh-stage execution-scope fixtures; no rendered/scientific acceptance claims."""
from contextlib import ExitStack, redirect_stdout, redirect_stderr
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.tools import t02_cache_check as cache
from src.envs import pair_table_checks as checks
from src.tools import merge_pair_v2 as merge


class FreshScopeTests(unittest.TestCase):
    def run_main(self, *, native_failure=False, mutate_after_auth=False, cached_process=False, gate_code=0):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name).resolve()
        evidence = root/'evidence.json';evidence.write_bytes(b'{"original":true}\n')
        reference = {'file':str(evidence),'sha256':hashlib.sha256(evidence.read_bytes()).hexdigest()}
        exported = SimpleNamespace(process_id='cached-process')
        table = SimpleNamespace(pair_id='same-original-id',dataset_key='dataset-key',
            shadow_metadata={'fixture':True},states={f's{index}':SimpleNamespace(captures=(exported,exported)) for index in range(200)})
        package = {'pair_id':table.pair_id,'scene':'FloorPlan1','source_config':{
            'grid_size':.25,'visibility_distance':3,'field_of_view':60},'cache_evidence_roots':[str(root)]}
        decoded, native_calls, workers = [], [], []
        def load_samples(table_value,row,directory,roots,state_ids,*,use_dataset_key):
            self.assertIs(table_value,table);self.assertTrue(use_dataset_key)
            self.assertEqual(cache.verified_read(reference,roots),evidence.read_bytes())
            decoded.append(tuple(state_ids))
            return [SimpleNamespace(fresh=(SimpleNamespace(process_id='cached-process' if cached_process else 'fresh-process'),),
                                    state_id=state) for state in state_ids]
        def authenticate(tables,samples,*,evidence_roots):
            self.assertEqual(tables,[table]);self.assertEqual([s.state_id for s in samples],list(decoded[-1]))
            self.assertEqual(checks.verified_read(reference,evidence_roots),evidence.read_bytes())
            native_calls.append(tuple(sample.state_id for sample in samples))
            if native_failure:raise ValueError('unchanged native checker failed')
            if mutate_after_auth:evidence.write_bytes(b'changed after the last scientific read')
            return {('dataset-key',s.state_id,'w0','fixture-render') for s in samples}
        class Worker:
            def __init__(self,function,config):
                workers.append(config)
                self.result={'pair_id':table.pair_id,'dataset_key':table.dataset_key,'status':'captured','cleanup_errors':[]}
                self.error=None;self.latest={};self.identity={'pid':10};self.unity_identity={'pid':11}
            def receive(self):pass
        def final_sample_report(samples,*,population,sampling_plan):
            self.assertEqual(sampling_plan['sample_size'],200)
            # This is a deliberately non-passing scientific boundary. The
            # fixture only checks original call order and byte reuse/cleanup.
            return {'cache_matches_live':False,'observed_distinct_states':len(samples),
                    'sample_complete':False,'missing_state_keys':[]}
        argv=['t02_cache_check','--name','t02_scope_fixture','--source','results/original.json',
              '--timeout-seconds','600','--pair-timeout-seconds','180','--gate-timeout-seconds','180']
        original_reader=checks.verified_read
        with ExitStack() as stack:
            spy=stack.enter_context(patch.object(checks,'verified_read',wraps=original_reader))
            stack.enter_context(patch.object(cache,'verified_read',spy))
            original_entries=(merge.read_reference,checks.verified_read,cache.verified_read)
            for target,value in (
                ('ROOT',root),('install_signals',lambda:None),('execution_environment',lambda *a:{'scratch':str(root/'scratch'),'pbs_jobid':'709186.hopper-m-02'}),
                ('logged_gate',lambda *a:({'_meta':{'exit_code':gate_code},'metrics':{'cleanup_errors':[]}}, {'fixture_gate':True},[])),
                ('gate_runtime',lambda *a:{}),('pair_runtime',lambda *a:{}),('Worker',Worker),
                ('stop_workers',lambda *a:[]),('load_exports',lambda *a:({'dataset-key':table},{'dataset-key':package},{},[],[{'whole_population':'fixture'}])),
                ('load_samples',load_samples),('authenticate_cache_sources',authenticate),('validate_cache_sample',final_sample_report)):
                stack.enter_context(patch.object(cache,target,value))
            stack.enter_context(patch.object(cache.sys,'argv',argv))
            stack.enter_context(patch.object(cache.os,'sched_getaffinity',create=True,return_value=set(range(12))))
            stack.enter_context(patch.object(cache.signal,'setitimer'))
            authorize=stack.enter_context(patch.object(cache,'authorize_tables'))
            with redirect_stdout(io.StringIO()),redirect_stderr(io.StringIO()):
                code=cache.main()
            self.assertEqual((merge.read_reference,checks.verified_read,cache.verified_read),original_entries)
            authorize.assert_not_called()
            reads=spy.call_count
        result=json.loads((root/'results/t02_scope_fixture.json').read_text())
        return code,result,decoded,native_calls,workers,reads

    def test_fresh_decode_and_entire_native_checker_both_run_with_one_original_byte_read(self):
        code,result,decoded,native,workers,reads=self.run_main()
        self.assertEqual(code,1)  # synthetic boundary never creates a real certificate
        self.assertEqual(decoded,native)
        self.assertEqual(len(decoded[0]),200)
        self.assertEqual(reads,1)
        self.assertEqual(len(workers),1)
        timings=result['metrics']['engineering_timings']
        for key in ('gate','load_exports','population_sampling','sample_validation_and_authorization'):
            self.assertTrue(timings[key]['returned_normally']);self.assertGreaterEqual(timings[key]['elapsed_seconds'],0)
        pair=timings['fresh_pairs'][0]
        self.assertEqual(pair['sampled_state_count'],200)
        for key in ('worker_execution_and_cleanup','decode_and_comparison','native_authentication','fresh_validation_including_final_rehash'):
            self.assertTrue(pair[key]['returned_normally']);self.assertGreaterEqual(pair[key]['elapsed_seconds'],0)
        self.assertIsNone(result['metrics'].get('authorization_manifest'))

    def test_native_failure_retains_error_and_failing_phase_time(self):
        code,result,decoded,native,workers,reads=self.run_main(native_failure=True)
        self.assertEqual(code,1);self.assertEqual(decoded,native);self.assertEqual(reads,1)
        self.assertIn('fresh_native_provenance_failed:dataset-key',result['metrics']['errors'])
        self.assertIn('unchanged native checker failed',result['metrics']['worker_runs'][0]['native_authentication_error'])
        self.assertFalse(result['metrics']['engineering_timings']['fresh_pairs'][0]['native_authentication']['returned_normally'])

    def test_final_hash_failure_cannot_publish_samples_or_authorization(self):
        code,result,decoded,native,workers,reads=self.run_main(mutate_after_auth=True)
        self.assertEqual(code,1);self.assertEqual(decoded,native)
        self.assertTrue(any('before cache-load completion' in error for error in result['metrics']['errors']))
        self.assertIsNone(result['metrics']['validation_report'])
        self.assertFalse(result['metrics']['engineering_timings']['fresh_pairs'][0]['fresh_validation_including_final_rehash']['returned_normally'])

    def test_original_cached_process_guard_remains_before_native_check(self):
        code,result,decoded,native,workers,reads=self.run_main(cached_process=True)
        self.assertEqual(code,1);self.assertEqual(native,[])
        self.assertTrue(any('Fresh process is also an export process' in error for error in result['metrics']['errors']))
        self.assertNotIn('native_authentication',result['metrics']['engineering_timings']['fresh_pairs'][0])

    def test_black_gate_does_not_fake_unreached_phase_times(self):
        code,result,decoded,native,workers,reads=self.run_main(gate_code=86)
        self.assertEqual(code,86);self.assertEqual((decoded,native,workers,reads),([],[],[],0))
        timings=result['metrics']['engineering_timings']
        self.assertEqual(set(timings),{'version','fresh_pairs','gate'})
        self.assertEqual(timings['fresh_pairs'],[])

    def test_failed_phase_records_monotonic_duration_and_propagates_original_error(self):
        timing={}
        with patch.object(cache.time,'perf_counter',side_effect=[10.25,12.5]),self.assertRaisesRegex(RuntimeError,'original'):
            with cache.engineering_phase(timing,'load_exports'):
                raise RuntimeError('original')
        self.assertEqual(timing,{'load_exports':{'elapsed_seconds':2.25,'returned_normally':False}})


if __name__=='__main__':unittest.main()
