"""Storage/selection/wiring fixtures; original provenance and native checks are explicit boundaries."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.tools import t02_test_compression as compression
from src.tools import t02_export as export
from src.tools import t02_export_parallel as parallel
from src.tools import gpu_shared_claim as shared
from src.tools import t02_rejected_raw as retention
from src.envs import pair_v2_configuration as cfg
from src.envs.pair_table_export import save_table, load_table
from test_pair_table_compression import handmade
from test_t02_export import FakeSim, package
from test_gpu_shared_claim import Fixture as QueueFixture
from test_t02_rejected_raw import Fixture as RetentionFixture


def write(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    body=json.dumps(value,sort_keys=True).encode();path.write_bytes(body)
    return {'file':str(path),'sha256':hashlib.sha256(body).hexdigest()}


class Fixture:
    def __init__(self, root):
        self.root=Path(root)
        self.source={'file':'results/source.json','sha256':'a'*64}
        self.pair={'file':'data/world_pairs/v2/test/pair.json','sha256':'b'*64}
        row={'scene':'FloorPlan21','view':{'position':{'x':0,'y':1,'z':0},'yaw':0,'horizon':0},
             'identities':{'anchor_name':'X','candidate_names':['A','B']}}
        configuration=cfg.from_pair(row)
        chosen={'configuration':configuration,'kind':'look_flip','proposal_identity':'c'*64,
                'checkpoint_reference':self.source,'pair_reference':self.pair,'scene':'FloorPlan21',
                'target_names':['A','B']}
        self.report={'proof':{'unique_proposals':[chosen], 'configuration_selection':cfg.select([chosen]),
                            'count_basis_authenticated':True},'selected_count':1}
        self.ref=write(self.root/'logs/review/diag_current_test_selected_99/report.json',self.report)
        self.logged={'file':'results/diag_current_test_selected_99.json','sha256':'d'*64}
        self.ref['logged_result']=self.logged
        path=self.root/'logs/runs/diag_current_test_selected_99.log';path.parent.mkdir(parents=True)
        path.write_text('CURRENT_TEST_SELECTED_JSON: '+json.dumps({'report':{k:self.ref[k] for k in ('file','sha256')},'selected_count':1})+'\n')
        self.data={'_meta':{'name':'diag_current_test_selected_99','end':'fixture','exit_code':0},
                   'metrics':{'command_exit_code':0}}
        self.package={'scene':'FloorPlan21','source_config':{'g01_partition':'test'},
            'source_provenance':{'container':self.source,'pair':self.pair},'row':row,
            'worlds':[{'agent_setup_command':configuration['key']['start_state']} for _ in range(2)],
            'kind':'look_flip','target_names':['A','B'],
            'independent_verification':{'proposal_identity':'c'*64}}

    def load(self):
        with patch.object(compression,'audited_result',return_value=(self.data,self.logged)):
            return compression.load_selection(self.ref,root=self.root)


class SelectionTests(unittest.TestCase):
    def test_actual_selected_snapshot_recalculation_and_complete_package_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            f=Fixture(directory);selected=f.load()
            proof=compression.validate_package(selected,f.package,root=f.root)
            self.assertEqual(proof['selected_entry'],f.report['proof']['configuration_selection']['selected'][0])
            for mutate in (lambda p:p['source_config'].update(g01_partition='train'),
                           lambda p:p['source_provenance']['pair'].update(sha256='e'*64),
                           lambda p:p.update(kind='look_stay'),
                           lambda p:p['row']['view'].update(yaw=45),
                           lambda p:p.update(target_names=['B','A'])):
                p=copy.deepcopy(f.package);mutate(p)
                with self.assertRaises(ValueError):compression.validate_package(selected,p,root=f.root)
            f.data['_meta']['exit_code']=1
            with self.assertRaises(ValueError):f.load()

    def test_snapshot_tamper_or_selection_replacement_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            f=Fixture(directory)
            body=Path(f.ref['file']).read_bytes();Path(f.ref['file']).write_bytes(body+b' ')
            with self.assertRaises(ValueError):f.load()
            f.report['proof']['configuration_selection']['selected'][0]['proposal_identity']='f'*64
            fresh=write(Path(f.ref['file']),f.report);f.ref.update(fresh)
            (f.root/'logs/runs/diag_current_test_selected_99.log').write_text('CURRENT_TEST_SELECTED_JSON: '+json.dumps({'report':fresh,'selected_count':1})+'\n')
            with self.assertRaises(ValueError):f.load()

    def test_explicit_flags_old_command_unchanged_and_train_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            f=Fixture(directory);selection=f.load()
            entry={'source':f.source,'pair':f.pair,'pair_id':'floorplan21_00001','scene':'FloorPlan21','proposal_identity':'c'*64}
            base=parallel.unit_command('t02_test',entry,900,660,180)
            self.assertNotIn('--compress-test-tables',base)
            argv=parallel.unit_command('t02_test',entry,900,660,180,compression_reference=f.ref)
            self.assertEqual(argv,base+compression.command_suffix(f.ref))
            compression.validate_entries(selection,{'partition':'test','entries':[entry]},root=f.root)
            for manifest in ({'partition':'train','entries':[entry]},{'partition':'test','entries':[entry,entry]}):
                with self.assertRaises(ValueError):compression.validate_entries(selection,manifest,root=f.root)
            for args in (SimpleNamespace(compress_test_tables=True,test_selection=None),
                         SimpleNamespace(compress_test_tables=False,test_selection=json.dumps(f.ref)),
                         SimpleNamespace(compression_smoke_dual=True)):
                with self.assertRaises(ValueError):compression.option_values(args)

    def test_same_renderer_dual_storage_runs_original_native_and_pair_checker(self):
        with tempfile.TemporaryDirectory() as directory:
            p=package();p['scene']='FloorPlan21'
            original=handmade(); original.source['synthetic']=False
            calls=[]
            def native(table,**kwargs):
                calls.append('native');table.export_verified=True;table.synthetic=False
                return {'passed':True}
            # Byte/graph construction is the synthetic boundary; production does one actual build_table.
            with patch.object(export,'build_table',return_value=original) as build, \
                 patch.object(export,'SimulatorTableRenderer',return_value=SimpleNamespace(restore_checks=[])), \
                 patch.object(export,'verify_render_sources',side_effect=native), \
                 patch.object(export,'check_pair',side_effect=lambda *a: calls.append('pair') or {'passed':True,'reasons':[]}):
                detail=export.export_checked(FakeSim(),p,Path(directory)/'table',[],1<<24,
                                             compression_proof={'fixture':'selection'},dual=True)
            self.assertEqual(build.call_count,1);self.assertEqual(calls,['native','pair'])
            proof=detail['test_compression']['dual_format_comparison']
            self.assertTrue(proof['rgb_bytes_equal']);self.assertTrue(proof['tau_and_tau_seen_equal'])
            self.assertIsNone(proof['cache_matches_live'])
            self.assertFalse(detail['downstream_allowed'])

    def test_shared_claim_scientific_keys_do_not_change_with_storage(self):
        with tempfile.TemporaryDirectory() as directory:
            q=QueueFixture(directory);unit=q.unit(1)
            path=q.root/'configs/m1.json';manifest=json.loads(path.read_text())
            manifest['partition']='test';manifest['entries'][0].update(scene='FloorPlan21',pair_id='floorplan21_00001')
            path.chmod(0o644);path.write_text(json.dumps(manifest));path.chmod(0o444)
            unit['command'][unit['command'].index('--manifest-sha256')+1]=hashlib.sha256(path.read_bytes()).hexdigest()
            before,keys=shared.t02_identity(unit,q.root)
            ref={'file':'logs/review/selected/report.json','sha256':'a'*64,'logged_result':{'file':'results/selection.json','sha256':'b'*64}}
            unit['command']+=compression.command_suffix(ref)
            after,newkeys=shared.t02_identity(unit,q.root)
            self.assertEqual(keys,newkeys);self.assertNotEqual(before,after)
            unit.update(identity=after,claim_keys=keys,unit_id=shared.canonical_hash(after))
            plan,planref=q.plan([unit]);allocation=q.parent(planref)
            claim=shared.SharedQueue(q.directory,plan,planref,root=q.root).claim(allocation,remaining_seconds=2000)
            env={'CVPR_SHARED_CLAIM_FILE':claim['claim_reference']['file'],'CVPR_SHARED_CLAIM_SHA256':claim['claim_reference']['sha256'],'CVPR_QUEUE_PARENT_MASK':q.gpu}
            identity={'pbs_jobid':allocation['pbs_jobid'],'node':allocation['node'],'scratch':str(q.scratch),'cuda_visible_devices':q.gpu,'cpu_affinity':list(range(12))}
            args=SimpleNamespace(name=unit['name']+'_u0000',pair_id=['floorplan21_00001'],source=Path('results/source1.json'),pair_file='data/world_pairs/v2/p1.json',timeout_seconds=900.,pair_timeout_seconds=660.,gate_timeout_seconds=180.,compress_test_tables=True,test_selection=compression.command_suffix(ref)[-1])
            self.assertEqual(shared.validate_t02_child_claim(env,identity,args,root=q.root)['t02_child_index'],0)
            args.compress_test_tables=False
            with self.assertRaises(ValueError):shared.validate_t02_child_claim(env,identity,args,root=q.root)

    def test_parallel_child_result_must_match_compression_command(self):
        from test_t02_export_parallel import Fixture as ParallelFixture
        with tempfile.TemporaryDirectory() as directory:
            f=ParallelFixture(directory,count=1,partition='test')
            ref={'file':'logs/review/selected/report.json','sha256':'a'*64,
                 'logged_result':{'file':'results/selection.json','sha256':'b'*64}}
            command=parallel.unit_command('t02_export_child',f.entries[0],900,660,180,compression_reference=ref)
            data=f.child_data(0,command)
            data['config'].update(compress_test_tables=True,test_selection=command[-1],
                                  test_compression_policy=copy.deepcopy(compression.codec.POLICY))
            record={'exit_code':0}
            parallel.child_outcome(record,data,f.identity,command,f.entries[0])
            for change in ({'compress_test_tables':False},{'test_selection':'{}'},
                           {'test_compression_policy':{}},{'compression_smoke_dual':True}):
                bad=copy.deepcopy(data);bad['config'].update(change)
                with self.assertRaises(ValueError):parallel.child_outcome(record,bad,f.identity,command,f.entries[0])

    def test_rejected_compressed_payload_retention_preserves_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            f=RetentionFixture(directory)
            # Replace only this synthetic private fixture's binary format before publication.
            (f.table/'abc.rgb').unlink();(f.table/'table.json').unlink();f.table.rmdir()
            ref=save_table(handmade(),f.table,compression=compression.codec.VERSION)
            f.row['table_manifest']=ref
            result=retention.apply_retention(f.row,f.owner,cleanup_confirmed=True)
            self.assertFalse(result['raw_retention']['binary_payloads_retained'])
            self.assertTrue(Path(ref['file']).exists());self.assertEqual(list(f.table.glob('*.npz')),[])
            self.assertFalse(result['table_payloads_available'])


if __name__=='__main__':unittest.main()
