"""Real-file storage lifecycle fixtures; no scientific/Unity acceptance is mocked as proven."""
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.tools import t02_rejected_raw as retention


class Fixture:
    def __init__(self, root, *, sampled=False):
        self.root = Path(root).resolve()
        self.parent_result = self.root / 'results/t02_fixture.json'
        self.run = self.root / 'scratch/t02_export/t02_fixture/700001.hopper-m-02'
        self.pair = self.run / 'pair_0000'
        self.pair.mkdir(parents=True)
        self.package = {'pair_id': 'pair0', 'scene': 'FloorPlan1',
                        'source_provenance': {'pair': {'file': '/original/pair.json', 'sha256': 'f' * 64}}}
        self.config = {'name': 't02_fixture', 'pbs_jobid': '700001.hopper-m-02',
                       'scratch_directory': str(self.run)}
        # Select an exact deterministic fixture identity; no global RNG patch.
        for i in range(1000):
            self.config['name'] = 't02_fixture_%d' % i
            identity = {'run_name': self.config['name'], 'pbs_jobid': self.config['pbs_jobid'],
                        'pair_index': 0, 'pair_id': 'pair0', 'scene': 'FloorPlan1',
                        'source_pair': self.package['source_provenance']['pair']}
            if retention.selection(identity)['selected'] is sampled:
                break
        self.owner = retention.start_attempt(self.config, self.package, self.pair, 0, self.parent_result)
        self.raw, self.table = self.pair / 'raw', self.pair / 'table'
        self.raw.mkdir(); self.table.mkdir()
        self.raw_binary = self.put(self.raw / 'capture_rgb.png', b'original image bytes')
        self.mask = self.put(self.raw / 'capture_masks.npz', b'original mask bytes')
        self.metadata = retention.save(self.raw / 'capture_metadata.json', {'original': 'metadata'})
        self.index = retention.save(self.raw / 'evidence_index.json', {
            'files': [self.raw_binary, self.mask, self.metadata], 'raw_files_not_deleted': True})
        self.incremental = self.put(self.raw / 'evidence_index.jsonl', b'unchanged index\n')
        table_binary = self.put(self.table / 'abc.rgb', b'raw table RGB')
        self.table_ref = retention.save(self.table / 'table.json', {
            'files': [{**table_binary, 'file': 'abc.rgb'}], 'states': {'unchanged': True}})
        self.detail = retention.save(self.pair / 'checks.json', {'original_science_result': 'rejected'})
        self.report = retention.save(self.pair / 'pair_report.json', {'status': 'rejected', 'raw_index': self.index})
        self.log = self.pair / 'unity_logs/Player.log'; self.log.parent.mkdir();self.log.write_text('original log')
        self.row = {'pair_id': 'pair0', 'status': 'rejected', 'rejection_reasons': ['graph_mismatch'],
                    'cleanup_errors': [], 'raw_index': self.index, 'raw_incremental_index': self.incremental,
                    'table_manifest': self.table_ref, 'detail': self.detail, 'report_reference': self.report,
                    'downstream_allowed': False, 'task_acceptance': None, 'error': None}
        self.protected = {p: p.read_bytes() for p in [Path(self.metadata['file']), Path(self.index['file']),
            Path(self.incremental['file']), Path(self.table_ref['file']), Path(self.detail['file']),
            Path(self.report['file']), self.log, Path(self.owner['file'])]}
        self.binaries = [Path(self.raw_binary['file']), Path(self.mask['file']), Path(table_binary['file'])]

    @staticmethod
    def put(path, body):
        path.write_bytes(body)
        return {'file': str(path), 'sha256': hashlib.sha256(body).hexdigest(), 'bytes': len(body)}


class RetentionTests(unittest.TestCase):
    def test_unsampled_rejected_omits_only_new_indexed_binary_after_cleanup(self):
        with tempfile.TemporaryDirectory() as folder:
            f = Fixture(folder)
            original = copy.deepcopy(f.row)
            result = retention.apply_retention(f.row, f.owner, cleanup_confirmed=True)
            self.assertEqual(f.row, original)
            self.assertEqual(result['status'], 'rejected')
            self.assertEqual(result['rejection_reasons'], ['graph_mismatch'])
            self.assertFalse(result['downstream_allowed'])
            self.assertTrue(all(not p.exists() for p in f.binaries))
            for p, body in f.protected.items():self.assertEqual(p.read_bytes(), body)
            self.assertNotIn('table_manifest', result)
            self.assertNotIn('raw_index', result)
            receipt = retention.read(result['raw_retention']['receipt_reference'])
            self.assertTrue(receipt['completed'])
            self.assertEqual(len(receipt['omitted_payloads']), 3)
            self.assertTrue(all('file' not in row and row['sha256'] for row in receipt['omitted_payloads']))
            self.assertEqual(retention.read(result['report_reference'])['status'], 'rejected')

    def test_sampled_rejected_and_accepted_keep_all_bytes(self):
        for sampled, status in ((True, 'rejected'), (False, 'exported')):
            with self.subTest(sampled=sampled,status=status), tempfile.TemporaryDirectory() as folder:
                f=Fixture(folder,sampled=sampled);f.row['status']=status
                result=retention.apply_retention(f.row,f.owner,cleanup_confirmed=True)
                self.assertTrue(all(p.exists() for p in f.binaries))
                self.assertEqual(result['table_manifest'],f.table_ref)
                self.assertTrue(result['raw_retention']['binary_payloads_retained'])
                self.assertFalse((f.pair/'raw_retention_intent.json').exists())

    def test_errors_black_timeout_unknown_cleanup_are_never_sampled(self):
        cases=[{'status':'error','error':{'type':'TimeoutError'}}, {'status':'error'},
               {'status':'black_frame'}, {'cleanup_errors':['unknown']},
               {'raw_index_error':'disk full'}, {'black_frame_reference':{'file':'black'}}]
        for change in cases:
            with self.subTest(change=change), tempfile.TemporaryDirectory() as folder:
                f=Fixture(folder);f.row.update(change)
                result=retention.apply_retention(f.row,f.owner,cleanup_confirmed=True)
                self.assertTrue(all(p.exists() for p in f.binaries))
                self.assertFalse(result['raw_retention']['eligible_scientific_rejection'])
        with tempfile.TemporaryDirectory() as folder:
            f=Fixture(folder);retention.apply_retention(f.row,f.owner,cleanup_confirmed=False)
            self.assertTrue(all(p.exists() for p in f.binaries))

    def test_shared_hardlink_symlink_hash_mismatch_fail_before_any_delete(self):
        for mutation in ('hardlink','symlink','changed','unindexed'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as folder:
                f=Fixture(folder)
                if mutation=='hardlink':os.link(f.binaries[0],f.root/'accepted_copy.png')
                if mutation=='symlink':
                    original=f.binaries[0].read_bytes();f.binaries[0].unlink()
                    target=f.root/'other.png';target.write_bytes(original);f.binaries[0].symlink_to(target)
                if mutation=='changed':f.binaries[0].write_bytes(b'changed')
                if mutation=='unindexed':(f.raw/'not_indexed.png').write_bytes(b'new')
                with self.assertRaises(ValueError):retention.apply_retention(f.row,f.owner,cleanup_confirmed=True)
                self.assertTrue(all(p.exists() for p in f.binaries))
                self.assertFalse((f.pair/'raw_retention_intent.json').exists())

    def test_published_or_other_attempt_is_not_current_private_deletion_scope(self):
        for mutation in ('published','pair','accepted'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as folder:
                f=Fixture(folder)
                if mutation=='published':f.parent_result.parent.mkdir();f.parent_result.write_text('{}')
                if mutation=='pair':f.row['pair_id']='other'
                if mutation=='accepted':f.row['downstream_allowed']=True
                with self.assertRaises(ValueError):retention.apply_retention(f.row,f.owner,cleanup_confirmed=True)
                self.assertTrue(all(p.exists() for p in f.binaries))

    def test_partial_unlink_error_is_recorded_without_changing_scientific_status(self):
        with tempfile.TemporaryDirectory() as folder:
            f=Fixture(folder);original=Path.unlink;calls=[]
            def fail_second(path,*args,**kwargs):
                calls.append(path)
                if len(calls)==2:raise OSError('fixture unlink failure')
                return original(path,*args,**kwargs)
            with patch.object(Path,'unlink',fail_second):
                result=retention.apply_retention(f.row,f.owner,cleanup_confirmed=True)
            self.assertEqual(result['status'],'rejected')
            self.assertEqual(result['raw_retention']['omitted_payload_count'],1)
            self.assertEqual(result['raw_retention']['retention_error']['type'],'OSError')
            receipt=retention.read(result['raw_retention']['receipt_reference'])
            self.assertFalse(receipt['completed'])
            self.assertEqual(len(receipt['omitted_payloads']),1)
            self.assertEqual(len(calls),2)

    def test_post_unlink_receipt_or_report_write_failure_keeps_inline_omission_truth(self):
        for filename in ('raw_retention_receipt.json', 'pair_retention_report.json'):
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as folder:
                f=Fixture(folder);original_save=retention.save
                def fail_write(path,value):
                    if Path(path).name==filename:raise OSError('quota fixture')
                    return original_save(path,value)
                with patch.object(retention,'save',side_effect=fail_write):
                    result=retention.apply_retention(f.row,f.owner,cleanup_confirmed=True)
                self.assertEqual(result['status'],'rejected')
                self.assertFalse(result['raw_payloads_available'])
                self.assertFalse(result['table_payloads_available'])
                self.assertNotIn('raw_index',result)
                self.assertNotIn('table_manifest',result)
                self.assertEqual(len(result['raw_retention']['omitted_payloads']),3)
                self.assertEqual(result['raw_retention']['retention_error']['type'],'OSError')
                self.assertTrue(Path(result['raw_retention']['intent_reference']['file']).exists())
                self.assertTrue(all(not p.exists() for p in f.binaries))
                if filename=='pair_retention_report.json':self.assertNotIn('report_reference',result)
                for p,body in f.protected.items():self.assertEqual(p.read_bytes(),body)

    def test_reason_counts_preserve_all_attempts_not_only_retained_examples(self):
        rows=[{'status':'rejected','rejection_reasons':['r','r']},
              {'status':'rejected','rejection_reasons':['r','s']},
              {'status':'error','error':{'type':'TimeoutError'}}, {'status':'exported'}]
        result=retention.summary(rows)
        self.assertEqual(result['attempts'],4)
        self.assertEqual(result['rejection_reason_counts'],{'r':2,'s':1})
        self.assertEqual(result['error_type_counts'],{'TimeoutError':1})
        self.assertEqual(result['status_counts'],{'rejected':2,'error':1,'exported':1})

    def test_draw_is_reproducible_and_bound_to_whole_attempt_identity(self):
        identity={'run':'a','pair':'p','source_sha':'a'*64}
        self.assertEqual(retention.selection(identity),retention.selection(copy.deepcopy(identity)))
        self.assertNotEqual(retention.selection(identity)['sha256'],
                            retention.selection({**identity,'source_sha':'b'*64})['sha256'])


    def test_parallel_opt_in_is_exact_and_child_declaration_is_bound(self):
        from src.tools import t02_export_parallel as parallel
        from test_t02_export_parallel import Fixture as ParallelFixture
        with tempfile.TemporaryDirectory() as folder:
            f=ParallelFixture(folder,count=1);entry=f.entries[0]
            plain=parallel.unit_command('t02_child',entry,900,660,180)
            opted=parallel.unit_command('t02_child',entry,900,660,180,sample_rejected_raw=True)
            self.assertEqual(opted,plain+['--sample-rejected-raw'])
            original=f.child_data(0,plain)
            parallel.child_outcome({'exit_code':0},original,f.identity,plain,entry)
            data=f.child_data(0,opted)
            with self.assertRaises(ValueError):
                parallel.child_outcome({'exit_code':0},data,f.identity,opted,entry)
            data['config'].update(sample_rejected_raw=True,raw_retention_policy=copy.deepcopy(retention.POLICY))
            parallel.child_outcome({'exit_code':0},data,f.identity,opted,entry)
            data['config']['raw_retention_policy']['seed']=1
            with self.assertRaises(ValueError):
                parallel.child_outcome({'exit_code':0},data,f.identity,opted,entry)

    def test_real_shared_claim_storage_option_never_duplicates_scientific_key(self):
        from src.tools import gpu_shared_claim as shared
        from test_gpu_shared_claim import Fixture as SharedFixture
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as folder:
            f=SharedFixture(Path(folder).resolve());unit=f.unit(1)
            old_identity=copy.deepcopy(unit['identity']);old_keys=unit['claim_keys'][:]
            unit['command'].append('--sample-rejected-raw')
            identity,keys=shared.t02_identity(unit,f.root)
            self.assertEqual(keys,old_keys)
            self.assertNotIn('raw_retention_policy',old_identity)
            self.assertEqual(identity['raw_retention_policy'],retention.POLICY)
            unit.update(identity=identity,claim_keys=keys,unit_id=shared.canonical_hash(identity))
            plan,reference=f.plan([unit]);allocation=f.parent(reference)
            claim=shared.SharedQueue(f.directory,plan,reference,root=f.root).claim(allocation,remaining_seconds=2000)
            actual={'pbs_jobid':allocation['pbs_jobid'],'node':allocation['node'],'scratch':str(f.scratch),
                    'cuda_visible_devices':f.gpu,'cpu_affinity':list(range(12))}
            env={'CVPR_SHARED_CLAIM_FILE':claim['claim_reference']['file'],
                 'CVPR_SHARED_CLAIM_SHA256':claim['claim_reference']['sha256'],'CVPR_QUEUE_PARENT_MASK':f.gpu}
            args=SimpleNamespace(name=unit['name']+'_u0000',pair_id=['floorplan1_00001'],
                source=Path('results/source1.json'),pair_file='data/world_pairs/v2/p1.json',
                timeout_seconds=900.0,pair_timeout_seconds=660.0,gate_timeout_seconds=180.0,
                sample_rejected_raw=False)
            with self.assertRaises(ValueError):shared.validate_t02_child_claim(env,actual,args,root=f.root)
            args.sample_rejected_raw=True
            self.assertEqual(shared.validate_t02_child_claim(env,actual,args,root=f.root)['t02_child_index'],0)
            duplicate=copy.deepcopy(unit);duplicate['command'].append('--sample-rejected-raw')
            with self.assertRaises(ValueError):shared.t02_identity(duplicate,f.root)


if __name__=='__main__':unittest.main()
