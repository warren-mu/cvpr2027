"""Handmade storage/reader equivalence only; real-pair validation is a separate logged smoke."""
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np
from src.envs import pair_table_codec as codec
from src.envs.pair_table_export import build_table, load_table, save_table
from src.envs.paired_env import table_environment
from test_pair_table import FixtureRenderer


def handmade(scene='FloorPlan21', partition='test'):
    return build_table(FixtureRenderer(leak=True), pair_id='test_fixture',scene=scene,
        kind='look_flip',instruction='Select the marked cup.',anchor_name='X',
        candidate_names=('A','B'),target_names=('A','B'),initial_position={'x':0,'y':0,'z':0},
        initial_heading=0,source={'synthetic':True,'g01_partition':partition})


class CodecTests(unittest.TestCase):
    def test_lossless_deterministic_dual_sha_and_declared_level(self):
        data=bytes(range(256))*100
        body,ref=codec.encode(data)
        self.assertEqual(codec.decode(body,ref),data)
        self.assertEqual(codec.encode(data),(body,ref))
        self.assertEqual(ref['raw_sha256'],hashlib.sha256(data).hexdigest())
        self.assertEqual(ref['sha256'],hashlib.sha256(body).hexdigest())
        self.assertEqual(ref['encoding']['compression_level'],6)
        for changed in ({'raw_sha256':'0'*64},{'raw_bytes':len(data)+1},{'sha256':'1'*64},
                        {'encoding':{**codec.POLICY,'compression_level':1}}):
            with self.subTest(changed=changed),self.assertRaises(ValueError):codec.decode(body,{**ref,**changed})
        with self.assertRaises(ValueError):codec.decode(body+b'x',ref)

    def test_pickle_wrong_dtype_shape_or_extra_member_never_loads(self):
        _,ref=codec.encode(b'1234')
        for array in (np.array([object()],dtype=object),np.array([1,2],dtype=np.uint16),
                      np.zeros((2,2),dtype=np.uint8)):
            raw=io.BytesIO();np.save(raw,array,allow_pickle=True)
            body=io.BytesIO()
            with zipfile.ZipFile(body,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
                z.writestr('payload.npy',raw.getvalue())
            encoded=body.getvalue();changed={**ref,'sha256':hashlib.sha256(encoded).hexdigest(),'bytes':len(encoded)}
            with self.assertRaises(ValueError):codec.decode(encoded,changed)
        body,ref=codec.encode(b'1234');out=io.BytesIO(body)
        with zipfile.ZipFile(out,'a',compression=zipfile.ZIP_DEFLATED) as z:z.writestr('extra',b'data')
        encoded=out.getvalue()
        with self.assertRaises(ValueError):codec.decode(encoded,{**ref,'sha256':hashlib.sha256(encoded).hexdigest(),'bytes':len(encoded)})

    def test_handmade_all_state_bytes_masks_labels_tau_and_seen_equal(self):
        original=handmade()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            refs=[save_table(original,root/'plain'),save_table(original,root/'compressed',compression=codec.VERSION)]
            tables=[load_table(r['file'],expected_sha256=r['sha256']) for r in refs]
            self.assertEqual(set(tables[0].states),set(tables[1].states))
            self.assertEqual(tables[0].labels,tables[1].labels)
            for key in tables[0].states:
                a,b=[table.states[key] for table in tables]
                self.assertEqual(a,b)
            for world in ('w0','w1'):
                for actions in (['turn_right','turn_left','turn_left'],['turn_left','turn_right','turn_right'],['turn_right']*4):
                    environments=[table_environment({t.pair_id:t},allow_synthetic=True) for t in tables]
                    observations=[env.reset(original.pair_id,world) for env in environments]
                    self.assertEqual(*observations)
                    for action in actions:
                        values=[env.step(action) for env in environments]
                        self.assertEqual(values[0][:3],values[1][:3])
                        for field in ('tau','x_seen_step','selectable_labels','state_id','original_world_id'):
                            self.assertEqual(values[0][3][field],values[1][3][field])
            plain=json.loads(Path(refs[0]['file']).read_text())
            compressed=json.loads(Path(refs[1]['file']).read_text())
            self.assertNotIn('storage_encoding',plain)
            self.assertEqual(compressed['storage_encoding'],codec.POLICY)
            self.assertTrue(all(ref['encoding']==codec.POLICY for ref in compressed['files']))
            # Old plaintext filenames and byte hashes remain their original format.
            self.assertTrue(all(r['file'].endswith(('.rgb','.mask')) and set(r)=={'file','sha256','bytes'} for r in plain['files']))

    def test_original_cache_predicates_receive_identical_decoded_bytes(self):
        from dataclasses import replace
        from src.envs.paired_env import RGBFrame
        from src.envs.pair_table_cache import check_cached_state
        from test_pair_table_cache import sample, change_view, mask
        original=sample(anchor_count=250)
        bad_masks=dict(original.fresh[1].masks,b=mask(500,4000))
        for item in (original,change_view(original,'fresh',1,masks=bad_masks)):
            views=[]
            for view in item.cached:
                rgb,ref=codec.encode(view.rgb.data)
                masks={}
                for name,data in view.masks.items():
                    body,metadata=codec.encode(data);masks[name]=codec.decode(body,metadata)
                views.append(replace(view,rgb=RGBFrame(codec.decode(rgb,ref),255,255),masks=masks))
            decoded=replace(item,cached=tuple(views))
            self.assertEqual(check_cached_state(decoded),check_cached_state(item))
            self.assertIsNone(check_cached_state(decoded)['cache_matches_live'])

    def test_train_and_unknown_codec_fail_before_writing_new_table(self):
        with tempfile.TemporaryDirectory() as directory:
            for i,value in enumerate((handmade('FloorPlan1','train'),handmade('FloorPlan21','train'))):
                path=Path(directory)/str(i)
                with self.assertRaises(ValueError):save_table(value,path,compression=codec.VERSION)
                self.assertFalse(path.exists())
            with self.assertRaises(ValueError):save_table(handmade(),Path(directory)/'lossy',compression='jpeg')

    def test_decode_failure_or_manifest_flag_omission_cannot_load(self):
        with tempfile.TemporaryDirectory() as directory:
            ref=save_table(handmade(),Path(directory)/'table',compression=codec.VERSION)
            path=Path(ref['file']);manifest=json.loads(path.read_text())
            manifest.pop('storage_encoding')
            body=json.dumps(manifest).encode();path.write_bytes(body)
            with self.assertRaises(ValueError):load_table(path,expected_sha256=hashlib.sha256(body).hexdigest())


if __name__=='__main__':unittest.main()
