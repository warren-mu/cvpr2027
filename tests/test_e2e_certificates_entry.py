"""Entry interfaces only; synthetic provenance is not a real qualification."""
import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import test_train_holdout_freeze as fixtures
from src.tools import train_holdout_freeze as holdout


path = Path(__file__).resolve().parents[1] / 'experiments/e2e_certificates83_one_update.py'
spec = importlib.util.spec_from_file_location('certificate_entry_fixture', path)
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


class CertificateEntryTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.HoldoutFreezeTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def test_actual_absolute_save_receipt_entry_and_unmodified_strict_loader(self):
        # Real freeze calls original save_json; only the external original-run
        # auditor is replaced by the established byte-backed fixture boundary.
        frozen, written, _ = self.fixture.freeze()
        self.assertTrue(Path(written['file']).is_absolute())
        receipt_ref = self.fixture.receipt(frozen, written)
        obtained = entry.frozen_reference_from_receipt(receipt_ref['file'])
        self.assertEqual(obtained, written)
        actual, verified = holdout.load_frozen_holdout(obtained, root=self.fixture.root)
        self.assertEqual(actual, frozen)
        self.assertEqual(verified, holdout.ref_identity(written))
        # Demonstrate that the original comparison is active, not mocked away.
        relative = dict(obtained, file=holdout.DESTINATION)
        with self.assertRaisesRegex(ValueError, 'matching successful original logged freeze'):
            holdout.load_frozen_holdout(relative, root=self.fixture.root)

    def test_failed_freeze_receipt_is_not_forwarded(self):
        frozen, written, _ = self.fixture.freeze()
        receipt_ref = self.fixture.receipt(frozen, written)
        path = Path(receipt_ref['file'])
        document = json.loads(path.read_bytes())
        document['_meta']['exit_code'] = 1
        path.write_text(json.dumps(document))
        with self.assertRaisesRegex(ValueError, 'successful original freeze receipt'):
            entry.frozen_reference_from_receipt(path)

    def test_same_original_offload_and_inputs_for_every_arm(self):
        manifest = {'file': '/synthetic/manifest.json', 'sha256': 'a' * 64}
        frozen = {'file': '/synthetic/holdout.json', 'sha256': 'b' * 64}
        rows = [entry.training_argv('smoke_e2e_pilot_' + arm + '_01', arm, manifest, frozen, 1800)
                for arm in ('T0', 'T4', 'T2')]
        rows.append(entry.training_argv('smoke_e2e_pilot_engineering_T4_retry1', 'T4',
                    manifest, frozen, 1800, engineering=True))
        for command in rows:
            self.assertNotIn('--critic-offload', command)
            self.assertEqual(json.loads(command[command.index('--manifest') + 1]), manifest)
            self.assertEqual(json.loads(command[command.index('--holdout-reference') + 1]), frozen)
        self.assertTrue(all('--engineering-smoke' not in command for command in rows[:3]))
        self.assertEqual(rows[-1][-2:], ['--engineering-smoke', 't4_one_update_engineering_v1'])

    def test_original_failure_code_and_submitter_stop_bound_remain(self):
        with patch.object(entry.subprocess, 'run', return_value=Mock(returncode=86)), patch('builtins.print'):
            with self.assertRaises(SystemExit) as observed:
                entry.run_checked(['synthetic'])
            self.assertEqual(observed.exception.code, 86)
        with self.assertRaisesRegex(ValueError, 'at most4h'):
            entry.submit_preview('smoke_e2e_pilot_T0_01', ['synthetic'], 14401)


if __name__ == '__main__':
    unittest.main()
