"""Synthetic path-only fixtures; root runs after commit via run_logged.

No Ray import, ray.init, socket bind, GPU call or service launch is needed.
"""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.tools.ray_scratch import SOCKET_SUFFIX, reserve_directory, validate_socket_budget


class RayScratchTests(unittest.TestCase):
    def test_observed_long_path_fails_but_short_scratch_root_fits(self):
        root = Path("/scratch/e1583364/cvpr2027")
        with self.assertRaises(ValueError):
            validate_socket_budget(root / "t00b_scale/t00b_scale8_baseline-706439.hopper-m-02/ray")
        record = validate_socket_budget(root / "r/abcdef")
        self.assertLessEqual(record["maximum_socket_path_bytes"], 107)
        self.assertTrue(record["maximum_socket_path"].endswith(SOCKET_SUFFIX))
        self.assertIn("_2147483647/sockets/plasma_store", record["maximum_socket_path"])

    def test_socket_limit_counts_utf8_bytes_and_includes_ray_subdirectory(self):
        prefix_bytes = 107 - len(("/" + SOCKET_SUFFIX).encode())
        directory = "/" + "a" * (prefix_bytes - 1)
        self.assertEqual(validate_socket_budget(directory)["maximum_socket_path_bytes"], 107)
        self.assertIn("/ray/session_", validate_socket_budget(directory)["maximum_socket_path"])
        with self.assertRaises(ValueError):
            validate_socket_budget(directory + "a")
        with self.assertRaises(ValueError):
            validate_socket_budget(directory[:-1] + "é")

    def test_variants_get_private_retained_directories_and_collision_does_not_reuse(self):
        with tempfile.TemporaryDirectory(prefix="r", dir="/tmp") as temporary:
            root = Path(temporary).resolve()
            original_environment = dict(os.environ)
            with patch("src.tools.ray_scratch.secrets.token_hex", side_effect=["aaaaaa", "aaaaaa", "bbbbbb"]):
                first = reserve_directory(root, "run_baseline", "fixture.pbs", root / "long/raw/first", root / "first.json")
                marker = Path(first["ray_tmpdir"]) / "preserve.txt"
                marker.write_text("retained")
                second = reserve_directory(root, "run_vllm04", "fixture.pbs", root / "long/raw/second", root / "second.json")
            self.assertNotEqual(first["ray_tmpdir"], second["ray_tmpdir"])
            self.assertEqual(marker.read_text(), "retained")
            self.assertEqual(Path(first["ray_tmpdir"]).stat().st_mode & 0o777, 0o700)
            self.assertEqual(json.loads((root / "first.json").read_text()), first)
            self.assertEqual(json.loads((Path(second["ray_tmpdir"]) / "reservation.json").read_text()), second)
            self.assertEqual(dict(os.environ), original_environment)

    def test_existing_provenance_is_not_overwritten(self):
        with tempfile.TemporaryDirectory(prefix="r", dir="/tmp") as temporary:
            root = Path(temporary).resolve()
            record = root / "record.json"
            record.write_text("old evidence")
            with self.assertRaises(FileExistsError):
                reserve_directory(root, "run", "fixture.pbs", root / "raw", record)
            self.assertEqual(record.read_text(), "old evidence")
            self.assertFalse((root / "r").exists())

    def test_long_or_escaped_scratch_path_fails_before_reservation(self):
        with tempfile.TemporaryDirectory(prefix="r", dir="/tmp") as temporary:
            root = Path(temporary).resolve()
            long = root / ("long" * 30)
            with self.assertRaises(ValueError):
                reserve_directory(long, "run", "fixture.pbs", long / "raw", root / "long.json")
            self.assertFalse(long.exists())
            with self.assertRaises(ValueError):
                reserve_directory(root, "run", "fixture.pbs", root.parent / "outside", root / "outside.json")
            (root / "r").symlink_to(root.parent, target_is_directory=True)
            with self.assertRaises(ValueError):
                reserve_directory(root, "run", "fixture.pbs", root / "raw", root / "escaped.json")


if __name__ == "__main__":
    unittest.main()
