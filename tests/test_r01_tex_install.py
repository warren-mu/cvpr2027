"""CPU synthetic safety fixtures; no downloads/install/TeX process."""
import io
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from src.tools.r01_tex_install import EXTRA_PACKAGES, REQUIRED_FILES, acquire_archive, file_hash, parse_tlpdb, safe_members


class TexLockTests(unittest.TestCase):
    def test_grfext_is_locked_and_checked_as_required_input(self):
        self.assertIn("grfext", EXTRA_PACKAGES)
        self.assertIn("grfext.sty", REQUIRED_FILES)

    def test_verified_cache_copy_preserves_source_and_avoids_download(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old, new = root / "old", root / "new"
            old.mkdir()
            source = old / "package.tar.xz"
            source.write_bytes(b"synthetic exact archive")
            expected = file_hash(source, "sha512")
            with patch("src.tools.r01_tex_install.download") as download:
                record = acquire_archive("https://example.invalid/package.tar.xz", new / source.name,
                                         expected, "sha512", old)
            download.assert_not_called()
            self.assertTrue(record["reused_verified"])
            self.assertEqual(file_hash(source, "sha512"), expected)
            self.assertEqual((new / source.name).read_bytes(), source.read_bytes())

    def test_old_revision_mismatch_is_not_copied_or_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = root / "old"
            old.mkdir()
            source = old / "package.tar.xz"
            source.write_bytes(b"older revision")
            target = root / "new" / source.name
            with patch("src.tools.r01_tex_install.download", return_value={"sha256": "new"}) as download:
                record = acquire_archive("https://example.invalid/package.tar.xz", target,
                                         "a" * 64, reuse_cache=old)
            download.assert_called_once_with("https://example.invalid/package.tar.xz", target, "a" * 64, "sha256")
            self.assertIn("not_reused_checksum_mismatch", record)
            self.assertFalse(target.exists())
            self.assertEqual(source.read_bytes(), b"older revision")

    def test_cache_symlink_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = root / "old"
            old.mkdir()
            outside = root / "outside"
            outside.write_bytes(b"synthetic archive")
            (old / "package.tar.xz").symlink_to(outside)
            with self.assertRaises(ValueError):
                acquire_archive("https://example.invalid/package.tar.xz", root / "new/package.tar.xz",
                                file_hash(outside), reuse_cache=old)

    def test_existing_destination_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old, new = root / "old", root / "new"
            old.mkdir()
            new.mkdir()
            source, target = old / "package.tar.xz", new / "package.tar.xz"
            source.write_bytes(b"exact archive")
            target.write_bytes(b"existing evidence")
            with self.assertRaises(ValueError):
                acquire_archive("https://example.invalid/package.tar.xz", target,
                                file_hash(source), reuse_cache=old)
            self.assertEqual(target.read_bytes(), b"existing evidence")

    def test_package_revisions_and_dependencies_retained(self):
        records = parse_tlpdb("name caption\nrevision 123\ndepend latex\ncontainerchecksum abc\n\nname latex\nrevision 456\n")
        self.assertEqual(records["caption"]["revision"], ["123"])
        self.assertEqual(records["caption"]["depend"], ["latex"])

    def test_duplicate_package_rejected(self):
        with self.assertRaises(ValueError):
            parse_tlpdb("name a\nrevision 1\n\nname a\nrevision 2\n")

    def archive(self, members):
        stream = io.BytesIO()
        with tarfile.open(fileobj=stream, mode="w") as archive:
            for member in members:
                archive.addfile(member)
        stream.seek(0)
        return tarfile.open(fileobj=stream)

    def test_safe_internal_binary_link(self):
        link = tarfile.TarInfo(".TinyTeX/bin/x86_64-linux/latex")
        link.type, link.linkname = tarfile.SYMTYPE, "pdftex"
        with self.archive([link]) as archive:
            self.assertEqual(len(safe_members(archive)), 1)

    def test_escaping_archive_link_rejected(self):
        link = tarfile.TarInfo(".TinyTeX/link")
        link.type, link.linkname = tarfile.SYMTYPE, "../../../outside"
        with self.archive([link]) as archive, self.assertRaises(ValueError):
            safe_members(archive)

    def test_archive_write_through_link_rejected(self):
        link = tarfile.TarInfo(".TinyTeX/link")
        link.type, link.linkname = tarfile.SYMTYPE, "inside"
        with self.archive([link, tarfile.TarInfo(".TinyTeX/link/file")]) as archive, self.assertRaises(ValueError):
            safe_members(archive)


if __name__ == "__main__":
    unittest.main()
