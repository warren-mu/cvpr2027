"""CPU synthetic safety fixtures; no downloads/install/TeX process."""
import io
import tarfile
import unittest

from src.tools.r01_tex_install import parse_tlpdb, safe_members


class TexLockTests(unittest.TestCase):
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
