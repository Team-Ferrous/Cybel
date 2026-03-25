import os
import shutil
import tempfile
import unittest

from tools.file_ops import FileOps


class TestFileOpsRollback(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.ops = FileOps(self.test_dir)
        self.path = os.path.join(self.test_dir, "sample.txt")
        with open(self.path, "w", encoding="utf-8") as handle:
            handle.write("v1\n")

        # Read first to satisfy stale-token policy, then write v2.
        _ = self.ops.read_file("sample.txt")
        self.ops.write_file("sample.txt", "v2\n")

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_list_backups_and_rollback(self):
        before = self.ops._state_ledger.delta_watermark()
        listing = self.ops.list_backups("sample.txt")
        self.assertIn(".bak", listing)

        result = self.ops.rollback_file("sample.txt")
        self.assertIn("Rolled back", result)

        with open(self.path, "r", encoding="utf-8") as handle:
            content = handle.read()
        self.assertEqual(content, "v1\n")
        changes = self.ops._state_ledger.changeset_since(before)
        self.assertEqual(changes["changed_files"], ["sample.txt"])


if __name__ == "__main__":
    unittest.main()
