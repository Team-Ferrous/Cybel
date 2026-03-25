import sys
import unittest
import os
import shutil
import tempfile

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.file_ops import FileOps


class TestExtendedFileOps(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.ops = FileOps(self.test_dir)

        # Create initial structure
        os.makedirs(os.path.join(self.test_dir, "subdir"))
        with open(os.path.join(self.test_dir, "file1.txt"), "w") as f:
            f.write("content1")
        with open(os.path.join(self.test_dir, "subdir", "file2.txt"), "w") as f:
            f.write("content2")

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_list_dir(self):
        # Test flat list
        output = self.ops.list_dir(".")
        self.assertIn("[FILE] file1.txt", output)
        self.assertIn("[DIR]  subdir", output)

        # Test recursive list
        output_recursive = self.ops.list_dir(".", recursive=True)
        self.assertIn("[FILE] subdir/file2.txt", output_recursive)

    def test_move_file(self):
        before = self.ops._state_ledger.delta_watermark()
        # Move file1.txt to file1_renamed.txt
        res = self.ops.move_file("file1.txt", "file1_renamed.txt")
        self.assertTrue("Successfully moved" in res)
        self.assertFalse(os.path.exists(os.path.join(self.test_dir, "file1.txt")))
        self.assertTrue(
            os.path.exists(os.path.join(self.test_dir, "file1_renamed.txt"))
        )
        changes = self.ops._state_ledger.changeset_since(before)
        self.assertEqual(changes["changed_files"], ["file1_renamed.txt"])
        self.assertEqual(changes["deleted_files"], ["file1.txt"])

    def test_delete_file(self):
        before = self.ops._state_ledger.delta_watermark()
        # Delete subdir/file2.txt
        res = self.ops.delete_file("subdir/file2.txt")
        self.assertTrue("Successfully deleted" in res)
        self.assertFalse(
            os.path.exists(os.path.join(self.test_dir, "subdir", "file2.txt"))
        )
        changes = self.ops._state_ledger.changeset_since(before)
        self.assertEqual(changes["deleted_files"], ["subdir/file2.txt"])
        # Backup should exist (though hard to test exact path without digging into BackupManager)


if __name__ == "__main__":
    unittest.main()
