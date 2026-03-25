import sys
import unittest
import os
import shutil
import tempfile

# Add project root to path
sys.path.append(os.getcwd())

from tools.file_ops import FileOps


class TestFileOpsFiltering(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.ops = FileOps(self.test_dir)

        # Create structure with noise
        os.makedirs(os.path.join(self.test_dir, "src"))
        os.makedirs(os.path.join(self.test_dir, "venv", "bin"))
        os.makedirs(os.path.join(self.test_dir, ".git"))
        os.makedirs(os.path.join(self.test_dir, "__pycache__"))

        with open(os.path.join(self.test_dir, "src", "main.py"), "w") as f:
            f.write("print('hello')")
        with open(os.path.join(self.test_dir, "venv", "bin", "python"), "w") as f:
            f.write("binary")
        with open(os.path.join(self.test_dir, "src", "main.pyc"), "w") as f:
            f.write("compiled")

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_list_dir_filtering_default(self):
        # Default should filter noise
        output = self.ops.list_dir(".", recursive=True)
        self.assertIn("[FILE] src/main.py", output)
        self.assertNotIn("[DIR]  venv", output)
        self.assertNotIn("[DIR]  .git", output)
        self.assertNotIn("[DIR]  __pycache__", output)
        self.assertNotIn("main.pyc", output)

    def test_list_dir_no_filtering(self):
        # Explicitly disabled filtering
        output = self.ops.list_dir(".", recursive=True, filter_noise=False)
        self.assertIn("[FILE] src/main.py", output)
        self.assertIn("[DIR]  venv", output)
        self.assertIn("[DIR]  .git", output)
        self.assertIn("[DIR]  __pycache__", output)
        self.assertIn("[FILE] src/main.pyc", output)


if __name__ == "__main__":
    unittest.main()
