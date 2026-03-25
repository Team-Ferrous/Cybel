import unittest
import os
import shutil
import tempfile
from rich.console import Console

# Import our new components
from saguaro.refactor.planner import RefactorPlanner, DependencyGraph
from saguaro.sentinel.verifier import SentinelVerifier


class TestPlanFailure(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.console = Console(quiet=True)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def create_file(self, filename, content):
        path = os.path.join(self.test_dir, filename)
        with open(path, "w") as f:
            f.write(content)
        return path

    def test_cycle_detection(self):
        """Test Scenario B: Circular Dependency detection."""

        # Create a cycle: a.py -> b.py -> a.py
        self.create_file("a.py", "import b\n\ndef foo():\n    pass")
        self.create_file("b.py", "import a\n\ndef bar():\n    pass")

        print(f"\nScanning directory: {self.test_dir}")

        # 1. Direct Graph Test
        planner = RefactorPlanner(self.test_dir)
        graph = planner._build_dependency_graph(
            [os.path.join(self.test_dir, "a.py"), os.path.join(self.test_dir, "b.py")]
        )

        cycles = planner.detect_cycles(graph)
        print(f"Cycles detected: {cycles}")

        self.assertTrue(len(cycles) > 0, "Should detect circular dependency")

        # 2. Sentinel Engine Test
        verifier = SentinelVerifier(self.test_dir, engines=["graph"])
        violations = verifier.verify_all()

        print(f"Sentinel violations: {violations}")

        found_cycle_error = any(v["rule_id"] == "GRAPH_001_CYCLE" for v in violations)
        self.assertTrue(found_cycle_error, "Sentinel should report GRAPH_001_CYCLE")

    def test_inconsistency_check(self):
        """Test Scenario A: Inconsistency (Reference to missing file - implicit via graph)."""
        # Graph builder handles existing files.
        # But if we reference a missing import, does it behave safely?

        self.create_file("c.py", "import non_existent_file\n")

        planner = RefactorPlanner(self.test_dir)
        graph = planner._build_dependency_graph([os.path.join(self.test_dir, "c.py")])

        # Should NOT crash
        print("Graph built successfully despite missing import")
        self.assertIsInstance(graph, DependencyGraph)


if __name__ == "__main__":
    unittest.main()
