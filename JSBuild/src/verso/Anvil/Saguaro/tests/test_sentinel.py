"""
Test script for SAGUARO Sentinel.
"""

import unittest
import tempfile
import shutil
import os
import sys
import yaml

# Ensure we can import from local
sys.path.append(os.getcwd())

from saguaro.sentinel.rules import Rule
from saguaro.sentinel.verifier import SentinelVerifier


class TestSentinel(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_rule_checker(self):
        rule = Rule(id="test", pattern="foo", message="No foo allowed")
        violations = rule.check("bar\nfoo\nbaz")
        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0][0], 2)  # Line 2

    def test_verifier_integration(self):
        # Create rules
        rules = {
            "rules": [
                {
                    "id": "no-secrets",
                    "pattern": "SECRET",
                    "message": "No secrets",
                    "severity": "ERROR",
                }
            ]
        }
        with open(os.path.join(self.test_dir, ".saguaro.rules"), "w") as f:
            yaml.dump(rules, f)

        # Create violations
        with open(os.path.join(self.test_dir, "bad.py"), "w") as f:
            f.write("print('SECRET_KEY')\n")

        # Verify
        previous = os.environ.get("SAGUARO_ALLOW_LEGACY_RULES")
        os.environ["SAGUARO_ALLOW_LEGACY_RULES"] = "1"
        try:
            verifier = SentinelVerifier(self.test_dir, engines=["native"])
            violations = verifier.verify_all()
        finally:
            if previous is None:
                os.environ.pop("SAGUARO_ALLOW_LEGACY_RULES", None)
            else:
                os.environ["SAGUARO_ALLOW_LEGACY_RULES"] = previous

        self.assertEqual(len(violations), 2)
        self.assertEqual(violations[0]["rule_id"], "no-secrets")


if __name__ == "__main__":
    unittest.main()
