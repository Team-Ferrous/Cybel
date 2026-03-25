import unittest
import numpy as np
from core.simd.simd_ops import SIMDOps


class TestSIMDSmoke(unittest.TestCase):
    def setUp(self):
        self.ops = SIMDOps()

    def test_library_load(self):
        self.assertTrue(self.ops.available, "SIMD library not available")

    def test_dot_product(self):
        if not self.ops.available:
            self.skipTest("SIMD lib not available")

        a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        b = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        got = self.ops.dot_product(a, b)
        self.assertAlmostEqual(got, 10.0, places=5)

    def test_mamba_ssm_scan_signature(self):
        if not self.ops.available:
            self.skipTest("SIMD lib not available")
        # Just check if the function exists in the library
        self.assertTrue(hasattr(self.ops.lib, "ssm_scan"), "ssm_scan symbol missing")

    def test_unified_attention_signature(self):
        if not self.ops.available:
            self.skipTest("SIMD lib not available")
        self.assertTrue(
            hasattr(self.ops.lib, "simd_flash_attention_forward"),
            "simd_flash_attention_forward symbol missing",
        )


if __name__ == "__main__":
    unittest.main()
