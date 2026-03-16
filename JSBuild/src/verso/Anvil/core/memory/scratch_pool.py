import ctypes
import numpy as np
from core.simd.simd_ops import SIMDOps


class ScratchPool:
    """
    Thread-local scratch buffer pool using HighNoon SIMD allocator.
    """

    def __init__(self):
        # Ensure library is loaded
        self._simd = SIMDOps()
        self._lib = SIMDOps._lib
        if self._lib:
            self._setup_signatures()

    def _setup_signatures(self):
        # float* scratch_get(int64_t size, int slot)
        self._lib.scratch_get.argtypes = [ctypes.c_int64, ctypes.c_int]
        self._lib.scratch_get.restype = ctypes.POINTER(ctypes.c_float)

        # void scratch_clear()
        self._lib.scratch_clear.argtypes = []
        self._lib.scratch_clear.restype = None

    def get(self, size_floats: int, slot: int = 0) -> np.ndarray:
        """
        Get a scratch buffer of at least size_floats.
        Returns a numpy array view (no copy) of the buffer.
        """
        if self._lib is None:
            # Fallback path if lib loading failed (mostly for dev/test without build)
            return np.zeros(size_floats, dtype=np.float32)

        ptr = self._lib.scratch_get(size_floats, slot)
        if not ptr:
            raise MemoryError("Failed to allocate scratch buffer")

        # Create numpy array from pointer without copying
        # We assume the pointer is valid as long as we don't clear the pool
        # and don't call get() again on the same slot with a larger size (which might realloc)
        buffer_from_memory = ctypes.cast(
            ptr, ctypes.POINTER(ctypes.c_float * size_floats)
        ).contents
        return np.frombuffer(buffer_from_memory, dtype=np.float32)

    def clear(self) -> None:
        """Clear the thread-local pool, freeing memory."""
        if self._lib:
            self._lib.scratch_clear()
