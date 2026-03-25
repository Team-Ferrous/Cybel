from __future__ import annotations

import pytest

from core.native.native_qsg_engine import NativeQSGEngine


def test_hybrid_graph_dispatch_is_disabled_in_strict_cpp_mode():
    engine = NativeQSGEngine.__new__(NativeQSGEngine)
    with pytest.raises(RuntimeError, match="disabled"):
        engine._get_logits_hybrid([7], start_pos=3)
