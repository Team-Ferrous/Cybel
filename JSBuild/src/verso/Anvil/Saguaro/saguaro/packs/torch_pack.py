"""Torch/deep-learning pack."""

from .base import PackSpec

TORCH_PACK = PackSpec(
    name="torch_pack",
    description="PyTorch, tensor, optimizer, and checkpoint heuristics.",
    keywords=["torch", "tensor", "optimizer", "state_dict", "attention", "checkpoint"],
    languages=["python", "cpp"],
    file_hints=["torch", "model", "trainer", "checkpoint"],
)
