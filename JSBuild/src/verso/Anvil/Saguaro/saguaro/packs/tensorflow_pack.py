"""TensorFlow pack."""

from .base import PackSpec

TENSORFLOW_PACK = PackSpec(
    name="tensorflow_pack",
    description="TensorFlow graphs, keras training loops, checkpoints, and device placement.",
    keywords=["tensorflow", "tf.", "keras", "savedmodel", "checkpoint", "gradienttape"],
    languages=["python", "cpp"],
    file_hints=["tensorflow", "keras", "saved_model", "checkpoint"],
)
