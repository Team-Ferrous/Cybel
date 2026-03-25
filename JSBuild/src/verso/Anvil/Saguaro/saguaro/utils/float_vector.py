"""Lightweight float vector with buffer protocol and shape metadata."""

from __future__ import annotations

from array import array


class FloatVector(array):
    """Array-backed float vector that preserves a minimal ndarray-like shape."""

    def __new__(cls, values: object = ()) -> "FloatVector":
        return super().__new__(cls, "f", values)

    @classmethod
    def zeros(cls, dim: int) -> "FloatVector":
        return cls([0.0] * max(0, int(dim)))

    @property
    def shape(self) -> tuple[int]:
        return (len(self),)
