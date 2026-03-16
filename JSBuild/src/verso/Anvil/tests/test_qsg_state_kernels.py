import numpy as np

from core.native.qsg_state_kernels_wrapper import (
    qsg_latent_decode_f16,
    qsg_latent_encode_f16,
    qsg_state_weighted_merge,
)


def test_qsg_latent_f16_codec_roundtrip() -> None:
    source = np.asarray(
        [[1.0, -2.0, 0.5, 1024.0], [0.25, -0.75, 3.5, -16.0]],
        dtype=np.float32,
    )

    encoded = qsg_latent_encode_f16(source)
    restored = qsg_latent_decode_f16(encoded)

    assert encoded.dtype == np.uint16
    assert restored.dtype == np.float32
    np.testing.assert_allclose(restored, source, atol=0.6)


def test_qsg_state_weighted_merge_matches_numpy_average() -> None:
    source = np.asarray(
        [[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0], [8.0, 6.0, 4.0, 2.0]],
        dtype=np.float32,
    )
    weights = np.asarray([0.2, 0.3, 0.5], dtype=np.float32)

    merged = qsg_state_weighted_merge(source, weights)
    expected = np.average(source, axis=0, weights=weights).astype(np.float32)

    np.testing.assert_allclose(merged, expected, atol=1e-5)
