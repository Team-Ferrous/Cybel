import numpy as np

from core.qsg.jacobi_refiner import JacobiRefiner


def test_jacobi_frontier_state_tracks_verified_candidates() -> None:
    refiner = JacobiRefiner(iterations=2)
    probs = np.full((1, 4, 6), 1.0e-4, dtype=np.float32)
    probs[0, 0, 2] = 0.82
    probs[0, 1, 2] = 0.78
    probs[0, 2, 2] = 0.75
    probs[0, 3, 4] = 0.73
    probs /= probs.sum(axis=-1, keepdims=True)
    coherence = np.asarray(
        [
            [1.0, 0.5, 0.0, 0.0],
            [0.5, 1.0, 0.4, 0.0],
            [0.0, 0.4, 1.0, 0.5],
            [0.0, 0.0, 0.5, 1.0],
        ],
        dtype=np.float32,
    )

    refined = refiner.refine(probs, coherence, frontier_width=2)
    frontier = refiner.last_frontier_state

    assert refined.shape == probs.shape
    assert frontier.frontier_width == 2
    assert frontier.branch_survival_rate > 0.0
    assert frontier.verify_cost_ms >= 0.0
    assert frontier.branch_entropy >= 0.0
    assert frontier.surviving_tokens
    assert int(np.argmax(refined[0, 1])) == 2
    assert refined[0, 3, 2] > probs[0, 3, 2]
