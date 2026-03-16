import sys
import os

# Add project root to sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from core.native.engine import RepetitionTracker


def test_single_token_loop():
    print("Running test_single_token_loop...")
    tracker = RepetitionTracker()
    tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    for t in tokens:
        res = tracker.add(t)
        # print(f"Added {t}, looping: {res}")
        assert not res

    print("Adding 9 identical tokens...")
    # Add 9 identical tokens
    for i in range(9):
        res = tracker.add(11)
        # print(f"Added 11 (count {i+1}), looping: {res}")
        if res:
            print(f"FAILURE: Detected loop prematurely at count {i+1}")
            print(f"Current tokens: {tracker.tokens}")
        assert not res

    # The 10th one should trigger it
    print("Adding 10th identical token...")
    res = tracker.add(11)
    if not res:
        print("FAILURE: Did not detect loop at 10th token")
        print(f"Current tokens: {tracker.tokens}")
    assert res
    print("✓ Single token loop detected")


def test_sequence_loop():
    print("\nRunning test_sequence_loop...")
    tracker = RepetitionTracker()
    # Sequence: [100, 101] (length 2) - MUST HAVE AT LEAST 2 DISTINCT TOKENS
    seq = [100, 101]

    # Add 2 iterations
    for i in range(2):
        for t in seq:
            res = tracker.add(t)
            assert not res

    # Add 3rd iteration
    print("Adding 3rd iteration...")
    res = tracker.add(100)
    assert not res
    res = tracker.add(101)  # Detected on 3rd full sequence
    assert res
    print("✓ Sequence loop (len 2) detected")


def test_sequence_loop_longer():
    print("\nRunning test_sequence_loop_longer...")
    tracker = RepetitionTracker()
    # Sequence: [200, 201, 202, 203] (length 4)
    seq = [200, 201, 202, 203]

    # Add 2 iterations
    for i in range(2):
        for t in seq:
            res = tracker.add(t)
            assert not res

    # Add 3rd iteration
    print("Adding 3rd iteration...")
    assert not tracker.add(200)
    assert not tracker.add(201)
    assert not tracker.add(202)
    res = tracker.add(203)
    assert res
    print("✓ Sequence loop (len 4) detected")


if __name__ == "__main__":
    try:
        test_single_token_loop()
        test_sequence_loop()
        test_sequence_loop_longer()
        print("\nALL REPETITION TRACKER TESTS PASSED")
    except AssertionError:
        print("\nTEST FAILED (AssertionError)")
        sys.exit(1)
    except Exception as e:
        print(f"\nTEST FAILED (Unexpected Exception: {e})")
        sys.exit(1)
