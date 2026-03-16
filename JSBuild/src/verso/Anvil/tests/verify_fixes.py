import sys
import os

# Add root to path
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "Saguaro"))

from core.unified_chat_loop import UnifiedChatLoop


class MockAgent:
    def __init__(self):
        self.console = None
        self.brain = None
        self.history = None
        self.registry = None
        self.semantic_engine = None
        self.approval_manager = None
        self.name = "MockAgent"
        self.root_dir = os.getcwd()


def test_critical_file_selection():
    agent = MockAgent()
    UnifiedChatLoop.__init__ = lambda self, agent, enhanced_mode=True: None
    loop = UnifiedChatLoop(agent)

    # Test candidates
    candidates = [
        "saguaro/native/ops/unified_memory_system_op.py",
        "saguaro/native/_limits.py",
        "tests/test_something.py",
        "core/unified_chat_loop.py",
        "README.md",
        "saguaro/native/ops/fused_quls_loss_op.py",
    ]

    query = "explain how the coconut native bridge works with unified memory"
    critical = loop._identify_critical_files_fast(query, candidates)

    print(f"Query: {query}")
    print(f"Critical files identified: {critical}")

    assert "core/unified_chat_loop.py" in critical
    assert len(critical) > 0
    print("Critical File Selection Test Passed!")


def test_complexity_score():
    agent = MockAgent()
    UnifiedChatLoop.__init__ = lambda self, agent, enhanced_mode=True: None
    loop = UnifiedChatLoop(agent)

    # Test cases for complexity
    # Case 1: Low complexity (1 file, simple query)
    score1 = loop._calculate_complexity_score(
        query="what is this?",
        num_files=1,
        skeletons={"test.py": "def foo(): pass"},
        search_rounds=1,
        question_type="simple",
    )
    print(f"Score 1 (Simple): {score1}")
    assert score1 < 8  # Tier 1

    # Case 2: Medium complexity (architecture query)
    score2 = loop._calculate_complexity_score(
        query="explain the architecture",
        num_files=5,
        skeletons={"f1.py": "class A: ...", "f2.py": "class B: ..."},
        search_rounds=2,
        question_type="architecture",
    )
    print(f"Score 2 (Architecture): {score2}")
    assert score2 >= 8  # Tier 2 (FileAnalyst)

    # Case 3: High complexity (many files + dense skeletons)
    dense_skeleton = "class Large:\n" + "    def method(self): pass\n" * 100
    score3 = loop._calculate_complexity_score(
        query="how does the whole system work?",
        num_files=12,
        skeletons={f"f{i}.py": dense_skeleton for i in range(12)},
        search_rounds=2,
        question_type="research",
    )
    print(f"Score 3 (High): {score3}")
    assert score3 >= 20  # Tier 3 (MultiAgent)

    print("Complexity Score Tests Passed!")


if __name__ == "__main__":
    test_critical_file_selection()
    test_complexity_score()
