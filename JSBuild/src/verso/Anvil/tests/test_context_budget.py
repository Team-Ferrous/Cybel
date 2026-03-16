from core.context import ContextBudgetAllocator


def test_budget_is_preserved():
    allocator = ContextBudgetAllocator(total_budget=400000)
    assert sum(allocator.budgets.values()) == 400000
    assert allocator.get_budget("master") > allocator.get_budget("system")


def test_subagent_pool_divides_by_active_count():
    allocator = ContextBudgetAllocator(total_budget=400000)
    single = allocator.get_subagent_budget(1)
    four = allocator.get_subagent_budget(4)
    assert single > four
    assert four == allocator.get_budget("subagent_pool") // 4


def test_coconut_latent_budget_is_independent():
    allocator = ContextBudgetAllocator(total_budget=400000)
    latent = allocator.coconut_latent_budget
    assert latent >= 200000
    assert latent not in allocator.budgets.values()
