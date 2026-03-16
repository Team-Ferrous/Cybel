from core.ownership.ownership_crdt import LWWEntry, OwnershipCRDT


def serialize_state(crdt: OwnershipCRDT):
    return {
        path: (
            entry.owner_agent_id,
            entry.owner_instance_id,
            entry.mode,
            entry.timestamp,
            entry.instance_id,
            entry.is_tombstone,
        )
        for path, entry in sorted(crdt.state.items())
    }


def test_lww_latest_timestamp_wins():
    crdt = OwnershipCRDT("inst-a")
    older = crdt.claim("core/a.py", "agent-a")
    newer = crdt.claim("core/a.py", "agent-b")

    replica = OwnershipCRDT("inst-b")
    replica.merge({"core/a.py": older})
    replica.merge({"core/a.py": newer})

    assert replica.state["core/a.py"].owner_agent_id == "agent-b"


def test_lww_tie_breaking_by_instance_id():
    crdt = OwnershipCRDT("inst-a")
    entry_a = LWWEntry(
        file_path="core/a.py",
        owner_agent_id="agent-a",
        owner_instance_id="inst-a",
        mode="exclusive",
        timestamp=10,
        instance_id="inst-a",
    )
    entry_b = LWWEntry(
        file_path="core/a.py",
        owner_agent_id="agent-b",
        owner_instance_id="inst-b",
        mode="exclusive",
        timestamp=10,
        instance_id="inst-b",
    )

    crdt.merge({"core/a.py": entry_a})
    crdt.merge({"core/a.py": entry_b})

    assert crdt.state["core/a.py"].owner_instance_id == "inst-b"


def test_merge_convergence():
    a = OwnershipCRDT("a")
    b = OwnershipCRDT("b")

    a.claim("core/a.py", "agent-a")
    b.claim("core/b.py", "agent-b")

    a.merge(b.state)
    b.merge(a.state)

    assert serialize_state(a) == serialize_state(b)


def test_merge_commutative():
    a = OwnershipCRDT("a")
    b = OwnershipCRDT("b")

    a.claim("core/a.py", "agent-a")
    b.claim("core/b.py", "agent-b")

    left = OwnershipCRDT("left")
    left.merge(a.state)
    left.merge(b.state)

    right = OwnershipCRDT("right")
    right.merge(b.state)
    right.merge(a.state)

    assert serialize_state(left) == serialize_state(right)


def test_merge_idempotent():
    a = OwnershipCRDT("a")
    a.claim("core/a.py", "agent-a")

    before = serialize_state(a)
    a.merge(a.state)
    after = serialize_state(a)

    assert before == after


def test_merge_associative():
    a = OwnershipCRDT("a")
    b = OwnershipCRDT("b")
    c = OwnershipCRDT("c")

    a.claim("core/a.py", "agent-a")
    b.claim("core/b.py", "agent-b")
    c.claim("core/c.py", "agent-c")

    left = OwnershipCRDT("left")
    left.merge(a.state)
    left.merge(b.state)
    left.merge(c.state)

    right = OwnershipCRDT("right")
    right.merge(a.state)
    tmp = OwnershipCRDT("tmp")
    tmp.merge(b.state)
    tmp.merge(c.state)
    right.merge(tmp.state)

    assert serialize_state(left) == serialize_state(right)


def test_tombstone_prevents_reclaim_before_gc():
    crdt = OwnershipCRDT("inst-a")
    crdt.claim("core/a.py", "agent-a")
    crdt.release("core/a.py", "agent-a")

    try:
        crdt.claim("core/a.py", "agent-b")
    except RuntimeError:
        pass
    else:
        raise AssertionError("Expected tombstone reclaim failure")

    crdt.gc_tombstones()
    entry = crdt.claim("core/a.py", "agent-b")
    assert entry.owner_agent_id == "agent-b"


def test_delta_sync_correctness():
    crdt = OwnershipCRDT("inst-a")
    first = crdt.claim("core/a.py", "agent-a")
    second = crdt.claim("core/b.py", "agent-b")

    delta = crdt.diff(other_clock=first.timestamp)

    assert "core/a.py" not in delta
    assert "core/b.py" in delta
    assert delta["core/b.py"].timestamp == second.timestamp
