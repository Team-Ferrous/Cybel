from core.agents.domain.em.electromagnetism_subagent import ElectromagnetismSubagent
from core.agents.domain.industrial.manufacturing_automation_subagent import (
    ManufacturingAutomationSubagent,
)
from core.agents.domain.robotics.drone_autonomy_subagent import DroneAutonomySubagent
from core.agents.domain.surrogates.cfd_surrogate_subagent import CFDSurrogateSubagent
from core.agents.domain.toolchain.dead_code_triage_subagent import (
    DeadCodeTriageSubagent,
)


def test_new_specialists_include_web_and_arxiv_research_baseline():
    for specialist in [
        ElectromagnetismSubagent,
        CFDSurrogateSubagent,
        DroneAutonomySubagent,
        ManufacturingAutomationSubagent,
    ]:
        tools = set(specialist.tools)
        assert {"web_search", "web_fetch", "search_arxiv", "fetch_arxiv_paper"} <= tools


def test_deadcode_triage_specialist_has_deadcode_tool():
    assert "deadcode" in DeadCodeTriageSubagent.tools
    assert "unwired" in DeadCodeTriageSubagent.tools
