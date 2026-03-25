from core.campaign.campaign_generator import CampaignGenerator
from core.loops.loop_builder import LoopValidator


def make_generator(tmp_path):
    generator = CampaignGenerator(agent=None)
    generator.config["generated_dir"] = str(tmp_path / "generated")
    return generator


def test_generated_code_passes_validation(tmp_path):
    generator = make_generator(tmp_path)
    path = generator.generate_from_description("Analyze repo for readiness", target_repo=str(tmp_path))

    code = open(path, "r", encoding="utf-8").read()
    valid, errors = LoopValidator().validate_all(code)
    assert valid is True
    assert errors == []


def test_generated_code_has_phases(tmp_path):
    generator = make_generator(tmp_path)
    path = generator.generate_from_description("Build a campaign", target_repo=str(tmp_path))
    code = open(path, "r", encoding="utf-8").read()

    assert "@phase(order=0" in code
    assert "def phase_baseline" in code


def test_generated_code_has_gates(tmp_path):
    generator = make_generator(tmp_path)
    path = generator.generate_from_description("Build a campaign", target_repo=str(tmp_path))
    code = open(path, "r", encoding="utf-8").read()

    assert "@gate(phase=\"phase_baseline\")" in code


def test_yaml_to_code_compilation(tmp_path):
    generator = make_generator(tmp_path)
    spec = tmp_path / "spec.yaml"
    spec.write_text(
        """
campaign:
  name: "Spec Campaign"
  phases:
    - id: "baseline"
      name: "Baseline"
      iteration: "none"
      gate:
        assertions:
          - "processed >= 0"
""",
        encoding="utf-8",
    )

    path = generator.generate_from_yaml(str(spec))
    code = open(path, "r", encoding="utf-8").read()

    assert "class SpecCampaign" in code
    assert "def phase_baseline" in code


def test_file_iteration_present_when_requested(tmp_path):
    generator = make_generator(tmp_path)
    spec = tmp_path / "iter.yaml"
    spec.write_text(
        """
campaign:
  name: "Iter Campaign"
  phases:
    - id: "file_analysis"
      name: "File Analysis"
      iteration: "all_files"
      per_file_agent_task: "Analyze this file"
      gate:
        assertions:
          - "processed >= total_files"
""",
        encoding="utf-8",
    )

    path = generator.generate_from_yaml(str(spec))
    code = open(path, "r", encoding="utf-8").read()
    assert "targets = self.discover_files" in code
    assert "for file_path in targets" in code
