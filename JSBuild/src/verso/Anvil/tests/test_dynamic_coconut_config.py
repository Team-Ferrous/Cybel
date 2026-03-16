from config.settings import DYNAMIC_COCONUT_CONFIG


def test_dynamic_coconut_config_has_required_thresholds():
    required_keys = {
        "adaptive_min_steps",
        "adaptive_max_steps",
        "adaptive_entropy_threshold",
        "adaptive_confidence_threshold",
        "max_subagent_slots",
    }
    assert required_keys.issubset(DYNAMIC_COCONUT_CONFIG.keys())
