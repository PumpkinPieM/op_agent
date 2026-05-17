from pathlib import Path

import yaml


SKILL_ROOT = Path(__file__).resolve().parents[1]
SKILLS_ROOT = SKILL_ROOT.parent
TESTS_DIR = Path(__file__).resolve().parent
ROUTING_CASES = TESTS_DIR / "routing_cases.yaml"
HUMAN_CASES = TESTS_DIR / "test_op_agent_routing_cases.md"
SKILL_MD = SKILL_ROOT / "SKILL.md"

CANONICAL_BUILDERS = {
    "cpu-native-builder",
    "cpu-plugin-builder",
    "gpu-builder",
    "aclnn-builder",
}
CANONICAL_BACKENDS = {"CPU", "GPU", "NPU"}
EXPECTED_CASE_IDS = {
    "cpu_plugin_default",
    "cpu_native_override",
    "npu_alias_ascend",
    "npu_alias_aclnn",
    "gpu_direct_route",
    "cpu_ambiguity_defaults_to_plugin",
    "npu_mint_api",
}


def load_routing_cases():
    return yaml.safe_load(ROUTING_CASES.read_text(encoding="utf-8"))


def case_map(data):
    return {case["id"]: case for case in data["cases"]}


def test_routing_cases_yaml_exists_and_has_expected_cases():
    assert ROUTING_CASES.exists(), f"Missing routing case contract: {ROUTING_CASES}"
    data = load_routing_cases()
    assert data["schema_version"] == "1.1.0"
    assert {case["id"] for case in data["cases"]} == EXPECTED_CASE_IDS


def test_all_cases_use_valid_backends_and_handoff_targets():
    data = load_routing_cases()
    for case in data["cases"]:
        assert isinstance(case["input"]["known_evidence"], str)
        expected = case["expected"]
        assert expected["normalized_backend"] in CANONICAL_BACKENDS
        assert expected["best_fit"] in CANONICAL_BUILDERS
        assert expected["handoff_skill"] in CANONICAL_BUILDERS
        assert expected["handoff_skill"] == expected["best_fit"]
        assert isinstance(expected["handoff_task"], str)
        assert expected["handoff_task"]
        assert expected["start_now"] is True
        assert expected["forbid_codegen"] is True


def test_handoff_targets_exist_as_real_skills():
    data = load_routing_cases()
    for case in data["cases"]:
        handoff_skill = case["expected"]["handoff_skill"]
        handoff_skill_md = SKILLS_ROOT / handoff_skill / "SKILL.md"
        assert handoff_skill_md.exists(), f"Missing handoff skill doc: {handoff_skill_md}"


def test_npu_aliases_normalize_to_npu_and_dispatch_to_aclnn():
    data = case_map(load_routing_cases())
    for case_id, raw_value in {
        "npu_alias_ascend": "Ascend",
        "npu_alias_aclnn": "aclnn",
    }.items():
        case = data[case_id]
        assert case["input"]["target_backend_raw"] == raw_value
        assert case["expected"]["normalized_backend"] == "NPU"
        assert case["expected"]["best_fit"] == "aclnn-builder"
        assert case["expected"]["handoff_skill"] == "aclnn-builder"


def test_cpu_gpu_and_default_dispatch_contracts():
    data = case_map(load_routing_cases())

    assert data["cpu_plugin_default"]["expected"]["best_fit"] == "cpu-plugin-builder"
    assert data["cpu_plugin_default"]["expected"]["handoff_skill"] == "cpu-plugin-builder"

    assert data["cpu_native_override"]["expected"]["best_fit"] == "cpu-native-builder"
    assert data["cpu_native_override"]["expected"]["handoff_skill"] == "cpu-native-builder"

    assert data["cpu_ambiguity_defaults_to_plugin"]["expected"]["best_fit"] == "cpu-plugin-builder"
    assert data["cpu_ambiguity_defaults_to_plugin"]["expected"]["handoff_skill"] == "cpu-plugin-builder"

    assert data["gpu_direct_route"]["expected"]["normalized_backend"] == "GPU"
    assert data["gpu_direct_route"]["expected"]["best_fit"] == "gpu-builder"
    assert data["gpu_direct_route"]["expected"]["handoff_skill"] == "gpu-builder"


def test_skill_md_contains_handoff_contract():
    text = SKILL_MD.read_text(encoding="utf-8")
    assert "## Normalization Rules" in text
    assert "`Ascend` and `aclnn` both map to `NPU`." in text
    assert "Use canonical builder names exactly" in text
    assert "cpu-plugin-builder" in text
    assert "cpu-native-builder" in text
    assert "gpu-builder" in text
    assert "aclnn-builder" in text
    assert "## Minimal Examples" in text
    assert "## Response Format" in text
    assert "Handoff:" in text
    assert "Load skill:" in text
    assert "Start now:" in text


def test_human_readable_cases_doc_is_retained():
    assert HUMAN_CASES.exists(), f"Missing human-readable routing guide: {HUMAN_CASES}"
    text = HUMAN_CASES.read_text(encoding="utf-8")
    assert "tests/routing_cases.yaml" in text
    assert "human-readable companion" in text
    assert "`target_backend_raw` shows the original user input before normalization" in text
    assert "target_backend_raw:" in text
