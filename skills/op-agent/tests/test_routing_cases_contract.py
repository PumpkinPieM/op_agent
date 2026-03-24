from pathlib import Path

import yaml


SKILL_ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = Path(__file__).resolve().parent
ROUTING_CASES = TESTS_DIR / "routing_cases.yaml"
HUMAN_CASES = TESTS_DIR / "test_op_agent_routing_cases.md"
SKILL_MD = SKILL_ROOT / "SKILL.md"

CANONICAL_BUILDERS = {
    "cpu-native-builder",
    "cpu-plugin-builder",
    "npu-native-builder",
    "npu-plugin-builder",
    "gpu-native-builder",
    "gpu-plugin-builder",
}
CANONICAL_BACKENDS = {"CPU", "GPU", "NPU"}
EXPECTED_CASE_IDS = {
    "cpu_plugin_default",
    "cpu_native_override",
    "npu_alias_ascend",
    "npu_alias_aclnn",
    "gpu_roadmap",
    "cpu_ambiguity",
    "npu_mint_api",
}


def load_routing_cases():
    return yaml.safe_load(ROUTING_CASES.read_text(encoding="utf-8"))


def case_map(data):
    return {case["id"]: case for case in data["cases"]}


def test_routing_cases_yaml_exists_and_has_expected_cases():
    assert ROUTING_CASES.exists(), f"Missing routing case contract: {ROUTING_CASES}"
    data = load_routing_cases()
    assert data["schema_version"] == "1.0.0"
    assert {case["id"] for case in data["cases"]} == EXPECTED_CASE_IDS


def test_all_cases_use_valid_backends_and_builder_targets():
    data = load_routing_cases()
    for case in data["cases"]:
        assert isinstance(case["input"]["known_evidence"], str)
        expected = case["expected"]
        assert expected["normalized_backend"] in CANONICAL_BACKENDS
        assert expected["best_fit"] in CANONICAL_BUILDERS | {"Roadmap"}
        assert isinstance(expected["ask_clarification"], bool)
        assert expected["forbid_codegen"] is True
        for option in expected["support_options"]:
            assert option["builder"] in CANONICAL_BUILDERS
            assert option["status"] in {"recommended", "available", "standard", "planned"}


def test_npu_aliases_normalize_to_npu():
    data = case_map(load_routing_cases())
    for case_id, raw_value in {
        "npu_alias_ascend": "Ascend",
        "npu_alias_aclnn": "aclnn",
    }.items():
        case = data[case_id]
        assert case["input"]["target_backend_raw"] == raw_value
        assert case["expected"]["normalized_backend"] == "NPU"
        assert case["expected"]["best_fit"] == "npu-native-builder"


def test_cpu_and_gpu_routing_contracts():
    data = case_map(load_routing_cases())

    assert data["cpu_plugin_default"]["expected"]["best_fit"] == "cpu-plugin-builder"
    assert data["cpu_plugin_default"]["expected"]["ask_clarification"] is False

    assert data["cpu_native_override"]["expected"]["best_fit"] == "cpu-native-builder"
    assert data["cpu_native_override"]["expected"]["ask_clarification"] is False

    assert data["cpu_ambiguity"]["expected"]["best_fit"] == "cpu-plugin-builder"
    assert data["cpu_ambiguity"]["expected"]["ask_clarification"] is True

    assert data["gpu_roadmap"]["expected"]["normalized_backend"] == "GPU"
    assert data["gpu_roadmap"]["expected"]["best_fit"] == "Roadmap"


def test_skill_md_contains_normalization_rules():
    text = SKILL_MD.read_text(encoding="utf-8")
    assert "## Normalization Rules" in text
    assert "`Ascend` and `aclnn` both map to `NPU`." in text
    assert "Use canonical builder names exactly" in text
    assert "cpu-plugin-builder (Mature / Recommended)" in text
    assert "npu-native-builder (Mature / Standard)" in text
    assert "Recommended/Available/Standard/Planned" in text
    assert "## Minimal Examples" in text
    assert "Best fit:" in text
    assert "cpu-plugin-builder" in text
    assert "npu-native-builder" in text


def test_human_readable_cases_doc_is_retained():
    assert HUMAN_CASES.exists(), f"Missing human-readable routing guide: {HUMAN_CASES}"
    text = HUMAN_CASES.read_text(encoding="utf-8")
    assert "tests/routing_cases.yaml" in text
    assert "human-readable companion" in text
    assert "`target_backend_raw` shows the original user input before normalization" in text
    assert "target_backend_raw:" in text
