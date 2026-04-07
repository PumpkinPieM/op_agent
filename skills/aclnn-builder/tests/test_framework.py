from __future__ import annotations

import json
import sys
from pathlib import Path
import subprocess

import pytest


TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from framework import (
    CaseSpec,
    CodexExecutor,
    ExecutionRequest,
    ExecutionResult,
    ExpectedChangeSet,
    Manifest,
    ManifestError,
    RepoSpec,
    SkillTestRunner,
    classify_repo_changes,
    load_manifest,
    prepare_repos,
    render_prompt,
    resolve_op_plugin_path,
    resolve_skill_path,
    resolve_repo_source,
    stage_skills_for_case,
)


def git(args: list[str], cwd: Path) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=cwd,
        text=True,
        capture_output=True,
        check=True,
    )
    return completed.stdout


def init_repo(path: Path, files: dict[str, str]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    git(["init"], cwd=path)
    git(["config", "user.email", "test@example.com"], cwd=path)
    git(["config", "user.name", "Test User"], cwd=path)
    for relative_path, content in files.items():
        file_path = path / relative_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
    git(["add", "."], cwd=path)
    git(["commit", "-m", "initial"], cwd=path)


class FakeExecutor:
    def __init__(self, mutation_map: dict[str, list[tuple[str, str, str | None]]]) -> None:
        self.mutation_map = mutation_map

    def run(self, request: ExecutionRequest) -> ExecutionResult:
        for action, repo_name, relative_path in self.mutation_map[request.case.case_id]:
            repo_root = request.prepared_repos[repo_name].checkout_path
            if action == "add":
                target = repo_root / str(relative_path)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(f"created by {request.case.case_id}\n", encoding="utf-8")
            elif action == "modify":
                target = repo_root / str(relative_path)
                target.write_text(f"modified by {request.case.case_id}\n", encoding="utf-8")
            elif action == "delete":
                target = repo_root / str(relative_path)
                target.unlink()
            elif action == "artifact":
                target = repo_root / str(relative_path)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("artifact\n", encoding="utf-8")
            else:
                raise AssertionError(f"Unsupported fake action: {action}")

        stdout_path = request.case_dir / "fake_stdout.txt"
        stderr_path = request.case_dir / "fake_stderr.txt"
        last_message_path = request.case_dir / "fake_last_message.txt"
        stdout_path.write_text("ok\n", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        last_message_path.write_text(f"finished {request.case.case_id}\n", encoding="utf-8")
        return ExecutionResult(
            command=("fake-executor", request.case.case_id),
            returncode=0,
            started_at=0.0,
            finished_at=1.0,
            timed_out=False,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            last_message_path=last_message_path,
        )


def test_manifest_template_loads():
    manifest = load_manifest(TESTS_DIR / "cases.yaml")
    assert manifest.schema_version == "1.0.0"
    assert manifest.cases[0].case_id == "example_aclnn_abs"
    assert manifest.cases[0].enabled is False


def test_render_prompt_replaces_repo_and_artifact_placeholders(tmp_path: Path):
    repo_root = tmp_path / "repo"
    op_plugin_root = tmp_path / "op-plugin"
    init_repo(repo_root, {"tracked.txt": "base\n"})
    op_plugin_root.mkdir(parents=True, exist_ok=True)
    repo_spec = RepoSpec(name="mindspore", source=str(repo_root))
    case = CaseSpec(
        case_id="prompt_case",
        prompt="repo={{repo:mindspore}} artifact={{artifact_dir}} op={{op_plugin_dir}}",
        repos=(repo_spec,),
    )
    prepared = prepare_repos(case, tmp_path / "sandbox", ms_root=repo_root, path_root=tmp_path)
    try:
        prompt = render_prompt(
            case.prompt,
            prepared_repos=prepared,
            artifact_dir=tmp_path / "artifacts",
            case_dir=tmp_path / "case",
            run_dir=tmp_path / "run",
            op_plugin_dir=op_plugin_root,
        )
        assert str(prepared["mindspore"].checkout_path) in prompt
        assert str(tmp_path / "artifacts") in prompt
        assert str(op_plugin_root) in prompt
    finally:
        from framework import cleanup_prepared_repos

        cleanup_prepared_repos(prepared)


def test_classify_repo_changes_separates_artifacts(tmp_path: Path):
    repo_root = tmp_path / "repo"
    init_repo(repo_root, {"tracked.txt": "base\n", "remove_me.txt": "bye\n"})
    repo_spec = RepoSpec(
        name="mindspore",
        source=str(repo_root),
        artifact_globs=("*_Feature.md", "*_pta_analysis.md"),
        expected_changes=ExpectedChangeSet(
            added=("new_file.cc",),
            modified=("tracked.txt",),
            deleted=("remove_me.txt",),
        ),
    )
    case = CaseSpec(case_id="diff_case", prompt="unused", repos=(repo_spec,))
    prepared = prepare_repos(case, tmp_path / "sandbox", ms_root=repo_root, path_root=tmp_path)
    try:
        checkout = prepared["mindspore"].checkout_path
        (checkout / "tracked.txt").write_text("changed\n", encoding="utf-8")
        (checkout / "new_file.cc").write_text("new\n", encoding="utf-8")
        (checkout / "remove_me.txt").unlink()
        (checkout / "abs_Feature.md").write_text("artifact\n", encoding="utf-8")
        outcome = classify_repo_changes(prepared["mindspore"])
        assert outcome.valid is True
        assert outcome.expected_changes.added == ("new_file.cc",)
        assert outcome.source_changes.added == ("new_file.cc",)
        assert outcome.source_changes.modified == ("tracked.txt",)
        assert outcome.source_changes.deleted == ("remove_me.txt",)
        assert outcome.artifact_changes == ("abs_Feature.md",)
    finally:
        from framework import cleanup_prepared_repos

        cleanup_prepared_repos(prepared)


def test_extra_changes_are_allowed_if_expected_bucketed_changes_exist(tmp_path: Path):
    repo_root = tmp_path / "repo"
    init_repo(repo_root, {"tracked.txt": "base\n", "remove_me.txt": "bye\n"})
    repo_spec = RepoSpec(
        name="mindspore",
        source=str(repo_root),
        expected_changes=ExpectedChangeSet(
            added=("new_file.cc",),
            modified=("tracked.txt",),
            deleted=("remove_me.txt",),
        ),
    )
    case = CaseSpec(case_id="extra_changes_case", prompt="unused", repos=(repo_spec,))
    prepared = prepare_repos(case, tmp_path / "sandbox", ms_root=repo_root, path_root=tmp_path)
    try:
        checkout = prepared["mindspore"].checkout_path
        (checkout / "tracked.txt").write_text("changed\n", encoding="utf-8")
        (checkout / "new_file.cc").write_text("new\n", encoding="utf-8")
        (checkout / "remove_me.txt").unlink()
        (checkout / "extra_added.cc").write_text("extra\n", encoding="utf-8")
        (checkout / "extra_modified.txt").write_text("extra tracked\n", encoding="utf-8")
        outcome = classify_repo_changes(prepared["mindspore"])
        assert outcome.valid is True
        assert "new_file.cc" in outcome.source_changes.added
        assert "tracked.txt" in outcome.source_changes.modified
        assert "remove_me.txt" in outcome.source_changes.deleted
        assert "extra_added.cc" in outcome.source_changes.added
    finally:
        from framework import cleanup_prepared_repos

        cleanup_prepared_repos(prepared)


def test_runner_executes_multiple_cases_with_isolation(tmp_path: Path):
    source_repo = tmp_path / "mindspore"
    skill_root = tmp_path / ".codex" / "skills"
    op_plugin_root = tmp_path / "op-plugin"
    init_repo(source_repo, {"base.txt": "base\n"})
    op_plugin_root.mkdir(parents=True, exist_ok=True)
    (skill_root / "aclnn-builder").mkdir(parents=True, exist_ok=True)
    (skill_root / "aclnn-builder" / "SKILL.md").write_text("skill\n", encoding="utf-8")
    manifest = Manifest(
        schema_version="1.0.0",
        default_artifact_globs=(),
        cases=(
            CaseSpec(
                case_id="case_one",
                prompt="first",
                repos=(
                    RepoSpec(
                        name="mindspore",
                        source=str(source_repo),
                        artifact_globs=("*_Feature.md",),
                        expected_changes=ExpectedChangeSet(
                            added=("alpha.txt",),
                            modified=(),
                            deleted=(),
                        ),
                    ),
                ),
            ),
            CaseSpec(
                case_id="case_two",
                prompt="second",
                repos=(
                    RepoSpec(
                        name="mindspore",
                        source=str(source_repo),
                        artifact_globs=("*_Feature.md",),
                        expected_changes=ExpectedChangeSet(
                            added=("beta.txt",),
                            modified=(),
                            deleted=(),
                        ),
                    ),
                ),
            ),
        ),
    )
    executor = FakeExecutor(
        {
            "case_one": [("add", "mindspore", "alpha.txt"), ("artifact", "mindspore", "alpha_Feature.md")],
            "case_two": [("add", "mindspore", "beta.txt"), ("artifact", "mindspore", "beta_Feature.md")],
        }
    )
    runner = SkillTestRunner(
        manifest=manifest,
        executor=executor,
        ms_root=source_repo,
        path_root=tmp_path,
        skill_path=skill_root,
        op_plugin_dir=op_plugin_root,
        runs_root=tmp_path / "runs",
    )
    run_dir, outcomes = runner.run()
    assert len(outcomes) == 2
    assert all(outcome.valid for outcome in outcomes)
    assert outcomes[0].repo_outcomes[0].source_changes.added == ("alpha.txt",)
    assert outcomes[1].repo_outcomes[0].source_changes.added == ("beta.txt",)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["run_id"]
    assert isinstance(summary["elapsed_time"], float)
    assert [item["case_id"] for item in summary["case_outcomes"]] == ["case_one", "case_two"]
    assert summary["case_outcomes"][0]["repo_outcomes"][0]["expected_changes"]["added"] == ["alpha.txt"]
    assert isinstance(summary["case_outcomes"][0]["execution_result"]["elapsed_time"], float)


def test_manifest_rejects_overlapping_expected_paths(tmp_path: Path):
    manifest_path = tmp_path / "bad.yaml"
    manifest_path.write_text(
        """
schema_version: "1.0.0"
cases:
  - id: "bad_case"
    prompt: "test"
    repos:
      - name: "mindspore"
        source: "relative/mindspore"
        expected_changes:
          added: ["same.cc"]
          modified: ["same.cc"]
          deleted: []
""".strip(),
        encoding="utf-8",
    )
    with pytest.raises(ManifestError):
        load_manifest(manifest_path)


def test_codex_executor_builds_ephemeral_command(tmp_path: Path):
    repo_root = tmp_path / "repo"
    op_plugin_root = tmp_path / "op-plugin"
    init_repo(repo_root, {"tracked.txt": "base\n"})
    op_plugin_root.mkdir(parents=True, exist_ok=True)
    repo_spec = RepoSpec(name="mindspore", source=str(repo_root))
    case = CaseSpec(case_id="cmd_case", prompt="test", repos=(repo_spec,))
    prepared = prepare_repos(case, tmp_path / "sandbox", ms_root=repo_root, path_root=tmp_path)
    try:
        executor = CodexExecutor(codex_bin="codex", model="gpt-5.4", extra_args=("--search",))
        request = ExecutionRequest(
            case=case,
            prompt="hello",
            prompt_path=tmp_path / "prompt.txt",
            case_dir=tmp_path / "case",
            run_dir=tmp_path / "run",
            artifact_dir=tmp_path / "artifacts",
            op_plugin_dir=op_plugin_root,
            prepared_repos=prepared,
        )
        command = executor.build_command(request)
        assert command[:4] == ["codex", "exec", "--json", "--sandbox"]
        assert "--ephemeral" in command
        assert "--output-last-message" in command
        assert "--search" in command
        assert str(prepared["mindspore"].checkout_path) in command
        assert str(op_plugin_root) in command
    finally:
        from framework import cleanup_prepared_repos

        cleanup_prepared_repos(prepared)


def test_mindspore_repo_defaults_to_ms_root(tmp_path: Path):
    ms_root = tmp_path / "mindspore"
    init_repo(ms_root, {"tracked.txt": "base\n"})
    repo_spec = RepoSpec(name="mindspore")
    assert resolve_repo_source(repo_spec, ms_root=ms_root, path_root=tmp_path) == ms_root.resolve()


def test_stage_skills_copies_into_isolated_mindspore_checkout(tmp_path: Path):
    repo_root = tmp_path / "mindspore"
    skill_root = tmp_path / "skills"
    init_repo(repo_root, {"tracked.txt": "base\n"})
    (skill_root / "aclnn-builder").mkdir(parents=True, exist_ok=True)
    (skill_root / "aclnn-builder" / "SKILL.md").write_text("skill\n", encoding="utf-8")
    repo_spec = RepoSpec(name="mindspore", source=str(repo_root))
    case = CaseSpec(case_id="skills_case", prompt="test", repos=(repo_spec,))
    prepared = prepare_repos(case, tmp_path / "sandbox", ms_root=repo_root, path_root=tmp_path)
    try:
        stage_skills_for_case(prepared, skill_root)
        copied = prepared["mindspore"].checkout_path / ".codex" / "skills" / "aclnn-builder" / "SKILL.md"
        assert copied.read_text(encoding="utf-8") == "skill\n"
    finally:
        from framework import cleanup_prepared_repos

        cleanup_prepared_repos(prepared)


def test_resolve_skill_path_defaults_to_ms_root_codex_skills(tmp_path: Path):
    ms_root = tmp_path / "mindspore"
    skill_root = ms_root / ".codex" / "skills"
    skill_root.mkdir(parents=True, exist_ok=True)
    assert resolve_skill_path(None, ms_root=ms_root, path_root=tmp_path) == skill_root.resolve()


def test_resolve_skill_path_errors_when_default_missing(tmp_path: Path):
    ms_root = tmp_path / "mindspore"
    ms_root.mkdir(parents=True, exist_ok=True)
    with pytest.raises(ManifestError):
        resolve_skill_path(None, ms_root=ms_root, path_root=tmp_path)


def test_resolve_op_plugin_path_returns_none_when_not_provided():
    assert resolve_op_plugin_path(None, path_root=Path.cwd()) is None


def test_resolve_op_plugin_path_resolves_relative_path(tmp_path: Path):
    op_plugin_root = tmp_path / "op-plugin"
    op_plugin_root.mkdir(parents=True, exist_ok=True)
    assert resolve_op_plugin_path(Path("op-plugin"), path_root=tmp_path) == op_plugin_root.resolve()


def test_render_prompt_requires_op_plugin_when_placeholder_is_used(tmp_path: Path):
    repo_root = tmp_path / "repo"
    init_repo(repo_root, {"tracked.txt": "base\n"})
    repo_spec = RepoSpec(name="mindspore", source=str(repo_root))
    case = CaseSpec(case_id="prompt_case", prompt="op={{op_plugin_dir}}", repos=(repo_spec,))
    prepared = prepare_repos(case, tmp_path / "sandbox", ms_root=repo_root, path_root=tmp_path)
    try:
        with pytest.raises(ManifestError):
            render_prompt(
                case.prompt,
                prepared_repos=prepared,
                artifact_dir=tmp_path / "artifacts",
                case_dir=tmp_path / "case",
                run_dir=tmp_path / "run",
                op_plugin_dir=None,
            )
    finally:
        from framework import cleanup_prepared_repos

        cleanup_prepared_repos(prepared)
