# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from google.agents.cli import _skills_check, _tools
from google.agents.cli.main import main
from google.agents.cli.scaffold.utils import version


@pytest.fixture(autouse=True)
def _disable_cli_startup_checks(monkeypatch) -> None:
    monkeypatch.setattr(version, "display_update_message", lambda: None)
    monkeypatch.setattr(_skills_check, "check_skills_version", lambda: None)
    monkeypatch.setattr(_tools, "require_tool", lambda _name: None)


def _write_pyproject(project: Path, name: str) -> None:
    (project / "pyproject.toml").write_text(
        f'[project]\nname = "{name}"\n', encoding="utf-8"
    )


def _add_agent(directory: Path, marker: str = "agent.py") -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / marker).write_text("", encoding="utf-8")


def _invoke(project: Path, *args: str, input: str | None = None):
    return CliRunner().invoke(main, ["init", str(project), *args], input=input)


def _manifest(result) -> dict[str, str]:
    assert result.exit_code == 0, result.output
    return yaml.safe_load(result.stdout)


@pytest.mark.parametrize(
    ("directory_name", "pyproject_name", "flag_name", "expected"),
    [
        ("folder-name", "pyproject-name", "Flag_Name", "flag-name"),
        ("folder-name", "Project_Name", None, "project-name"),
        ("Folder_Name", None, None, "folder-name"),
    ],
)
def test_name_precedence_and_normalization(
    tmp_path, directory_name, pyproject_name, flag_name, expected
) -> None:
    project = tmp_path / directory_name
    project.mkdir()
    _add_agent(project / "app")
    if pyproject_name:
        _write_pyproject(project, pyproject_name)

    args = ["--dry-run"]
    if flag_name:
        args.extend(["--name", flag_name])
    result = _invoke(project, *args)

    assert _manifest(result)["name"] == expected
    assert "Note: Project names are normalized" in result.stderr


def test_valid_app_has_first_priority(tmp_path) -> None:
    project = tmp_path / "sample"
    project.mkdir()
    _write_pyproject(project, "sample")
    _add_agent(project / "app")
    _add_agent(project / "sample")
    _add_agent(project / "other")

    result = _invoke(project, "--dry-run")

    assert _manifest(result)["agent_directory"] == "app"


@pytest.mark.parametrize("layout", ["root", "src"])
def test_pyproject_package_candidate_precedes_direct_children(tmp_path, layout) -> None:
    project = tmp_path / "sample"
    project.mkdir()
    _write_pyproject(project, "My-Agent")
    package = project / "my_agent" if layout == "root" else project / "src/my_agent"
    _add_agent(package)
    _add_agent(project / "other")

    result = _invoke(project, "--dry-run")

    expected = "my_agent" if layout == "root" else "src/my_agent"
    assert _manifest(result)["agent_directory"] == expected


def test_unique_direct_child_candidate_supports_yaml_marker(tmp_path) -> None:
    project = tmp_path / "sample"
    project.mkdir()
    _add_agent(project / "custom", "root_agent.yaml")

    result = _invoke(project, "--dry-run")

    assert _manifest(result)["agent_directory"] == "custom"


def test_explicit_agent_directory_wins_and_is_relative(tmp_path) -> None:
    project = tmp_path / "sample"
    project.mkdir()
    _add_agent(project / "app")
    _add_agent(project / "nested/custom")

    result = _invoke(
        project, "--agent-directory", "nested/custom", "--dry-run"
    )

    assert _manifest(result)["agent_directory"] == "nested/custom"


@pytest.mark.parametrize("agent_directory", ["missing", "../outside"])
def test_explicit_agent_directory_must_be_valid_and_inside_path(
    tmp_path, agent_directory
) -> None:
    project = tmp_path / "sample"
    project.mkdir()
    (tmp_path / "outside").mkdir()

    result = _invoke(
        project, "--agent-directory", agent_directory, "--dry-run"
    )

    assert result.exit_code != 0
    assert "--agent-directory" in result.output
    assert "Traceback" not in result.output


def test_explicit_agent_directory_rejects_absolute_path(tmp_path) -> None:
    project = tmp_path / "sample"
    project.mkdir()
    _add_agent(project / "app")

    result = _invoke(
        project, "--agent-directory", str(project / "app"), "--dry-run"
    )

    assert result.exit_code != 0
    assert "relative" in result.output


@pytest.mark.parametrize("extra", [(), ("--agent-directory", "app")])
def test_agent_directory_symlink_escape_is_rejected(tmp_path, extra) -> None:
    project = tmp_path / "sample"
    outside = tmp_path / "outside"
    project.mkdir()
    _add_agent(outside)
    (project / "app").symlink_to(outside, target_is_directory=True)

    result = _invoke(project, "--dry-run", *extra)

    assert result.exit_code != 0
    assert "outside PATH" in result.output
    assert "Traceback" not in result.output


def test_multiple_candidates_prompt_on_tty(tmp_path, monkeypatch) -> None:
    from google.agents.cli import cmd_init

    project = tmp_path / "sample"
    project.mkdir()
    _add_agent(project / "alpha")
    _add_agent(project / "beta")
    monkeypatch.setattr(cmd_init, "_is_interactive", lambda: True)

    result = _invoke(project, "--dry-run", input="2\n")

    assert _manifest(result)["agent_directory"] == "beta"
    assert "Select agent directory" in result.stderr


@pytest.mark.parametrize(
    ("interactive", "extra"), [(False, ()), (True, ("--yes",))]
)
def test_multiple_candidates_fail_closed_without_prompt(
    tmp_path, monkeypatch, interactive, extra
) -> None:
    from google.agents.cli import cmd_init

    project = tmp_path / "sample"
    project.mkdir()
    _add_agent(project / "alpha")
    _add_agent(project / "beta")
    monkeypatch.setattr(cmd_init, "_is_interactive", lambda: interactive)

    result = _invoke(project, "--dry-run", *extra)

    assert result.exit_code != 0
    assert "--agent-directory" in result.output
    assert "Traceback" not in result.output


def test_multiple_pyproject_candidates_fail_at_their_priority(tmp_path) -> None:
    project = tmp_path / "sample"
    project.mkdir()
    _write_pyproject(project, "sample")
    _add_agent(project / "sample")
    _add_agent(project / "src/sample")

    result = _invoke(project, "--dry-run", "--yes")

    assert result.exit_code != 0
    assert "--agent-directory" in result.output


def test_no_candidate_fails_without_creating_app(tmp_path) -> None:
    project = tmp_path / "sample"
    project.mkdir()

    result = _invoke(project, "--dry-run")

    assert result.exit_code != 0
    assert "agent.py or root_agent.yaml" in result.output
    assert not (project / "app").exists()


def test_dry_run_outputs_only_yaml_and_never_writes(tmp_path) -> None:
    project = tmp_path / "sample"
    project.mkdir()
    _add_agent(project / "app")
    manifest = project / "agents-cli-manifest.yaml"
    manifest.write_bytes(b"existing: bytes\n")

    result = _invoke(project, "--name", "New_Name", "--dry-run")

    assert _manifest(result) == {
        "name": "new-name",
        "agent_directory": "app",
        "deployment_target": "none",
    }
    assert result.stdout == yaml.safe_dump(
        yaml.safe_load(result.stdout), sort_keys=False
    )
    assert manifest.read_bytes() == b"existing: bytes\n"
    assert "normalized" not in result.stdout
    assert "normalized" in result.stderr


@pytest.mark.parametrize("extra", [(), ("--yes",)])
def test_existing_manifest_is_byte_identical_without_force(tmp_path, extra) -> None:
    project = tmp_path / "sample"
    project.mkdir()
    _add_agent(project / "app")
    manifest = project / "agents-cli-manifest.yaml"
    original = b"name: do-not-touch\n"
    manifest.write_bytes(original)

    result = _invoke(project, *extra)

    assert result.exit_code != 0
    assert "--force" in result.output
    assert manifest.read_bytes() == original


def test_force_atomically_replaces_manifest_from_same_directory(
    tmp_path, monkeypatch
) -> None:
    from google.agents.cli import cmd_init

    project = tmp_path / "sample"
    project.mkdir()
    _add_agent(project / "app")
    manifest = project / "agents-cli-manifest.yaml"
    manifest.write_text("old: value\n", encoding="utf-8")
    real_replace = cmd_init.os.replace
    calls = []

    def record_replace(source, destination) -> None:
        calls.append((Path(source), Path(destination)))
        real_replace(source, destination)

    monkeypatch.setattr(cmd_init.os, "replace", record_replace)

    result = _invoke(project, "--force")

    assert result.exit_code == 0, result.output
    assert yaml.safe_load(manifest.read_text(encoding="utf-8")) == {
        "name": "sample",
        "agent_directory": "app",
        "deployment_target": "none",
    }
    assert len(calls) == 1
    assert calls[0][0].parent == project.resolve()
    assert calls[0][1] == manifest.resolve()


def test_failed_atomic_replace_preserves_manifest(tmp_path, monkeypatch) -> None:
    from google.agents.cli import cmd_init

    project = tmp_path / "sample"
    project.mkdir()
    _add_agent(project / "app")
    manifest = project / "agents-cli-manifest.yaml"
    original = b"name: original\n"
    manifest.write_bytes(original)

    def fail_replace(_source, _destination) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(cmd_init.os, "replace", fail_replace)

    result = _invoke(project, "--force")

    assert result.exit_code != 0
    assert "replace failed" in result.output
    assert manifest.read_bytes() == original
    assert list(project.glob(".agents-cli-manifest.yaml.*")) == []


def test_init_only_processes_the_requested_path(tmp_path) -> None:
    selected = tmp_path / "selected"
    sibling = tmp_path / "sibling"
    selected.mkdir()
    sibling.mkdir()
    _add_agent(selected / "app")
    _add_agent(sibling / "app")

    result = _invoke(selected)

    assert result.exit_code == 0, result.output
    assert (selected / "agents-cli-manifest.yaml").exists()
    assert not (sibling / "agents-cli-manifest.yaml").exists()


@pytest.mark.parametrize("marker", ["package.json", "pom.xml"])
@pytest.mark.parametrize("extra", [(), ("--agent-directory", "app")])
def test_explicit_non_python_project_is_unsupported(tmp_path, marker, extra) -> None:
    project = tmp_path / "sample"
    project.mkdir()
    (project / marker).write_text("{}\n", encoding="utf-8")
    _add_agent(project / "app")

    result = _invoke(project, "--dry-run", *extra)

    assert result.exit_code != 0
    assert "unsupported" in result.output.lower()
    assert "Python" in result.output


@pytest.mark.parametrize(
    ("project_name", "extra", "expected"),
    [
        (None, ("--name", ""), "--name"),
        ('""', (), "[project].name"),
        (123, (), "[project].name"),
    ],
)
def test_project_name_must_be_a_non_empty_string(
    tmp_path, project_name, extra, expected
) -> None:
    project = tmp_path / "sample"
    project.mkdir()
    _add_agent(project / "app")
    if project_name is not None:
        (project / "pyproject.toml").write_text(
            f"[project]\nname = {project_name}\n", encoding="utf-8"
        )

    result = _invoke(project, "--dry-run", *extra)

    assert result.exit_code != 0
    assert expected in result.output
    assert "non-empty string" in result.output
    assert "Traceback" not in result.output


def test_pyproject_allows_auxiliary_non_python_markers(tmp_path) -> None:
    project = tmp_path / "sample"
    project.mkdir()
    _write_pyproject(project, "sample")
    (project / "pom.xml").write_text("<project/>\n", encoding="utf-8")
    _add_agent(project / "app")

    result = _invoke(project, "--dry-run")

    assert _manifest(result)["agent_directory"] == "app"


def test_path_defaults_to_current_directory(tmp_path) -> None:
    _add_agent(tmp_path / "app")

    with CliRunner().isolated_filesystem(temp_dir=tmp_path) as directory:
        current = Path(directory)
        _add_agent(current / "app")
        result = CliRunner().invoke(main, ["init", "--dry-run"])

    assert _manifest(result)["agent_directory"] == "app"
