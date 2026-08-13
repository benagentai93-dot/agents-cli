# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import click
import pytest
from click.testing import CliRunner

from google.agents.cli import _skills_check, _tools
from google.agents.cli.main import main
from google.agents.cli.scaffold.commands import create as create_module
from google.agents.cli.scaffold.utils import backup, template, version


@pytest.fixture(autouse=True)
def _disable_cli_startup_checks(monkeypatch) -> None:
    monkeypatch.setattr(version, "display_update_message", lambda: None)
    monkeypatch.setattr(_skills_check, "check_skills_version", lambda: None)
    monkeypatch.setattr(_tools, "require_tool", lambda _name: None)
    monkeypatch.setattr(create_module, "resolve_gcp_project", lambda: None)


def _invoke_create(runner: CliRunner, output_dir, *extra: str):
    return runner.invoke(
        main,
        [
            "create",
            "sample-agent",
            "--agent",
            "adk",
            "--prototype",
            "--yes",
            "--skip-checks",
            "--output-dir",
            str(output_dir),
            *extra,
        ],
    )


@pytest.mark.parametrize(
    ("project_name", "prepare", "expected"),
    [
        ("x" * 27, lambda _path: None, "exceeds 26 characters"),
        ("existing", lambda path: path.mkdir(), "already exists"),
    ],
)
def test_create_preconditions_fail_with_usage_error(
    tmp_path, project_name, prepare, expected
) -> None:
    prepare(tmp_path / project_name)

    result = CliRunner().invoke(
        main,
        ["create", project_name, "--output-dir", str(tmp_path), "--skip-welcome"],
    )

    assert result.exit_code == 2
    assert expected in result.output
    assert "Traceback" not in result.output


def test_create_rejects_invalid_agent_without_traceback(tmp_path) -> None:
    result = CliRunner().invoke(
        main,
        [
            "create",
            "sample-agent",
            "--agent",
            "not-an-agent",
            "--output-dir",
            str(tmp_path),
            "--skip-welcome",
        ],
    )

    assert result.exit_code == 2
    assert "Invalid agent name or number" in result.output
    assert "Traceback" not in result.output


def test_create_rejects_invalid_base_template_without_traceback(tmp_path) -> None:
    local_template = tmp_path / "local-template"
    local_template.mkdir()

    result = CliRunner().invoke(
        main,
        [
            "create",
            "sample-agent",
            "--agent",
            f"local@{local_template}",
            "--base-template",
            "not-a-template",
            "--output-dir",
            str(tmp_path),
            "--skip-welcome",
        ],
    )

    assert result.exit_code == 2
    assert "Base template 'not-a-template' not found" in result.output
    assert "Traceback" not in result.output


def test_create_rejects_invalid_agent_directory_without_traceback(
    monkeypatch, tmp_path
) -> None:
    def validate_cli_override(**kwargs) -> None:
        template.validate_agent_directory_name(
            kwargs["cli_overrides"]["settings"]["agent_directory"]
        )

    monkeypatch.setattr(template, "process_template", validate_cli_override)

    result = _invoke_create(CliRunner(), tmp_path, "--agent-directory", "bad-dir")

    assert result.exit_code == 2
    assert "contains hyphens" in result.output
    assert "Traceback" not in result.output


def test_create_wraps_generation_errors_without_traceback(monkeypatch, tmp_path) -> None:
    def fail_generation(**_kwargs) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(template, "process_template", fail_generation)

    result = _invoke_create(CliRunner(), tmp_path)

    assert result.exit_code == 1
    assert "Error: disk full" in result.output
    assert "Traceback" not in result.output


def test_create_wraps_remote_fetch_errors_without_traceback(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(
        create_module.remote_template,
        "parse_agent_spec",
        lambda _agent: type("RemoteSpec", (), {"is_adk_samples": False})(),
    )

    def fail_fetch(*_args, **_kwargs):
        raise RuntimeError("remote unavailable")

    monkeypatch.setattr(
        create_module.remote_template, "fetch_remote_template", fail_fetch
    )

    result = CliRunner().invoke(
        main,
        [
            "create",
            "sample-agent",
            "--agent",
            "github.com/example/repo",
            "--output-dir",
            str(tmp_path),
            "--skip-welcome",
        ],
    )

    assert result.exit_code == 1
    assert "Error: remote unavailable" in result.output
    assert "Traceback" not in result.output


def test_create_propagates_interactive_cancellation(monkeypatch, tmp_path) -> None:
    def cancel_backup(*_args, **_kwargs) -> None:
        raise click.Abort()

    monkeypatch.setattr(backup, "create_project_backup", cancel_backup)
    ctx = click.Context(create_module.create, info_name="create")

    with pytest.raises(click.Abort):
        ctx.invoke(
            create_module.create,
            project_name="sample-agent",
            output_dir=str(tmp_path),
            in_folder=True,
            interactive=True,
            skip_welcome=True,
        )
