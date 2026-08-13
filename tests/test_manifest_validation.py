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

import json

import pytest
from click.testing import CliRunner

from google.agents.cli import _skills_check, _tools
from google.agents.cli.info import cmd_info
from google.agents.cli.main import main
from google.agents.cli.scaffold.utils import version


@pytest.fixture(autouse=True)
def _disable_cli_startup_checks(monkeypatch) -> None:
    monkeypatch.setattr(version, "display_update_message", lambda: None)
    monkeypatch.setattr(_skills_check, "check_skills_version", lambda: None)
    monkeypatch.setattr(_tools, "require_tool", lambda _name: None)
    monkeypatch.setattr(cmd_info, "get_installed_skills", lambda: [])


@pytest.mark.parametrize(
    "manifest",
    [
        "",
        "null\n",
        "scalar\n",
        "- item\n",
        "create_params: null\n",
        "create_params: scalar\n",
        "create_params:\n  - item\n",
    ],
)
def test_info_rejects_invalid_manifest_without_traceback(
    monkeypatch, tmp_path, manifest
) -> None:
    (tmp_path / "agents-cli-manifest.yaml").write_text(manifest, encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(main, ["info", "--json"])

    assert result.exit_code == 1
    assert result.output.count("Error:") == 1
    assert result.output.count("Invalid agents-cli-manifest.yaml") == 1
    assert "Traceback" not in result.output


def test_info_accepts_valid_manifest(monkeypatch, tmp_path) -> None:
    (tmp_path / "agents-cli-manifest.yaml").write_text(
        "name: valid-project\n"
        "agent_directory: agent\n"
        "region: europe-west1\n"
        "create_params:\n"
        "  deployment_target: cloud_run\n"
        "  is_a2a: true\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(main, ["info", "--json"])

    assert result.exit_code == 0
    info = json.loads(result.stdout)
    assert info["project_name"] == "valid-project"
    assert info["agent_directory"] == "agent"
    assert info["region"] == "europe-west1"
    assert info["deployment_target"] == "cloud_run"
    assert info["is_a2a"] is True
