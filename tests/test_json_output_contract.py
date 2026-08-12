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

from click.testing import CliRunner

from google.agents.cli import _project, _skills_check, _tools
from google.agents.cli.info import cmd_info
from google.agents.cli.main import main
from google.agents.cli.scaffold.utils import version


def test_info_json_keeps_notices_on_stderr(monkeypatch, tmp_path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "json-contract"\n'
        '[tool.agents-cli]\nagent_directory = "app"\nacli_version = "0.0.1"\n'
        '[tool.agents-cli.create_params]\ndeployment_target = "none"\n',
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(_project, "_WARNED_LEGACY_CONFIG", False)
    monkeypatch.setattr(_tools, "require_tool", lambda _name: None)
    monkeypatch.setattr(version, "_update_check_is_due", lambda: True)
    monkeypatch.setattr(version, "check_for_updates", lambda: (True, "1.3.1", "1.3.2"))
    monkeypatch.setattr(version, "_record_update_check", lambda: None)
    monkeypatch.setattr(_skills_check, "_is_ci", lambda: False)
    monkeypatch.setattr(_skills_check, "_skills_check_is_due", lambda: True)
    monkeypatch.setattr(
        _skills_check,
        "_find_installed_skills",
        lambda: {"google-agents-cli-test": "0.0.1"},
    )
    monkeypatch.setattr(_skills_check, "_record_skills_check", lambda: None)
    monkeypatch.setattr(cmd_info, "get_installed_skills", lambda: [])

    result = CliRunner().invoke(main, ["info", "--json"])

    assert result.exit_code == 0
    assert json.loads(result.stdout)["project_name"] == "json-contract"
    assert "Update available" in result.stderr
    assert "Skills version mismatch" in result.stderr
    assert "Legacy configuration detected" in result.stderr
    assert "Version mismatch" in result.stderr
