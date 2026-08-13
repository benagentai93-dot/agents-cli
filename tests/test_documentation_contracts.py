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

from pathlib import Path

import pytest
from click.testing import CliRunner

from google.agents.cli import _skills_check, _tools
from google.agents.cli.main import main
from google.agents.cli.scaffold.utils import version

ROOT = Path(__file__).parents[1]
README = ROOT / "README.md"
MIGRATION_GUIDE = ROOT / "docs/src/reference/from-agent-starter-pack.md"


@pytest.fixture(autouse=True)
def _disable_cli_startup_checks(monkeypatch) -> None:
    monkeypatch.setattr(version, "display_update_message", lambda: None)
    monkeypatch.setattr(_skills_check, "check_skills_version", lambda: None)
    monkeypatch.setattr(_tools, "require_tool", lambda _name: None)


def test_documented_create_command_has_help() -> None:
    result = CliRunner().invoke(main, ["create", "--help"])

    assert result.exit_code == 0, result.output
    assert "Usage:" in result.output


def test_readme_uses_the_create_command() -> None:
    readme = README.read_text(encoding="utf-8")

    assert readme.count("`agents-cli create <name>`") == 2
    assert "`agents-cli scaffold <name>`" not in readme

