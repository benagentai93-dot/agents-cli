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
DEPLOY_SKILL = ROOT / "skills/google-agents-cli-deploy/SKILL.md"
OBSERVABILITY_SKILL = ROOT / "skills/google-agents-cli-observability/SKILL.md"

DEPLOYMENT_ORDER = (
    "| Basic Agent Runtime and Cloud Run | Run `agents-cli deploy` directly. |",
    "| GKE | `agents-cli deploy` runs the required targeted Terraform. |",
    "| Terraform-managed observability | For every target, run "
    "`agents-cli infra single-project --apply` before `agents-cli deploy`. |",
    "| Existing imperative deployment | Do not apply Terraform afterward; import "
    "or delete it before switching, or keep it imperative and configure "
    "observability manually. |",
)
CONFLICTING_DEPLOYMENT_GUIDANCE = (
    "Do NOT run `agents-cli infra single-project` before deploying.",
    "For `deployment_target = agent_runtime`, run "
    "`agents-cli infra single-project` **before** the first `agents-cli deploy`.",
)


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


def test_migration_guide_uses_safe_ordered_commands() -> None:
    guide = MIGRATION_GUIDE.read_text(encoding="utf-8")
    steps = (
        "1. `agents-cli info` reads the current configuration without changing it.",
        "2. `agents-cli scaffold upgrade --dry-run` previews the migration.",
        "3. `agents-cli scaffold upgrade` applies the migration.",
    )

    info_help = CliRunner().invoke(main, ["info", "--help"])
    upgrade_help = CliRunner().invoke(main, ["scaffold", "upgrade", "--help"])

    assert info_help.exit_code == 0, info_help.output
    assert upgrade_help.exit_code == 0, upgrade_help.output
    assert "--dry-run" in upgrade_help.output
    assert all(step in guide for step in steps)
    assert [guide.index(step) for step in steps] == sorted(
        guide.index(step) for step in steps
    )
    assert "sed -i ''" not in guide


@pytest.mark.parametrize(
    "path", [README, MIGRATION_GUIDE, DEPLOY_SKILL, OBSERVABILITY_SKILL]
)
def test_deployment_order_is_consistent(path: Path) -> None:
    text = path.read_text(encoding="utf-8")

    deploy_help = CliRunner().invoke(main, ["deploy", "--help"])
    infra_help = CliRunner().invoke(main, ["infra", "single-project", "--help"])

    assert deploy_help.exit_code == 0, deploy_help.output
    assert infra_help.exit_code == 0, infra_help.output
    assert "--apply" in infra_help.output
    assert all(rule in text for rule in DEPLOYMENT_ORDER)
    assert not any(rule in text for rule in CONFLICTING_DEPLOYMENT_GUIDANCE)
