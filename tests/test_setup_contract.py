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

from subprocess import CompletedProcess

from click.testing import CliRunner

from google.agents.cli.setup.cmd_setup import cmd_setup


def test_setup_reports_cli_install_failure_after_partial_success(monkeypatch) -> None:
    def fail_commands(args, **kwargs):
        return CompletedProcess(args, returncode=1, stdout="", stderr="install failed")

    monkeypatch.setattr("google.agents.cli.setup.cmd_setup.run", fail_commands)
    monkeypatch.setattr(
        "google.agents.cli.setup.cmd_setup.run_npx_skills", lambda *args: None
    )

    result = CliRunner().invoke(cmd_setup, ["--workspace", "--skip-auth"])

    assert result.exit_code != 0
    assert "CLI:    Not installed" in result.output
    assert "Skills: Installed" in result.output
    assert "Error: agents-cli installation failed" in result.output
    assert "Done." not in result.output
