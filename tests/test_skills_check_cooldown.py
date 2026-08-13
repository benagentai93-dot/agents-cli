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
from subprocess import CompletedProcess, TimeoutExpired

import pytest

from google.agents.cli import __version__, _runner, _skills_check


@pytest.mark.parametrize(
    "outcome", ["success", "empty", "timeout", "nonzero", "malformed"]
)
def test_attempted_skills_check_starts_cooldown(
    monkeypatch, tmp_path, outcome: str
) -> None:
    skill_dir = tmp_path / "installed-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        f"---\nmetadata:\n  version: {__version__}\n---\n", encoding="utf-8"
    )
    calls = 0

    def fake_run_resolved(args: list[str], **_kwargs: object) -> CompletedProcess[str]:
        nonlocal calls
        calls += 1
        if outcome == "timeout":
            raise TimeoutExpired(args, 15)
        if outcome == "nonzero":
            return CompletedProcess(args, returncode=1, stdout="", stderr="failed")
        if outcome == "malformed":
            return CompletedProcess(args, returncode=0, stdout="{", stderr="")
        entries = (
            [{"name": "google-agents-cli-test", "path": str(skill_dir)}]
            if outcome == "success"
            else []
        )
        return CompletedProcess(
            args, returncode=0, stdout=json.dumps(entries), stderr=""
        )

    stamp = tmp_path / ".agents" / ".acli_skills_check"
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(_runner, "run_resolved", fake_run_resolved)
    monkeypatch.setattr(_skills_check, "_SKILLS_CHECK_STAMP", stamp)
    monkeypatch.setattr(_skills_check, "_is_ci", lambda: False)
    monkeypatch.setattr(_skills_check.time, "time", lambda: 1_000.0)

    _skills_check.check_skills_version()
    _skills_check.check_skills_version()

    assert (calls, stamp.is_file()) == (1, True)
