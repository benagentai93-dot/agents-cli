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

from google.agents.cli import _runner, _skills_check


def test_installed_skills_fallback_uses_pinned_package(monkeypatch, tmp_path) -> None:
    captured: list[list[str]] = []
    assert _skills_check.SKILLS_NPX_PACKAGE == "skills@1.5.9"

    def fake_run_resolved(args: list[str], **_kwargs: object) -> CompletedProcess[str]:
        captured.append(args)
        return CompletedProcess(args, returncode=0, stdout="[]", stderr="")

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(_skills_check, "SKILLS_NPX_PACKAGE", "skills@test-pin")
    monkeypatch.setattr(_runner, "run_resolved", fake_run_resolved)

    _skills_check._find_installed_skills()

    assert captured == [["npx", "-y", "skills@test-pin", "list", "--json"]]
