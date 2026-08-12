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

from __future__ import annotations

import importlib.util
import logging
import os
import sys
import tomllib
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from google.agents.cli.run import _local_server
from google.agents.cli.scaffold.utils import template

_OTEL_ENV = "AGENTS_CLI_OTEL_TO_CLOUD"


def _render_local_project(tmp_path: Path) -> Path:
    template.process_template(
        agent_name="adk",
        template_dir=template.get_template_path("adk"),
        project_name="local-contract",
        deployment_target="none",
        cicd_runner="skip",
        session_type="in_memory",
        output_dir=tmp_path,
    )
    return tmp_path / "local-contract"


def _module(name: str, *, package: bool = False, **attrs: object) -> ModuleType:
    module = ModuleType(name)
    if package:
        module.__path__ = []  # type: ignore[attr-defined]
    for attr, value in attrs.items():
        setattr(module, attr, value)
    return module


def _install_generated_app_stubs(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    captured: dict[str, object] = {}

    class FakeApp:
        def __init__(self) -> None:
            self.state = SimpleNamespace()
            self.title = ""
            self.description = ""

        def post(self, _path: str):
            return lambda function: function

    def get_fast_api_app(**kwargs: object) -> FakeApp:
        captured.update(kwargs)
        return FakeApp()

    def reject_cloud_setup(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("local template attempted cloud authentication/logging")

    def load_dotenv() -> None:
        # The private launcher flag must be read before project .env files.
        os.environ[_OTEL_ENV] = "1"

    modules = {
        "a2a": _module("a2a", package=True),
        "a2a.server": _module("a2a.server", package=True),
        "a2a.server.tasks": _module(
            "a2a.server.tasks", InMemoryTaskStore=type("InMemoryTaskStore", (), {})
        ),
        "dotenv": _module("dotenv", load_dotenv=load_dotenv),
        "fastapi": _module("fastapi", FastAPI=FakeApp),
        "google.adk": _module("google.adk", package=True),
        "google.adk.cli": _module("google.adk.cli", package=True),
        "google.adk.cli.fast_api": _module(
            "google.adk.cli.fast_api", get_fast_api_app=get_fast_api_app
        ),
        "google.adk.runners": _module("google.adk.runners", Runner=object),
        "google.auth": _module("google.auth", default=reject_cloud_setup),
        "google.cloud": _module("google.cloud", package=True),
        "google.cloud.logging": _module(
            "google.cloud.logging", Client=reject_cloud_setup
        ),
        "app": _module("app", package=True),
        "app.app_utils": _module("app.app_utils", package=True),
        "app.app_utils.services": _module(
            "app.app_utils.services",
            ARTIFACT_SERVICE_URI="shared://artifact",
            SESSION_SERVICE_URI="shared://session",
            get_artifact_service=lambda: object(),
            get_session_service=lambda: object(),
        ),
        "app.app_utils.a2a": _module(
            "app.app_utils.a2a", attach_a2a_routes=lambda *_args, **_kwargs: None
        ),
        "app.app_utils.typing": _module(
            "app.app_utils.typing", Feedback=type("Feedback", (), {})
        ),
    }
    modules["app.app_utils"].services = modules["app.app_utils.services"]
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    return captured


def test_local_template_uses_no_cloud_logging_dependency(tmp_path: Path) -> None:
    project = _render_local_project(tmp_path)
    dependencies = tomllib.loads((project / "pyproject.toml").read_text())["project"][
        "dependencies"
    ]

    assert not any(dep.startswith("google-cloud-logging") for dep in dependencies)


@pytest.mark.parametrize(
    ("initial_flag", "expected"),
    [(None, False), ("1", True), ("true", False)],
)
def test_local_template_import_needs_no_adc_and_honors_launcher_otel_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    initial_flag: str | None,
    expected: bool,
) -> None:
    project = _render_local_project(tmp_path)
    captured = _install_generated_app_stubs(monkeypatch)
    if initial_flag is None:
        monkeypatch.delenv(_OTEL_ENV, raising=False)
    else:
        monkeypatch.setenv(_OTEL_ENV, initial_flag)

    spec = importlib.util.spec_from_file_location(
        f"generated_local_fast_api_app_{expected}_{initial_flag}",
        project / "app" / "fast_api_app.py",
    )
    assert spec and spec.loader
    generated = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generated)

    assert captured["otel_to_cloud"] is expected
    assert isinstance(generated.logger, logging.Logger)


@pytest.mark.parametrize("trace_to_cloud", [False, True])
def test_server_child_only_receives_explicit_cloud_otel_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    trace_to_cloud: bool,
) -> None:
    agent_dir = tmp_path / "app"
    agent_dir.mkdir()
    (agent_dir / "fast_api_app.py").write_text("app = None\n", encoding="utf-8")
    monkeypatch.setenv(_OTEL_ENV, "inherited")
    child: dict[str, object] = {}

    def fake_popen(cmd: list[str], **kwargs: object) -> SimpleNamespace:
        child["cmd"] = cmd
        child["env"] = kwargs["env"]
        return SimpleNamespace(pid=1234)

    monkeypatch.setattr(_local_server, "popen_resolved_detached", fake_popen)

    _local_server._start_server(
        project_root=tmp_path,
        agent_dir="app",
        port=18080,
        trace_to_cloud=trace_to_cloud,
    )

    env = child["env"]
    assert isinstance(env, dict)
    if trace_to_cloud:
        assert env[_OTEL_ENV] == "1"
    else:
        assert _OTEL_ENV not in env
    assert "ignored when booting" not in caplog.text


def test_adk_fallback_retains_cloud_otel_flag(tmp_path: Path) -> None:
    command = _local_server._build_serve_command(
        project_root=tmp_path,
        agent_dir="app",
        port=18080,
        trace_to_cloud=True,
    )

    assert "--otel_to_cloud" in command
