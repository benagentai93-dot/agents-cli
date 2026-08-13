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

from pathlib import Path

import pytest
import requests
from rich.console import Console

from google.agents.cli._project import ProjectConfig
from google.agents.cli.eval import cmd_generate
from google.agents.cli.run import _local_server, cmd_run
from google.agents.cli.run._local_server import ServerInfo


@pytest.mark.parametrize("owned_pid", [None, 111])
@pytest.mark.parametrize("fail", [False, True])
def test_eval_stops_only_server_started_by_this_invocation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    owned_pid: int | None,
    fail: bool,
) -> None:
    stopped: list[tuple[Path, int | None]] = []
    monkeypatch.setattr(
        cmd_generate,
        "ensure_server",
        lambda *_args, **_kwargs: ServerInfo(18080, owned_pid=owned_pid),
    )
    monkeypatch.setattr(
        cmd_generate,
        "stop_server",
        lambda root, *, expected_pid=None: stopped.append((root, expected_pid)),
    )

    def run_http(**_kwargs) -> None:
        if fail:
            raise RuntimeError("inference failed")

    monkeypatch.setattr(cmd_generate, "_run_http", run_http)

    def call() -> None:
        cmd_generate._run_against_local_server(
            console=Console(quiet=True),
            project_root=tmp_path,
            cfg=ProjectConfig(agent_directory="app"),
            dataset="dataset.json",
            eval_cases=[],
            output_path=tmp_path / "traces.json",
            concurrency=1,
            custom_headers=(),
        )

    if fail:
        with pytest.raises(RuntimeError, match="inference failed"):
            call()
    else:
        call()

    assert stopped == ([(tmp_path, owned_pid)] if owned_pid is not None else [])


@pytest.mark.parametrize("owned_pid", [None, 111])
@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("keep_server", [False, True])
def test_run_cleanup_respects_ownership_and_keep_server(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    owned_pid: int | None,
    fail: bool,
    keep_server: bool,
) -> None:
    stopped: list[tuple[Path, int | None]] = []
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        cmd_run,
        "_resolve_dispatch_target",
        lambda **_kwargs: cmd_run._DispatchTarget(
            service_url="http://127.0.0.1:18080",
            headers={},
            mode="adk",
            app_name="app",
            owned_server_pid=owned_pid,
        ),
    )

    def dispatch(**_kwargs) -> None:
        if fail:
            raise requests.ConnectionError("lost")

    monkeypatch.setattr(cmd_run, "_dispatch_query", dispatch)
    monkeypatch.setattr(
        cmd_run,
        "stop_server",
        lambda root, *, expected_pid=None: stopped.append((root, expected_pid)),
    )

    callback = cmd_run.cmd_run.callback
    assert callback is not None

    def call() -> None:
        callback(
            message="hello",
            url=None,
            mode=None,
            app_name=None,
            files=(),
            session_id=None,
            custom_headers=(),
            start_server=keep_server,
            otel_to_cloud=False,
            trace_to_cloud=False,
            verbose=False,
        )

    if fail:
        with pytest.raises(requests.ConnectionError, match="lost"):
            call()
    else:
        call()

    should_stop = owned_pid is not None and not keep_server
    assert stopped == ([(tmp_path, owned_pid)] if should_stop else [])


def test_automatic_cleanup_does_not_stop_replacement_server(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cleaned: list[dict] = []
    _local_server._write_pid_file(tmp_path, pid=222, port=18081)
    monkeypatch.setattr(
        _local_server, "_cleanup", lambda _root, info: cleaned.append(info)
    )

    stopped = _local_server.stop_server(tmp_path, expected_pid=111)

    assert stopped is False
    assert cleaned == []


def test_explicit_cleanup_still_stops_current_server(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cleaned: list[dict] = []
    _local_server._write_pid_file(tmp_path, pid=222, port=18081)
    monkeypatch.setattr(
        _local_server, "_cleanup", lambda _root, info: cleaned.append(info)
    )

    stopped = _local_server.stop_server(tmp_path)

    assert stopped is True
    assert [info["pid"] for info in cleaned] == [222]
