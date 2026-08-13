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

import json
from pathlib import Path

import click
import pytest

from google.agents.cli.run import _local_server


def _existing_server(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    metadata_mode: bool | None,
) -> dict[str, object]:
    metadata: dict[str, object] = {
        "pid": 123,
        "port": 18080,
        "last_activity": "2999-01-01T00:00:00+00:00",
    }
    if metadata_mode is not None:
        metadata["use_in_memory_session"] = metadata_mode
    stopped: list[dict] = []
    monkeypatch.setattr(_local_server, "_read_pid_file", lambda _root: metadata)
    monkeypatch.setattr(_local_server, "_is_server_alive", lambda *_args: True)
    monkeypatch.setattr(_local_server, "_update_activity", lambda _root: None)
    monkeypatch.setattr(
        _local_server, "_cleanup", lambda _root, info: stopped.append(info)
    )
    return {"metadata": metadata, "stopped": stopped}


@pytest.mark.parametrize("mode", [True, False])
def test_server_reuses_only_matching_session_mode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mode: bool
) -> None:
    state = _existing_server(monkeypatch, tmp_path, metadata_mode=mode)

    info = _local_server.ensure_server(
        tmp_path, "app", use_in_memory_session=mode
    )

    assert info == _local_server.ServerInfo(18080, started=False)
    assert state["stopped"] == []


@pytest.mark.parametrize(
    ("metadata_mode", "requested_mode"), [(True, False), (False, True)]
)
def test_server_rejects_mismatched_session_mode_without_stopping_existing_process(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    metadata_mode: bool,
    requested_mode: bool,
) -> None:
    state = _existing_server(monkeypatch, tmp_path, metadata_mode=metadata_mode)

    with pytest.raises(click.ClickException, match="different session mode"):
        _local_server.ensure_server(
            tmp_path, "app", use_in_memory_session=requested_mode
        )

    assert state["stopped"] == []


def test_legacy_metadata_reuses_historical_in_memory_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    state = _existing_server(monkeypatch, tmp_path, metadata_mode=None)

    info = _local_server.ensure_server(
        tmp_path, "app", use_in_memory_session=True
    )

    assert info == _local_server.ServerInfo(18080, started=False)
    assert state["stopped"] == []


def test_legacy_metadata_fails_closed_for_persistent_eval_session(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    state = _existing_server(monkeypatch, tmp_path, metadata_mode=None)

    with pytest.raises(click.ClickException, match="different session mode"):
        _local_server.ensure_server(
            tmp_path, "app", use_in_memory_session=False
        )

    assert state["stopped"] == []


@pytest.mark.parametrize("mode", [True, False])
def test_pid_metadata_records_session_mode(tmp_path: Path, mode: bool) -> None:
    _local_server._write_pid_file(
        tmp_path,
        pid=123,
        port=18080,
        use_in_memory_session=mode,
    )

    metadata = json.loads(
        (tmp_path / ".google-agents-cli" / "run_server.json").read_text()
    )
    assert metadata["use_in_memory_session"] is mode
