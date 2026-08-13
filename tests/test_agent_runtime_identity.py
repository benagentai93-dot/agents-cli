# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import click
import pytest

from google.agents.cli._project import ProjectConfig
from google.agents.cli.deploy import _operation, agent_runtime

PROJECT_NUMBER = "123456789"
LOCATION = "us-central1"
RESOURCE_NAME = (
    f"projects/{PROJECT_NUMBER}/locations/{LOCATION}/reasoningEngines/runtime-1"
)


def _agent(name: str, display_name: str = "sample-agent") -> SimpleNamespace:
    deployment_spec = SimpleNamespace(env=[], resource_limits={})
    spec = SimpleNamespace(deployment_spec=deployment_spec)
    return SimpleNamespace(
        api_resource=SimpleNamespace(
            name=name,
            display_name=display_name,
            spec=spec,
        )
    )


class _AgentEngines:
    def __init__(
        self,
        listed: list[SimpleNamespace] | tuple[SimpleNamespace, ...] = (),
        fetched=None,
        operation_error=None,
        clear_error=None,
    ):
        self.listed = list(listed)
        self.fetched = fetched
        self.operation_error = operation_error
        self.clear_error = clear_error
        self.calls = []

    def list(self):
        self.calls.append(("list",))
        return self.listed

    def get(self, *, name):
        self.calls.append(("get", name))
        if isinstance(self.fetched, Exception):
            raise self.fetched
        return self.fetched or _agent(name)

    def _create_config(self, **kwargs):
        return kwargs

    def _create(self, *, config):
        self.calls.append(("_create", config))
        return SimpleNamespace(name=f"{RESOURCE_NAME}/operations/create")

    def _update(self, *, name, config):
        self.calls.append(("_update", name, config))
        action = "clear" if config.get("update_mask") else "update"
        return SimpleNamespace(name=f"{name}/operations/{action}")

    def create(self, *, config):
        self.calls.append(("create_identity", config))
        return _agent(RESOURCE_NAME)

    def _get_agent_operation(self, *, operation_name):
        return SimpleNamespace(
            name=operation_name,
            done=True,
            error=(
                self.clear_error
                if operation_name.endswith("/operations/clear")
                else self.operation_error
            ),
        )


def _deploy(monkeypatch, tmp_path, engines: _AgentEngines, metadata=None, **kwargs):
    tmp_path.joinpath("Dockerfile").write_text("FROM scratch\n", encoding="utf-8")
    if metadata is not None:
        tmp_path.joinpath("deployment_metadata.json").write_text(
            json.dumps(metadata), encoding="utf-8"
        )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        agent_runtime.vertexai,
        "Client",
        lambda **_kwargs: SimpleNamespace(agent_engines=engines),
    )
    monkeypatch.setattr(agent_runtime.vertexai, "init", lambda **_kwargs: None)
    no_wait = kwargs.pop("no_wait", True)
    return agent_runtime.deploy_agent_runtime(
        cfg=ProjectConfig(project_name="sample-agent", deployment_target="agent_runtime"),
        project="sample-project",
        display_name="sample-agent",
        location=LOCATION,
        source_packages=["./Dockerfile"],
        no_wait=no_wait,
        **kwargs,
    )


def test_deploy_prefers_valid_metadata_identity_over_display_name_list(
    monkeypatch, tmp_path
) -> None:
    engines = _AgentEngines(
        listed=[
            _agent(RESOURCE_NAME),
            _agent("projects/other/locations/other/reasoningEngines/wrong"),
        ],
        fetched=_agent(RESOURCE_NAME),
    )

    _deploy(
        monkeypatch,
        tmp_path,
        engines,
        {
            "remote_agent_runtime_id": RESOURCE_NAME,
            "deployment_target": "agent_runtime",
        },
    )

    assert ("get", RESOURCE_NAME) in engines.calls
    assert engines.calls.index(("get", RESOURCE_NAME)) < engines.calls.index(("list",))
    updates = [call for call in engines.calls if call[0] == "_update"]
    assert len(updates) == 1
    assert updates[0][1] == RESOURCE_NAME


def test_metadata_identity_does_not_require_gcloud_or_resource_manager_api(
    monkeypatch, tmp_path
) -> None:
    engines = _AgentEngines(listed=[_agent(RESOURCE_NAME)], fetched=_agent(RESOURCE_NAME))
    monkeypatch.setenv("PATH", "")
    monkeypatch.setattr(
        agent_runtime.resourcemanager_v3,
        "ProjectsClient",
        lambda: (_ for _ in ()).throw(AssertionError("must not use Resource Manager")),
    )

    _deploy(
        monkeypatch,
        tmp_path,
        engines,
        {
            "remote_agent_runtime_id": RESOURCE_NAME,
            "deployment_target": "agent_runtime",
        },
    )

    assert ("get", RESOURCE_NAME) in engines.calls


def test_deploy_rejects_metadata_display_mismatch_before_mutation(
    monkeypatch, tmp_path
) -> None:
    engines = _AgentEngines(fetched=_agent(RESOURCE_NAME, "different-agent"))

    with pytest.raises(click.ClickException, match="display"):
        _deploy(
            monkeypatch,
            tmp_path,
            engines,
            {
                "remote_agent_runtime_id": RESOURCE_NAME,
                "deployment_target": "agent_runtime",
            },
        )

    assert not {"_create", "_update", "create_identity"} & {
        call[0] for call in engines.calls
    }


def test_deploy_rejects_duplicate_display_names_before_identity_or_mutation(
    monkeypatch, tmp_path
) -> None:
    engines = _AgentEngines(
        listed=[
            _agent(
                f"projects/{PROJECT_NUMBER}/locations/{LOCATION}/reasoningEngines/one"
            ),
            _agent(
                f"projects/{PROJECT_NUMBER}/locations/{LOCATION}/reasoningEngines/two"
            ),
        ]
    )

    with pytest.raises(click.ClickException, match=r"[Mm]ultiple|[Dd]uplicate"):
        _deploy(monkeypatch, tmp_path, engines, agent_identity=True)

    assert not {"_create", "_update", "create_identity"} & {
        call[0] for call in engines.calls
    }


@pytest.mark.parametrize(
    ("remote_id", "target", "fetched", "message"),
    [
        ("malformed", "agent_runtime", None, "Invalid"),
        ("", "agent_runtime", None, "Invalid"),
        (None, "agent_runtime", None, "Invalid"),
        (123, "agent_runtime", None, "Invalid"),
        (RESOURCE_NAME, "cloud_run", None, "deployment_target"),
        (
            RESOURCE_NAME.replace(PROJECT_NUMBER, "987654321"),
            "agent_runtime",
            None,
            "project and location",
        ),
        (
            RESOURCE_NAME.replace(LOCATION, "europe-west1"),
            "agent_runtime",
            None,
            "project and location",
        ),
        (RESOURCE_NAME, "agent_runtime", RuntimeError("gone"), "could not be loaded"),
        (
            RESOURCE_NAME,
            "agent_runtime",
            _agent(RESOURCE_NAME.replace("runtime-1", "runtime-2")),
            "resource name or display name",
        ),
        (
            RESOURCE_NAME,
            "agent_runtime",
            _agent(RESOURCE_NAME, "different-agent"),
            "resource name or display name",
        ),
    ],
    ids=(
        "malformed-id empty-id null-id non-string-id wrong-target wrong-project "
        "wrong-location stale-id actual-name-mismatch display-mismatch"
    ).split(),
)
def test_deploy_rejects_invalid_metadata_before_mutation(
    monkeypatch, tmp_path, remote_id, target, fetched, message
) -> None:
    engines = _AgentEngines(fetched=fetched)

    with pytest.raises(click.ClickException, match=message):
        _deploy(
            monkeypatch,
            tmp_path,
            engines,
            {"remote_agent_runtime_id": remote_id, "deployment_target": target},
        )

    assert not {"_create", "_update", "create_identity"} & {
        call[0] for call in engines.calls
    }


@pytest.mark.parametrize(
    ("metadata", "listed", "mutation"),
    [
        (None, [], "_create"),
        (None, [_agent(RESOURCE_NAME)], "_update"),
        (
            {"remote_agent_runtime_id": "None", "deployment_target": "cloud_run"},
            [_agent(RESOURCE_NAME)],
            "_update",
        ),
    ],
    ids="no-match-creates one-match-updates none-sentinel-falls-back".split(),
)
def test_deploy_without_usable_metadata_mutates_only_unambiguous_target(
    monkeypatch, tmp_path, metadata, listed, mutation
) -> None:
    engines = _AgentEngines(listed=listed)

    _deploy(monkeypatch, tmp_path, engines, metadata)

    mutations = [call[0] for call in engines.calls if call[0] in {"_create", "_update"}]
    assert mutations == [mutation]


def test_deploy_revalidates_target_after_claim_before_mutation(
    monkeypatch, tmp_path
) -> None:
    listed = [_agent(RESOURCE_NAME, "different-agent")]
    engines = _AgentEngines(listed=listed)
    metadata = {
        "remote_agent_runtime_id": RESOURCE_NAME,
        "deployment_target": "agent_runtime",
    }
    original_claim = agent_runtime.claim_operation

    def claim_after_other_deploy(*args):
        tmp_path.joinpath("deployment_metadata.json").write_text(
            json.dumps(metadata), encoding="utf-8"
        )
        listed[0] = _agent(RESOURCE_NAME)
        engines.listed = listed
        engines.fetched = _agent(RESOURCE_NAME)
        return original_claim(*args)

    monkeypatch.setattr(agent_runtime, "claim_operation", claim_after_other_deploy)

    with pytest.raises(click.ClickException, match=r"changed.*retry"):
        _deploy(monkeypatch, tmp_path, engines)

    assert not {"_create", "_update", "create_identity"} & {
        call[0] for call in engines.calls
    }
    assert (
        json.loads(tmp_path.joinpath("deployment_metadata.json").read_text()) == metadata
    )


def test_write_deployment_metadata_produces_valid_json(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)

    agent_runtime.write_deployment_metadata(
        _agent(RESOURCE_NAME),
        ProjectConfig(project_name="sample-agent", deployment_target="agent_runtime"),
    )

    metadata = json.loads(tmp_path.joinpath("deployment_metadata.json").read_text())
    assert metadata["remote_agent_runtime_id"] == RESOURCE_NAME
    assert metadata["deployment_target"] == "agent_runtime"


def test_write_deployment_metadata_replace_failure_preserves_old_bytes(
    monkeypatch, tmp_path
) -> None:
    path = tmp_path.joinpath("deployment_metadata.json")
    old = b'{"remote_agent_runtime_id":"old"}\n'
    path.write_bytes(old)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        _operation.os, "replace", lambda *_args: (_ for _ in ()).throw(OSError("full"))
    )

    with pytest.raises(OSError, match="full"):
        agent_runtime.write_deployment_metadata(
            _agent(RESOURCE_NAME),
            ProjectConfig(project_name="sample-agent", deployment_target="agent_runtime"),
        )

    assert path.read_bytes() == old


def test_clear_operation_preserves_metadata_written_while_pending(
    monkeypatch, tmp_path
) -> None:
    path = tmp_path.joinpath("deployment_metadata.json")
    original = {"remote_agent_runtime_id": RESOURCE_NAME}
    path.write_text(json.dumps(original), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    _operation.write_operation(
        f"{RESOURCE_NAME}/operations/create",
        "sample-project",
        LOCATION,
        "agent_runtime",
    )
    current = json.loads(path.read_text())
    current["some_other_writer"] = "keep"
    _operation.write_metadata(current)

    _operation.clear_operation(current["pending_operation"]["claim_id"])

    assert json.loads(path.read_text()) == {**original, "some_other_writer": "keep"}


def test_legacy_status_cleanup_does_not_clear_a_new_operation(
    monkeypatch, tmp_path
) -> None:
    path = tmp_path.joinpath("deployment_metadata.json")
    legacy_name = f"{RESOURCE_NAME}/operations/legacy"
    path.write_text(
        json.dumps({"pending_operation": {"operation_name": legacy_name}}),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    observed = _operation.read_operation()
    assert observed is not None
    path.write_text(
        json.dumps(
            {
                "pending_operation": {
                    "claim_id": "new-owner",
                    "operation_name": f"{RESOURCE_NAME}/operations/new",
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(_operation.OperationPendingError):
        _operation.clear_operation(
            observed.get("claim_id"),
            operation_name=observed["operation_name"],
        )

    assert json.loads(path.read_text())["pending_operation"]["claim_id"] == "new-owner"


def test_legacy_operation_finishes_by_exact_operation_name(monkeypatch, tmp_path) -> None:
    path = tmp_path.joinpath("deployment_metadata.json")
    operation_name = f"{RESOURCE_NAME}/operations/legacy"
    path.write_text(
        json.dumps({"pending_operation": {"operation_name": operation_name}}),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    _operation.finish_operation(
        None,
        {"remote_agent_runtime_id": RESOURCE_NAME},
        operation_name=operation_name,
    )

    assert json.loads(path.read_text()) == {"remote_agent_runtime_id": RESOURCE_NAME}


def test_concurrent_operation_claim_has_exactly_one_owner(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    barrier = threading.Barrier(2)
    mutations = []
    monkeypatch.setattr(
        agent_runtime,
        "_start_deploy_operation",
        lambda *_args, **_kwargs: (
            mutations.append("started")
            or SimpleNamespace(name=f"{RESOURCE_NAME}/operations/create")
        ),
    )
    monkeypatch.setattr(
        agent_runtime, "_prepare_deploy_config", lambda *_args, **_kwargs: {}
    )
    monkeypatch.setattr(
        agent_runtime, "_select_agent_runtime", lambda *_args, **_kwargs: []
    )

    def deploy() -> bool:
        barrier.wait()
        try:
            agent_runtime._start_and_record_operation(
                SimpleNamespace(),
                agent_runtime.AgentEngineConfig(display_name="sample-agent"),
                [],
                "sample-project",
                LOCATION,
            )
        except click.ClickException:
            return False
        return True

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _index: deploy(), range(2)))

    assert results.count(True) == 1
    assert mutations == ["started"]
    assert _operation.read_operation() is not None


def test_concurrent_processes_have_exactly_one_operation_claim(
    tmp_path,
) -> None:
    script = """
from google.agents.cli.deploy import _operation

try:
    _operation.claim_operation("sample-project", "us-central1", "agent_runtime")
except _operation.OperationPendingError:
    print("blocked")
else:
    print("claimed")
"""
    env = {
        **os.environ,
        "PYTHONPATH": str(Path(__file__).parents[1] / "src"),
    }
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", script],
            cwd=tmp_path,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        for _ in range(2)
    ]
    outputs = [process.communicate(timeout=10) for process in processes]

    assert [process.returncode for process in processes] == [0, 0]
    assert sorted(stdout.strip() for stdout, _stderr in outputs) == [
        "blocked",
        "claimed",
    ]


def test_pending_deploy_rejects_second_mutation_and_preserves_operation(
    monkeypatch, tmp_path
) -> None:
    pending = {
        "operation_name": f"{RESOURCE_NAME}/operations/first",
        "project": "sample-project",
        "location": LOCATION,
        "deployment_target": "agent_runtime",
        "started_at": "2026-08-13T00:00:00+00:00",
    }
    metadata = {"pending_operation": pending}
    engines = _AgentEngines()
    monkeypatch.setattr(
        agent_runtime,
        "setup_agent_identity",
        lambda *_args: (
            engines.calls.append(("create_identity",)) or _agent(RESOURCE_NAME)
        ),
    )

    with pytest.raises(click.ClickException, match=r"deploy --status"):
        _deploy(monkeypatch, tmp_path, engines, metadata, agent_identity=True)

    assert not {"_create", "_update", "create_identity"} & {
        call[0] for call in engines.calls
    }
    assert (
        json.loads(tmp_path.joinpath("deployment_metadata.json").read_text()) == metadata
    )


def test_successful_deploy_preserves_metadata_written_while_pending(
    monkeypatch, tmp_path
) -> None:
    path = tmp_path.joinpath("deployment_metadata.json")
    original = {
        "remote_agent_runtime_id": RESOURCE_NAME,
        "deployment_target": "agent_runtime",
    }
    engines = _AgentEngines(listed=[_agent(RESOURCE_NAME)], fetched=_agent(RESOURCE_NAME))

    def await_operation(**_kwargs):
        current = json.loads(path.read_text())
        current["some_other_writer"] = "keep"
        _operation.write_metadata(current)

    monkeypatch.setattr(
        agent_runtime._agent_engines_utils,
        "_await_operation",
        await_operation,
    )

    _deploy(monkeypatch, tmp_path, engines, original, no_wait=False)

    metadata = json.loads(path.read_text())
    assert metadata["some_other_writer"] == "keep"
    assert metadata["remote_agent_runtime_id"] == RESOURCE_NAME
    assert "pending_operation" not in metadata


def test_finish_operation_is_one_atomic_metadata_transition(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.chdir(tmp_path)
    claim_id = _operation.claim_operation("sample-project", LOCATION, "agent_runtime")
    _operation.update_metadata({"some_other_writer": "keep"})
    entered_write = threading.Event()
    release_write = threading.Event()
    original_write = _operation._write_metadata

    def blocking_write(data) -> None:
        if (
            data.get("remote_agent_runtime_id") == RESOURCE_NAME
            and "pending_operation" not in data
        ):
            entered_write.set()
            assert release_write.wait(timeout=5)
        original_write(data)

    monkeypatch.setattr(_operation, "_write_metadata", blocking_write)

    with ThreadPoolExecutor(max_workers=2) as executor:
        finish = executor.submit(
            _operation.finish_operation,
            claim_id,
            {"remote_agent_runtime_id": RESOURCE_NAME},
        )
        assert entered_write.wait(timeout=5)
        next_claim = executor.submit(
            _operation.claim_operation,
            "sample-project",
            LOCATION,
            "agent_runtime",
        )
        with pytest.raises(TimeoutError):
            next_claim.result(timeout=0.1)
        release_write.set()
        finish.result(timeout=5)
        next_claim_id = next_claim.result(timeout=5)

    metadata = json.loads(tmp_path.joinpath("deployment_metadata.json").read_text())
    assert metadata["some_other_writer"] == "keep"
    assert metadata["remote_agent_runtime_id"] == RESOURCE_NAME
    assert metadata["pending_operation"]["claim_id"] == next_claim_id


def test_metadata_lock_is_not_created_in_packaged_project(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)

    _operation.write_metadata({"deployment_target": "agent_runtime"})

    assert [path.name for path in tmp_path.iterdir()] == ["deployment_metadata.json"]


def test_operation_record_failure_reports_remote_operation(monkeypatch, tmp_path) -> None:
    engines = _AgentEngines()
    monkeypatch.setattr(
        agent_runtime,
        "write_operation",
        lambda **_kwargs: (_ for _ in ()).throw(OSError("disk full")),
    )

    with pytest.raises(click.ClickException) as exc_info:
        _deploy(monkeypatch, tmp_path, engines)

    message = str(exc_info.value)
    assert f"{RESOURCE_NAME}/operations/create" in message
    assert "deploy --status" in message
    assert [call[0] for call in engines.calls].count("_create") == 1


def test_local_operation_config_failure_restores_previous_metadata(
    monkeypatch, tmp_path
) -> None:
    path = tmp_path.joinpath("deployment_metadata.json")
    original = b'{"some_other_writer":"keep"}\n'
    path.write_bytes(original)
    engines = _AgentEngines()
    monkeypatch.setattr(
        engines,
        "_create_config",
        lambda **_kwargs: (_ for _ in ()).throw(ValueError("bad local config")),
    )

    with pytest.raises(ValueError, match="bad local config"):
        _deploy(monkeypatch, tmp_path, engines)

    assert path.read_bytes() == original
    assert not {"_create", "_update"} & {call[0] for call in engines.calls}
    claim_id = _operation.claim_operation("sample-project", LOCATION, "agent_runtime")
    _operation.clear_operation(claim_id)


def test_remote_operation_submit_failure_keeps_claim_fail_closed(
    monkeypatch, tmp_path
) -> None:
    engines = _AgentEngines()

    def fail_submit(*, config):
        engines.calls.append(("_create", config))
        raise RuntimeError("submission outcome unknown")

    monkeypatch.setattr(engines, "_create", fail_submit)

    with pytest.raises(RuntimeError, match="outcome unknown"):
        _deploy(monkeypatch, tmp_path, engines)

    monkeypatch.chdir(tmp_path)
    pending = _operation.read_operation()
    assert pending is not None
    assert pending["state"] == "starting"
    assert "operation_name" not in pending
    with pytest.raises(_operation.OperationPendingError):
        _operation.claim_operation("sample-project", LOCATION, "agent_runtime")


def test_identity_setup_failure_keeps_claim_fail_closed(monkeypatch, tmp_path) -> None:
    engines = _AgentEngines()

    def fail_identity_create(*, config):
        engines.calls.append(("create_identity", config))
        raise RuntimeError("identity create outcome unknown")

    monkeypatch.setattr(engines, "create", fail_identity_create)

    with pytest.raises(RuntimeError, match="outcome unknown"):
        _deploy(monkeypatch, tmp_path, engines, agent_identity=True)

    pending = json.loads(
        tmp_path.joinpath("deployment_metadata.json").read_text(encoding="utf-8")
    )["pending_operation"]
    assert pending["state"] == "starting"
    assert "operation_name" not in pending
    assert [call[0] for call in engines.calls].count("create_identity") == 1


def test_failed_deployment_restores_previous_metadata_bytes(
    monkeypatch, tmp_path
) -> None:
    metadata = {
        "remote_agent_runtime_id": RESOURCE_NAME,
        "deployment_target": "agent_runtime",
    }
    engines = _AgentEngines(
        listed=[_agent(RESOURCE_NAME)],
        fetched=_agent(RESOURCE_NAME),
        operation_error="deployment failed",
    )

    with pytest.raises(click.ClickException, match="Deployment failed"):
        _deploy(monkeypatch, tmp_path, engines, metadata, no_wait=False)

    expected = json.dumps(metadata).encode()
    assert tmp_path.joinpath("deployment_metadata.json").read_bytes() == expected


def test_failed_secret_clear_restores_previous_metadata_bytes(
    monkeypatch, tmp_path
) -> None:
    metadata = {
        "remote_agent_runtime_id": RESOURCE_NAME,
        "deployment_target": "agent_runtime",
    }
    engines = _AgentEngines(
        listed=[_agent(RESOURCE_NAME)],
        fetched=_agent(RESOURCE_NAME),
        clear_error="secret clear failed",
    )

    with pytest.raises(click.ClickException, match="clear Agent Runtime secrets"):
        _deploy(
            monkeypatch,
            tmp_path,
            engines,
            metadata,
            no_wait=False,
            set_secrets="",
        )

    expected = json.dumps(metadata).encode()
    assert tmp_path.joinpath("deployment_metadata.json").read_bytes() == expected
