# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import json
from types import SimpleNamespace

import click
import pytest

from google.agents.cli import _gcp_project
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
        self, listed=(), fetched=None, operation_error=None, clear_error=None
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
    monkeypatch.setattr(_gcp_project, "get_gcp_project_number", lambda _project: PROJECT_NUMBER)
    monkeypatch.setattr(
        agent_runtime,
        "get_gcp_project_number",
        lambda _project: PROJECT_NUMBER,
        raising=False,
    )
    no_wait = kwargs.pop("no_wait", True)
    return agent_runtime.deploy_agent_runtime(
        cfg=ProjectConfig(
            project_name="sample-agent", deployment_target="agent_runtime"
        ),
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
        listed=[_agent("projects/other/locations/other/reasoningEngines/wrong")],
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

    assert ("list",) not in engines.calls
    assert ("get", RESOURCE_NAME) in engines.calls
    updates = [call for call in engines.calls if call[0] == "_update"]
    assert len(updates) == 1
    assert updates[0][1] == RESOURCE_NAME


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
            ProjectConfig(
                project_name="sample-agent", deployment_target="agent_runtime"
            ),
        )

    assert path.read_bytes() == old


def test_failed_deployment_restores_previous_metadata_bytes(
    monkeypatch, tmp_path
) -> None:
    metadata = {
        "remote_agent_runtime_id": RESOURCE_NAME,
        "deployment_target": "agent_runtime",
    }
    engines = _AgentEngines(
        fetched=_agent(RESOURCE_NAME), operation_error="deployment failed"
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
        fetched=_agent(RESOURCE_NAME), clear_error="secret clear failed"
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
