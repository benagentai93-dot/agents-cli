# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from types import SimpleNamespace

import pytest
from vertexai._genai import _agent_engines_utils
from vertexai.agent_engines.templates.adk import AdkApp

from google.agents.cli._project import ProjectConfig
from google.agents.cli.deploy import agent_runtime

RESOURCE_NAME = "projects/123/locations/us-central1/reasoningEngines/runtime-1"


def _agent() -> SimpleNamespace:
    deployment_spec = SimpleNamespace(env=[], resource_limits={})
    return SimpleNamespace(
        api_resource=SimpleNamespace(
            name=RESOURCE_NAME,
            display_name="sample-agent",
            spec=SimpleNamespace(deployment_spec=deployment_spec),
        )
    )


class _AgentEngines:
    def __init__(self, existing: bool):
        self.existing = existing
        self.class_methods = None
        self.started = None

    def list(self):
        return [_agent()] if self.existing else []

    def get(self, *, name):
        assert name == RESOURCE_NAME
        return _agent()

    def _create_config(self, **kwargs):
        self.class_methods = kwargs["class_methods"]
        return kwargs

    def _create(self, *, config):
        self.started = "create"
        return SimpleNamespace(name=f"{RESOURCE_NAME}/operations/create")

    def _update(self, *, name, config):
        assert name == RESOURCE_NAME
        self.started = "update"
        return SimpleNamespace(name=f"{RESOURCE_NAME}/operations/update")


@pytest.mark.parametrize("existing", [False, True], ids=["create", "update"])
def test_deploy_wires_complete_adk_class_methods_without_self(
    monkeypatch, tmp_path, existing
) -> None:
    operations = _agent_engines_utils._get_registered_operations(
        agent=object.__new__(AdkApp)
    )
    expected = _agent_engines_utils._generate_class_methods_spec_or_raise(
        agent=object.__new__(AdkApp), operations=operations
    )
    expected_names = {
        _agent_engines_utils._to_dict(method)["name"] for method in expected
    }
    engines = _AgentEngines(existing)
    tmp_path.joinpath("Dockerfile").write_text("FROM scratch\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        agent_runtime.vertexai,
        "Client",
        lambda **_kwargs: SimpleNamespace(agent_engines=engines),
    )
    monkeypatch.setattr(agent_runtime.vertexai, "init", lambda **_kwargs: None)

    agent_runtime.deploy_agent_runtime(
        cfg=ProjectConfig(
            project_name="sample-agent", deployment_target="agent_runtime"
        ),
        project="sample-project",
        display_name="sample-agent",
        location="us-central1",
        source_packages=["./Dockerfile"],
        no_wait=True,
    )

    assert engines.started == ("update" if existing else "create")
    assert engines.class_methods
    assert len(engines.class_methods) == 13
    assert {method["name"] for method in engines.class_methods} == expected_names
    assert {
        "async_create_session",
        "async_get_session",
        "async_list_sessions",
        "async_stream_query",
    } <= expected_names
    assert all(
        "self" not in method["parameters"].get("properties", {})
        for method in engines.class_methods
    )
