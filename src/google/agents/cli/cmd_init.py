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

"""Initialize an existing Python ADK project for agents-cli."""

from __future__ import annotations

import contextlib
import os
import sys
import tempfile
import tomllib
from pathlib import Path

import click
import yaml

from google.agents.cli.scaffold.commands.create import normalize_project_name

_MANIFEST = "agents-cli-manifest.yaml"
_AGENT_MARKERS = ("agent.py", "root_agent.yaml")
_NON_PYTHON_PROJECT_MARKERS = ("package.json", "Cargo.toml", "go.mod", "pom.xml")


def _is_agent_directory(path: Path) -> bool:
    return path.is_dir() and any((path / marker).is_file() for marker in _AGENT_MARKERS)


def _is_interactive() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def _read_project_name(root: Path) -> object | None:
    pyproject = root / "pyproject.toml"
    if not pyproject.is_file():
        return None
    try:
        with pyproject.open("rb") as file:
            data = tomllib.load(file)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise click.ClickException(f"Could not read pyproject.toml: {error}") from error
    project = data.get("project")
    return project.get("name") if isinstance(project, dict) else None


def _validated_name(value: object, source: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise click.UsageError(f"{source} must be a non-empty string.")
    return value


def _normalize_name(name: str) -> str:
    # normalize_project_name reports changes; keep dry-run stdout machine-readable.
    with contextlib.redirect_stdout(sys.stderr):
        return normalize_project_name(name)


def _require_python_project(root: Path) -> None:
    if not (root / "pyproject.toml").exists() and any(
        (root / marker).exists() for marker in _NON_PYTHON_PROJECT_MARKERS
    ):
        raise click.ClickException(
            "Unsupported project language; init currently supports Python ADK projects only."
        )


def _relative_agent_directory(
    root: Path, candidate: Path, *, explicit: bool = False
) -> str:
    if explicit and candidate.is_absolute():
        raise click.UsageError("--agent-directory must be a relative path inside PATH.")
    resolved = (root / candidate).resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as error:
        raise click.UsageError(
            f"{'--agent-directory' if explicit else 'Agent directory'} resolves outside PATH."
        ) from error
    if not _is_agent_directory(resolved):
        raise click.UsageError(
            f"{'--agent-directory' if explicit else 'Agent directory'} must be an existing directory containing "
            "agent.py or root_agent.yaml."
        )
    return relative.as_posix()


def _choose(candidates: list[str], yes: bool) -> str:
    if len(candidates) == 1:
        return candidates[0]
    if not yes and _is_interactive():
        click.echo("Select agent directory:", err=True)
        for index, path in enumerate(candidates, 1):
            click.echo(f"  {index}. {path}", err=True)
        selection = click.prompt(
            "Selection", type=click.IntRange(1, len(candidates)), err=True
        )
        return candidates[selection - 1]
    raise click.UsageError(
        "Multiple agent directories found; specify --agent-directory."
    )


def _detect_agent_directory(root: Path, project_name: str | None, yes: bool) -> str:
    app = root / "app"
    if _is_agent_directory(app):
        return _relative_agent_directory(root, app)

    if project_name:
        package_name = _normalize_name(project_name).replace("-", "_")
        package_candidates = [root / package_name, root / "src" / package_name]
        valid_packages = [
            _relative_agent_directory(root, path)
            for path in package_candidates
            if _is_agent_directory(path)
        ]
        if valid_packages:
            return _choose(valid_packages, yes)

    direct_candidates = sorted(
        _relative_agent_directory(root, path)
        for path in root.iterdir()
        if _is_agent_directory(path)
    )
    if direct_candidates:
        return _choose(direct_candidates, yes)

    raise click.ClickException(
        "No Python ADK agent directory containing agent.py or root_agent.yaml was found."
    )


def _write_manifest(path: Path, contents: str) -> None:
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as file:
            temporary = file.name
            file.write(contents)
        os.replace(temporary, path)
    except OSError as error:
        if temporary:
            with contextlib.suppress(OSError):
                Path(temporary).unlink(missing_ok=True)
        raise click.ClickException(f"Could not write {_MANIFEST}: {error}") from error


@click.command("init")
@click.argument(
    "path",
    default=".",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option("--name", help="Project name for the manifest.")
@click.option(
    "--agent-directory",
    help="Relative existing agent directory within PATH.",
)
@click.option("--dry-run", is_flag=True, help="Print the manifest without writing it.")
@click.option("--yes", is_flag=True, help="Disable interactive prompts.")
@click.option("--force", is_flag=True, help="Atomically replace an existing manifest.")
def cmd_init(
    *,
    path: Path,
    name: str | None,
    agent_directory: str | None,
    dry_run: bool,
    yes: bool,
    force: bool,
) -> None:
    """Initialize an existing Python ADK project."""
    root = path.resolve()
    _require_python_project(root)
    raw_project_name = _read_project_name(root)
    project_name = (
        _validated_name(raw_project_name, "pyproject.toml [project].name")
        if raw_project_name is not None
        else None
    )
    selected_name = (
        _validated_name(name, "--name")
        if name is not None
        else project_name or root.name
    )
    normalized_name = _normalize_name(selected_name)
    selected_directory = (
        _relative_agent_directory(root, Path(agent_directory), explicit=True)
        if agent_directory is not None
        else _detect_agent_directory(root, project_name, yes)
    )
    contents = yaml.safe_dump(
        {
            "name": normalized_name,
            "agent_directory": selected_directory,
            "deployment_target": "none",
        },
        sort_keys=False,
    )
    manifest = root / _MANIFEST
    if dry_run:
        click.echo(contents, nl=False)
        return
    if manifest.exists() and not force:
        raise click.ClickException(f"{_MANIFEST} already exists; use --force to replace it.")
    _write_manifest(manifest, contents)
    click.echo(f"Created {manifest}", err=True)
