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

"""agents-cli run command — run agent with a single prompt."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import mimetypes
import uuid
from pathlib import Path
from typing import NamedTuple

import click
import httpx
import requests
from a2a.client import ClientConfig, create_client
from a2a.types import Message, Part, Role, SendMessageRequest
from a2a.utils.constants import (
    AGENT_CARD_WELL_KNOWN_PATH,
    PROTOCOL_VERSION_1_0,
    VERSION_HEADER,
    TransportProtocol,
)
from google.protobuf.json_format import MessageToDict

from google.agents.cli._adk_client import create_session, run_sse
from google.agents.cli._agent_runtime_a2a import (
    build_agent_runtime_a2a_base_url,
    build_agent_runtime_a2a_card_url,
)
from google.agents.cli._project import (
    chdir_project_root,
    read_project_config,
    require_agent_directory,
)
from google.agents.cli._remote import (
    build_remote_headers,
    is_raw_agent_runtime_url,
    parse_agent_runtime_service_url,
    validate_agent_runtime_url,
)
from google.agents.cli.run._local_server import ensure_server, stop_server
from google.agents.cli.run._multimodal import (
    build_a2a_parts,
    build_adk_sse_parts,
    build_agent_runtime_message,
)

_ARTIFACTS_DIR = Path(".google-agents-cli") / "artifacts"


class _DispatchTarget(NamedTuple):
    service_url: str
    headers: dict
    mode: str
    app_name: str
    # Set only when this invocation started the local server.
    owned_server_pid: int | None = None


def _resolve_dispatch_target(
    url: str | None,
    mode: str | None,
    app_name: str | None,
    custom_headers: tuple[str, ...],
    *,
    trace_to_cloud: bool = False,
) -> _DispatchTarget:
    """Resolve where and how to dispatch a query.

    Remote (``--url``): validates ``--mode`` first so missing flags don't
    surface as "no pyproject.toml". Reads the local project for
    ``app_name`` only when ``--app-name`` was not provided.

    Local: starts (or reuses) a background server and points at it.
    Always uses ADK SSE.
    """
    if url:
        if not mode:
            raise click.UsageError(
                "--mode is required when using --url. Choose from: a2a, adk"
            )
        validate_agent_runtime_url(url)
        if app_name:
            resolved = app_name
        else:
            chdir_project_root()
            cfg = read_project_config()
            require_agent_directory(cfg)
            resolved = cfg.agent_directory
        return _DispatchTarget(
            service_url=url,
            headers=build_remote_headers(custom_headers, url),
            mode=mode,
            app_name=resolved,
        )

    chdir_project_root()
    cfg = read_project_config()
    require_agent_directory(cfg)
    server = ensure_server(Path.cwd(), cfg.agent_directory, trace_to_cloud=trace_to_cloud)
    return _DispatchTarget(
        service_url=f"http://127.0.0.1:{server.port}",
        headers={},
        mode="adk",
        app_name=app_name or cfg.agent_directory,
        owned_server_pid=server.owned_pid,
    )


def _handle_stop_server(ctx: click.Context, _param: click.Parameter, value: bool) -> None:
    """Eager callback for ``--stop-server``: stop and exit before argument parsing."""
    if not value:
        return
    chdir_project_root()
    if stop_server(Path.cwd()):
        ctx.exit(0)
    raise click.ClickException("No local server is running.")


@click.command("run")
@click.argument("message")
@click.option(
    "--url",
    default=None,
    help="URL of a remote agent to query. If specified, no local server is started.",
)
@click.option(
    "--mode",
    type=click.Choice(["a2a", "adk"], case_sensitive=False),
    default=None,
    help=(
        "Protocol for --url queries: 'a2a' or 'adk'. "
        "Required when using --url. Local runs always use ADK SSE."
    ),
)
@click.option(
    "--app-name",
    default=None,
    help=(
        "Agent name for remote ADK SSE / A2A endpoints. "
        "Defaults to the local project's agent_directory; "
        "specify to query a different agent or to run from outside a project."
    ),
)
@click.option(
    "--file",
    "-f",
    "files",
    multiple=True,
    type=click.Path(exists=True, readable=True),
    help="Attach a file (image, PDF, audio, video). Repeatable.",
)
@click.option(
    "--session-id",
    default=None,
    help=(
        "Resume an existing session. A local session lives in the running "
        "server's memory — keep the server alive with --start-server to "
        "resume it across runs."
    ),
)
@click.option(
    "--header",
    "-H",
    "custom_headers",
    multiple=True,
    help="Custom HTTP header (format: 'Key: Value'). Repeatable. Overrides auto-detected auth.",
)
@click.option(
    "--start-server",
    "start_server",
    is_flag=True,
    default=False,
    help=(
        "Keep the local server running after execution. The server persists until "
        "stopped with --stop-server, giving subsequent run requests less overhead "
        "and keeping in-memory sessions alive between runs."
    ),
)
@click.option(
    "--stop-server",
    "stop_server_flag",
    is_flag=True,
    default=False,
    is_eager=True,
    expose_value=False,
    callback=_handle_stop_server,
    help="Stop the local background server and exit.",
)
@click.option(
    "--otel-to-cloud",
    is_flag=True,
    default=False,
    help=(
        "Export OpenTelemetry traces/logs to Google Cloud. "
        "Takes effect when the local server starts; ignored with --url."
    ),
)
# TODO: b/533949139
@click.option(
    "--trace-to-cloud",
    is_flag=True,
    default=False,
    hidden=True,
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    default=False,
    help="Print full JSON event payloads.",
)
def cmd_run(
    message: str,
    *,
    url: str | None,
    mode: str | None,
    app_name: str | None,
    files: tuple[str, ...],
    session_id: str | None,
    custom_headers: tuple[str, ...],
    start_server: bool,
    otel_to_cloud: bool,
    trace_to_cloud: bool,
    verbose: bool,
):
    """Run the agent with a single prompt (non-interactive).

    MESSAGE is the prompt to send to the agent.

    \b
    Run from your project directory to query the agent locally. A plain run
    starts a one-off server and shuts it down when it finishes; pass
    --start-server to keep the server running in the background instead.
    Once a server is running, later plain runs reuse it (and leave it
    running), so repeated requests are faster. Stop a persistent server with
    --stop-server. After 30 minutes idle, the next request restarts it.

    \b
    Use --url to query a deployed agent instead. Requires
    --mode to choose the protocol:

    \b
      a2a   A2A protocol
      adk   ADK SSE (/run_sse, or :streamQuery for Agent Runtime)

    \b
    Supports --file for multimodal input and --session-id for
    conversation continuity. A local session lives in the running server's
    memory, so keep the server alive with --start-server to resume it
    across separate runs.

    \b
    Binary artifacts (images, audio, files) returned by the agent are
    saved under '.google-agents-cli/artifacts/' in the project root and
    listed in an 'Artifacts:' footer at the end of the response.
    File references returned by URI are not downloaded.
    """
    if url and start_server:
        click.secho(
            "Warning: --start-server has no effect when using --url.",
            fg="yellow",
            err=True,
        )
    # TODO: b/533949139
    if trace_to_cloud:
        logging.warning(
            "--trace-to-cloud is deprecated and will be removed in a future "
            "release. Use --otel-to-cloud instead."
        )
    export_otel = otel_to_cloud or trace_to_cloud
    if url and export_otel:
        click.secho(
            "Warning: --otel-to-cloud has no effect when using --url.",
            fg="yellow",
            err=True,
        )

    target = _resolve_dispatch_target(
        url=url,
        mode=mode,
        app_name=app_name,
        custom_headers=custom_headers,
        trace_to_cloud=export_otel,
    )
    if url:
        click.echo(f"Querying remote agent: {url} (mode: {target.mode})")

    # Only tear down a server this invocation started; a reused persistent
    # server (e.g. from --start-server) is left running.
    should_stop_server = (
        not url and not start_server and target.owned_server_pid is not None
    )
    try:
        _dispatch_query(
            service_url=target.service_url,
            message=message,
            files=files,
            headers=target.headers,
            mode=target.mode,
            app_name=target.app_name,
            session_id=session_id,
            verbose=verbose,
        )
    except (
        requests.ConnectionError,
        requests.Timeout,
        httpx.TransportError,
    ) as exc:
        if not url:
            raise
        raise click.ClickException(
            f"Could not reach remote agent at: {url}\n"
            f"  {exc}\n"
            "  Check that the URL is correct and the service is running."
        ) from exc
    finally:
        if should_stop_server:
            # cwd is the project root here (set by _resolve_dispatch_target).
            stop_server(Path.cwd(), expected_pid=target.owned_server_pid)


def _dispatch_query(
    service_url: str,
    message: str,
    files: tuple[str, ...],
    headers: dict,
    *,
    mode: str,
    app_name: str,
    session_id: str | None = None,
    verbose: bool = False,
) -> None:
    """Route a query to the right protocol handler.

    Used by both local (``mode='adk'``, localhost ``service_url``,
    empty ``headers``) and remote (``mode`` from ``--mode``, deployed
    URL, auth headers) flows so the two paths can't drift.

    Modes:
      - ``a2a``: A2A protocol.  If the URL points to Agent Runtime,
        automatically constructs the ``/a2a`` sub-path; otherwise
        appends ``/a2a/{app_name}``.
      - ``adk``: ADK SSE.  Uses ``:streamQuery`` for Agent Runtime
        URLs, ``/run_sse`` for everything else.
    """
    if mode == "a2a":
        if is_raw_agent_runtime_url(service_url):
            location, runtime_resource = parse_agent_runtime_service_url(service_url)
            a2a_base = build_agent_runtime_a2a_base_url(
                location, runtime_resource, app_name
            )
            card_url = build_agent_runtime_a2a_card_url(
                location, runtime_resource, app_name
            )
        else:
            a2a_base = f"{service_url}/a2a/{app_name}"
            card_url = f"{a2a_base}{AGENT_CARD_WELL_KNOWN_PATH}"
        _query_a2a(
            card_url=card_url,
            parts=build_a2a_parts(message, files),
            headers=headers,
            url=a2a_base,
            session_id=session_id,
            verbose=verbose,
        )
    elif mode == "adk":
        if is_raw_agent_runtime_url(service_url):
            _query_agent_runtime_sse(
                service_url=service_url,
                message=build_agent_runtime_message(message, files),
                headers=headers,
                session_id=session_id,
                verbose=verbose,
            )
        else:
            _query_adk_sse(
                service_url=service_url,
                parts=build_adk_sse_parts(message, files),
                headers=headers,
                app_name=app_name,
                session_id=session_id,
                verbose=verbose,
            )


def _print_session_id(session_id: str | None) -> None:
    """Print session ID footer with a copy-pasteable resume command."""
    if not session_id:
        return
    click.echo()
    click.secho(f"Session: {session_id}", dim=True)
    click.secho(
        f'  Resume with: agents-cli run "<message>" --session-id {session_id}',
        dim=True,
    )


def _print_artifacts(paths: list[str]) -> None:
    """Print an Artifacts footer listing binary artifacts saved to disk."""
    if not paths:
        return
    click.echo()
    click.secho("Artifacts:", dim=True)
    for path in paths:
        click.secho(f"  {path}", dim=True)


def _write_artifact(data: bytes, mime_type: str | None) -> str:
    """Write artifact bytes to the artifacts dir and return a display path."""
    mime = mime_type or "application/octet-stream"
    ext = mimetypes.guess_extension(mime) or ""
    artifacts_dir = Path.cwd() / _ARTIFACTS_DIR
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    path = artifacts_dir / f"{uuid.uuid4().hex[:8]}{ext}"
    path.write_bytes(data)
    # Prefer a relative path so terminals can cmd+click and shells can tab-complete.
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def _save_inline_artifact(data_b64: str, mime_type: str | None) -> str | None:
    """Decode URL-safe base64 inline data and save it as an artifact.

    Returns the saved path, or ``None`` if decoding failed.
    """
    mime = mime_type or "application/octet-stream"
    try:
        decoded = base64.urlsafe_b64decode(data_b64)
    except ValueError as exc:
        click.secho(
            f"Warning: could not decode inline {mime} artifact: {exc}",
            err=True,
            fg="yellow",
        )
        return None

    return _write_artifact(decoded, mime_type)


def _print_author_tag(author: str | None, last_author: str | None) -> str | None:
    """Print ``[author]:`` tag when author changes. Returns updated last_author."""
    if author and author != last_author:
        if last_author is not None:
            click.echo()
        click.echo(f"[{author}]: ", nl=False)
        return author
    return last_author


def _print_sse_event(
    event: dict,
    last_author: str | None,
    verbose: bool,
    artifacts: list[str],
) -> str | None:
    """Process and print a single SSE/NDJSON event. Returns updated last_author.
    Appends paths of any saved binary artifacts to ``artifacts``.
    """
    author = event.get("author")
    last_author = _print_author_tag(author, last_author)
    content = event.get("content")
    if isinstance(content, dict):
        for part in content.get("parts", []):
            _print_sse_part(part, artifacts)
    if verbose:
        click.echo()
        click.secho(json.dumps(event, indent=2), dim=True)
    return last_author


def _query_adk_sse(
    service_url: str,
    parts: list[dict],
    headers: dict,
    *,
    app_name: str,
    session_id: str | None = None,
    verbose: bool = False,
) -> None:
    """Create a session and stream an SSE response from an ADK FastAPI agent.

    Thin CLI wrapper around :mod:`google.agents.cli._adk_client`: creates a
    session if none is supplied, streams events from ``/run_sse``, and
    renders each event to the terminal as it arrives.
    """
    if not session_id:
        try:
            session_id = create_session(
                service_url, app_name, "cli-user", headers=headers
            )
        except requests.HTTPError as exc:
            response = exc.response
            status = response.status_code if response is not None else "unknown"
            body = response.text if response is not None else str(exc)
            hint = ""
            if response is not None and response.status_code in (404, 405):
                hint = "\n  If this is an A2A agent, try --mode a2a instead."
            raise click.ClickException(
                f"Failed to create session (HTTP {status}):\n  {body}{hint}"
            ) from exc

    # Print user message (text part only for display)
    user_text = " ".join(p.get("text", "") for p in parts if "text" in p).strip()
    if user_text:
        click.echo(f"[user]: {user_text}")

    last_author = None
    artifacts: list[str] = []
    user_message = {"role": "user", "parts": parts}
    try:
        for event in run_sse(
            service_url,
            app_name,
            session_id,
            user_message=user_message,
            headers=headers,
            user_id="cli-user",
        ):
            last_author = _print_sse_event(event, last_author, verbose, artifacts)
    except requests.HTTPError as exc:
        response = exc.response
        status = response.status_code if response is not None else "unknown"
        body = response.text if response is not None else str(exc)
        raise click.ClickException(
            f"Failed to run agent (HTTP {status}):\n  {body}"
        ) from exc

    click.echo()
    _print_artifacts(artifacts)
    _print_session_id(session_id)


def _create_agent_runtime_session(
    service_url: str, headers: dict, user_id: str = "cli-user"
) -> str:
    resp = requests.post(
        f"{service_url}:query",
        headers=headers,
        json={
            "class_method": "async_create_session",
            "input": {"user_id": user_id},
        },
        timeout=30,
    )
    if not resp.ok:
        raise click.ClickException(
            f"Failed to create Agent Runtime session (HTTP {resp.status_code}):\n"
            f"  {resp.text}"
        )
    session_id = resp.json().get("output", {}).get("id")
    if not session_id:
        raise click.ClickException("Agent Runtime returned a session with no ID.")
    return session_id


def _query_agent_runtime_sse(
    service_url: str,
    message: str | dict,
    headers: dict,
    session_id: str | None = None,
    verbose: bool = False,
) -> None:
    """Stream a query to an Agent Runtime via the ``:streamQuery`` HTTP endpoint.

    Uses the ``async_stream_query`` class method.  The response is
    newline-delimited JSON (one JSON object per line).  Creates a session
    via ``async_create_session`` when ``session_id`` is not provided.
    """
    if not session_id:
        session_id = _create_agent_runtime_session(service_url, headers)

    stream_url = f"{service_url}:streamQuery"

    input_payload: dict = {
        "user_id": "cli-user",
        "session_id": session_id,
        "message": message,
    }

    payload = {
        "class_method": "async_stream_query",
        "input": input_payload,
    }

    # Print user message
    user_text = (
        message
        if isinstance(message, str)
        else " ".join(
            p.get("text", "") for p in message.get("parts", []) if "text" in p
        ).strip()
    )
    if user_text:
        click.echo(f"[user]: {user_text}")

    with requests.post(
        stream_url, headers=headers, json=payload, stream=True, timeout=120
    ) as resp:
        if not resp.ok:
            hint = ""
            if resp.status_code in (400, 404, 405, 500):
                hint = (
                    "\n  This agent may not support the ADK :streamQuery API."
                    "\n  If this is an A2A agent, try --mode a2a instead."
                )
            raise click.ClickException(
                f"Failed to query Agent Runtime (HTTP {resp.status_code}):\n"
                f"  {resp.text}{hint}"
            )
        last_author = None
        artifacts: list[str] = []
        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            last_author = _print_sse_event(event, last_author, verbose, artifacts)

    click.echo()
    _print_artifacts(artifacts)
    _print_session_id(session_id)


def _query_a2a(
    *,
    card_url: str,
    parts: list[Part],
    headers: dict,
    url: str | None = None,
    session_id: str | None = None,
    verbose: bool = False,
) -> None:
    """Fetch an A2A agent card and query the agent."""
    resp = httpx.get(card_url, headers=headers, timeout=30)
    if resp.status_code != 200:
        hint = ""
        if resp.status_code in (404, 405):
            hint = "\n  If this is an ADK agent, try --mode adk instead."
        raise click.ClickException(
            f"Failed to fetch agent card (HTTP {resp.status_code}):\n  {resp.text}{hint}"
        )
    base_url = url or card_url.removesuffix(AGENT_CARD_WELL_KNOWN_PATH)
    asyncio.run(
        _query_a2a_async(base_url, parts, headers, session_id=session_id, verbose=verbose)
    )


async def _query_a2a_async(
    base_url: str,
    parts: list[Part],
    headers: dict,
    session_id: str | None = None,
    verbose: bool = False,
) -> None:
    """Async implementation — sends a message and prints the response (a2a-sdk 1.0).

    The client is resolved from the reachable ``base_url`` (which may differ from
    the card's advertised URL, e.g. kubectl port-forward / Agent Runtime).
    """
    agent_name = "agent"

    # Print user message (text parts only for display).
    user_text = " ".join(p.text for p in parts if p.text)
    if user_text:
        click.echo(f"[user]: {user_text}")

    req_headers = dict(headers)
    # Note that on the 0.3 legacy compatibility path, this header gets overwritten,
    # so including it should be safely backwards compatible.
    req_headers.setdefault(VERSION_HEADER, PROTOCOL_VERSION_1_0)

    async with httpx.AsyncClient(headers=req_headers, timeout=120) as http_client:
        config = ClientConfig(
            httpx_client=http_client,
            # INVARIANT: keep JSONRPC here. The a2a-sdk v0.3 compat transport is
            # JSON-RPC only; dropping it silently breaks consuming v0.3 agents.
            supported_protocol_bindings=[
                TransportProtocol.JSONRPC,
                TransportProtocol.HTTP_JSON,
            ],
        )
        try:
            a2a_client = await create_client(base_url, config)
        except Exception as exc:
            raise click.ClickException(
                f"Could not resolve an A2A agent at {base_url}: {exc}\n"
                "  If this is an older agent, upgrade it with "
                "'agents-cli scaffold upgrade', or try --mode adk."
            ) from exc

        msg = Message(
            message_id=str(uuid.uuid4()),
            role=Role.ROLE_USER,
            parts=parts,
            context_id=session_id or "",
        )

        last_author = None
        response_session_id = None
        artifacts: list[str] = []
        async for chunk in a2a_client.send_message(SendMessageRequest(message=msg)):
            if chunk.HasField("artifact_update"):
                if not response_session_id and chunk.artifact_update.context_id:
                    response_session_id = chunk.artifact_update.context_id
                for part in chunk.artifact_update.artifact.parts:
                    last_author = _print_author_tag(agent_name, last_author)
                    _print_a2a_part(part, artifacts)
            elif chunk.HasField("task"):
                if not response_session_id and chunk.task.context_id:
                    response_session_id = chunk.task.context_id
                for artifact in chunk.task.artifacts:
                    for part in artifact.parts:
                        last_author = _print_author_tag(agent_name, last_author)
                        _print_a2a_part(part, artifacts)
            elif chunk.HasField("message"):
                for part in chunk.message.parts:
                    last_author = _print_author_tag(agent_name, last_author)
                    _print_a2a_part(part, artifacts)

            if verbose:
                click.echo()
                click.secho(str(chunk), dim=True)

    click.echo()
    _print_artifacts(artifacts)
    _print_session_id(response_session_id)


def _print_a2a_part(part: Part, artifacts: list[str]) -> None:
    """Print an A2A 1.0 response part. Appends saved artifact paths to ``artifacts``."""
    if part.text:
        click.echo(part.text, nl=False)
    elif part.url:
        click.echo(f"\n[file: {part.url}]", nl=False)
    elif part.raw:
        artifacts.append(_write_artifact(part.raw, part.media_type))
    elif part.HasField("data"):
        click.echo(f"\n{json.dumps(MessageToDict(part.data), indent=2)}", nl=False)


def _print_sse_part(part: dict, artifacts: list[str]) -> None:
    """Print an ADK SSE response part to the terminal."""
    text = part.get("text")
    if text:
        click.echo(text, nl=False)
        return
    inline_data = part.get("inlineData")
    if inline_data:
        path = _save_inline_artifact(inline_data["data"], inline_data.get("mimeType"))
        if path is not None:
            artifacts.append(path)
        return
    file_data = part.get("fileData")
    if file_data:
        uri = file_data.get("fileUri")
        if uri:
            click.echo(f"\n[file: {uri}]", nl=False)
        return
    function_call = part.get("functionCall")
    if function_call:
        name = function_call.get("name", "")
        args = function_call.get("args", {})
        click.echo(f"\n[tool_call: {name}({json.dumps(args)})]", nl=False)
        return
    function_response = part.get("functionResponse")
    if function_response:
        name = function_response.get("name", "")
        response = function_response.get("response", {})
        click.echo(f"\n[tool_response: {name} -> {json.dumps(response)}]", nl=False)
        return
