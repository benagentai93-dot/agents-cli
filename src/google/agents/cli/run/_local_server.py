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

"""Background local server management for the ``run`` command."""

from __future__ import annotations

import json
import logging
import os
import socket
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import NamedTuple

import click
import psutil

from google.agents.cli._runner import popen_resolved_detached, redact_cmd

_PID_DIR = ".google-agents-cli"
_PID_FILENAME = "run_server.json"
_LOG_FILENAME = "run_server.log"
_BASE_PORT = 18080
_MAX_PORT_ATTEMPTS = 10
_DEFAULT_IDLE_TIMEOUT = 1800  # 30 minutes
_STARTUP_TIMEOUT_POSIX = 30
_STARTUP_TIMEOUT_WINDOWS = 90
_DEFAULT_STARTUP_TIMEOUT = (
    _STARTUP_TIMEOUT_WINDOWS if os.name == "nt" else _STARTUP_TIMEOUT_POSIX
)


class ServerInfo(NamedTuple):
    """A running local server and any process owned by this call.

    ``owned_pid`` is set only when ``ensure_server`` launched a new process;
    it is ``None`` when an already-running server was reused. Callers use it
    to avoid tearing down a server someone else is keeping alive.
    """

    port: int
    owned_pid: int | None = None


def ensure_server(
    project_root: Path,
    agent_dir: str,
    *,
    idle_timeout: int = _DEFAULT_IDLE_TIMEOUT,
    trace_to_cloud: bool = False,
    use_in_memory_session: bool = True,
) -> ServerInfo:
    """Return a running local server's port, starting one if needed.

    If an existing server has been idle longer than *idle_timeout* seconds
    (based on the ``last_activity`` timestamp in the PID file), it is
    stopped and a fresh one is started.

    Args:
        project_root: The project root directory (cwd when running).
        agent_dir: The agent directory name (e.g. ``"investment_agent"``).
        idle_timeout: Seconds of inactivity before the server is considered
            stale and replaced.  Defaults to 30 minutes.
        trace_to_cloud: When ``True``, export traces to Cloud Trace.
            Only takes effect when a new server is started.
        use_in_memory_session: Sets ``USE_IN_MEMORY_SESSION`` in the
            server's env to ``true`` (default) or ``false``. Only takes
            effect when a new server is started; a reused server keeps
            whatever env it was started with.

    Returns:
        A :class:`ServerInfo` with the port and any PID owned by this call.
    """
    info = _read_pid_file(project_root)

    if info:
        if _is_server_alive(info["pid"], info["port"]):
            existing_session_mode = info.get("use_in_memory_session", True)
            if existing_session_mode != use_in_memory_session:
                raise click.ClickException(
                    "The existing local server uses a different session mode.\n"
                    "  Run 'agents-cli run --stop-server' first, then retry."
                )
            # Check idle timeout — stop the server if it's been idle too long.
            if _is_idle(info, idle_timeout):
                _cleanup(project_root, info)
            else:
                if trace_to_cloud and not info.get("trace_to_cloud"):
                    click.secho(
                        "Warning: reusing existing server that was started "
                        "without --trace-to-cloud.\n"
                        "  Run 'agents-cli run --stop-server' first to "
                        "restart with tracing enabled.",
                        fg="yellow",
                        err=True,
                    )
                _update_activity(project_root)
                return ServerInfo(info["port"], owned_pid=None)
        else:
            # Stale PID file — clean up before starting fresh.
            _cleanup(project_root, info)

    port = _find_free_port()
    pid = _start_server(
        project_root=project_root,
        agent_dir=agent_dir,
        port=port,
        trace_to_cloud=trace_to_cloud,
        use_in_memory_session=use_in_memory_session,
    )
    _wait_for_port(project_root, port, pid=pid)
    _write_pid_file(
        project_root,
        pid=pid,
        port=port,
        trace_to_cloud=trace_to_cloud,
        use_in_memory_session=use_in_memory_session,
    )
    click.secho(f"Local server started on port {port} (PID {pid})", dim=True)
    click.secho("  Stop with: agents-cli run --stop-server", dim=True)
    return ServerInfo(port, owned_pid=pid)


def stop_server(project_root: Path, *, expected_pid: int | None = None) -> bool:
    """Stop the background server.

    When *expected_pid* is provided, stop the server only if the current PID
    metadata still identifies that process. Explicit user stops omit it.

    Returns:
        ``True`` if a server was found and stopped.
    """
    info = _read_pid_file(project_root)
    if not info or (expected_pid is not None and info.get("pid") != expected_pid):
        return False
    _cleanup(project_root, info)
    click.secho("Local server stopped.", dim=True)
    return True


def get_server_port(project_root: Path) -> int | None:
    """Return the port of the running local server, or ``None`` if absent.

    Returns ``None`` when no server has been started for *project_root*,
    or when the recorded process is no longer alive.
    """
    info = _read_pid_file(project_root)
    if not info:
        return None
    if not _is_server_alive(info["pid"], info["port"]):
        return None
    return info["port"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _find_free_port(
    base: int = _BASE_PORT, max_attempts: int = _MAX_PORT_ATTEMPTS
) -> int:
    """Find a free local port starting from *base*."""
    for offset in range(max_attempts):
        port = base + offset
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", port))
                return port
        except OSError:
            continue
    raise click.ClickException(
        f"No free port found in range {base}–{base + max_attempts - 1}.\n"
        "  Stop other servers or use --url to query a remote agent."
    )


def _has_fast_api_app(project_root: Path, agent_dir: str) -> bool:
    return (project_root / agent_dir / "fast_api_app.py").is_file()


def _build_serve_command(
    *,
    project_root: Path,
    agent_dir: str,
    port: int,
    trace_to_cloud: bool = False,
) -> list[str]:
    """Return the command list for booting the local server.

    Prefers running this projects `fast_api_app.py` under uvicorn when the file exists, otherwise falls back to `adk api_server`.
    """
    if _has_fast_api_app(project_root, agent_dir):
        return [
            "uv",
            "run",
            "uvicorn",
            f"{agent_dir}.fast_api_app:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ]

    cmd = [
        "uv",
        "run",
        "adk",
        "api_server",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--reload_agents",
        "--no-reload",
    ]
    if trace_to_cloud:
        cmd.append("--otel_to_cloud")
    cmd.append(".")
    return cmd


def _start_server(
    *,
    project_root: Path,
    agent_dir: str,
    port: int,
    trace_to_cloud: bool = False,
    use_in_memory_session: bool = True,
) -> int:
    """Start the local server as a detached background process.

    Returns the PID.

    `use_in_memory_session` sets `USE_IN_MEMORY_SESSION` to `true` or `false`
    in the child's env. `true` lets the server start without cloud dependencies
    (e.g. Agent Runtime session type); `false` makes it use the agent's real
    session service.
    """
    adk_dir = project_root / _PID_DIR
    adk_dir.mkdir(exist_ok=True)
    log_path = adk_dir / _LOG_FILENAME

    cmd = _build_serve_command(
        project_root=project_root,
        agent_dir=agent_dir,
        port=port,
        trace_to_cloud=trace_to_cloud,
    )

    env = os.environ.copy()
    env.pop("AGENTS_CLI_OTEL_TO_CLOUD", None)
    if trace_to_cloud:
        env["AGENTS_CLI_OTEL_TO_CLOUD"] = "1"
    env["USE_IN_MEMORY_SESSION"] = "true" if use_in_memory_session else "false"
    env.setdefault("PYTHONUNBUFFERED", "1")

    log_file = open(log_path, "a", encoding="utf-8")
    stderr_file = None
    try:
        log_file.write(f"=== Starting server at {datetime.now(UTC).isoformat()} ===\n")
        log_file.write(f"Command: {redact_cmd(cmd)}\n")
        log_file.write(f"CWD: {project_root}\n")
        log_file.flush()

        if sys.platform == "win32":
            stderr_path = adk_dir / "run_server.stderr.log"
            stderr_file = open(stderr_path, "a", encoding="utf-8")
            stderr_file.write(
                f"=== Starting server stderr at {datetime.now(UTC).isoformat()} ===\n"
            )
            stderr_file.write(f"Command: {redact_cmd(cmd)}\n")
            stderr_file.flush()
            child_stdout = log_file
            child_stderr = stderr_file
        else:
            child_stdout = log_file
            child_stderr = log_file

        proc = popen_resolved_detached(
            cmd,
            cwd=str(project_root),
            stdout=child_stdout,
            stderr=child_stderr,
            env=env,
        )
        log_file.write(f"=== Started server process {proc.pid} ===\n")
        log_file.flush()
    finally:
        # Close the parent's copy of the fd — the child inherits its own.
        log_file.close()
        if stderr_file:
            stderr_file.close()
    return proc.pid


def _wait_for_port(
    project_root: Path,
    port: int,
    timeout: int = _DEFAULT_STARTUP_TIMEOUT,
    pid: int | None = None,
) -> None:
    """Wait until a local server is ready to handle HTTP requests.

    Polls with an HTTP GET so the server's lifespan (which registers
    routes like A2A endpoints) has time to complete.
    """
    import urllib.error
    import urllib.request

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        # Fail fast if the server process has already crashed.
        # os.kill(pid, 0) is unreliable on Windows, so use psutil.pid_exists.
        if pid is not None and not psutil.pid_exists(pid):
            raise click.ClickException(
                "Local server process exited during startup.\n"
                f"  Check logs: {_PID_DIR}/{_LOG_FILENAME}"
            )
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=1)
            return
        except urllib.error.HTTPError:
            # Any HTTP response (even 404/405) means the server is ready.
            return
        except (urllib.error.URLError, OSError):
            time.sleep(0.3)

    if sys.platform == "win32":
        log_path = project_root / _PID_DIR / "run_server.stderr.log"
    else:
        log_path = project_root / _PID_DIR / _LOG_FILENAME
    log_content = ""
    if log_path.exists():
        try:
            lines = []
            truncated = False
            with open(log_path, encoding="utf-8", errors="replace") as f:
                for line in f:
                    lines.append(line)
                    if len(lines) > 50:
                        lines.pop(0)
                        truncated = True
            log_content = "".join(lines)
            if truncated:
                log_content = "... (truncated) ...\n" + log_content
        except Exception as e:
            log_content = f"<Failed to read log file: {e}>"
    else:
        log_content = "<Log file does not exist>"

    raise click.ClickException(
        f"Local server did not start within {timeout}s.\n"
        f"  Check logs: {_PID_DIR}/{_LOG_FILENAME}\n"
        f"  Log content:\n{log_content}"
    )


# --- PID file helpers ---


def _pid_file_path(project_root: Path) -> Path:
    return project_root / _PID_DIR / _PID_FILENAME


def _read_pid_file(project_root: Path) -> dict | None:
    path = _pid_file_path(project_root)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _write_pid_file(
    project_root: Path,
    *,
    pid: int,
    port: int,
    trace_to_cloud: bool = False,
    use_in_memory_session: bool = True,
) -> None:
    now = datetime.now(UTC).isoformat()
    data = {
        "pid": pid,
        "port": port,
        "started_at": now,
        "last_activity": now,
        "trace_to_cloud": trace_to_cloud,
        "use_in_memory_session": use_in_memory_session,
    }
    path = _pid_file_path(project_root)
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def _update_activity(project_root: Path) -> None:
    """Stamp ``last_activity`` so idle detection resets."""
    path = _pid_file_path(project_root)
    try:
        data = json.loads(path.read_text())
        data["last_activity"] = datetime.now(UTC).isoformat()
        path.write_text(json.dumps(data, indent=2) + "\n")
    except (json.JSONDecodeError, OSError):
        pass


def _is_idle(info: dict, idle_timeout: int) -> bool:
    """Return ``True`` if the server has been idle longer than *idle_timeout*."""
    try:
        last = datetime.fromisoformat(info["last_activity"])
        idle = (datetime.now(UTC) - last).total_seconds()
        return idle > idle_timeout
    except (KeyError, ValueError):
        # Treat missing or unparseable timestamps as stale.
        return True


def _is_server_alive(pid: int, port: int) -> bool:
    """Return ``True`` if the process exists AND the port is open."""
    # os.kill(pid, 0) is unreliable on Windows, so use psutil.pid_exists.
    if not psutil.pid_exists(pid):
        return False
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            return True
    except OSError:
        return False


def _cleanup(project_root: Path, info: dict) -> None:
    """Terminate the server process and remove the PID file."""
    pid = info.get("pid")
    if pid:
        try:
            parent = psutil.Process(pid)
            children = parent.children(recursive=True)
            for child in children:
                try:
                    child.terminate()
                except psutil.NoSuchProcess:
                    pass
            parent.terminate()
            psutil.wait_procs([*children, parent], timeout=3)
        except psutil.NoSuchProcess:
            logging.warning(
                "Local server process with PID %d not found, skipping termination.", pid
            )
    path = _pid_file_path(project_root)
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        logging.warning("Failed to remove PID file %s: %s", path, exc)
