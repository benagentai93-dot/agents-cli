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

"""Read/write helpers for pending deploy operations in METADATA_FILE.

When a deployment starts (sync or ``--no-wait``), the long-running operation
name and metadata are persisted as a ``pending_operation`` field inside
``METADATA_FILE`` so that ``deploy --status`` can poll it later.
"""

from __future__ import annotations

import base64
import contextlib
import datetime
import hashlib
import json
import logging
import os
import tempfile
import uuid
from collections.abc import Iterator
from typing import Any

METADATA_FILE = "deployment_metadata.json"
_PREVIOUS_METADATA = "_previous_metadata"


class OperationPendingError(RuntimeError):
    """Raised when another deploy already owns the pending-operation claim."""


@contextlib.contextmanager
def _metadata_lock() -> Iterator[None]:
    """Serialize metadata read-modify-write sequences across CLI processes."""
    metadata_path = os.path.realpath(os.path.abspath(METADATA_FILE))
    digest = hashlib.sha256(metadata_path.encode()).hexdigest()
    lock_path = os.path.join(tempfile.gettempdir(), f"agents-cli-{digest}.lock")
    with open(lock_path, "a+b") as lock_file:
        if os.name == "nt":
            import msvcrt

            lock_file.seek(0)
            if not lock_file.read(1):
                lock_file.write(b"\0")
                lock_file.flush()
            lock_file.seek(0)
            msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _read_metadata() -> dict[str, Any]:
    """Read METADATA_FILE, tolerating a missing or corrupt file.

    A malformed or zero-byte file (left by an interrupted run, a partial
    write, or a manual edit) is treated as empty so a single bad file can't
    permanently block every subsequent deploy. Returns ``{}`` when the file
    is missing or unreadable.
    """
    if not os.path.exists(METADATA_FILE):
        return {}
    try:
        with open(METADATA_FILE, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logging.warning("Ignoring corrupt %s (%s); treating as empty.", METADATA_FILE, e)
        return {}
    if not isinstance(data, dict):
        logging.warning(
            "Ignoring %s with unexpected top-level %s; treating as empty.",
            METADATA_FILE,
            type(data).__name__,
        )
        return {}
    return data


def read_deployment_metadata() -> dict[str, Any] | None:
    """Read deployment metadata strictly when selecting a mutation target."""
    with _metadata_lock():
        if not os.path.exists(METADATA_FILE):
            return None
        try:
            with open(METADATA_FILE, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            raise ValueError(f"Could not read {METADATA_FILE}: {e}") from e
        if not isinstance(data, dict):
            raise ValueError(f"{METADATA_FILE} must contain a JSON object")
        return data


def _replace_metadata_bytes(content: bytes) -> None:
    directory = os.path.dirname(os.path.abspath(METADATA_FILE))
    fd, temporary = tempfile.mkstemp(prefix=f".{METADATA_FILE}.", dir=directory)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, METADATA_FILE)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _write_metadata(data: dict[str, Any]) -> None:
    content = (json.dumps(data, indent=2) + "\n").encode()
    _replace_metadata_bytes(content)


def write_metadata(data: dict[str, Any]) -> None:
    """Atomically replace METADATA_FILE with *data*."""
    with _metadata_lock():
        _write_metadata(data)


def update_metadata(updates: dict[str, Any]) -> None:
    """Atomically merge *updates* into the latest deployment metadata."""
    with _metadata_lock():
        data = _read_metadata()
        data.update(updates)
        _write_metadata(data)


def _pending_operation(
    project: str,
    location: str,
    deployment_target: str,
    previous: bytes,
    existed: bool,
) -> dict[str, Any]:
    return {
        "claim_id": uuid.uuid4().hex,
        "state": "starting",
        "project": project,
        "location": location,
        "deployment_target": deployment_target,
        "started_at": datetime.datetime.now(tz=datetime.UTC).isoformat(),
        _PREVIOUS_METADATA: {
            "existed": existed,
            "content": base64.b64encode(previous).decode("ascii"),
        },
    }


def claim_operation(project: str, location: str, deployment_target: str) -> str:
    """Atomically claim the right to start one remote deployment mutation."""
    with _metadata_lock():
        data = _read_metadata()
        if isinstance(data.get("pending_operation"), dict):
            raise OperationPendingError("a deployment operation is already pending")
        existed = os.path.exists(METADATA_FILE)
        previous = b""
        if existed:
            with open(METADATA_FILE, "rb") as f:
                previous = f.read()
        pending = _pending_operation(
            project, location, deployment_target, previous, existed
        )
        data["pending_operation"] = pending
        _write_metadata(data)
        return pending["claim_id"]


def write_operation(
    operation_name: str,
    project: str,
    location: str,
    deployment_target: str,
    claim_id: str | None = None,
) -> None:
    """Persist a pending deploy operation to METADATA_FILE."""
    with _metadata_lock():
        data = _read_metadata()
        pending = data.get("pending_operation")
        if claim_id is not None:
            if not isinstance(pending, dict) or pending.get("claim_id") != claim_id:
                raise OperationPendingError("deployment operation claim was replaced")
        elif isinstance(pending, dict):
            raise OperationPendingError("a deployment operation is already pending")
        else:
            existed = os.path.exists(METADATA_FILE)
            previous = b""
            if existed:
                with open(METADATA_FILE, "rb") as f:
                    previous = f.read()
            pending = _pending_operation(
                project, location, deployment_target, previous, existed
            )
        pending.update(
            {
                "operation_name": operation_name,
                "state": "pending",
                "project": project,
                "location": location,
                "deployment_target": deployment_target,
            }
        )
        data["pending_operation"] = pending
        _write_metadata(data)


def read_operation() -> dict[str, Any] | None:
    """Read a pending operation from METADATA_FILE, or None."""
    with _metadata_lock():
        pending = _read_metadata().get("pending_operation")
        return pending if isinstance(pending, dict) else None


def _owns_operation(
    pending: dict[str, Any],
    claim_id: str | None,
    operation_name: str | None,
) -> bool:
    if claim_id is not None:
        return pending.get("claim_id") == claim_id
    return (
        pending.get("claim_id") is None
        and operation_name is not None
        and pending.get("operation_name") == operation_name
    )


def finish_operation(
    claim_id: str | None,
    updates: dict[str, Any],
    *,
    operation_name: str | None = None,
) -> None:
    """Merge successful deployment metadata and release its claim atomically."""
    with _metadata_lock():
        data = _read_metadata()
        pending = data.get("pending_operation")
        if not isinstance(pending, dict) or not _owns_operation(
            pending, claim_id, operation_name
        ):
            raise OperationPendingError("deployment operation claim was replaced")
        data.update(updates)
        del data["pending_operation"]
        _write_metadata(data)


def clear_operation(
    claim_id: str | None = None, *, operation_name: str | None = None
) -> None:
    """Remove the pending_operation field from METADATA_FILE."""
    with _metadata_lock():
        data = _read_metadata()
        pending = data.get("pending_operation")
        if not isinstance(pending, dict):
            return
        if not _owns_operation(pending, claim_id, operation_name):
            raise OperationPendingError("deployment operation claim was replaced")
        previous = pending.get(_PREVIOUS_METADATA)
        del data["pending_operation"]
        if isinstance(previous, dict) and isinstance(previous.get("existed"), bool):
            try:
                content = base64.b64decode(previous.get("content", ""), validate=True)
                previous_data = json.loads(content) if content else {}
            except (json.JSONDecodeError, ValueError, TypeError):
                previous_data = None
            if previous["existed"] and data == previous_data:
                _replace_metadata_bytes(content)
                return
            if not previous["existed"] and not data:
                os.unlink(METADATA_FILE)
                return
        if data:
            _write_metadata(data)
        else:
            os.unlink(METADATA_FILE)
