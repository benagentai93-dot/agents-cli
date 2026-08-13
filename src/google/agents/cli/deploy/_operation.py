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
import datetime
import json
import logging
import os
import tempfile
from typing import Any

METADATA_FILE = "deployment_metadata.json"
_PREVIOUS_METADATA = "_previous_metadata"


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


def write_metadata(data: dict[str, Any]) -> None:
    """Atomically replace METADATA_FILE with *data*."""
    content = (json.dumps(data, indent=2) + "\n").encode()
    _replace_metadata_bytes(content)


def write_operation(
    operation_name: str,
    project: str,
    location: str,
    deployment_target: str,
) -> None:
    """Persist a pending deploy operation to METADATA_FILE."""
    pending = {
        "operation_name": operation_name,
        "project": project,
        "location": location,
        "deployment_target": deployment_target,
        "started_at": datetime.datetime.now(tz=datetime.UTC).isoformat(),
    }

    # Merge into existing metadata if present
    existed = os.path.exists(METADATA_FILE)
    previous = b""
    if existed:
        with open(METADATA_FILE, "rb") as f:
            previous = f.read()
    pending[_PREVIOUS_METADATA] = {
        "existed": existed,
        "content": base64.b64encode(previous).decode("ascii"),
    }
    data = _read_metadata()
    data["pending_operation"] = pending
    write_metadata(data)


def read_operation() -> dict[str, Any] | None:
    """Read a pending operation from METADATA_FILE, or None."""
    return _read_metadata().get("pending_operation")


def clear_operation() -> None:
    """Remove the pending_operation field from METADATA_FILE."""
    data = _read_metadata()
    pending = data.get("pending_operation")
    if not isinstance(pending, dict):
        return
    previous = pending.get(_PREVIOUS_METADATA)
    if isinstance(previous, dict) and isinstance(previous.get("existed"), bool):
        if previous["existed"]:
            try:
                content = base64.b64decode(previous.get("content", ""), validate=True)
            except (ValueError, TypeError) as e:
                raise ValueError("Invalid previous deployment metadata") from e
            _replace_metadata_bytes(content)
        else:
            os.unlink(METADATA_FILE)
        return
    del data["pending_operation"]
    write_metadata(data)
