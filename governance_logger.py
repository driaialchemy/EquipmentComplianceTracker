"""Python governance logger with synchronous blocking support."""

import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Any, Optional

try:
    import requests
except ImportError:
    requests = None  # type: ignore

DEFAULT_GOVERNOR_BACKEND = r"C:\Users\msell\OneDrive\AIAlchemy\aiagentgovernance\backend"
_governor_start_attempted = False


def _governor_url() -> str:
    return os.environ.get("GOVERNANCE_URL", "http://localhost:3000")


def _timeout_seconds() -> float:
    return int(os.environ.get("GOVERNANCE_TIMEOUT", "5000")) / 1000


def _governor_backend_dir() -> str:
    return os.environ.get("GOVERNANCE_BACKEND_DIR", DEFAULT_GOVERNOR_BACKEND)


def _governor_is_healthy() -> bool:
    if requests is None:
        return False
    try:
        response = requests.get(f"{_governor_url()}/health", timeout=2)
        return response.status_code == 200
    except Exception:
        return False


def ensure_governor_running(wait_seconds: int = 45) -> bool:
    """Start the governor if it is not already running."""
    global _governor_start_attempted

    if _governor_is_healthy():
        return True

    if _governor_start_attempted:
        for _ in range(wait_seconds):
            if _governor_is_healthy():
                return True
            time.sleep(1)
        return False

    _governor_start_attempted = True
    backend_dir = _governor_backend_dir()
    if not os.path.isdir(backend_dir):
        print(f"[GovernanceLogger] Governor backend not found: {backend_dir}")
        return False

    print("[GovernanceLogger] Governor not running - starting it now ...")
    command = (
        f"Set-Location '{backend_dir}'; "
        "Write-Host 'GOVERNOR - auto-started by agent' -ForegroundColor Green; "
        "npx tsx src/server.ts"
    )

    popen_kwargs: dict[str, Any] = {}
    if sys.platform == "win32":
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_CONSOLE

    subprocess.Popen(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", command],
        **popen_kwargs,
    )

    for _ in range(wait_seconds):
        if _governor_is_healthy():
            print("[GovernanceLogger] Governor is ready.")
            return True
        time.sleep(1)

    print("[GovernanceLogger] Governor did not become healthy in time.")
    return False


def _parse_response(response: "requests.Response") -> dict[str, Any]:
    try:
        body = response.json()
    except ValueError:
        body = {}

    return {
        "success": response.status_code == 200 and body.get("success", True),
        "allowed": body.get("allowed", response.status_code == 200),
        "violation": body.get("violation"),
        "violatedPolicy": body.get("violatedPolicy"),
        "escalationId": body.get("escalationId"),
        "data": body.get("data"),
    }


def log_success(agent_id: str, description: str, output: Optional[Any] = None) -> dict[str, Any]:
    """Log activity and return governor blocking decision."""
    if requests is None:
        return {
            "success": False,
            "allowed": False,
            "violation": "requests package not installed",
        }

    if not ensure_governor_running():
        return {
            "success": False,
            "allowed": False,
            "violation": "Governor is not running and could not be started",
        }

    activity = {
        "agentId": agent_id,
        "actionType": "agent_execution_complete",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "description": description,
        "result": {"success": True, "output": output},
    }

    try:
        response = requests.post(
            f"{_governor_url()}/agents/{agent_id}/activity",
            json=activity,
            timeout=_timeout_seconds(),
        )
        return _parse_response(response)
    except Exception as exc:
        print(f"[GovernanceLogger] Failed to log activity for {agent_id}: {exc}")
        return {
            "success": False,
            "allowed": False,
            "violation": f"Failed to get governance approval: {exc}",
        }


def log_error(agent_id: str, error: str) -> dict[str, Any]:
    """Log an agent error to the governance governor."""
    if requests is None:
        return {"success": False, "allowed": False, "violation": "requests package not installed"}

    if not ensure_governor_running():
        return {
            "success": False,
            "allowed": False,
            "violation": "Governor is not running and could not be started",
        }

    activity = {
        "agentId": agent_id,
        "actionType": "error_occurred",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "description": f"Error: {error}",
        "result": {"success": False, "error": error},
    }

    try:
        response = requests.post(
            f"{_governor_url()}/agents/{agent_id}/activity",
            json=activity,
            timeout=_timeout_seconds(),
        )
        return _parse_response(response)
    except Exception as exc:
        print(f"[GovernanceLogger] Failed to log error for {agent_id}: {exc}")
        return {
            "success": False,
            "allowed": False,
            "violation": f"Failed to get governance approval: {exc}",
        }


def require_approval(agent_id: str, description: str, output: Optional[Any] = None) -> dict[str, Any]:
    """Log activity and exit if governor blocks the pipeline."""
    approval = log_success(agent_id, description, output)
    if not approval.get("allowed", False):
        print(f"BLOCKED: {approval.get('violation', 'Governance violation')}")
        if approval.get("escalationId"):
            print(f"   Escalation: {approval['escalationId']}")
        sys.exit(1)
    return approval
