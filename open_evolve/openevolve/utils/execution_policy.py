"""Explicit operator consent for local execution; not an OS sandbox."""
import os


def require_execution_consent():
    if os.environ.get("OPENEVOLVE_ALLOW_UNSANDBOXED_EXECUTION") != "1":
        raise PermissionError(
            "Execution is disabled. Unreviewed/generated code requires an external OS sandbox. "
            "Only for reviewed trusted local code, or inside that sandbox, set "
            "OPENEVOLVE_ALLOW_UNSANDBOXED_EXECUTION=1. This grants host-user privileges; "
            "timeouts and subprocesses are not isolation."
        )
