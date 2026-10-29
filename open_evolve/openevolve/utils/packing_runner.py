"""Numeric IPC for trusted local circle-packing candidates.

This removes parent-process pickle loading. It is not a hostile-code sandbox.
"""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile

import numpy as np


_CHILD = '''import importlib.util, json, pathlib, sys
import numpy as np
program, output = sys.argv[1:]
try:
    sys.path.insert(0, str(pathlib.Path(program).parent))
    spec = importlib.util.spec_from_file_location("candidate", program)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    centers, radii, total = module.run_packing()
    payload = {"centers": np.asarray(centers).tolist(),
               "radii": np.asarray(radii).tolist(), "sum_radii": float(total)}
    text = json.dumps(payload, allow_nan=False)
except Exception as error:
    text = json.dumps({"error": str(error)[:1000]})
pathlib.Path(output).write_text(text, encoding="utf-8")
'''


def run_numeric_candidate(program_path, timeout_seconds=20):
    """Run a trusted candidate without provider credentials or executable result data."""
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    program = str(Path(program_path).resolve(strict=True))
    # Keep platform runtime paths; never pass API keys, PYTHONPATH or loader overrides.
    env = {key: os.environ[key] for key in ("PATH", "SYSTEMROOT", "WINDIR", "TMPDIR", "TEMP", "TMP")
           if key in os.environ}
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    with tempfile.TemporaryDirectory(prefix="packing-") as temporary:
        result_path = Path(temporary) / "result.json"
        process = subprocess.Popen(
            [sys.executable, "-c", _CHILD, program, str(result_path)],
            cwd=temporary, env=env, stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=os.name == "posix",
        )
        try:
            try:
                process.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired as error:
                raise TimeoutError(f"Candidate exceeded {timeout_seconds} seconds") from error
            if process.returncode != 0:
                raise RuntimeError(f"Candidate exited with code {process.returncode}")
            with result_path.open("rb") as source:
                payload = source.read(65537)
            if len(payload) > 65536:
                raise ValueError("Candidate result exceeds 64 KiB")
            result = json.loads(payload)
            if not isinstance(result, dict):
                raise ValueError("Candidate result must be an object")
            if "error" in result:
                raise RuntimeError(f"Candidate failed: {result['error']}")
            centers = np.asarray(result["centers"], dtype=float)
            radii = np.asarray(result["radii"], dtype=float)
            total = float(result["sum_radii"])
            if centers.shape != (26, 2) or radii.shape != (26,):
                raise ValueError("Candidate must return 26 circles")
            if not np.isfinite(centers).all() or not np.isfinite(radii).all() or not np.isfinite(total):
                raise ValueError("Candidate result must be finite")
            return centers, radii, total
        finally:
            if os.name == "posix":
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            elif process.poll() is None:
                process.kill()
            process.wait()
