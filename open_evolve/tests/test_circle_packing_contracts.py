"""Feasibility and trusted-local runner regression contracts, without LLM calls."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def evaluator(name):
    path = ROOT / ("best_result" if name == "saved_circle_packing" else "examples") / ("circle_packing" if name == "saved_circle_packing" else name) / "evaluator.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", ["circle_packing", "circle_packing_with_artifacts", "saved_circle_packing"])
@pytest.mark.parametrize("centers,radii", [
    (np.full((26, 2), np.nan), np.full(26, .1)),
    (np.full((26, 2), .5), np.full(26, -1.0)),
    (np.zeros((2, 3)), np.zeros(2)),
    (np.zeros((26, 2)), np.full(26, np.inf)),
])
def test_invalid_geometry_is_never_feasible(name, centers, radii):
    result = evaluator(name).validate_packing(centers, radii)
    assert (result[0] if isinstance(result, tuple) else result) is False


@pytest.mark.parametrize("name", ["circle_packing", "circle_packing_with_artifacts", "saved_circle_packing"])
def test_zero_radius_grid_is_a_valid_boundary_case(name):
    centers = np.column_stack([np.linspace(.1, .9, 26), np.full(26, .5)])
    result = evaluator(name).validate_packing(centers, np.zeros(26))
    assert (result[0] if isinstance(result, tuple) else result) is True


@pytest.mark.parametrize("name", ["circle_packing", "circle_packing_with_artifacts", "saved_circle_packing"])
def test_runner_does_not_inherit_provider_credentials(name, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-sentinel-not-a-secret")
    program = tmp_path / "candidate's packing.py"
    program.write_text('''import os
import numpy as np
def run_packing():
    assert 'OPENAI_API_KEY' not in os.environ
    return np.zeros((26, 2)), np.zeros(26), 0.0
''')
    centers, radii, total = evaluator(name).run_with_timeout(str(program), 10)
    assert np.asarray(centers).shape == (26, 2)
    assert np.asarray(radii).shape == (26,)
    assert total == 0.0


@pytest.mark.parametrize("name", ["circle_packing", "circle_packing_with_artifacts", "saved_circle_packing"])
def test_candidate_timeout_is_bounded(name, tmp_path):
    program = tmp_path / "sleep.py"
    program.write_text("import time\ntime.sleep(30)\n")
    module = evaluator(name)
    with pytest.raises(module.TimeoutError):
        module.run_with_timeout(str(program), .2)
