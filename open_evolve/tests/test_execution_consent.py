"""No evaluator/plugin/candidate code may run without an explicit trust decision."""
from pathlib import Path

import pytest

from openevolve.config import EvaluatorConfig
from openevolve.evaluator import Evaluator
from openevolve.utils.packing_runner import run_numeric_candidate


def test_evaluator_denies_plugin_import_without_consent(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENEVOLVE_ALLOW_UNSANDBOXED_EXECUTION", raising=False)
    marker = tmp_path / "executed"
    evaluator = tmp_path / "evaluator.py"
    evaluator.write_text(f"from pathlib import Path\nPath({str(marker)!r}).write_text('ran')\ndef evaluate(path):return {{'score':1}}\n")
    with pytest.raises(PermissionError, match="UNSANDBOXED"):
        Evaluator(EvaluatorConfig(), str(evaluator))
    assert not marker.exists()


def test_numeric_runner_denies_before_candidate_execution(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENEVOLVE_ALLOW_UNSANDBOXED_EXECUTION", raising=False)
    marker = tmp_path / "executed"
    candidate = tmp_path / "candidate.py"
    candidate.write_text(f"from pathlib import Path\nPath({str(marker)!r}).write_text('ran')\nimport numpy as np\ndef run_packing():return np.zeros((26,2)),np.zeros(26),0\n")
    with pytest.raises(PermissionError, match="UNSANDBOXED"):
        run_numeric_candidate(candidate)
    assert not marker.exists()


def standalone_entrypoints():
    import ast
    root = Path(__file__).resolve().parents[1]
    paths = sorted([*root.glob("examples/*/evaluator.py"), *root.glob("best_result/*/evaluator.py")])
    return [(str(path.relative_to(root)), node.lineno) for path in paths
            for node in ast.walk(ast.parse(path.read_text()))
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name in {"evaluate", "evaluate_stage1", "evaluate_stage2", "evaluate_stage3"}]


@pytest.mark.parametrize("relative,line", standalone_entrypoints())
def test_standalone_entrypoint_denies_before_runtime_helpers(relative, line, monkeypatch):
    # Extract the real entrypoint without initializing optional GPU/provider libraries.
    # No runtime helpers are available: denial must precede their use or candidate execution.
    import ast
    import asyncio
    import __future__
    import inspect
    from openevolve.utils.execution_policy import require_execution_consent

    monkeypatch.delenv("OPENEVOLVE_ALLOW_UNSANDBOXED_EXECUTION", raising=False)
    path = Path(__file__).resolve().parents[1] / relative
    node = next(node for node in ast.walk(ast.parse(path.read_text()))
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.lineno == line)
    scope = {"require_execution_consent": require_execution_consent}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec",
                 flags=__future__.annotations.compiler_flag), scope)
    function = scope[node.name]
    parameters = inspect.signature(function).parameters.values()
    arguments = [None for parameter in parameters if parameter.default is inspect.Parameter.empty]
    with pytest.raises(PermissionError, match="UNSANDBOXED"):
        result = function(*arguments)
        if inspect.isawaitable(result):
            asyncio.run(result)


def test_revoked_consent_blocks_future_evaluation(tmp_path, monkeypatch):
    import asyncio
    marker = tmp_path / "executed"
    plugin = tmp_path / "plugin.py"
    plugin.write_text(f"from pathlib import Path\ndef evaluate(path):\n Path({str(marker)!r}).touch()\n return {{'score':1}}\n")
    evaluator = Evaluator(EvaluatorConfig(), str(plugin))
    monkeypatch.delenv("OPENEVOLVE_ALLOW_UNSANDBOXED_EXECUTION", raising=False)
    with pytest.raises(PermissionError, match="UNSANDBOXED"):
        asyncio.run(evaluator.evaluate_program("print('trusted fixture')"))
    assert not marker.exists()
