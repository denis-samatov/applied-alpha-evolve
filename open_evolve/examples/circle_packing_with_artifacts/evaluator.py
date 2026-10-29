"""
Evaluator for circle packing example (n=26) with improved timeout handling
Enhanced with artifacts to demonstrate execution feedback
"""

import numpy as np
import time
import traceback
import builtins

# Import EvaluationResult for artifacts support
from openevolve.evaluation_result import EvaluationResult


class TimeoutError(Exception):
    pass


def timeout_handler(signum, frame):
    """Handle timeout signal"""
    raise TimeoutError("Function execution timed out")


def validate_packing(centers, radii):
    """
    Validate that circles don't overlap and are inside the unit square

    Args:
        centers: np.array of shape (n, 2) with (x, y) coordinates
        radii: np.array of shape (n) with radius of each circle

    Returns:
        Tuple of (is_valid: bool, validation_details: dict)
    """
    try:
        centers = np.asarray(centers, dtype=float)
        radii = np.asarray(radii, dtype=float)
        valid_input = (centers.shape == (26, 2) and radii.shape == (26,)
                       and np.isfinite(centers).all() and np.isfinite(radii).all()
                       and (radii >= 0).all())
    except (TypeError, ValueError):
        valid_input = False
    if not valid_input:
        return False, {"is_valid": False, "error": "invalid shape, non-finite or negative radius"}
    n = centers.shape[0]
    validation_details = {
        "total_circles": n,
        "boundary_violations": [],
        "overlaps": [],
        "min_radius": float(np.min(radii)),
        "max_radius": float(np.max(radii)),
        "avg_radius": float(np.mean(radii)),
    }

    # Check if circles are inside the unit square
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if x - r < -1e-6 or x + r > 1 + 1e-6 or y - r < -1e-6 or y + r > 1 + 1e-6:
            violation = (
                f"Circle {i} at ({x:.6f}, {y:.6f}) with radius {r:.6f} is outside unit square"
            )
            validation_details["boundary_violations"].append(violation)
            print(violation)

    # Check for overlaps
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if dist < radii[i] + radii[j] - 1e-6:  # Allow for tiny numerical errors
                overlap = (
                    f"Circles {i} and {j} overlap: dist={dist:.6f}, r1+r2={radii[i]+radii[j]:.6f}"
                )
                validation_details["overlaps"].append(overlap)
                print(overlap)

    is_valid = (
        len(validation_details["boundary_violations"]) == 0
        and len(validation_details["overlaps"]) == 0
    )
    validation_details["is_valid"] = is_valid

    return is_valid, validation_details


def run_with_timeout(program_path, timeout_seconds=20):
    """Run trusted local code using bounded numeric JSON IPC, not pickle."""
    from openevolve.utils.packing_runner import run_numeric_candidate

    try:
        return run_numeric_candidate(program_path, timeout_seconds)
    except builtins.TimeoutError as error:
        raise TimeoutError(str(error)) from error


def evaluate(program_path):
    """
    Evaluate the program by running it once and checking the sum of radii

    Args:
        program_path: Path to the program file

    Returns:
        EvaluationResult with metrics and artifacts
    """
    # Target value from the paper
    TARGET_VALUE = 2.635  # AlphaEvolve result for n=26

    try:
        # For constructor-based approaches, a single evaluation is sufficient
        # since the result is deterministic
        start_time = time.time()

        # Use subprocess to run with timeout
        centers, radii, reported_sum = run_with_timeout(
            program_path, timeout_seconds=600  # Single timeout
        )

        end_time = time.time()
        eval_time = end_time - start_time

        # Ensure centers and radii are numpy arrays
        if not isinstance(centers, np.ndarray):
            centers = np.array(centers)
        if not isinstance(radii, np.ndarray):
            radii = np.array(radii)

        # Validate solution
        valid, validation_details = validate_packing(centers, radii)

        # Check shape and size
        shape_valid = centers.shape == (26, 2) and radii.shape == (26,)
        if not shape_valid:
            shape_error = f"Invalid shapes: centers={centers.shape}, radii={radii.shape}, expected (26, 2) and (26,)"
            print(shape_error)

            return EvaluationResult(
                metrics={
                    "sum_radii": 0.0,
                    "target_ratio": 0.0,
                    "validity": 0.0,
                    "eval_time": float(eval_time),
                    "combined_score": 0.0,
                },
                artifacts={
                    "stderr": shape_error,
                    "failure_stage": "shape_validation",
                    "expected_shapes": "centers: (26, 2), radii: (26,)",
                    "actual_shapes": f"centers: {centers.shape}, radii: {radii.shape}",
                    "execution_time": f"{eval_time:.2f}s",
                },
            )

        # Calculate sum
        sum_radii = np.sum(radii) if valid else 0.0

        # Make sure reported_sum matches the calculated sum
        sum_mismatch = abs(sum_radii - reported_sum) > 1e-6
        if sum_mismatch:
            mismatch_warning = (
                f"Warning: Reported sum {reported_sum} doesn't match calculated sum {sum_radii}"
            )
            print(mismatch_warning)

        # Target ratio (how close we are to the target)
        target_ratio = sum_radii / TARGET_VALUE if valid else 0.0

        # Validity score
        validity = 1.0 if valid else 0.0

        # Combined score - higher is better
        combined_score = target_ratio * validity

        print(
            f"Evaluation: valid={valid}, sum_radii={sum_radii:.6f}, target={TARGET_VALUE}, ratio={target_ratio:.6f}, time={eval_time:.2f}s"
        )

        # Prepare artifacts with packing details
        artifacts = {
            "execution_time": f"{eval_time:.2f}s",
            "packing_summary": f"Sum of radii: {sum_radii:.6f}/{TARGET_VALUE} = {target_ratio:.4f}",
            "validation_report": f"Valid: {valid}, Violations: {len(validation_details.get('boundary_violations', []))} boundary, {len(validation_details.get('overlaps', []))} overlaps",
        }

        # Add validation details if there are issues
        if not valid:
            if validation_details.get("boundary_violations"):
                artifacts["boundary_violations"] = "\n".join(
                    validation_details["boundary_violations"]
                )
            if validation_details.get("overlaps"):
                artifacts["overlap_violations"] = "\n".join(validation_details["overlaps"])
            artifacts["failure_stage"] = "geometric_validation"

        # Add sum mismatch warning if present
        if sum_mismatch:
            artifacts["sum_mismatch"] = f"Reported: {reported_sum:.6f}, Calculated: {sum_radii:.6f}"

        # Add successful packing stats for good solutions
        if valid and target_ratio > 0.95:  # Near-optimal solutions
            artifacts["stdout"] = f"Excellent packing! Achieved {target_ratio:.1%} of target value"
            artifacts[
                "radius_stats"
            ] = f"Min: {validation_details['min_radius']:.6f}, Max: {validation_details['max_radius']:.6f}, Avg: {validation_details['avg_radius']:.6f}"

        return EvaluationResult(
            metrics={
                "sum_radii": float(sum_radii),
                "target_ratio": float(target_ratio),
                "validity": float(validity),
                "eval_time": float(eval_time),
                "combined_score": float(combined_score),
            },
            artifacts=artifacts,
        )

    except TimeoutError as e:
        error_msg = f"Evaluation timed out: {str(e)}"
        print(error_msg)
        return EvaluationResult(
            metrics={
                "sum_radii": 0.0,
                "target_ratio": 0.0,
                "validity": 0.0,
                "eval_time": 600.0,  # Timeout duration
                "combined_score": 0.0,
            },
            artifacts={
                "stderr": error_msg,
                "failure_stage": "execution_timeout",
                "timeout_duration": "600s",
                "suggestion": "Consider optimizing the packing algorithm for faster convergence",
            },
        )
    except Exception as e:
        error_msg = f"Evaluation failed completely: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        return EvaluationResult(
            metrics={
                "sum_radii": 0.0,
                "target_ratio": 0.0,
                "validity": 0.0,
                "eval_time": 0.0,
                "combined_score": 0.0,
            },
            artifacts={
                "stderr": error_msg,
                "traceback": traceback.format_exc(),
                "failure_stage": "program_execution",
                "suggestion": "Check for syntax errors, import issues, or runtime exceptions",
            },
        )


# Stage-based evaluation for cascade evaluation
def evaluate_stage1(program_path):
    """
    First stage evaluation - quick validation check
    Enhanced with artifacts for debugging
    """
    try:
        # Use the simplified subprocess approach
        try:
            start_time = time.time()
            centers, radii, sum_radii = run_with_timeout(program_path, timeout_seconds=600)
            eval_time = time.time() - start_time

            # Ensure centers and radii are numpy arrays
            if not isinstance(centers, np.ndarray):
                centers = np.array(centers)
            if not isinstance(radii, np.ndarray):
                radii = np.array(radii)

            # Validate solution (shapes and constraints)
            shape_valid = centers.shape == (26, 2) and radii.shape == (26,)
            if not shape_valid:
                shape_error = f"Invalid shapes: centers={centers.shape}, radii={radii.shape}"
                print(shape_error)
                return EvaluationResult(
                    metrics={"validity": 0.0, "combined_score": 0.0},
                    artifacts={
                        "stderr": shape_error,
                        "failure_stage": "stage1_shape_validation",
                        "expected_shapes": "centers: (26, 2), radii: (26,)",
                        "actual_shapes": f"centers: {centers.shape}, radii: {radii.shape}",
                        "execution_time": f"{eval_time:.2f}s",
                    },
                )

            valid, validation_details = validate_packing(centers, radii)

            # Calculate sum
            actual_sum = np.sum(radii) if valid else 0.0

            # Target from paper
            target = 2.635

            # Simple combined score for stage 1
            combined_score = (actual_sum / target) if valid else 0.0

            # Prepare artifacts for stage 1
            artifacts = {
                "execution_time": f"{eval_time:.2f}s",
                "stage": "quick_validation",
                "packing_summary": f"Sum: {actual_sum:.6f}, Ratio: {actual_sum/target:.4f}",
            }

            # Add validation issues if any
            if not valid:
                artifacts[
                    "stderr"
                ] = f"Validation failed: {len(validation_details.get('boundary_violations', []))} boundary violations, {len(validation_details.get('overlaps', []))} overlaps"
                artifacts["failure_stage"] = "stage1_geometric_validation"
                if validation_details.get("boundary_violations"):
                    artifacts["boundary_issues"] = validation_details["boundary_violations"][
                        0
                    ]  # Just first issue
                if validation_details.get("overlaps"):
                    artifacts["overlap_issues"] = validation_details["overlaps"][
                        0
                    ]  # Just first issue

            # Return evaluation metrics
            return EvaluationResult(
                metrics={
                    "validity": 1.0 if valid else 0.0,
                    "sum_radii": float(actual_sum),
                    "target_ratio": float(actual_sum / target if valid else 0.0),
                    "combined_score": float(combined_score),
                },
                artifacts=artifacts,
            )

        except TimeoutError as e:
            error_msg = f"Stage 1 evaluation timed out: {e}"
            print(error_msg)
            return EvaluationResult(
                metrics={"validity": 0.0, "combined_score": 0.0},
                artifacts={
                    "stderr": error_msg,
                    "failure_stage": "stage1_timeout",
                    "timeout_duration": "600s",
                    "suggestion": "Algorithm may be too slow for stage 1 - consider simpler heuristics",
                },
            )
        except Exception as e:
            error_msg = f"Stage 1 evaluation failed: {e}"
            print(error_msg)
            print(traceback.format_exc())
            return EvaluationResult(
                metrics={"validity": 0.0, "combined_score": 0.0},
                artifacts={
                    "stderr": error_msg,
                    "traceback": traceback.format_exc(),
                    "failure_stage": "stage1_execution",
                    "suggestion": "Check basic syntax and imports before attempting full evaluation",
                },
            )

    except Exception as e:
        error_msg = f"Stage 1 evaluation failed completely: {e}"
        print(error_msg)
        print(traceback.format_exc())
        return EvaluationResult(
            metrics={"validity": 0.0, "combined_score": 0.0},
            artifacts={
                "stderr": error_msg,
                "traceback": traceback.format_exc(),
                "failure_stage": "stage1_critical_failure",
                "suggestion": "Major issues detected - check program structure and dependencies",
            },
        )


def evaluate_stage2(program_path):
    """
    Second stage evaluation - full evaluation
    """
    # Full evaluation as in the main evaluate function
    return evaluate(program_path)
