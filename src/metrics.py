"""Metric helpers. Accuracy matrix convention: rows=training stage, cols=evaluation task.
Only entries with task <= stage are normally populated; use None elsewhere.
"""
from __future__ import annotations
from typing import Sequence


def backward_transfer(matrix: Sequence[Sequence[float | None]]) -> float:
    """Mean BWT: final performance on each prior task minus its first-learned score.

    Negative values indicate forgetting. Excludes the final task, which has no
    later training stage. The matrix must be square and lower-triangular observed.
    """
    n = len(matrix)
    if n < 2 or any(len(row) != n for row in matrix):
        raise ValueError("matrix must be square with at least two tasks")
    differences = []
    for task in range(n - 1):
        initial = matrix[task][task]
        final = matrix[n - 1][task]
        if initial is None or final is None:
            raise ValueError("missing diagonal or final prior-task accuracy")
        differences.append(final - initial)
    return sum(differences) / len(differences)


def forward_transfer(matrix: Sequence[Sequence[float | None]],
                     baseline: Sequence[float]) -> float:
    """Mean pre-training transfer to a not-yet-trained task vs. its baseline.

    matrix[i][j] is evaluation on task j after training through task i.
    baseline[j] must be measured before training on task j using the same model
    and evaluation split. FWT averages j=1..n-1 at stage j-1.
    """
    n = len(matrix)
    if n < 2 or len(baseline) != n or any(len(row) != n for row in matrix):
        raise ValueError("incompatible task matrix/baseline")
    vals = []
    for task in range(1, n):
        value = matrix[task - 1][task]
        if value is None:
            raise ValueError("missing pre-training evaluation for future task")
        vals.append(value - baseline[task])
    return sum(vals) / len(vals)


def adaptation_speed(curve: Sequence[float], threshold: float) -> int | None:
    """First one-based interaction index at/above threshold, else None."""
    for i, value in enumerate(curve, start=1):
        if value >= threshold:
            return i
    return None
