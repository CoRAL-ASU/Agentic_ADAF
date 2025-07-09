"""Utility helpers for applying simple operations to tables."""

from __future__ import annotations

from typing import Any, Iterable, Sequence


def _to_float(value: Any) -> float | None:
    try:
        return float(str(value).replace(",", "").replace("$", ""))
    except (TypeError, ValueError):
        return None


def apply_tabular_op(table: Iterable[Sequence[Any]], op: Sequence[Any]):
    """Apply a simple operation ``op`` on ``table``.

    Parameters
    ----------
    table:
        The table represented as an iterable of rows (each row is a sequence of
        cell values).
    op:
        A tuple describing the operation as produced by :class:`TableAgent`.
    """

    kind = op[0] if op else "noop"

    if kind == "row" and len(op) >= 2:
        idx = op[1]
        rows = list(table)
        if -len(rows) <= idx < len(rows):
            return rows[idx]
        return None

    if kind == "column" and len(op) >= 2:
        idx = op[1]
        result = []
        for row in table:
            if len(row) > idx:
                result.append(row[idx])
        return result

    if kind == "cell" and len(op) >= 3:
        r, c = op[1], op[2]
        rows = list(table)
        if -len(rows) <= r < len(rows) and len(rows[r]) > c >= -len(rows[r]):
            return rows[r][c]
        return None

    if kind == "sum_column" and len(op) >= 2:
        idx = op[1]
        total = 0.0
        found = False
        for row in table:
            if len(row) > idx:
                val = _to_float(row[idx])
                if val is not None:
                    total += val
                    found = True
        return total if found else None

    if kind == "noop":
        # Simply return the original table structure
        return list(list(r) for r in table)

    return None
