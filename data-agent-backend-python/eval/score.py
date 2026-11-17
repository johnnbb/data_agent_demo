from __future__ import annotations

import math
import sqlite3
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Optional

import sqlglot
from sqlglot import exp


def extract_tables_from_sql(sql: str, dialect: str = "sqlite") -> set[str]:
    """Parse SQL with sqlglot and return referenced physical table names (lowercased), excluding CTE aliases."""
    clean = (sql or "").strip().rstrip(";")
    if not clean:
        return set()
    try:
        parsed = sqlglot.parse_one(clean, read=dialect)
    except Exception:
        return set()
    if parsed is None:
        return set()

    cte_names = {
        cte.alias_or_name.lower()
        for cte in parsed.find_all(exp.CTE)
        if cte.alias_or_name
    }
    table_names = {
        t.name.lower()
        for t in parsed.find_all(exp.Table)
        if t.name
    }
    return table_names - cte_names


def execute_sqlite_readonly(
    db_path: Path,
    sql: str,
    timeout_s: float = 10.0,
) -> tuple[bool, Optional[list[tuple[Any, ...]]], Optional[str]]:
    """Execute a SQLite query against a read-only URI connection with a progress-handler timeout."""
    if not db_path.exists():
        return False, None, f"Database file not found: {db_path}"
    clean = (sql or "").strip()
    if not clean:
        return False, None, "Empty SQL statement"

    conn: Optional[sqlite3.Connection] = None
    try:
        db_uri = f"file:{db_path.resolve()}?mode=ro"
        conn = sqlite3.connect(db_uri, uri=True, timeout=timeout_s)
        deadline = time.monotonic() + timeout_s

        def _progress_handler() -> int:
            return 1 if time.monotonic() > deadline else 0

        conn.set_progress_handler(_progress_handler, 10000)
        cursor = conn.cursor()
        cursor.execute(clean)
        rows = [tuple(row) for row in cursor.fetchall()]
        cursor.close()
        return True, rows, None
    except sqlite3.OperationalError as exc:
        msg = str(exc)
        if "interrupted" in msg.lower():
            return False, None, f"SQLite query timed out after {timeout_s:.1f}s"
        return False, None, msg
    except Exception as exc:
        return False, None, str(exc)
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def _normalize_cell(value: Any) -> tuple[int, str]:
    """Normalize a cell value to a comparable, deterministically sortable token."""
    if value is None:
        return (0, "")
    if isinstance(value, bool):
        return (1, "1" if value else "0")
    if isinstance(value, (int, float)):
        fval = float(value)
        if math.isnan(fval) or math.isinf(fval):
            return (2, str(fval))
        rounded = round(fval, 4)
        if rounded == 0.0:
            rounded = 0.0
        return (2, f"{rounded:.4f}")
    if isinstance(value, bytes):
        return (3, value.decode("utf-8", errors="replace"))
    return (3, str(value))


def _normalize_rows(rows: Iterable[Iterable[Any]]) -> Counter[tuple[tuple[int, str], ...]]:
    """Convert result rows into a row-order- and column-order-insensitive multiset."""
    normalized = []
    for row in rows:
        cells = tuple(sorted(_normalize_cell(cell) for cell in row))
        normalized.append(cells)
    return Counter(normalized)


def compute_ex(
    pred_rows: Optional[list[tuple[Any, ...]]],
    gold_rows: Optional[list[tuple[Any, ...]]],
) -> tuple[bool, bool]:
    """
    Compare predicted and gold SQLite result sets under BIRD EX semantics.

    Returns (is_match, is_empty_match), where is_empty_match is True iff both
    pred_rows and gold_rows are empty lists.
    """
    if pred_rows is None or gold_rows is None:
        return False, False
    if len(gold_rows) == 0 and len(pred_rows) == 0:
        return True, True
    if len(pred_rows) != len(gold_rows):
        return False, False
    return _normalize_rows(pred_rows) == _normalize_rows(gold_rows), False
