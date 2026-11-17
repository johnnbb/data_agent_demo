import sqlite3
from pathlib import Path

from eval.score import compute_ex, execute_sqlite_readonly, extract_tables_from_sql


def test_extract_tables_from_sql_cte_subtraction():
    sql = """
    WITH eligible_rates AS (
        SELECT CDSCode, `Free Meal Count (K-12)` * 1.0 / `Enrollment (K-12)` AS rate
        FROM frpm
        WHERE `County Name` = 'Alameda'
    ),
    top_school AS (
        SELECT CDSCode FROM eligible_rates ORDER BY rate DESC LIMIT 1
    )
    SELECT s.School
    FROM schools AS s
    JOIN top_school AS ts ON s.CDSCode = ts.CDSCode
    """
    tables = extract_tables_from_sql(sql)
    assert tables == {"frpm", "schools"}


def test_compute_ex_equivalent_aliases_column_order_and_floats(tmp_path: Path):
    db_path = tmp_path / "sample.sqlite"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE schools (id INTEGER, name TEXT, rate REAL)")
    conn.executemany(
        "INSERT INTO schools VALUES (?, ?, ?)",
        [(1, "Alpha", 0.123456), (2, "Beta", 0.5)],
    )
    conn.commit()
    conn.close()

    ok1, rows1, err1 = execute_sqlite_readonly(
        db_path, "SELECT name AS school_name, rate AS free_rate FROM schools"
    )
    ok2, rows2, err2 = execute_sqlite_readonly(
        db_path,
        "SELECT ROUND(rate, 4) AS r, name AS n FROM schools ORDER BY id DESC",
    )
    assert ok1 and err1 is None
    assert ok2 and err2 is None

    is_match, is_empty = compute_ex(rows1, rows2)
    assert is_match is True
    assert is_empty is False


def test_compute_ex_mismatched_row_count():
    gold = [(1, "Alpha"), (1, "Alpha")]
    pred = [(1, "Alpha")]
    is_match, is_empty = compute_ex(pred, gold)
    assert is_match is False
    assert is_empty is False


def test_compute_ex_empty_set_match_flag():
    is_match, is_empty = compute_ex([], [])
    assert is_match is True
    assert is_empty is True

    is_match2, is_empty2 = compute_ex([], [(1,)])
    assert is_match2 is False
    assert is_empty2 is False
