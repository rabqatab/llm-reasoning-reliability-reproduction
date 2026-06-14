"""SQLite execution + comparison utilities for the BIRD text-to-SQL extension.

These functions provide an execution-match comparison (BIRD-style execution
accuracy) used to replace the sympy-based numeric comparison of the original
RPC math evaluators. Result sets are normalized to an order-insensitive
frozenset of row-tuples, so two queries match iff they return the same multiset
of rows up to ordering (we use a *set*, matching BIRD's standard exec-acc which
compares ``set(predicted) == set(gold)``).

sqlite3 is part of the Python standard library, so this module has no third
party dependencies and runs on CPU.
"""

from __future__ import annotations

import sqlite3
import threading
from typing import Optional

# A frozenset of row tuples, or None when the query failed / timed out.
ResultSet = Optional[frozenset]


def _execute(db_path: str, sql: str, out: dict) -> None:
    """Run ``sql`` against ``db_path`` and store a normalized result set in
    ``out['result']``. Intended to be run inside a worker thread so that the
    caller can enforce a wall-clock timeout. On any error, leaves
    ``out['result']`` as None.
    """
    try:
        # ``uri`` is False; open read-only would be nicer but BIRD DB paths are
        # plain files. We never write, so a normal connection is fine.
        conn = sqlite3.connect(db_path)
        # BIRD databases contain non-UTF8 bytes in a few text columns; decode
        # leniently so a stray byte does not abort an otherwise valid query.
        conn.text_factory = lambda b: b.decode("utf-8", "replace")
        try:
            cursor = conn.execute(sql)
            rows = cursor.fetchall()
            # Normalize: each row -> tuple, whole result -> order-insensitive set.
            out["result"] = frozenset(tuple(row) for row in rows)
        finally:
            conn.close()
    except Exception:
        out["result"] = None


def run_sql(db_path: str, sql: str, timeout: int = 30) -> ResultSet:
    """Execute ``sql`` against the SQLite DB at ``db_path``.

    Returns a ``frozenset`` of row-tuples (order-insensitive) on success, or
    ``None`` on any error or if execution exceeds ``timeout`` seconds.

    A daemon worker thread is used so that a runaway / long-running query does
    not block the caller indefinitely. Note that sqlite3 will keep executing in
    the background thread after a timeout, but the thread is a daemon and the
    process is free to continue; for the short BIRD dev queries this is fine.
    """
    out: dict = {"result": None}
    worker = threading.Thread(target=_execute, args=(db_path, sql, out), daemon=True)
    worker.start()
    worker.join(timeout)
    if worker.is_alive():
        # Timed out: leave result as None.
        return None
    return out["result"]


def exec_match(db_path: str, sql_a: str, sql_b: str, timeout: int = 30) -> bool:
    """Return True iff both SQL strings run successfully and produce the same
    (order-insensitive) result set on the database at ``db_path``.

    If either query errors out or times out (result is None), returns False.
    """
    res_a = run_sql(db_path, sql_a, timeout=timeout)
    if res_a is None:
        return False
    res_b = run_sql(db_path, sql_b, timeout=timeout)
    if res_b is None:
        return False
    return res_a == res_b
