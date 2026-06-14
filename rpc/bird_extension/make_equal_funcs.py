"""Factory that builds RPC-compatible ``equal_func`` and ``check_equal``
closures backed by SQLite execution-match, for the BIRD text-to-SQL task.

The original RPC evaluators (compute_perp / compute_sc / compute_rpc) cluster
reasoning paths with ``equal_func(ai, aj, ci, cj)`` and judge correctness with
``check_equal(ans_i, answer)``, where the answers are math expressions compared
with sympy. For text-to-SQL we instead:

  * store each candidate SQL string as a ``predict`` entry,
  * store the gold SQL string as ``answer``,
  * compare two candidates by executing both on the problem's DB and checking
    result-set equality (``exec_match``),
  * compare a candidate vs gold the same way.

Because ``exec_match`` needs the *per-problem* database path, the evaluators'
fixed ``equal_func`` / ``check_equal`` signatures (which carry no problem index)
are a problem. We solve it the way the RPC code itself does for its cache: by
building the closures *per problem index* via :func:`make_funcs_for_problem`,
and driving the evaluation loop ourselves (see ``run_bird.py``) so the right
closure is installed for each problem.

A convenience :class:`SQLFuncFactory` wraps the mapping from problem index to
its database path.
"""

from __future__ import annotations

import os
from typing import Callable, Dict, List, Tuple

from sql_exec import exec_match

EqualFunc = Callable[..., bool]
CheckEqualFunc = Callable[..., bool]


def db_path_for(db_base: str, db_id: str) -> str:
    """Return the .sqlite path for a BIRD ``db_id`` under ``db_base``
    (the ``dev_databases`` directory)."""
    return os.path.join(db_base, db_id, f"{db_id}.sqlite")


def make_funcs_for_problem(db_path: str, timeout: int = 30) -> Tuple[EqualFunc, CheckEqualFunc]:
    """Build ``(equal_func, check_equal)`` closures bound to a single DB.

    ``equal_func(ai, aj, ci, cj, cache_dict=None)`` compares two candidate SQL
    strings ``ai`` and ``aj`` (the completions ``ci``/``cj`` are ignored, kept
    only for signature compatibility with the RPC evaluators).

    ``check_equal(ans, gold, cache_dict=None)`` compares a candidate SQL ``ans``
    against the gold SQL ``gold``.

    Both accept an optional ``cache_dict`` (``{ "a<##>b": bool }``) so they can
    be wrapped with the same caching convention RPC uses; ``None`` disables it.
    """

    def _cached(a: str, b: str, cache_dict) -> bool:
        if cache_dict is not None:
            key = str(a) + "<##>" + str(b)
            if key in cache_dict:
                return cache_dict[key]
            rev = str(b) + "<##>" + str(a)
            if rev in cache_dict:
                return cache_dict[rev]
            val = exec_match(db_path, a, b, timeout=timeout)
            cache_dict[key] = val
            cache_dict[rev] = val
            return val
        return exec_match(db_path, a, b, timeout=timeout)

    def equal_func(ai, aj, ci=None, cj=None, cache_dict=None) -> bool:
        return _cached(ai, aj, cache_dict)

    def check_equal(ans, gold, cache_dict=None) -> bool:
        return _cached(ans, gold, cache_dict)

    return equal_func, check_equal


class SQLFuncFactory:
    """Holds the per-problem-index DB paths and hands out closures.

    Build it from a meta list mapping problem index -> {question_id, db_id},
    plus the ``dev_databases`` base path.
    """

    def __init__(self, db_ids_by_index: List[str], db_base: str, timeout: int = 30):
        self.db_base = db_base
        self.timeout = timeout
        self.db_paths: List[str] = [db_path_for(db_base, d) for d in db_ids_by_index]

    @classmethod
    def from_meta(cls, meta: List[Dict], db_base: str, timeout: int = 30) -> "SQLFuncFactory":
        db_ids = [m["db_id"] for m in meta]
        return cls(db_ids, db_base, timeout=timeout)

    def funcs(self, problem_index: int) -> Tuple[EqualFunc, CheckEqualFunc]:
        """Return ``(equal_func, check_equal)`` for the given problem index."""
        return make_funcs_for_problem(self.db_paths[problem_index], timeout=self.timeout)
