"""CPU test for sql_exec.py against the real BIRD dev databases.

Runs three checks over a sample of dev items:
  1. gold-vs-gold exec_match should be True (a passing query equals itself).
  2. gold-vs-broken exec_match should be False (a deliberately mangled query).
  3. run_sql returns None for syntactically invalid SQL.

Run (sqlite3 is stdlib, so plain python works):
    python3 test_sql_exec.py
or with the RPC uv project:
    cd ../RPC && uv run --project . python ../bird_extension/test_sql_exec.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sql_exec import exec_match, run_sql  # noqa: E402

BIRD_ROOT = "/mnt/nfs/ssd2/bird_data/dev_20240627"
DEV_JSON = os.path.join(BIRD_ROOT, "dev.json")
DB_DIR = os.path.join(BIRD_ROOT, "dev_databases")

SAMPLE = 60  # number of dev items to test (spread across DBs / difficulties)


def db_path_for(db_id: str) -> str:
    return os.path.join(DB_DIR, db_id, f"{db_id}.sqlite")


def main() -> int:
    with open(DEV_JSON, "r", encoding="utf-8") as f:
        dev = json.load(f)

    # Sample evenly across the dataset so we hit every db_id and difficulty.
    step = max(1, len(dev) // SAMPLE)
    items = dev[::step][:SAMPLE]

    gg_pass = 0      # gold-vs-gold matched
    gg_total = 0
    gg_failed_items = []  # items where gold did not even run / match itself
    broken_correct = 0   # gold-vs-broken correctly returned False
    broken_total = 0

    for it in items:
        db_id = it["db_id"]
        gold = it["SQL"]
        dbp = db_path_for(db_id)
        if not os.path.exists(dbp):
            print(f"WARN missing db {dbp}")
            continue

        # 1. gold vs gold
        gg_total += 1
        if exec_match(dbp, gold, gold):
            gg_pass += 1
        else:
            # Either the gold query errored, or returned None.
            res = run_sql(dbp, gold)
            gg_failed_items.append((it["question_id"], db_id, res is None))

        # 2. gold vs broken (append an impossible filter -> empty / different set
        #    OR break syntax). We wrap to force a different result set.
        broken = gold.rstrip().rstrip(";") + " WHERE 1=0"
        # Only meaningful if broken actually runs differently; if broken errors,
        # exec_match returns False which is still the desired outcome.
        broken_total += 1
        if not exec_match(dbp, gold, broken):
            broken_correct += 1

    # 3. invalid SQL -> None
    any_db = db_path_for(items[0]["db_id"])
    invalid_is_none = run_sql(any_db, "SELEC * FRM nope") is None

    print("=" * 60)
    print(f"Sampled items: {gg_total}")
    print(f"gold-vs-gold exec-match pass: {gg_pass}/{gg_total} "
          f"= {100.0 * gg_pass / max(1, gg_total):.1f}%")
    if gg_failed_items:
        print(f"  gold-vs-gold failures (qid, db, ran_to_None): {gg_failed_items}")
    print(f"gold-vs-broken correctly NOT matched: {broken_correct}/{broken_total} "
          f"= {100.0 * broken_correct / max(1, broken_total):.1f}%")
    print(f"invalid SQL -> None: {invalid_is_none}")
    print("=" * 60)

    ok = (gg_pass == gg_total) and (broken_correct == broken_total) and invalid_is_none
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
