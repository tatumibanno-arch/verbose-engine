"""config/rules.yaml に従って今日の対象レースを選び、Supabase の picks に保存する。

使い方:
  python scripts/select_races.py            # 今日
  python scripts/select_races.py --hd 20261102 --dry-run

TODO: filters / conditions の評価は項目確定後に実装。現状は venues / race_numbers だけ効く。
"""
from __future__ import annotations

import argparse
import json

from common import load_rules, supabase_client, today_hd


def apply_rules(racelist_rows: list[dict], rules: dict) -> list[dict]:
    f = rules.get("filters", {})
    venues = set(f.get("venues") or [])
    rnos = set(f.get("race_numbers") or [])

    races: dict[tuple, list[dict]] = {}
    for r in racelist_rows:
        races.setdefault((r["hd"], r["jcd"], r["rno"]), []).append(r)

    picks: list[dict] = []
    for (hd, jcd, rno), boats in races.items():
        if venues and jcd not in venues:
            continue
        if rnos and rno not in rnos:
            continue
        # TODO: conditions（級別・2連率など）をここで評価
        picks.append({
            "hd": hd,
            "jcd": jcd,
            "rno": rno,
            "rules_version": rules.get("version"),
            "bet_type": rules["bet"]["type"],
            "pattern": rules["formation"]["pattern"],
            "unit_yen": rules["bet"]["unit_yen"],
            "reason": "filters only (conditions not implemented yet)",
        })
    return picks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hd", default=today_hd())
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rules = load_rules()
    sb = supabase_client()
    rows = sb.table("racelist").select("*").eq("hd", args.hd).execute().data
    picks = apply_rules(rows, rules)

    if args.dry_run:
        print(json.dumps(picks, ensure_ascii=False, indent=2))
        return 0

    if picks:
        sb.table("picks").upsert(picks, on_conflict="hd,jcd,rno").execute()
    print(f"picked {len(picks)} races")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
