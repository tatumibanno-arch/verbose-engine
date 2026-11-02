"""翌日の出走表から rules.yaml に従って買い目を作り、picks に保存して notes/picks/ に書く。

使い方:
  python scripts/select_races.py                 # 翌日
  python scripts/select_races.py --date 20261103 --dry-run

買い目は「紙の上の購入（paper=1）」として記録する。実際の購入は人が朝に行う。
"""
from __future__ import annotations

import argparse
import itertools
import json

from common import NOTES_DIR, from_hd, load_rules, to_hd, tomorrow, venue_name
from db import connect, upsert


def expand_formation(formation: str) -> list[str]:
    """'1-23-234' → ['1-2-3','1-2-4','1-3-2','1-3-4']（同じ艇の重複は除く）"""
    slots = [list(s) for s in formation.split("-")]
    combos = []
    for c in itertools.product(*slots):
        if len(set(c)) == len(c):
            combos.append("-".join(c))
    return combos


def candidate_races(con, hd: str, rules: dict) -> list[dict]:
    """フィルタと1号艇条件を満たすレースを返す。"""
    f = rules.get("filters", {})
    l1 = rules.get("lane1", {})
    rows = con.execute(
        "select jcd, rno, lane, grade, race_name, close_time, data from entries where hd=? order by jcd, rno, lane",
        (hd,),
    ).fetchall()
    races: dict[tuple, list] = {}
    for r in rows:
        races.setdefault((r[0], r[1]), []).append(r)

    out = []
    for (jcd, rno), boats in races.items():
        if f.get("venues") and jcd not in f["venues"]:
            continue
        if f.get("race_numbers") and rno not in f["race_numbers"]:
            continue
        lane1 = next((b for b in boats if b[2] == 1), None)
        if not lane1:
            continue
        if l1.get("grades") and (lane1[3] or "") not in l1["grades"]:
            continue
        # TODO: min_win_rate_lane1 / min_motor_2ren は data 列の項目名が確定したら評価する
        out.append({"jcd": jcd, "rno": rno, "race_name": lane1[4], "close_time": lane1[5],
                    "lane1_grade": lane1[3]})
    return out


def build_picks(hd: str, races: list[dict], rules: dict) -> list[dict]:
    b = rules["budget"]
    bet = rules["bet"]
    races = races[: b["max_races"]]            # TODO: 自信度順に並べてから切る
    if not races:
        return []
    combos = expand_formation(bet["formation"])
    per_race = b["daily_yen"] // len(races)
    per_point = max(b["unit_yen"], (per_race // len(combos)) // b["unit_yen"] * b["unit_yen"])
    picks = []
    for r in races:
        for c in combos:
            picks.append({
                "hd": hd, "jcd": r["jcd"], "rno": r["rno"],
                "ticket_type": bet["ticket_type"], "combination": c, "amount": per_point,
                "rules_version": rules["version"],
                "reason": f"lane1 {r['lane1_grade']}",
                "paper": 1,
            })
    return picks


def write_note(hd: str, picks: list[dict], rules: dict) -> None:
    d = from_hd(hd)
    path = NOTES_DIR / "picks" / f"{d.isoformat()}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    total = sum(p["amount"] for p in picks)
    lines = [f"# {d.isoformat()} 選定（rules v{rules['version']}）", "",
             f"- 合計 ¥{total} / 予算 ¥{rules['budget']['daily_yen']}", ""]
    cur = None
    for p in picks:
        key = (p["jcd"], p["rno"])
        if key != cur:
            lines.append(f"## {venue_name(p['jcd'])} {p['rno']}R　{p['reason']}")
            cur = key
        lines.append(f"- {p['ticket_type']} {p['combination']} ¥{p['amount']}")
    lines += ["", "## 実購入メモ（朝に人が書く）", "", "- 買った / 見送った / 変更した："]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=to_hd(tomorrow()))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rules = load_rules()
    con = connect()
    races = candidate_races(con, args.date, rules)
    picks = build_picks(args.date, races, rules)
    print(f"{args.date}: 候補 {len(races)} レース → 買い目 {len(picks)} 点 / ¥{sum(p['amount'] for p in picks)}")
    if args.dry_run:
        print(json.dumps(picks, ensure_ascii=False, indent=2))
        return 0
    con.execute("delete from picks where hd=? and paper=1", (args.date,))
    upsert(con, "picks", picks)
    write_note(args.date, picks, rules)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
