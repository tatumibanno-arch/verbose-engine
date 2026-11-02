"""前日（または指定日）の picks を results と突き合わせ、的中率と回収率を出す。

使い方:
  python scripts/verify.py                  # 今日
  python scripts/verify.py --hd 20261101
  python scripts/verify.py --hd 20261101 --write-notes   # notes/daily/ にも書く

出力は JSON（通知と日次ログの両方で使う）。
TODO: 払戻の突き合わせは results の項目確定後に実装。現状は枠だけ。
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from common import ROOT, supabase_client, today_hd


def evaluate(picks: list[dict], results: list[dict]) -> dict:
    res_by_key = {(r["hd"], r["jcd"], r["rno"]): r for r in results}
    n = 0
    hits = 0
    bet_total = 0
    payout_total = 0
    details = []
    for p in picks:
        r = res_by_key.get((p["hd"], p["jcd"], p["rno"]))
        if not r:
            continue
        n += 1
        points = 1  # TODO: pattern から点数を展開
        bet = points * p["unit_yen"]
        payout = 0  # TODO: r の払戻から pattern 該当分を取り出す
        hit = payout > 0
        hits += int(hit)
        bet_total += bet
        payout_total += payout
        details.append({"jcd": p["jcd"], "rno": p["rno"], "hit": hit, "bet": bet, "payout": payout})
    return {
        "hd": picks[0]["hd"] if picks else None,
        "races": n,
        "hits": hits,
        "hit_rate": round(hits / n, 3) if n else None,
        "bet_total": bet_total,
        "payout_total": payout_total,
        "roi": round(payout_total / bet_total, 3) if bet_total else None,
        "details": details,
    }


def write_daily_note(summary: dict) -> Path:
    hd = summary["hd"]
    d = date(int(hd[:4]), int(hd[4:6]), int(hd[6:]))
    path = ROOT / "notes" / "daily" / f"{d.isoformat()}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# {d.isoformat()} 検証",
        "",
        f"- 対象: {summary['races']}レース / 的中: {summary['hits']} / 的中率: {summary['hit_rate']}",
        f"- 購入: ¥{summary['bet_total']} / 払戻: ¥{summary['payout_total']} / 回収率: {summary['roi']}",
        "",
        "## 気づき",
        "",
        "- （ここに書く）",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hd", default=today_hd())
    ap.add_argument("--write-notes", action="store_true")
    args = ap.parse_args()

    sb = supabase_client()
    picks = sb.table("picks").select("*").eq("hd", args.hd).execute().data
    results = sb.table("results").select("*").eq("hd", args.hd).execute().data
    summary = evaluate(picks, results)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.write_notes and summary["hd"]:
        sb.table("verifications").upsert(
            {k: v for k, v in summary.items() if k != "details"}, on_conflict="hd"
        ).execute()
        p = write_daily_note(summary)
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
