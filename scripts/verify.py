"""picks と payouts を突き合わせ、的中率と回収率を出す。日次ノートも書く。

使い方:
  python scripts/verify.py                    # 今日
  python scripts/verify.py --date 20261102
  python scripts/verify.py --date 20261102 --write-notes
  python scripts/verify.py --backtest --from 20231101 --to 20261101   # 現行ルールを過去に当てる

出力は JSON。回収率 = 払戻合計 ÷ 購入合計（1.0 が損益分岐）。
"""
from __future__ import annotations

import argparse
import json
from datetime import timedelta

from common import NOTES_DIR, from_hd, load_rules, to_hd, today, venue_name
from db import connect, upsert


def evaluate(con, hd: str, picks: list[dict] | None = None) -> dict:
    if picks is None:
        picks = [dict(zip(["jcd", "rno", "ticket_type", "combination", "amount", "rules_version"], r))
                 for r in con.execute(
                     "select jcd, rno, ticket_type, combination, amount, rules_version from picks where hd=?",
                     (hd,)).fetchall()]
    pay = {(r[0], r[1], r[2], r[3]): r[4] for r in con.execute(
        "select jcd, rno, ticket_type, combination, payout from payouts where hd=?", (hd,)).fetchall()}
    done_races = {(r[0], r[1]) for r in con.execute(
        "select distinct jcd, rno from payouts where hd=?", (hd,)).fetchall()}

    bets = hits = bet_total = payout_total = 0
    races = set()
    details = []
    for p in picks:
        key = (p["jcd"], p["rno"])
        if key not in done_races:
            continue  # 中止・未確定
        races.add(key)
        bets += 1
        bet_total += p["amount"]
        # 払戻は100円あたり。購入額に比例
        unit_payout = pay.get((p["jcd"], p["rno"], p["ticket_type"], p["combination"]), 0)
        got = unit_payout * p["amount"] // 100
        if got > 0:
            hits += 1
        payout_total += got
        details.append({"race": f"{venue_name(p['jcd'])}{p['rno']}R", "combo": p["combination"],
                        "bet": p["amount"], "payout": got})
    return {
        "hd": hd,
        "rules_version": picks[0]["rules_version"] if picks else None,
        "races": len(races), "bets": bets, "hits": hits,
        "bet_total": bet_total, "payout_total": payout_total,
        "hit_rate": round(hits / bets, 3) if bets else None,
        "roi": round(payout_total / bet_total, 3) if bet_total else None,
        "details": details,
    }


def rolling(con, hd: str, days: int) -> dict | None:
    d1 = from_hd(hd)
    d0 = d1 - timedelta(days=days - 1)
    r = con.execute(
        "select count(*), sum(bet_total), sum(payout_total) from verifications where hd between ? and ?",
        (to_hd(d0), hd)).fetchone()
    if not r or not r[1]:
        return None
    return {"days": r[0], "bet": r[1], "payout": r[2], "roi": round(r[2] / r[1], 3)}


def month_to_date(con, hd: str) -> dict | None:
    """暦月の1日から hd までの累計（月次回収率の途中経過）。"""
    r = con.execute(
        "select count(*), sum(bet_total), sum(payout_total) from verifications where hd between ? and ?",
        (hd[:6] + "01", hd)).fetchone()
    if not r or not r[1]:
        return None
    return {"days": r[0], "bet": r[1], "payout": r[2], "roi": round(r[2] / r[1], 3)}


def write_daily_note(con, s: dict, rules: dict) -> None:
    d = from_hd(s["hd"])
    path = NOTES_DIR / "daily" / f"{d.isoformat()}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {d.isoformat()} 検証（rules v{s['rules_version']}）", "",
             f"- 対象 {s['races']}レース / {s['bets']}点 / 的中 {s['hits']} / 的中率 {s['hit_rate']}",
             f"- 購入 ¥{s['bet_total']} / 払戻 ¥{s['payout_total']} / **回収率 {s['roi']}**", ""]
    m = month_to_date(con, s["hd"])
    if m:
        lines.append(f"- **今月（{d.year}年{d.month}月）累計: {m['days']}日分 / 購入 ¥{m['bet']} / "
                     f"払戻 ¥{m['payout']} / 月次回収率 {m['roi']}**")
    for n in rules["evaluation"]["rolling_days"]:
        r = rolling(con, s["hd"], n)
        if r:
            lines.append(f"- 直近{n}日: {r['days']}日分 / 購入 ¥{r['bet']} / 払戻 ¥{r['payout']} / 回収率 {r['roi']}")
    lines += ["", "## 内訳", ""]
    for x in s["details"]:
        mark = "◎" if x["payout"] else "×"
        lines.append(f"- {mark} {x['race']} {x['combo']} ¥{x['bet']} → ¥{x['payout']}")
    lines += ["", "## 気づき（夜に埋める）", "", "- ", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def backtest(con, rules: dict, d0, d1) -> dict:
    """現行ルールを過去の出走表に当て、同じ買い方で回収率を見る。"""
    from select_races import build_picks, candidate_races

    keys = ("races", "bets", "hits", "bet_total", "payout_total")
    agg = {"days": 0, "races": 0, "bets": 0, "hits": 0, "bet_total": 0, "payout_total": 0}
    months: dict[str, dict] = {}
    d = d0
    while d <= d1:
        hd = to_hd(d)
        picks = build_picks(hd, candidate_races(con, hd, rules), rules)
        if picks:
            s = evaluate(con, hd, picks)
            if s["bets"]:
                agg["days"] += 1
                m = months.setdefault(hd[:6], {"month": f"{hd[:4]}-{hd[4:6]}", "days": 0, **{k: 0 for k in keys}})
                m["days"] += 1
                for k in keys:
                    agg[k] += s[k]
                    m[k] += s[k]
        d += timedelta(days=1)
    # 評価の単位は暦月。月次回収率 = その月の払戻合計 ÷ 購入合計（rules.yaml evaluation 参照）
    target = rules["evaluation"].get("target_monthly_roi", 1.0)
    monthly = []
    for m in months.values():
        m["roi"] = round(m["payout_total"] / m["bet_total"], 3) if m["bet_total"] else None
        m["win"] = m["roi"] is not None and m["roi"] > target
        monthly.append(m)
    agg["monthly"] = monthly
    agg["months"] = len(monthly)
    agg["win_months"] = sum(m["win"] for m in monthly)
    rois = sorted(m["roi"] for m in monthly if m["roi"] is not None)
    agg["monthly_roi_median"] = rois[len(rois) // 2] if rois else None
    agg["monthly_roi_worst"] = rois[0] if rois else None
    agg["hit_rate"] = round(agg["hits"] / agg["bets"], 3) if agg["bets"] else None
    agg["roi"] = round(agg["payout_total"] / agg["bet_total"], 3) if agg["bet_total"] else None
    agg["rules_version"] = rules["version"]
    return agg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=to_hd(today()))
    ap.add_argument("--write-notes", action="store_true")
    ap.add_argument("--backtest", action="store_true")
    ap.add_argument("--from", dest="date_from")
    ap.add_argument("--to", dest="date_to")
    args = ap.parse_args()

    rules = load_rules()
    con = connect()

    if args.backtest:
        if not (args.date_from and args.date_to):
            ap.error("--backtest には --from/--to が必要")
        print(json.dumps(backtest(con, rules, from_hd(args.date_from), from_hd(args.date_to)),
                         ensure_ascii=False, indent=2))
        return 0

    s = evaluate(con, args.date)
    print(json.dumps(s, ensure_ascii=False, indent=2))
    if args.write_notes and s["bets"]:
        upsert(con, "verifications", [{k: v for k, v in s.items() if k != "details"}])
        write_daily_note(con, s, rules)
        print(f"wrote notes/daily/{from_hd(args.date).isoformat()}.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
