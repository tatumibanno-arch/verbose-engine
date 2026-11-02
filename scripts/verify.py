"""picks と payouts を突き合わせ、的中率と回収率を出す（ルール別と合計）。日次ノートも書く。

使い方:
  python scripts/verify.py                    # 今日
  python scripts/verify.py --date 20261102
  python scripts/verify.py --date 20261102 --write-notes
  python scripts/verify.py --monthly                     # 月次回収率（ルール別、verifications_by_rule から）
  python scripts/verify.py --backtest --from 20231101 --to 20261101                       # lane1_a1 を過去に当てる
  python scripts/verify.py --backtest --rule tansho_ev_v1 --from 20241101 --to 20261031   # モデル系は OOS 予測の履歴で

出力は JSON。回収率 = 払戻合計 ÷ 購入合計（1.0 が損益分岐）。
ルール名は picks.reason の先頭（「<ルール名> | 詳細」。select_races.rule_of）。
返還: 買い目に欠場艇（results で position も st_timing も NULL）が入っていたら購入0・払戻0（分析と同じ扱い）。
"""
from __future__ import annotations

import argparse
import json
from datetime import timedelta

from common import NOTES_DIR, from_hd, load_rules, to_hd, today, venue_name
from db import connect, upsert
from select_races import rule_of

SUM_KEYS = ("races", "bets", "hits", "refunds", "bet_total", "payout_total")


def _summ(hd, rule, version, rows) -> dict:
    s = {"hd": hd, "rule": rule, "rules_version": version, "races": len({r["race_key"] for r in rows if r["bet"]}),
         "bets": sum(1 for r in rows if r["bet"]), "hits": sum(1 for r in rows if r["payout"] > 0),
         "refunds": sum(1 for r in rows if r["refund"]),
         "bet_total": sum(r["bet"] for r in rows), "payout_total": sum(r["payout"] for r in rows)}
    s["hit_rate"] = round(s["hits"] / s["bets"], 3) if s["bets"] else None
    s["roi"] = round(s["payout_total"] / s["bet_total"], 3) if s["bet_total"] else None
    return s


def evaluate(con, hd: str, picks: list[dict] | None = None) -> dict:
    if picks is None:
        picks = [dict(zip(["jcd", "rno", "ticket_type", "combination", "amount", "rules_version", "reason"], r))
                 for r in con.execute(
                     "select jcd, rno, ticket_type, combination, amount, rules_version, reason from picks where hd=? "
                     "order by jcd, rno, combination", (hd,)).fetchall()]
    pay = {(r[0], r[1], r[2], r[3]): r[4] for r in con.execute(
        "select jcd, rno, ticket_type, combination, payout from payouts where hd=?", (hd,)).fetchall()}
    done_races = {(r[0], r[1]) for r in con.execute(
        "select distinct jcd, rno from payouts where hd=?", (hd,)).fetchall()}
    void = {(r[0], r[1], r[2]) for r in con.execute(
        "select jcd, rno, lane from results where hd=? and position is null "
        "and json_extract(data,'$.st_timing') is null", (hd,)).fetchall()}

    rows = []
    for p in picks:
        key = (p["jcd"], p["rno"])
        if key not in done_races:
            continue  # 中止・未確定
        lanes = [int(c) for c in str(p["combination"]).replace("=", "-").split("-") if c.isdigit()]
        refund = any((p["jcd"], p["rno"], ln) in void for ln in lanes)
        # 払戻は100円あたり。購入額に比例
        unit_payout = pay.get((p["jcd"], p["rno"], p["ticket_type"], p["combination"]), 0)
        got = 0 if refund else unit_payout * p["amount"] // 100
        rows.append({"rule": rule_of(p.get("reason")), "race_key": key, "refund": refund,
                     "bet": 0 if refund else p["amount"], "payout": got,
                     "race": f"{venue_name(p['jcd'])}{p['rno']}R", "combo": p["combination"],
                     "ticket_type": p["ticket_type"]})
    version = picks[0]["rules_version"] if picks else None
    total = _summ(hd, "all", version, rows)
    total["by_rule"] = {r: _summ(hd, r, version, [x for x in rows if x["rule"] == r])
                        for r in sorted({x["rule"] for x in rows})}
    total["details"] = [{k: x[k] for k in ("rule", "race", "ticket_type", "combo", "bet", "payout", "refund")} for x in rows]
    return total


def _agg(con, where: str, params: tuple, rule: str | None):
    if rule:
        r = con.execute(f"select count(*), sum(bet_total), sum(payout_total) from verifications_by_rule "
                        f"where rule=? and {where}", (rule, *params)).fetchone()
    else:
        r = con.execute(f"select count(*), sum(bet_total), sum(payout_total) from verifications where {where}",
                        params).fetchone()
    if not r or not r[1]:
        return None
    return {"days": r[0], "bet": r[1], "payout": r[2], "roi": round(r[2] / r[1], 3)}


def rolling(con, hd: str, days: int, rule: str | None = None) -> dict | None:
    d0 = from_hd(hd) - timedelta(days=days - 1)
    return _agg(con, "hd between ? and ?", (to_hd(d0), hd), rule)


def month_to_date(con, hd: str, rule: str | None = None) -> dict | None:
    """暦月の1日から hd までの累計（月次回収率の途中経過）。"""
    return _agg(con, "hd between ? and ?", (hd[:6] + "01", hd), rule)


def monthly(con, rule: str | None = None) -> list[dict]:
    """ルール別の月次回収率（verifications_by_rule から）。"""
    q = ("select rule, substr(hd,1,6) m, count(*), sum(bets), sum(hits), sum(bet_total), sum(payout_total) "
         "from verifications_by_rule " + ("where rule=? " if rule else "") + "group by rule, m order by rule, m")
    out = []
    for r in con.execute(q, (rule,) if rule else ()).fetchall():
        out.append({"rule": r[0], "month": f"{r[1][:4]}-{r[1][4:]}", "days": r[2], "bets": r[3], "hits": r[4],
                    "bet_total": r[5], "payout_total": r[6], "roi": round(r[6] / r[5], 3) if r[5] else None})
    return out


def write_daily_note(con, s: dict, rules: dict) -> None:
    d = from_hd(s["hd"])
    path = NOTES_DIR / "daily" / f"{d.isoformat()}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {d.isoformat()} 検証（rules v{s['rules_version']}）", ""]
    for rule, x in s["by_rule"].items():
        lines += [f"## {rule}", "",
                  f"- 対象 {x['races']}レース / {x['bets']}点 / 的中 {x['hits']} / 的中率 {x['hit_rate']}"
                  + (f" / 返還 {x['refunds']}点" if x["refunds"] else ""),
                  f"- 購入 ¥{x['bet_total']} / 払戻 ¥{x['payout_total']} / **回収率 {x['roi']}**"]
        m = month_to_date(con, s["hd"], rule)
        if m:
            lines.append(f"- **今月（{d.year}年{d.month}月）累計: {m['days']}日分 / 購入 ¥{m['bet']} / "
                         f"払戻 ¥{m['payout']} / 月次回収率 {m['roi']}**")
        for n in rules["evaluation"]["rolling_days"]:
            r = rolling(con, s["hd"], n, rule)
            if r:
                lines.append(f"- 直近{n}日: {r['days']}日分 / 購入 ¥{r['bet']} / 払戻 ¥{r['payout']} / 回収率 {r['roi']}")
        lines.append("")
    if len(s["by_rule"]) > 1:
        lines += ["## 合計（全ルール）", "",
                  f"- {s['bets']}点 / 的中 {s['hits']} / 購入 ¥{s['bet_total']} / 払戻 ¥{s['payout_total']} / 回収率 {s['roi']}", ""]
    lines += ["## 内訳", ""]
    for x in s["details"]:
        mark = "返" if x["refund"] else ("◎" if x["payout"] else "×")
        lines.append(f"- {mark} [{x['rule']}] {x['race']} {x['ticket_type']} {x['combo']} ¥{x['bet']} → ¥{x['payout']}")
    lines += ["", "## 気づき（夜に埋める）", "", "- ", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def backtest(con, rules: dict, d0, d1) -> dict:
    """lane1_a1（budget/bet/filters/lane1）を過去の出走表に当て、同じ買い方で回収率を見る。"""
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
    return _finish_backtest(agg, list(months.values()), rules, rules.get("name", "lane1_a1"))


def _finish_backtest(agg: dict, monthly_rows: list[dict], rules: dict, rule: str) -> dict:
    # 評価の単位は暦月。月次回収率 = その月の払戻合計 ÷ 購入合計（rules.yaml evaluation 参照）
    target = rules["evaluation"].get("target_monthly_roi", 1.0)
    for m in monthly_rows:
        m["roi"] = round(m["payout_total"] / m["bet_total"], 3) if m["bet_total"] else None
        m["win"] = m["roi"] is not None and m["roi"] > target
    agg["monthly"] = monthly_rows
    agg["months"] = len(monthly_rows)
    agg["win_months"] = sum(m["win"] for m in monthly_rows)
    rois = sorted(m["roi"] for m in monthly_rows if m["roi"] is not None)
    agg["monthly_roi_median"] = rois[len(rois) // 2] if rois else None
    agg["monthly_roi_worst"] = rois[0] if rois else None
    agg["hit_rate"] = round(agg["hits"] / agg["bets"], 3) if agg["bets"] else None
    agg["roi"] = round(agg["payout_total"] / agg["bet_total"], 3) if agg["bet_total"] else None
    agg["rules_version"] = rules["version"]
    agg["rule"] = rule
    return agg


def backtest_strategy(rules: dict, rule: str, d0, d1) -> dict:
    """モデル系（strategies.<rule>）: data/models/<dir>/history.parquet の OOS 予測（各月、その月より前で学習）に
    現行のしきい値を当てる。予測の無い月は出ない。"""
    import model as M
    cfg = rules["strategies"][rule]
    g = M.backtest_history(cfg, f"{d0:%Y-%m}", f"{d1:%Y-%m}")
    rows = [{"month": r.month, "bets": int(r.bets), "hits": int(r.hits), "bet_total": int(r.cost),
             "payout_total": int(round(r.payout))} for r in g.itertuples()]
    agg = {"bets": sum(r["bets"] for r in rows), "hits": sum(r["hits"] for r in rows),
           "bet_total": sum(r["bet_total"] for r in rows), "payout_total": sum(r["payout_total"] for r in rows),
           "note": "暦月単位（--from/--to の月を含む月全体）。history の OOS 予測を使う"}
    return _finish_backtest(agg, rows, rules, rule)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=to_hd(today()))
    ap.add_argument("--write-notes", action="store_true")
    ap.add_argument("--backtest", action="store_true")
    ap.add_argument("--monthly", action="store_true")
    ap.add_argument("--rule", help="--backtest / --monthly の対象ルール（既定: backtest は lane1_a1、monthly は全部）")
    ap.add_argument("--from", dest="date_from")
    ap.add_argument("--to", dest="date_to")
    args = ap.parse_args()

    rules = load_rules()
    con = connect()

    if args.backtest:
        if not (args.date_from and args.date_to):
            ap.error("--backtest には --from/--to が必要")
        d0, d1 = from_hd(args.date_from), from_hd(args.date_to)
        if args.rule and args.rule in (rules.get("strategies") or {}):
            res = backtest_strategy(rules, args.rule, d0, d1)
        else:
            res = backtest(con, rules, d0, d1)
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return 0
    if args.monthly:
        print(json.dumps(monthly(con, args.rule), ensure_ascii=False, indent=2))
        return 0

    s = evaluate(con, args.date)
    print(json.dumps(s, ensure_ascii=False, indent=2))
    if args.write_notes and s["bets"]:
        upsert(con, "verifications", [{"hd": s["hd"], "rules_version": s["rules_version"],
                                       **{k: s[k] for k in ("races", "bets", "hits", "bet_total", "payout_total",
                                                            "hit_rate", "roi")}}])
        upsert(con, "verifications_by_rule", [{k: v for k, v in x.items()} for x in s["by_rule"].values()])
        write_daily_note(con, s, rules)
        print(f"wrote notes/daily/{from_hd(args.date).isoformat()}.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
