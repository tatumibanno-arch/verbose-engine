"""翌日の出走表から rules.yaml に従って買い目を作り、picks に保存して notes/picks/ に書く。

使い方:
  python scripts/select_races.py                 # 翌日
  python scripts/select_races.py --date 20261103 --dry-run
  python scripts/select_races.py --date 20261103 --only tansho_ev_v1

ルールは2系統:
  * lane1_a1（rules.yaml の budget / bet / filters / lane1。v1 からある「1号艇A1」）
  * strategies.<名前>（例 tansho_ev_v1。1着確率モデル × 予想単勝配当。scripts/model.py）
picks.reason は「<ルール名> | 詳細」の形。verify.py はこの先頭でルール別に集計する。
買い目は「紙の上の購入（paper=1）」として記録する。実際の購入は人が朝に行う。
"""
from __future__ import annotations

import argparse
import itertools
import json

from common import NOTES_DIR, from_hd, load_rules, to_hd, tomorrow, venue_name
from db import connect, upsert

LANE_MARK = {1: "①", 2: "②", 3: "③", 4: "④", 5: "⑤", 6: "⑥"}
TICKET_JA = {"tansho": "単勝", "fukusho": "複勝", "nirentan": "2連単", "nirenfuku": "2連複",
             "sanrentan": "3連単", "sanrensho": "3連単", "sanrenfuku": "3連複"}
REASON_SEP = " | "


def rule_of(reason: str | None) -> str:
    """picks.reason → ルール名（区切りが無い古い行は lane1_a1）。"""
    if reason and REASON_SEP in reason:
        return reason.split(REASON_SEP, 1)[0].strip()
    return "lane1_a1"


def expand_formation(formation: str) -> list[str]:
    """'1-23-234' → ['1-2-3','1-2-4','1-3-2','1-3-4']（同じ艇の重複は除く）"""
    slots = [list(s) for s in formation.split("-")]
    combos = []
    for c in itertools.product(*slots):
        if len(set(c)) == len(c):
            combos.append("-".join(c))
    return combos


# ============================================================== lane1_a1（v1）
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


def spent_so_far(con) -> int:
    """これまでの購入累計（検証済みの日の合計）。"""
    r = con.execute("select coalesce(sum(bet_total), 0) from verifications").fetchone()
    return int(r[0])


def build_picks(hd: str, races: list[dict], rules: dict, spent: int = 0) -> list[dict]:
    b = rules["budget"]
    bet = rules["bet"]
    name = rules.get("name", "lane1_a1")
    if b.get("max_races"):
        races = races[: b["max_races"]]        # TODO: 自信度順に並べてから切る
    if not races:
        return []
    combos = expand_formation(bet["formation"])
    if b.get("allocation") == "equal" and b.get("daily_yen"):
        per_race = b["daily_yen"] // len(races)
        per_point = max(b["unit_yen"], (per_race // len(combos)) // b["unit_yen"] * b["unit_yen"])
    else:
        per_point = b["unit_yen"]              # fixed: 1点 unit_yen
    # 実験全体の上限を超える日は選定しない
    today_cost = per_point * len(combos) * len(races)
    if b.get("total_yen") and spent + today_cost > b["total_yen"]:
        print(f"予算上限: 累計 ¥{spent} + 今日 ¥{today_cost} > ¥{b['total_yen']} のため選定なし")
        return []
    picks = []
    for r in races:
        for c in combos:
            picks.append({
                "hd": hd, "jcd": r["jcd"], "rno": r["rno"],
                "ticket_type": bet["ticket_type"], "combination": c, "amount": per_point,
                "rules_version": rules["version"],
                "reason": f"{name}{REASON_SEP}lane1 {r['lane1_grade']}",
                "paper": 1,
            })
    return picks


# ============================================================== strategies（モデル系）
def build_strategy_picks(con, hd: str, name: str, cfg: dict, rules: dict, record: bool = True) -> tuple[list[dict], str | None]:
    """strategies.<name> の買い目。返り値 (picks, 選定できなかった理由 or None)。"""
    import model as M
    try:
        lanes = M.predict_day(cfg, hd, record=record)
    except FileNotFoundError as e:
        return [], str(e)
    sel = M.select_lanes(lanes, cfg)
    picks = []
    for r in sel.itertuples(index=False):
        picks.append({
            "hd": hd, "jcd": r.jcd, "rno": int(r.rno), "ticket_type": cfg["ticket_type"],
            "combination": str(int(r.lane)), "amount": int(cfg["unit_yen"]),
            "rules_version": rules["version"],
            "reason": f"{name}{REASON_SEP}p={r.p1:.3f} g={r.g:.0f} ev={r.ev:.3f} model={r.model_month}",
            "paper": int(cfg.get("paper", 1)),
        })
    return picks, None


def enabled_strategies(rules: dict, only: list[str] | None = None) -> dict:
    out = {}
    for name, cfg in (rules.get("strategies") or {}).items():
        if cfg.get("enabled", True) and (not only or name in only):
            out[name] = cfg
    return out


# ============================================================== ノート
def _parse_detail(reason: str) -> dict:
    d = {}
    for tok in reason.split(REASON_SEP, 1)[-1].split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            d[k] = v
    return d


def write_note(con, hd: str, picks: list[dict], rules: dict, sections: dict[str, dict]) -> None:
    """朝にスマホで見て買う用。ルールごとに見出し、場ごとに R・締切・艇番・選手・確率・予想配当。"""
    d = from_hd(hd)
    path = NOTES_DIR / "picks" / f"{d.isoformat()}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    info = {(r[0], r[1], r[2]): (r[3], r[4]) for r in con.execute(
        "select jcd, rno, lane, racer_name, close_time from entries where hd=?", (hd,)).fetchall()}
    total = sum(p["amount"] for p in picks)
    lines = [f"# {d.isoformat()} 選定（rules v{rules['version']}）", "",
             f"- 合計 {len(picks)}点 / ¥{total:,}（すべて紙上 paper=1 で記録。実際に買うかは人が決める）"]
    for name, s in sections.items():
        n = [p for p in picks if rule_of(p["reason"]) == name]
        lines.append(f"- {name}: {len(n)}点 / ¥{sum(p['amount'] for p in n):,}" + (f"（{s['skip']}）" if s.get("skip") else ""))
    lines.append("")
    for name, s in sections.items():
        ps = sorted([p for p in picks if rule_of(p["reason"]) == name],
                    key=lambda p: (p["jcd"], p["rno"], p["combination"]))
        lines += [f"## {name}　{s['title']}", ""]
        if s.get("desc"):
            lines += [s["desc"], ""]
        if not ps:
            lines += [f"- 候補なし{'（' + s['skip'] + '）' if s.get('skip') else ''}", ""]
            continue
        cur = None
        for p in ps:
            if p["jcd"] != cur:
                if cur is not None:
                    lines.append("")
                lines.append(f"### {venue_name(p['jcd'])}")
                cur = p["jcd"]
            tt = TICKET_JA.get(p["ticket_type"], p["ticket_type"])
            if p["ticket_type"] == "tansho":
                lane = int(p["combination"])
                racer, close = info.get((p["jcd"], p["rno"], lane), ("?", "?"))
                x = _parse_detail(p["reason"])
                lines.append(f"- {p['rno']}R（締切 {close}）{LANE_MARK.get(lane, lane)} {racer}　{tt} ¥{p['amount']}"
                             f"　確率 {float(x['p']):.0%} / 予想 {float(x['g']) / 100:.1f}倍 / 期待値 {float(x['ev']):.2f}")
            else:
                close = info.get((p["jcd"], p["rno"], 1), ("", "?"))[1]
                lines.append(f"- {p['rno']}R（締切 {close}）{tt} {p['combination']} ¥{p['amount']}")
        lines.append("")
    lines += ["## 実購入メモ（朝に人が書く）", "", "- 買った / 見送った / 変更した：", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=to_hd(tomorrow()))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", help="このルールだけ（カンマ区切り。lane1_a1 / tansho_ev_v1 …）")
    ap.add_argument("--no-history", action="store_true", help="モデルの予測を history に残さない（過去日の確認用）")
    args = ap.parse_args()

    rules = load_rules()
    con = connect()
    only = args.only.split(",") if args.only else None
    picks: list[dict] = []
    sections: dict[str, dict] = {}

    base = rules.get("name", "lane1_a1")
    if rules.get("enabled", True) and (not only or base in only):
        races = candidate_races(con, args.date, rules)
        p = build_picks(args.date, races, rules, spent=spent_so_far(con))
        picks += p
        sections[base] = {"title": f"1号艇{'/'.join(rules['lane1'].get('grades') or [])}・"
                                   f"{TICKET_JA.get(rules['bet']['ticket_type'], '')} {rules['bet']['formation']}",
                          "desc": "1号艇の級別で拾うルール（v1 から）。"}
        print(f"{args.date} {base}: 候補 {len(races)} レース → 買い目 {len(p)} 点 / ¥{sum(x['amount'] for x in p)}")

    for name, cfg in enabled_strategies(rules, only).items():
        p, skip = build_strategy_picks(con, args.date, name, cfg, rules, record=not (args.dry_run or args.no_history))
        picks += p
        sections[name] = {"title": f"{TICKET_JA.get(cfg['ticket_type'], cfg['ticket_type'])}・"
                                   f"1着確率{cfg['min_win_prob']:.0%}以上 かつ 期待値{cfg['min_ev']}以上",
                          "desc": "確率＝モデルの1着確率、予想＝前夜に推定した単勝配当（当たったときの倍率）、"
                                  "期待値＝確率×予想。オッズは見ていないので、締切オッズが予想より大きく低い艇は見送ってもよい"
                                  "（見送ったらメモに書く）。",
                          "skip": skip}
        print(f"{args.date} {name}: 買い目 {len(p)} 点 / ¥{sum(x['amount'] for x in p)}" + (f"（{skip}）" if skip else ""))

    if args.dry_run:
        print(json.dumps(picks, ensure_ascii=False, indent=2))
        return 0
    # 作り直すルールの紙上の買い目だけ消す（--only のときは他のルールを残す）
    for name in sections:
        rows = con.execute("select jcd, rno, ticket_type, combination, reason from picks where hd=? and paper=1",
                           (args.date,)).fetchall()
        for jcd, rno, tt, comb, reason in rows:
            if rule_of(reason) == name:
                con.execute("delete from picks where hd=? and jcd=? and rno=? and ticket_type=? and combination=? and paper=1",
                            (args.date, jcd, rno, tt, comb))
    con.commit()
    upsert(con, "picks", picks)
    # ノートは picks テーブルにある全ルール分で書く（--only でも他のルールの行を消さない）
    allp = [dict(zip(["hd", "jcd", "rno", "ticket_type", "combination", "amount", "reason"], r)) for r in con.execute(
        "select hd, jcd, rno, ticket_type, combination, amount, reason from picks where hd=? and paper=1", (args.date,))]
    write_note(con, args.date, allp, rules, sections)
    print(f"wrote notes/picks/{from_hd(args.date).isoformat()}.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
