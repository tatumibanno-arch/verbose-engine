"""締切前アラート（strategies.live_tansho_poolgap_v1）。舟券は買わない。知らせるだけ。

しくみ:
  前夜  prepare: 翌日のレース一覧（締切時刻・選手名・モデルの1着確率）を data/live/<日付>.csv に書く（nightly が実行）
  当日  run:     締切 lead_minutes 分前に単勝・3連単のオッズを取り、
                 3連単から出した1着確率 ÷ 単勝から出した1着確率 ≥ min_gap かつ 3連単からの1着確率 ≥ min_p3
                 の艇を LINE に流す。全レースのオッズと判定を data/live/log/<日付>_s<枠>.csv に残す（後で検証する材料）
  夜    report:  ログと払戻を突き合わせて紙上の成績を出す（nightly が実行）

使い方:
  python scripts/live_alert.py prepare --date 20261104
  python scripts/live_alert.py run                # 今の時刻から受け持ちの枠（rules.yaml slots）を自動で選ぶ
  python scripts/live_alert.py run --test         # 次に締切が来るレースを1つだけ判定して、テストとして LINE に送る
  python scripts/live_alert.py report --date 20261104
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

import requests

from common import DATA_DIR, JST, VENUES, load_rules, to_hd, today, tomorrow
from line_push import send_line

RULE = "live_tansho_poolgap_v1"
LIVE_DIR = DATA_DIR / "live"
LOG_DIR = LIVE_DIR / "log"
POLL_SECONDS = 20
LOG_FIELDS = ["hd", "jcd", "rno", "close", "checked_at", "lane", "name", "tansho_odds", "p_tansho", "p_3t",
              "gap", "p_model", "alert", "sent"]


def cfg() -> dict:
    return load_rules()["strategies"][RULE]


def now_jst() -> datetime:
    return datetime.now(JST)


def hm_to_dt(day: datetime, hm: str) -> datetime:
    h, m = map(int, hm.split(":"))
    return day.replace(hour=h, minute=m, second=0, microsecond=0)


# ------------------------------------------------------------------ prepare（前夜）
def cmd_prepare(hd: str) -> int:
    import pandas as pd

    from db import connect
    con = connect()
    e = pd.read_sql("select jcd, rno, lane, racer_name, grade, close_time, race_name from entries where hd=?", con, params=(hd,))
    if e.empty:
        print(f"{hd}: 番組表がありません", file=sys.stderr)
        return 1
    races = e.groupby(["jcd", "rno"]).agg(close=("close_time", "first"), race_name=("race_name", "first")).reset_index()
    for lane in range(1, 7):
        nm = e[e.lane == lane].set_index(["jcd", "rno"]).racer_name
        races[f"name_{lane}"] = [nm.get((j, r), "") for j, r in zip(races.jcd, races.rno)]
    hist = Path(DATA_DIR / "models" / "tansho_ev" / "history.parquet")
    if hist.exists():
        h = pd.read_parquet(hist)
        h = h[h.hd == hd].drop_duplicates(["jcd", "rno"], keep="last").set_index(["jcd", "rno"])
        for lane in range(1, 7):
            races[f"p_{lane}"] = [h[f"p1r_{lane}"].get((j, r)) if (j, r) in h.index else None
                                  for j, r in zip(races.jcd, races.rno)]
    races["n_a1"] = [int(((e.jcd == j) & (e.rno == r) & (e.grade == "A1")).sum()) for j, r in zip(races.jcd, races.rno)]
    w = load_rules()["strategies"].get(RULE_WATCH) or {}
    p1 = races["p_1"] if "p_1" in races else pd.Series([None] * len(races))
    races["watch"] = [int(bool(w.get("enabled")) and a >= w.get("min_a1", 99) and (p or 0) >= w.get("min_p1", 1))
                      for a, p in zip(races.n_a1, pd.to_numeric(p1, errors="coerce").fillna(0))]
    races = races.sort_values(["close", "jcd", "rno"])
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    out = LIVE_DIR / f"{hd}.csv"
    races.to_csv(out, index=False)
    print(f"{out}: {len(races)} races")
    return 0


def load_races(hd: str) -> list[dict]:
    path = LIVE_DIR / f"{hd}.csv"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["jcd"] = str(r["jcd"]).zfill(2)
        r["rno"] = int(r["rno"])
    return rows


# ------------------------------------------------------------------ 判定
ODDS_FIELDS = ["hd", "jcd", "rno", "checked_at", "ticket_type", "combination", "odds"]


def market_probs(hd: str, jcd: str, rno: int, session: requests.Session, snap: list | None = None):
    """(単勝オッズ[6], 単勝由来の1着確率[6], 3連単由来の1着確率[6])。取れなければ None。
    snap にリストを渡すと、取った単勝・2連単・3連単のオッズをすべて追加する（2連単・3連単の研究用）。"""
    from scrape_web import fetch, parse_odds2tf, parse_odds3t, parse_oddstf
    tf = parse_oddstf(fetch("oddstf", hd, jcd, rno, session) or "")
    t3 = parse_odds3t(fetch("odds3t", hd, jcd, rno, session) or "")
    t2 = parse_odds2tf(fetch("odds2tf", hd, jcd, rno, session) or "")
    if snap is not None:
        at = now_jst().strftime("%H:%M:%S")
        snap += [{"hd": hd, "jcd": jcd, "rno": rno, "checked_at": at, "ticket_type": tt, "combination": comb,
                  "odds": o} for tt, comb, o, _ in tf + t2 + t3 if tt in ("tansho", "nirensho", "sanrensho")]
    if not tf or not t3:
        return None
    odds = [None] * 6
    for tt, comb, o, _ in tf:
        if tt == "tansho":
            odds[int(comb) - 1] = o
    inv = [1 / o if o and o > 0 else 0.0 for o in odds]
    s = sum(inv)
    p_tan = [v / s if s else 0.0 for v in inv]
    inv3 = [0.0] * 6
    for _, comb, o, _ in t3:
        if o and o > 0:
            inv3[int(comb[0]) - 1] += 1 / o
    s3 = sum(inv3)
    p3 = [v / s3 if s3 else 0.0 for v in inv3]
    return odds, p_tan, p3


def judge(race: dict, hd: str, c: dict, session: requests.Session, checked_at: datetime, snap: list | None = None):
    res = market_probs(hd, race["jcd"], race["rno"], session, snap)
    rows = []
    if res is None:
        return rows
    odds, p_tan, p3 = res
    for i in range(6):
        gap = p3[i] / p_tan[i] if p_tan[i] > 0 else None
        alert = bool(c.get("enabled", True) and odds[i] and odds[i] > 0 and gap is not None and gap >= c["min_gap"] and p3[i] >= c["min_p3"])
        pm = race.get(f"p_{i + 1}")
        rows.append({
            "hd": hd, "jcd": race["jcd"], "rno": race["rno"], "close": race["close"],
            "checked_at": checked_at.strftime("%H:%M:%S"), "lane": i + 1, "name": race.get(f"name_{i + 1}", ""),
            "tansho_odds": odds[i], "p_tansho": round(p_tan[i], 4), "p_3t": round(p3[i], 4),
            "gap": round(gap, 3) if gap is not None else None, "p_model": pm or None,
            "alert": int(alert), "sent": 0,
        })
    return rows


def message(race: dict, rows: list[dict], test: bool) -> str:
    head = "🧪【テスト】" if test else "🚤【紙上】"
    v = VENUES.get(race["jcd"], race["jcd"])
    lines = [f"{head}{race['close']}締切 {v}{race['rno']}R"]
    hits = [r for r in rows if r["alert"]]
    show = hits or ([max(rows, key=lambda r: r["gap"] or 0)] if test else [])
    for r in show:
        lines.append(f"単勝 {r['lane']}号艇 {r['name']}  オッズ {r['tansho_odds']}倍（{r['checked_at'][:5]}時点）")
        lines.append(f"3連単から見た1着率 {r['p_3t']:.0%} / 単勝の人気 {r['p_tansho']:.0%}（ずれ {r['gap']:.1f}倍）")
    if test and not hits:
        lines.append("※条件（ずれ・確率の下限）には届いていない。テスト表示です")
    return "\n".join(lines)


def append_log(path: Path, rows: list[dict], fields: list[str] = LOG_FIELDS) -> None:
    new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerows(rows)


# ------------------------------------------------------------------ 2連単（3連単とのずれ）
RULE_2T = "live_2t_poolgap_try"
PICK2_FIELDS = ["hd", "jcd", "rno", "close", "checked_at", "combination", "odds", "q2", "p3", "ratio", "ev", "sent"]


def picks_2t(snap: list[dict], c2: dict) -> list[dict]:
    """3連単の市場より2連単の方が安い（人気が低い）2連単。ev ≈ 3連単由来の確率 × 2連単オッズ。"""
    o2 = {x["combination"]: x["odds"] for x in snap if x["ticket_type"] == "nirensho" and x["odds"]}
    o3 = {x["combination"]: x["odds"] for x in snap if x["ticket_type"] == "sanrensho" and x["odds"]}
    i2 = {k: 1 / v for k, v in o2.items() if v > 0}
    i3 = {k: 1 / v for k, v in o3.items() if v > 0}
    s2, s3 = sum(i2.values()), sum(i3.values())
    if not s2 or not s3:
        return []
    out = []
    for pair, v in i2.items():
        q2 = v / s2
        p3 = sum(w for k, w in i3.items() if k.startswith(pair + "-")) / s3
        if p3 >= c2["min_p3"] and p3 / q2 >= c2["min_ratio"]:
            out.append({"combination": pair, "odds": o2[pair], "q2": round(q2, 4), "p3": round(p3, 4),
                        "ratio": round(p3 / q2, 3), "ev": round(p3 * o2[pair], 3)})
    return sorted(out, key=lambda x: -x["ev"])[: c2["max_per_race"]]


def message_2t(race: dict, picks: list[dict], at: str) -> str:
    v = VENUES.get(race["jcd"], race["jcd"])
    lines = [f"🎯【試し・2連単】{race['close']}締切 {v}{race['rno']}R"]
    for p in picks:
        lines.append(f"2連単 {p['combination']}  {p['odds']}倍（{at}時点）")
        lines.append(f"3連単から見た確率 {p['p3']:.0%} / 2連単の人気 {p['q2']:.0%}（推定期待値 {p['ev']:.2f}）")
    lines.append("※検証前の試し。根拠は未確認")
    return "\n".join(lines)


# ------------------------------------------------------------------ 候補レースの直前レポート
RULE_WATCH = "live_watch_report"
WATCH_LAST_TRY_MINUTES = 7      # 展示がまだ出ていなくても、この分数を切ったら出ている分だけで送る
CIRCLED = "①②③④⑤⑥"


def watch_list_text(hd: str) -> str:
    """朝の一覧（nightly の LINE に入れる）。"""
    rows = [r for r in load_races(hd) if str(r.get("watch")) == "1"]
    day = f"{int(hd[4:6])}/{int(hd[6:])}"
    if not rows:
        return f"🔎 {day} 直前レポートの候補レース: なし"
    out = [f"🔎 {day} 直前レポートの候補（A1多め・1号艇有力）{len(rows)}レース。締切12分前ごろに展示とオッズを送ります"]
    for r in rows:
        p = float(r.get("p_1") or 0)
        out.append(f"{r['close']} {VENUES.get(r['jcd'], r['jcd'])}{r['rno']}R {r.get('race_name', '')} ①{r.get('name_1', '')} 1着予想{p:.0%}")
    return "\n".join(out)


def load_deep(hd: str, jcd: str) -> dict | None:
    p = LIVE_DIR / f"{hd}_deep_{jcd}.json"
    if not p.exists():
        return None
    import json
    return json.loads(p.read_text(encoding="utf-8"))


def pick_lines(d: dict, odds: dict, rank: dict) -> list[str]:
    """展開予想と、本命・穴の目安（合成オッズつき）。"""
    pk = (load_rules()["strategies"].get(RULE_WATCH) or {}).get("pick") or {}
    combos = ["-".join(str(x) for x in p) for p in
              [(a, b, c) for a in range(1, 7) for b in range(1, 7) for c in range(1, 7) if len({a, b, c}) == 3]]
    P = dict(zip(combos, d["p120"]))
    inv = {c: 1 / o for c, o in odds.items() if o}
    s3 = sum(inv.values())
    q = {c: inv.get(c, 0) / s3 for c in combos} if s3 else {c: 0 for c in combos}
    f = {}
    for L in d["lanes"]:
        k = 1.0
        r = rank.get(L["lane"])
        if r and r <= 2:
            k *= pk.get("ex_top", 1.1)
        elif r and r >= 5:
            k *= pk.get("ex_bottom", 0.9)
        if L.get("flag") == "◎":
            k *= pk.get("good", 1.1)
        elif L.get("flag") == "↑":
            k *= pk.get("up", 1.05)
        elif L.get("flag") == "△":
            k *= pk.get("bad", 0.85)
        f[L["lane"]] = k
    w = pk.get("model_weight", 0.5)
    adj = {}
    for c in combos:
        a, b, cc = map(int, c.split("-"))
        adj[c] = (w * P[c] + (1 - w) * q[c]) * f[a] * f[b] ** 0.5 * f[cc] ** 0.25
    t = sum(adj.values())
    adj = {c: v / t for c, v in adj.items()}
    win = [sum(v for c, v in adj.items() if c.startswith(f"{i}-")) for i in range(1, 7)]
    order = sorted(range(6), key=lambda i: -win[i])
    kind = {0: "逃げ", 1: "差し", 2: "まくり・まくり差し", 3: "まくり・まくり差し", 4: "外からの突き抜け", 5: "外からの突き抜け"}
    out = [f"■展開予想: {CIRCLED[order[0]]}の{kind[order[0]]}が本線（1着{win[order[0]]:.0%}）、"
           f"対抗は{CIRCLED[order[1]]}の{kind[order[1]]}（{win[order[1]]:.0%}）、{CIRCLED[order[2]]}（{win[order[2]]:.0%}）"]
    hon = sorted([c for c in combos if c in odds], key=lambda c: -adj[c])[: pk.get("honmei_points", 3)]
    if hon:
        syn = 1 / sum(1 / odds[c] for c in hon)
        out.append(f"■本命 {len(hon)}点: " + " / ".join(f"{c}（{odds[c]}倍）" for c in hon) +
                   f" → 合成{syn:.1f}倍・来る確率{sum(adj[c] for c in hon):.0%}")
    ana = [c for c in combos if c in odds and c not in hon and pk.get("ana_min_odds", 20) <= odds[c] <= pk.get("ana_max_odds", 100)
           and q[c] > 0 and adj[c] >= pk.get("ana_min_prob", 0.01) and adj[c] / q[c] >= pk.get("ana_min_ratio", 1.2)]
    ana = sorted(ana, key=lambda c: -adj[c] * odds[c])[: pk.get("ana_points", 3)]
    if ana:
        syn = 1 / sum(1 / odds[c] for c in ana)
        out.append(f"■穴 {len(ana)}点: " + " / ".join(f"{c}（{odds[c]}倍）" for c in ana) +
                   f" → 合成{syn:.1f}倍・来る確率{sum(adj[c] for c in ana):.0%}（市場の見立て{sum(q[c] for c in ana):.0%}）")
    else:
        out.append("■穴: 条件（20〜100倍・市場より1.2倍以上高く見える）に合う目なし")
    return out


def deep_report(race: dict, deep: dict, bi: list, t3: list) -> str:
    """前夜の材料（選手・モーター・コース別・今節・前検メモ・モデル）＋ 直前（展示・風・オッズ）。"""
    v = VENUES.get(race["jcd"], race["jcd"])
    d = deep["races"].get(str(int(race["rno"])))
    lines = [f"🔎【{v}{race['rno']}R {race['close']}締切 {d.get('race_name', '')}】"]
    bmap = {b["lane"]: b for b in (bi or [])}
    ex = [b["exhibition_time"] for b in bi or [] if b.get("exhibition_time")]
    rank = {b["lane"]: 1 + sum(1 for t in ex if b.get("exhibition_time") and t < b["exhibition_time"]) for b in bi or []}
    # 風
    w0 = (bi or [{}])[0] if bi else {}
    ws = w0.get("wind_speed")
    if ws is not None:
        bins = deep["wind_bins"]
        k = next((i for i in range(len(bins) - 1) if bins[i] < ws <= bins[i + 1]), 0)
        l1 = deep["wind_l1"].get(str(k))
        calm = deep["wind_l1"].get("0")
        lines.append(f"風{ws}m 波{w0.get('wave_height')}cm → この場でこの風速の1号艇1着率 {l1:.0%}（無風 {calm:.0%}）" if l1 else f"風{ws}m 波{w0.get('wave_height')}cm")
    # 進入
    st = sorted([b for b in bi or [] if b.get("start_course")], key=lambda x: x["start_course"])
    if st:
        course = "".join(str(b["lane"]) for b in st)
        lines.append(f"スタ展 進入{course}" + ("（枠なり）" if course == "123456" else "（★枠なりでない）"))
    # 展開（前夜モデルの1着確率）
    p1 = d.get("p1")
    if p1:
        lines.append(f"展開（前夜モデル）: 逃げ{p1[0]:.0%} 差し{p1[1]:.0%} まくり系{p1[2] + p1[3]:.0%}（③{p1[2]:.0%}④{p1[3]:.0%}） 外{p1[4] + p1[5]:.0%}")
    # 各艇
    for L in d["lanes"]:
        b = bmap.get(L["lane"], {})
        parts = [f"{CIRCLED[L['lane'] - 1]}{L['name'].replace(' ', '')}({L['grade']}){L['flag']}"]
        if L.get("c_win") is not None:
            parts.append(f"{L['lane']}C勝{L['c_win']:.0f}% ST{str(L['c_st']).lstrip('0')}")
        parts.append(f"M{L['motor']}" + (f" 当地3内{L['m_local3']:.0%}" if L.get("m_local3") is not None else ""))
        if b.get("exhibition_time"):
            parts.append(f"展{b['exhibition_time']}({rank.get(L['lane'])}位)")
        if b.get("start_st_raw"):
            parts.append(f"スタ展{b['start_st_raw']}")
        if b.get("parts_changed"):
            parts.append("交換:" + "・".join(b["parts_changed"]))
        if L.get("meet") and L["meet"] != "-":
            parts.append(f"今節{L['meet']}")
        if L.get("flag_note"):
            parts.append(f"[{L['flag_note']}]")
        lines.append(" ".join(parts))
    # オッズ × モデル
    if t3:
        odds = {c: o for _, c, o, _ in t3 if o and o > 0}
        inv = {c: 1 / o for c, o in odds.items()}; s3 = sum(inv.values())
        p1m = sum(w for c, w in inv.items() if c.startswith("1-")) / s3 if s3 else 0
        lines.append(f"3連単から見た1号艇1着率 {p1m:.0%}" + (f"（モデル {p1[0]:.0%}）" if p1 else ""))
        if d.get("top"):
            ev = [f"{c} {odds.get(c, '-')}倍 確率{p:.1%} 期待値{p * odds[c]:.2f}" for c, p in d["top"][:6] if c in odds]
            lines.append("モデル上位×今のオッズ:")
            lines += ["  " + x for x in ev]
        pop = sorted(odds.items(), key=lambda x: x[1])[:5]
        lines.append("人気上位: " + " / ".join(f"{c} {o}倍" for c, o in pop))
        if d.get("p120"):
            lines += pick_lines(d, odds, rank)
    lines.append("◎=好気配 ↑=上向き △=劣勢気配。期待値・本命・穴はモデルと目安の補正による参考（過去の検証では勝てていない）")
    return "\n".join(lines)


def watch_report(race: dict, hd: str, cw: dict, session: requests.Session) -> str | None:
    from scrape_web import fetch, parse_beforeinfo, parse_odds3t
    bi = parse_beforeinfo(fetch("beforeinfo", hd, race["jcd"], race["rno"], session) or "")
    t3 = parse_odds3t(fetch("odds3t", hd, race["jcd"], race["rno"], session) or "")
    v = VENUES.get(race["jcd"], race["jcd"])
    lines = [f"🔎【直前レポート】{race['close']}締切 {v}{race['rno']}R {race.get('race_name', '')}（A1×{race.get('n_a1', '?')}）"]
    has_ex = bool(bi) and any(b.get("exhibition_time") for b in bi)
    if not has_ex and not race.get("_final_try"):
        return None                       # 展示前。少し待ってから取り直す
    deep = load_deep(hd, race["jcd"])
    if deep and str(int(race["rno"])) in deep.get("races", {}):
        return deep_report(race, deep, bi if has_ex else [], t3)
    if has_ex:
        bi = sorted(bi, key=lambda x: x["lane"])
        ex = [b["exhibition_time"] for b in bi if b.get("exhibition_time")]
        rank = {b["lane"]: 1 + sum(1 for t in ex if b.get("exhibition_time") and t < b["exhibition_time"]) for b in bi}
        lines.append("展示: " + " ".join(f"{CIRCLED[b['lane']-1]}{b.get('exhibition_time') or '-'}({rank[b['lane']]}位)" for b in bi))
        st = sorted([b for b in bi if b.get("start_course")], key=lambda x: x["start_course"])
        if st:
            lines.append("スタ展: 進入 " + "".join(str(b["lane"]) for b in st) + " / ST " +
                         " ".join(f"{CIRCLED[b['lane']-1]}{b.get('start_st_raw') or '-'}" for b in st))
        parts = [f"{CIRCLED[b['lane']-1]}{'・'.join(b['parts_changed'])}" for b in bi if b.get("parts_changed")]
        if parts:
            lines.append("部品交換: " + " ".join(parts))
        lines.append("チルト: " + " ".join(f"{CIRCLED[b['lane']-1]}{b.get('tilt')}" for b in bi))
        w = bi[0]
        if w.get("wind_speed") is not None:
            lines.append(f"水面: 風{w.get('wind_speed')}m 波{w.get('wave_height')}cm")
    else:
        lines.append("（直前情報はまだ出ていません）")
    p1 = race.get("p_1")
    if p1:
        lines.append(f"モデルの1号艇1着率 {float(p1):.0%}")
    if t3:
        one = sorted([(o, c) for tt, c, o, _ in t3 if c.startswith("1-") and o and o > 0])[: cw["top_combos"]]
        inv = [1 / o for o, _ in t3_all] if (t3_all := [(o, c) for tt, c, o, _ in t3 if o and o > 0]) else []
        s3 = sum(inv)
        p1m = sum(1 / o for o, c in t3_all if c.startswith("1-")) / s3 if s3 else 0
        lines.append(f"3連単から見た1号艇1着率 {p1m:.0%}")
        lines.append("1着①の人気上位: " + " / ".join(f"{c} {o}倍" for o, c in one))
        syn = []
        acc = 0.0
        for k, (o, c) in enumerate(one, 1):
            acc += 1 / o
            if k in (1, 2, 3, 4, 6):
                syn.append(f"{k}点 {1 / acc:.1f}倍")
        lines.append("上位から買った時の合成オッズ: " + " / ".join(syn))
    else:
        lines.append("（オッズが取れませんでした）")
    return "\n".join(lines)


# ------------------------------------------------------------------ run（当日）
def pick_slot(c: dict, now: datetime) -> int | None:
    for i, (a, b) in reversed(list(enumerate(c["slots"]))):  # 枠が重なる時間は後の枠を選ぶ
        if hm_to_dt(now, a) - timedelta(minutes=45) <= now < hm_to_dt(now, b):
            return i
    return None


def cmd_run(slot: int | None, test: bool, dry: bool) -> int:
    c = cfg()
    now = now_jst()
    hd = to_hd(today())
    races = load_races(hd)
    if not races:
        msg = f"⚠️ {hd} の締切前アラート: レース一覧（data/live/{hd}.csv）がありません。前夜の処理を確認してください"
        print(msg)
        if not dry and slot in (None, 0):
            send_line(msg)
        return 1
    session = requests.Session()

    if test:
        later = [r for r in races if hm_to_dt(now, r["close"]) > now + timedelta(minutes=1)]
        if not later:
            print("今日はもう締切前のレースがありません")
            return 0
        race = later[0]
        rows = judge(race, hd, c, session, now_jst())
        text = message(race, rows, test=True) if rows else f"🧪【テスト】{race['close']} {race['jcd']} {race['rno']}R オッズが取れませんでした"
        print(text)
        if not dry:
            if send_line(text):
                print("LINE に送信しました")
            else:
                print("WARN: LINE_USER_ID が空なので送っていません（GitHub の Secrets を確認）")
                return 1
        return 0

    if slot is None:
        slot = pick_slot(c, now)
    if slot is None:
        print("受け持ちの枠がない時刻です")
        return 0
    s_from, s_to = c["slots"][slot]
    mine = [r for r in races if s_from <= r["close"] < s_to]
    win_from, win_to = c["alert_window"]
    log_path = LOG_DIR / f"{hd}_s{slot}.csv"
    done: set = set()
    if log_path.exists():  # 再実行したときに同じレースを二重に知らせない
        with open(log_path, encoding="utf-8") as f:
            done = {(r["jcd"], int(r["rno"])) for r in csv.DictReader(f)}
    end = hm_to_dt(now, s_to) + timedelta(minutes=2)
    print(f"{hd} slot {slot} {s_from}-{s_to}: races {len(mine)} (done {len(done)})", flush=True)
    n_alert = 0
    cw = load_rules()["strategies"].get(RULE_WATCH) or {}
    watch_path = LOG_DIR / f"{hd}_s{slot}_watch.csv"
    watch_done: set = set()
    if watch_path.exists():
        with open(watch_path, encoding="utf-8") as f:
            watch_done = {(r["jcd"], int(r["rno"])) for r in csv.DictReader(f)}
    deep_v = {str(x).zfill(2) for x in cw.get("deep_venues") or []}
    watch = [r for r in mine if str(r.get("watch")) == "1" or r["jcd"] in deep_v] if cw.get("enabled") else []
    with ThreadPoolExecutor(4) as ex:
        while now_jst() < end:
            t = now_jst()
            for r in watch:
                key = (r["jcd"], r["rno"])
                if key in watch_done:
                    continue
                left = (hm_to_dt(t, r["close"]) - t).total_seconds() / 60
                if left < c["min_lead_minutes"]:
                    watch_done.add(key)
                elif left <= cw["report_minutes"]:
                    r["_final_try"] = left <= WATCH_LAST_TRY_MINUTES
                    try:
                        text = watch_report(r, hd, cw, session)
                    except Exception as e:  # noqa: BLE001
                        print(f"WARN watch {key}: {e}", flush=True)
                        continue
                    if text is None:          # 展示がまだ。次の見回りで取り直す
                        continue
                    watch_done.add(key)
                    print(text, flush=True)
                    sent = 0
                    if not dry and cw.get("send_line"):
                        try:
                            sent = int(bool(send_line(text)))
                        except Exception as e:  # noqa: BLE001
                            print(f"WARN LINE: {e}", flush=True)
                    if not dry:
                        append_log(watch_path, [{"jcd": r["jcd"], "rno": r["rno"], "close": r["close"],
                                                 "sent_at": now_jst().strftime("%H:%M:%S"), "sent": sent,
                                                 "text": text.replace("\n", " | ")}],
                                   ["jcd", "rno", "close", "sent_at", "sent", "text"])
            due = []
            for r in mine:
                key = (r["jcd"], r["rno"])
                if key in done:
                    continue
                left = (hm_to_dt(t, r["close"]) - t).total_seconds() / 60
                if left < c["min_lead_minutes"]:
                    done.add(key)          # 間に合わない（ジョブの開始が遅れたなど）
                elif left <= c["lead_minutes"]:
                    due.append(r)
            if not due:
                if all((r["jcd"], r["rno"]) in done for r in mine) and all((r["jcd"], r["rno"]) in watch_done for r in watch):
                    break
                time.sleep(POLL_SECONDS)
                continue
            snaps = {}
            results = list(ex.map(lambda r: (r, judge(r, hd, c, session, now_jst(), snaps.setdefault((r["jcd"], r["rno"]), []))), due))
            c2 = load_rules()["strategies"].get(RULE_2T) or {}
            for race, rows in results:
                snap = snaps.get((race["jcd"], race["rno"])) or []
                if not dry and snap:
                    append_log(LOG_DIR / f"{hd}_s{slot}_odds.csv", snap, ODDS_FIELDS)
                if c2.get("enabled") and snap:
                    p2 = picks_2t(snap, c2)
                    if p2:
                        at = snap[0]["checked_at"][:5]
                        text = message_2t(race, p2, at)
                        print(text, flush=True)
                        sent = 0
                        if not dry and c2.get("send_line") and win_from <= race["close"] < win_to:
                            try:
                                sent = int(bool(send_line(text)))
                            except Exception as e:
                                print(f"WARN LINE: {e}", flush=True)
                        if not dry:
                            append_log(LOG_DIR / f"{hd}_s{slot}_2t.csv",
                                       [dict(p, hd=hd, jcd=race["jcd"], rno=race["rno"], close=race["close"],
                                             checked_at=snap[0]["checked_at"], sent=sent) for p in p2], PICK2_FIELDS)
                done.add((race["jcd"], race["rno"]))
                if not rows:
                    print(f"  {race['close']} {race['jcd']} {race['rno']}R: オッズ取得失敗", flush=True)
                    continue
                top = max(rows, key=lambda r: r["gap"] or 0)
                print(f"  {race['close']} {race['jcd']} {race['rno']}R: 最大ずれ {top['lane']}号艇 {top['gap']} "
                      f"(3連単 {top['p_3t']:.0%} / 単勝 {top['p_tansho']:.0%}, {top['tansho_odds']}倍)"
                      f"{' → アラート' if any(r['alert'] for r in rows) else ''}", flush=True)
                if c.get("send_line", True) and any(r["alert"] for r in rows) and win_from <= race["close"] < win_to:
                    text = message(race, rows, test=False)
                    print(text, flush=True)
                    if not dry:
                        try:
                            if send_line(text):
                                for r in rows:
                                    r["sent"] = r["alert"]
                        except Exception as e:  # 1通の失敗で全体を止めない
                            print(f"WARN LINE: {e}", flush=True)
                    n_alert += 1
                if not dry:
                    append_log(log_path, rows)
    print(f"slot {slot} done: alerts {n_alert}", flush=True)
    return 0


# ------------------------------------------------------------------ report（夜）
def cmd_report(hd: str) -> int:
    import pandas as pd

    from db import connect
    logs = sorted(p for p in LOG_DIR.glob(f"{hd}_s*.csv") if not p.name.endswith(("_odds.csv", "_2t.csv", "_watch.csv")))
    if not logs:
        print(f"{hd}: ログなし")
        return 0
    d = pd.concat([pd.read_csv(p, dtype={"jcd": str}) for p in logs])
    d["jcd"] = d.jcd.str.zfill(2)
    con = connect()
    pay = pd.read_sql("select jcd, rno, combination, payout from payouts where hd=? and ticket_type='tansho'",
                      con, params=(hd,))
    pay["lane"] = pay.combination.astype(int)
    d = d.merge(pay[["jcd", "rno", "lane", "payout"]], on=["jcd", "rno", "lane"], how="left")
    d["ret"] = d.payout.fillna(0)
    a = d[d.alert == 1]
    c = cfg()
    lines = [f"# 締切前アラート {hd}（{RULE}、紙上）", "",
             f"- 判定したレース {d.groupby(['jcd', 'rno']).ngroups} / アラート {len(a)}点（送信 {int(a.sent.sum())}）",
             f"- 的中 {int((a.ret > 0).sum())} / 購入 {len(a) * c['unit_yen']}円 / 払戻 {int(a.ret.sum() * c['unit_yen'] / 100)}円"
             + (f" / 回収率 {a.ret.sum() / (len(a) * 100):.2f}" if len(a) else ""), ""]
    for r in a.sort_values("close").itertuples():
        lines.append(f"- {r.close} {VENUES.get(r.jcd, r.jcd)}{r.rno}R {r.lane}号艇 {r.name} "
                     f"{r.tansho_odds}倍 ずれ{r.gap} → {'◎' + str(int(r.ret)) + '円' if r.ret > 0 else '×'}")
    text = "\n".join(lines)
    out = Path(__file__).resolve().parent.parent / "notes" / "live" / f"{hd}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare", "run", "report", "watchlist"])
    ap.add_argument("--date")
    ap.add_argument("--slot", type=int)
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.cmd == "prepare":
        return cmd_prepare(a.date or to_hd(tomorrow()))
    if a.cmd == "watchlist":
        print(watch_list_text(a.date or to_hd(tomorrow())))
        return 0
    if a.cmd == "report":
        return cmd_report(a.date or to_hd(today()))
    return cmd_run(a.slot, a.test, a.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
