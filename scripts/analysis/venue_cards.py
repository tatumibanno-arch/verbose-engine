"""場ごとのカルテ（notes/venues/）を作る。読み取り専用・再実行可。

使い方:
    python scripts/analysis/venue_cards.py            # notes/venues/ を作り直す（約2〜3分）

出力:
    notes/venues/README.md          … 24場の一覧（難易度・判定）と計算方法
    notes/venues/{jcd}_{場名}.md     … 1場1ページのカルテ

方針:
  * 期間は roi_scan と同じ: 発見用 2023-11-01〜2025-10-31 / 検証用 2025-11-01〜2026-10-31。
  * 読み込み・返還の扱い・回収率の計算は scripts/analysis/roi_scan.py の load() と Book をそのまま使う
    （欠場艇を含む買い目は返還＝買わなかった扱い、position NULL でも ST がある艇は外れ扱い）。
  * 判定（狙う／様子見／買わない）は「発見用と検証用で同じ向き」の数字だけを根拠にする。採用ではなく仮説。
  * 当日情報（風・波）は参考として載せるだけで、判定・仮説には使わない。
  * DB は読み取り専用で開く。
"""
from __future__ import annotations

import math
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))
from common import VENUES  # noqa: E402
import roi_scan as rs  # noqa: E402
import venue_course as vc  # noqa: E402

OUT_DIR = ROOT / "notes" / "venues"
JCDS = [f"{i:02d}" for i in range(1, 25)]
NAT = "全国"

# ---- 判定・仮説のしきい値（分析用。選定ルールではないので rules.yaml には置かない）
EDGE = 0.03          # 定番の買い方で「全国との差」がこれ以上（両期間とも）なら意味のある差とみなす
EDGE_Z = 2.0         # 3年通算の全国差の z 値がこれ以上
HYP_MIN_D, HYP_MIN_V = 200, 100   # 仮説セルの最低レース数（発見／検証）
HYP_GOOD = 0.95      # 単勝1 が両期間ともこれ以上 → 「損益分岐に近い」仮説
EXA_MIN = 0.85       # 出やすい2連単の出目を仮説に載せる最低回収率（両期間）
HYP_BAD = 0.82       # 単勝1 が両期間ともこれ以下 → 「1号艇を買わない」仮説
UPSET = 10000        # 万舟（円）
FAV_PAY = 200        # 「人気通り」の近似: 単勝払戻が200円未満（2倍未満の艇が勝った）

STRATS = [  # (key, 表示名, 券種, 買い目)
    ("t1", "単勝 1", "tansho", ["1"]),
    ("e12", "2連単 1-2", "nirensho", ["1-2"]),
    ("e1x", "2連単 1-全(5点)", "nirensho", [f"1-{b}" for b in range(2, 7)]),
    ("s123", "3連単 1-2-3", "sanrensho", ["1-2-3"]),
    ("s1xx", "3連単 1-全-全(20点)", "sanrensho",
     [f"1-{b}-{c}" for b in range(2, 7) for c in range(2, 7) if b != c]),
]
CORE = ["t1", "e12", "s123"]   # 「定番」の判定に使う3つ（1号艇の頭・本線）


def vn(j):
    return VENUES[j]


# ================================================================ 読み込み
def load_all():
    con = rs.connect()
    races, void, _void_all, pay, ent = rs.load(con)
    t0 = time.time()
    key = races.set_index(["hd", "jcd", "rno"])["rid"]
    res = pd.read_sql(
        "select hd, jcd, rno, lane, position pos, json_extract(data,'$.entrance_position') course, "
        "json_extract(data,'$.exhibition_time') ex from results where hd between ? and ?",
        con, params=(rs.DISC[0], rs.VAL[1]))
    res = res.join(key, on=["hd", "jcd", "rno"], how="inner")
    odds = pd.read_sql(
        "select hd, jcd, rno, combination comb, odds from odds where ticket_type='tansho' "
        "and hd between ? and ?", con, params=(rs.DISC[0], rs.VAL[1]))
    con.close()
    odds = odds.join(key, on=["hd", "jcd", "rno"], how="inner")
    print(f"[venue_cards] results/odds loaded ({time.time()-t0:.0f}s)", file=sys.stderr)
    return races, void, pay, ent, res, odds


def build_race_table(races, pay, ent, res):
    R = races.copy()
    R["month"] = R.hd.str[:6]
    R["per"] = np.where(R.hd <= rs.DISC[1], "d", "v")
    R["cat"] = R.race_name.map(rs.race_cat).fillna("その他(愛称戦)")
    R["rband"] = pd.cut(R.rno, [0, 4, 8, 12], labels=["1-4R", "5-8R", "9-12R"]).astype(str)
    # 払戻: 出目と配当（同着は先頭の1件）
    for tt, col in [("sanrensho", "tri"), ("nirensho", "exa"), ("tansho", "win")]:
        p = pay[pay.tt == tt].sort_values("payout").drop_duplicates("rid")
        R = R.merge(p[["rid", "comb", "payout"]].rename(columns={"comb": col, "payout": col + "_pay"}),
                    on="rid", how="left")
    R["winlane"] = pd.to_numeric(R.tri.str[0], errors="coerce")
    # 進入
    r = res.copy()
    r["course"] = pd.to_numeric(r.course, errors="coerce")
    w = r[r.pos == 1].drop_duplicates("rid")[["rid", "course"]].rename(columns={"course": "wincourse"})
    R = R.merge(w, on="rid", how="left")
    rc = r[r.course.notna()].copy()
    rc["same"] = (rc.course == rc.lane)
    rc["mz"] = (rc.course < rc.lane)
    g = rc.groupby("rid").agg(wakunari=("same", "all"), maezuke=("mz", "any"), nb=("lane", "size"))
    R = R.merge(g, on="rid", how="left")
    # 1号艇の級と勝率順位（番組表）
    W, has = rs.wide_entries(len(races), ent)
    R["g1"] = W["grade"][R.rid.to_numpy(), 0]
    nwr = W["nwr"][R.rid.to_numpy()]
    R["l1top"] = np.where(has[R.rid.to_numpy()], nwr[:, 0] >= np.nanmax(nwr[:, 1:], axis=1), np.nan)
    R = R.sort_values("rid").reset_index(drop=True)
    return R


# ================================================================ 集計の部品
def roi_parts(cost, ret, mask):
    c, r = cost[mask], ret[mask]
    C = c.sum()
    if C == 0:
        return np.nan, np.nan, 0
    roi = r.sum() / C
    se = math.sqrt(((r - roi * c) ** 2).sum()) / C
    return roi, se, int(mask.sum())


def win_months(cost, ret, months, mask):
    df = pd.DataFrame({"m": months[mask], "c": cost[mask], "r": ret[mask]}).groupby("m").sum()
    df = df[df.c > 0]
    return int((df.r / df.c > 1.0).sum()), len(df)


def _sf(z):
    """標準正規の上側確率。"""
    return 0.5 * math.erfc(z / math.sqrt(2))


def entropy_bits(series):
    p = series.value_counts(normalize=True).to_numpy()
    return float(-(p * np.log2(p)).sum())


def pct(x, d=1):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100*x:.{d}f}"


def r3(x):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"


def r2(x):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.2f}"


def yen(x):
    return f"{int(round(x)):,}円"


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(out)


# ================================================================ 本体
def compute(R, book, odds):
    months = R.month.to_numpy()
    jc = R.jcd.to_numpy()
    per = R.per.to_numpy()
    allm = np.ones(len(R), bool)
    S = {k: book.static(tt, cs) for k, _, tt, cs in STRATS}
    S["all120"] = book.static("sanrensho", rs.combos("sanrensho"))
    ok = R.tri.notna().to_numpy()

    V = {}
    scopes = {NAT: allm}
    scopes.update({j: jc == j for j in JCDS})
    for j, m in scopes.items():
        d = {}
        m = m & ok
        Rm = R[m]
        d["n"] = int(m.sum())
        d["n_d"], d["n_v"] = int((m & (per == "d")).sum()), int((m & (per == "v")).sum())
        d["in1"] = (Rm.wincourse == 1).mean()
        d["l1win"] = (Rm.winlane == 1).mean()
        d["course_win"] = Rm.wincourse.value_counts(normalize=True).reindex(range(1, 7)).fillna(0).to_dict()
        d["lane_win"] = Rm.winlane.value_counts(normalize=True).reindex(range(1, 7)).fillna(0).to_dict()
        d["wakunari"] = Rm.wakunari.mean()
        d["maezuke"] = Rm.maezuke.mean()
        d["tri_rate"] = Rm.tri.value_counts(normalize=True)
        d["exa_rate"] = Rm.exa.value_counts(normalize=True)
        d["tri_rate_p"] = {p: Rm[Rm.per == p].tri.value_counts(normalize=True) for p in "dv"}
        d["exa_rate_p"] = {p: Rm[Rm.per == p].exa.value_counts(normalize=True) for p in "dv"}
        d["n_p"] = {p: int((Rm.per == p).sum()) for p in "dv"}
        d["upset"] = (Rm.tri_pay >= UPSET).mean()
        d["upset_p"] = {p: (Rm[Rm.per == p].tri_pay >= UPSET).mean() for p in "dv"}
        d["med"] = Rm.tri_pay.median()
        d["mean"] = Rm.tri_pay.mean()
        d["exa_med"] = Rm.exa_pay.median()
        d["top1"] = d["tri_rate"].iloc[0]
        d["top1_comb"] = d["tri_rate"].index[0]
        d["entropy"] = entropy_bits(Rm.tri)
        d["fav"] = (Rm.win_pay < FAV_PAY).mean()
        d["fav_p"] = {p: (Rm[Rm.per == p].win_pay < FAV_PAY).mean() for p in "dv"}
        # 回収率
        roi = {}
        for k in [s[0] for s in STRATS] + ["all120"]:
            cost, ret = S[k]
            row = {}
            for p in "dv":
                mm = m & (per == p)
                row[p], row[p + "_se"], _ = roi_parts(cost, ret, mm)
                row[p + "_wm"] = win_months(cost, ret, months, mm)
            row["a"], row["a_se"], _ = roi_parts(cost, ret, m)
            roi[k] = row
        d["roi"] = roi
        # 区分ごと（前夜に分かる条件）
        rows = []
        for lab, sub in [(b, Rm[Rm.rband == b]) for b in ["1-4R", "5-8R", "9-12R"]] + \
                         [(c, Rm[Rm.cat == c]) for c in ["予選", "予選特", "一般", "一般特", "選抜特選",
                                                          "ドリーム", "準優勝戦", "優勝戦", "その他(愛称戦)"]]:
            if len(sub) >= 100:
                rows.append((lab, (sub.wincourse == 1).mean(), (sub.tri_pay >= UPSET).mean(), sub.tri_pay.median(), len(sub)))
        d["by_cond"] = rows
        rno = Rm.groupby("rno").agg(in1=("wincourse", lambda s: (s == 1).mean()),
                                    up=("tri_pay", lambda s: (s >= UPSET).mean()), n=("rid", "size"))
        d["by_rno"] = rno
        # 当日条件（参考）
        d["wind"] = day_conditions(Rm)
        V[j] = d

    # 人気通り指標の妥当性チェック（単勝オッズがある期間だけ）
    V["_fav_check"] = fav_check(R, odds)
    V["_S"] = S
    return V


def day_conditions(Rm):
    out = []
    ws = Rm.wind_speed
    wv = Rm.wave_height
    for lab, mk in [("風 0-2m", ws <= 2), ("風 3-4m", (ws > 2) & (ws <= 4)), ("風 5m以上", ws >= 5),
                    ("追い風 4m以上", (Rm.rel == "追い") & (ws >= 4)),
                    ("向かい風 4m以上", (Rm.rel == "向かい") & (ws >= 4)),
                    ("波 0-2cm", wv <= 2), ("波 3-5cm", (wv > 2) & (wv <= 5)), ("波 6cm以上", wv >= 6)]:
        sub = Rm[mk.fillna(False)]
        out.append((lab, (sub.wincourse == 1).mean() if len(sub) else np.nan, len(sub)))
    return out


def fav_check(R, odds):
    """単勝オッズが取れているレースで、1番人気の勝率と「単勝払戻<200円」率を場ごとに比べる。"""
    if odds.empty:
        return None
    o = odds.dropna(subset=["odds"])
    o = o[o.odds > 0]
    fav = o.sort_values(["rid", "odds"]).drop_duplicates("rid")[["rid", "comb"]]
    m = R.merge(fav, on="rid")
    m = m[m.winlane.notna()]
    m["fav_won"] = m.comb.astype(str) == m.winlane.astype(int).astype(str)
    m["cheap"] = m.win_pay < FAV_PAY
    g = m.groupby("jcd").agg(fav_won=("fav_won", "mean"), cheap=("cheap", "mean"), n=("rid", "size"))
    if len(g) < 5:
        return None
    return {"n": int(len(m)), "corr": float(np.corrcoef(g.fav_won, g.cheap)[0, 1]),
            "fav_won_nat": float(m.fav_won.mean()), "cheap_nat": float(m.cheap.mean()),
            "from": m.hd.min(), "to": m.hd.max(), "by": g}


# ---------------------------------------------------------------- 難易度
def difficulty(V):
    rows = []
    for j in JCDS:
        d = V[j]
        rows.append({"jcd": j, "top1": d["top1"], "entropy": d["entropy"], "l1win": d["l1win"], "fav": d["fav"]})
    D = pd.DataFrame(rows).set_index("jcd")
    Z = pd.DataFrame(index=D.index)
    for c, sign in [("top1", 1), ("entropy", -1), ("l1win", 1), ("fav", 1)]:
        Z[c] = sign * (D[c] - D[c].mean()) / D[c].std(ddof=0)
    D["easy_z"] = Z.mean(axis=1)
    D["rank"] = D.easy_z.rank(ascending=False).astype(int)      # 1 = 一番やさしい
    for c, asc in [("top1", False), ("entropy", True), ("l1win", False), ("fav", False)]:
        D[c + "_rk"] = D[c].rank(ascending=asc, method="min").astype(int)
    D["label"] = np.select([D["rank"] <= 8, D["rank"] <= 16], ["やさしい", "ふつう"], "難しい")
    return D


# ---------------------------------------------------------------- 判定
def verdict(j, V, D):
    d, n = V[j], V[NAT]
    reasons_good, reasons_bad = [], []
    for k in CORE + ["e1x", "s1xx"]:
        a, b = d["roi"][k], n["roi"][k]
        dd, dv = a["d"] - b["d"], a["v"] - b["v"]
        z = (a["a"] - b["a"]) / a["a_se"] if a["a_se"] > 0 else 0
        name = dict((s[0], s[1]) for s in STRATS)[k]
        txt = f"{name} 発見 {r3(a['d'])}／検証 {r3(a['v'])}（全国 {r3(b['d'])}／{r3(b['v'])}）"
        if dd >= EDGE and dv >= EDGE and z >= EDGE_Z:
            reasons_good.append((z, k, txt))
        if dd <= -EDGE and dv <= -EDGE and z <= -EDGE_Z:
            reasons_bad.append((-z, k, txt))
    core_bad = [r for r in reasons_bad if r[1] in CORE]
    core_good = [r for r in reasons_good if r[1] in CORE]
    # 荒れるのに配当が見合わない: 難しい側 かつ 3連単全通り回収率が両期間とも全国より低い
    a120, n120 = d["roi"]["all120"], n["roi"]["all120"]
    hard_cheap = (D.loc[j, "label"] == "難しい") and a120["d"] < n120["d"] - 0.01 and a120["v"] < n120["v"] - 0.01
    t1_bad = any(r[1] == "t1" for r in core_bad)
    if core_good and not core_bad:
        v = "狙う"
    elif t1_bad or len(core_bad) >= 2 or (core_bad and hard_cheap):
        v = "買わない"
    else:
        v = "様子見"
    return v, sorted(reasons_good, reverse=True), sorted(reasons_bad, reverse=True), hard_cheap


# ---------------------------------------------------------------- 出目の偏り
def skewed_combos(d, n, which, min_nat=0.008, top=3):
    """全国の出現率が min_nat 以上の出目で、両期間とも同じ向きに全国からずれているもの。"""
    out = []
    rate_p, nrate_p = d[which + "_rate_p"], n[which + "_rate_p"]
    nat = n[which + "_rate"]
    for comb, nr in nat.items():
        if nr < min_nat:
            continue
        diffs, zs = [], []
        for p in "dv":
            vr = rate_p[p].get(comb, 0.0)
            q = nrate_p[p].get(comb, 0.0)
            se = math.sqrt(max(q * (1 - q), 1e-9) / max(d["n_p"][p], 1))
            diffs.append(vr - q)
            zs.append((vr - q) / se)
        if np.sign(diffs[0]) == np.sign(diffs[1]) and diffs[0] != 0:
            z = (zs[0] + zs[1]) / math.sqrt(2)
            out.append((comb, d[which + "_rate"].get(comb, 0.0), nr, z,
                        rate_p["d"].get(comb, 0.0), rate_p["v"].get(comb, 0.0)))
    more = sorted([o for o in out if o[3] >= 2.0], key=lambda o: -o[3])[:top]
    less = sorted([o for o in out if o[3] <= -2.0], key=lambda o: o[3])[:top]
    return more, less


# ---------------------------------------------------------------- 仮説（前夜に分かる条件）
def hypotheses(j, R, V, book):
    d, n = V[j], V[NAT]
    S = V["_S"]
    cost, ret = S["t1"]
    per = R.per.to_numpy()
    base = R.jcd.to_numpy() == j
    conds = [(f"{r}R", R.rno.to_numpy() == r) for r in range(1, 13)]
    conds += [(b, (R.rband == b).to_numpy()) for b in ["1-4R", "5-8R", "9-12R"]]
    conds += [(f"{c}のレース", (R.cat == c).to_numpy()) for c in ["予選", "一般", "選抜特選", "準優勝戦", "優勝戦"]]
    conds += [(f"1号艇が{g}のレース", (R.g1 == g).to_numpy()) for g in ["A1", "A2", "B1", "B2"]]
    conds += [("1号艇が全国勝率1位", (R.l1top == 1).to_numpy()), ("1号艇が全国勝率2位以下", (R.l1top == 0).to_numpy())]
    good, bad = [], []
    exp = {"good": 0.0, "bad": 0.0, "cells": 0}
    base_roi = {p: roi_parts(cost, ret, base & (per == p))[0] for p in "dv"}
    for lab, cm in conds:
        m = base & cm
        nd, nv = int((m & (per == "d")).sum()), int((m & (per == "v")).sum())
        if nd < HYP_MIN_D or nv < HYP_MIN_V:
            continue
        rd, sd, _ = roi_parts(cost, ret, m & (per == "d"))
        rv, sv, _ = roi_parts(cost, ret, m & (per == "v"))
        # 帰無（この条件に場の平均以上の効果はない）で偶然そうなる確率
        exp["cells"] += 1
        exp["good"] += _sf((HYP_GOOD - base_roi["d"]) / sd) * _sf((HYP_GOOD - base_roi["v"]) / sv)
        exp["bad"] += _sf((base_roi["d"] - HYP_BAD) / sd) * _sf((base_roi["v"] - HYP_BAD) / sv)
        wd = (R.winlane[m & (per == "d")] == 1).mean()
        wv = (R.winlane[m & (per == "v")] == 1).mean()
        wm = win_months(cost, ret, R.month.to_numpy(), m & (per == "v"))
        rec = (lab, rd, rv, nd, nv, wd, wv, wm)
        if rd >= HYP_GOOD and rv >= HYP_GOOD:
            good.append(rec)
        if rd <= HYP_BAD and rv <= HYP_BAD:
            bad.append(rec)
    good.sort(key=lambda x: -min(x[1], x[2]))
    bad.sort(key=lambda x: max(x[1], x[2]))
    hyps = []
    # 1) 構造（イン）
    diff = d["in1"] - n["in1"]
    a1 = R[base & (R.g1 == "A1").to_numpy() & R.winlane.notna().to_numpy()]
    a1n = R[(R.g1 == "A1") & R.winlane.notna()]
    hyps.append(
        f"**イン{'が強い' if diff > 0.03 else ('が弱い' if diff < -0.03 else 'は全国並み')}**: イン1着率 {pct(d['in1'])}%"
        f"（全国 {pct(n['in1'])}、{diff*100:+.1f}pt）。1号艇A1の1着率 {pct((a1.winlane == 1).mean())}%"
        f"（全国 {pct((a1n.winlane == 1).mean())}、n={len(a1):,}）。"
        + ("ただし単勝1の回収率は全国並み以下なので、強さはオッズに織り込まれている。"
           if diff > 0.03 and d["roi"]["t1"]["a"] <= n["roi"]["t1"]["a"] + 0.01 else "")
        + ("1号艇の頭を外す（2〜4号艇の頭・差し）展開が多い場。"
           if diff < -0.05 else ""))
    for g in good[:2]:
        hyps.append(f"**{g[0]}は1号艇の単勝が損益分岐に近い**: 単勝1 回収率 発見 {r3(g[1])}（{g[3]:,}R）／検証 {r3(g[2])}"
                    f"（{g[4]:,}R、勝ち月 {g[7][0]}/{g[7][1]}）。1号艇1着 {pct(g[5])}%／{pct(g[6])}%。")
    for b in bad[:1]:
        hyps.append(f"**{b[0]}は1号艇を買わない**: 単勝1 回収率 発見 {r3(b[1])}（{b[3]:,}R）／検証 {r3(b[2])}"
                    f"（{b[4]:,}R）。1号艇1着 {pct(b[5])}%／{pct(b[6])}%。")
    # 企画レースなど、R番号でインが大きく変わるもの（両期間とも場平均から10pt以上）
    if len(hyps) < 4:
        Rb = R[base & R.winlane.notna().to_numpy()]
        avg = {p: (Rb[Rb.per == p].winlane == 1).mean() for p in "dv"}
        best = None
        for r in range(1, 13):
            x = Rb[Rb.rno == r]
            w = {p: (x[x.per == p].winlane == 1).mean() for p in "dv"}
            if min(w["d"] - avg["d"], w["v"] - avg["v"]) >= 0.10:
                gap = min(w["d"] - avg["d"], w["v"] - avg["v"])
                if best is None or gap > best[0]:
                    best = (gap, r, w, len(x))
        if best:
            _, r, w, k = best
            mr = (R.rno.to_numpy() == r) & base
            ro = [roi_parts(cost, ret, mr & (per == p))[0] for p in "dv"]
            hyps.append(f"**{r}Rはインが別格**（番組で1号艇に有力選手を置くレース＝企画レースやメインの選抜戦とみられる）: 1号艇1着 発見 {pct(w['d'])}%／検証 {pct(w['v'])}%"
                        f"（場平均 {pct(avg['d'])}／{pct(avg['v'])}、n={k:,}）。ただし単勝1は {r3(ro[0])}／{r3(ro[1])}"
                        + ("で、オッズに織り込まれている。" if max(ro) < 0.95 else "。"))
    # 出目の偏り（2連単）で回収率も両期間で全国より上のもの
    more, _ = skewed_combos(d, n, "exa", min_nat=0.01, top=5)
    for comb, rate, nr, z, rd_, rv_ in more:
        if len(hyps) >= 4:
            break
        c, r_ = book.static("nirensho", [comb])
        a = [roi_parts(c, r_, base & (per == p))[0] for p in "dv"]
        b_ = [roi_parts(c, r_, per == p)[0] for p in "dv"]
        if a[0] > b_[0] and a[1] > b_[1] and min(a) >= EXA_MIN:
            hyps.append(f"**2連単 {comb} が出やすい**: 出現 {pct(rate)}%（全国 {pct(nr)}、発見 {pct(rd_)}／検証 {pct(rv_)}）。"
                        f"回収率 発見 {r3(a[0])}／検証 {r3(a[1])}（全国 {r3(b_[0])}／{r3(b_[1])}）。")
    return hyps[:4], good, bad, exp


# ================================================================ 出力
def card(j, R, V, D, book, verdicts):
    d, n = V[j], V[NAT]
    name = vn(j)
    v, rg, rb, hard_cheap = verdicts[j]
    dj = D.loc[j]
    L = []
    L.append(f"# {j} {name}")
    L.append("")
    L.append(f"**判定: {v}**（仮説）／難易度 **{dj['rank']}位**/24（{dj['label']}）／{d['n']:,}レース"
             f"（発見 {d['n_d']:,}・検証 {d['n_v']:,}）")
    L.append("")
    L.append("期間 2023-11〜2026-10。発見＝2023-11〜2025-10、検証＝2025-11〜2026-10。回収率は確定配当ベース。")
    L.append("")
    # 判定の根拠
    L.append("## ひとことで")
    L.append("")
    L.append(f"- イン1着率 **{pct(d['in1'])}%**（全国 {pct(n['in1'])}、{(d['in1']-n['in1'])*100:+.1f}pt）"
             f"／万舟率 **{pct(d['upset'])}%**（全国 {pct(n['upset'])}）／3連単中央値 {yen(d['med'])}")
    t1 = d["roi"]["t1"]
    L.append(f"- 単勝1の回収率 発見 {r3(t1['d'])}／検証 **{r3(t1['v'])}**（全国 {r3(n['roi']['t1']['d'])}／{r3(n['roi']['t1']['v'])}）")
    if rg:
        L.append("- 全国より上（両期間とも）: " + "; ".join(r[2] for r in rg[:2]))
    if rb:
        L.append("- 全国より下（両期間とも）: " + "; ".join(r[2] for r in rb[:2]))
    if hard_cheap:
        a, b = d["roi"]["all120"], n["roi"]["all120"]
        L.append(f"- 荒れるのに配当が見合わない: 3連単全通り回収率 {r3(a['d'])}／{r3(a['v'])}（全国 {r3(b['d'])}／{r3(b['v'])}）")
    if not rg and not rb:
        L.append("- 定番の買い方で、両期間そろって全国と差がつくものはない")
    L.append("")
    # 基本
    L.append("## 基本（進入とコース）")
    L.append("")
    L.append(f"- 枠なり率 {pct(d['wakunari'])}%（全国 {pct(n['wakunari'])}）、前付けのあるレース {pct(d['maezuke'])}%"
             f"（全国 {pct(n['maezuke'])}）。1号艇の1着率 {pct(d['l1win'])}%")
    L.append("")
    L.append(table(["コース", "1着率", "全国", "差"],
                   [[f"{c}", pct(d['course_win'][c]), pct(n['course_win'][c]),
                     f"{(d['course_win'][c]-n['course_win'][c])*100:+.1f}"] for c in range(1, 7)]))
    L.append("")
    # 出目
    L.append("## 出目")
    L.append("")
    L.append("1着の艇番: " + " / ".join(f"{k}号艇 {pct(d['lane_win'][k])}" for k in range(1, 7)) + "（%）")
    L.append("")
    for which, title in [("tri", "3連単"), ("exa", "2連単")]:
        top = d[which + "_rate"].head(10)
        L.append(f"**{title} 上位10**（出現率%）")
        L.append("")
        L.append(table(["出目", "出現率", "全国"],
                       [[c, pct(r), pct(n[which + '_rate'].get(c, 0.0))] for c, r in top.items()]))
        L.append("")
        more, less = skewed_combos(d, n, which, min_nat=0.008 if which == "tri" else 0.01)
        if more:
            L.append(f"- 全国より多い（両期間とも）: " + "、".join(f"{c} {pct(r)}%（全国 {pct(q)}）" for c, r, q, *_ in more))
        if less:
            L.append(f"- 全国より少ない（両期間とも）: " + "、".join(f"{c} {pct(r)}%（全国 {pct(q)}）" for c, r, q, *_ in less))
        if not more and not less:
            L.append("- 両期間そろって全国から目立ってずれる出目はない")
        L.append("")
    # 荒れ方
    L.append("## 荒れ方（前夜に分かる条件）")
    L.append("")
    L.append(f"- 万舟率 {pct(d['upset'])}%（発見 {pct(d['upset_p']['d'])}／検証 {pct(d['upset_p']['v'])}、全国 {pct(n['upset'])}）。"
             f"3連単の中央値 {yen(d['med'])}・平均 {yen(d['mean'])}（全国 {yen(n['med'])}・{yen(n['mean'])}）。2連単の中央値 {yen(d['exa_med'])}")
    L.append("")
    L.append(table(["条件", "イン1着", "万舟", "n"],
                   [[lab, pct(a), pct(u), f"{k:,}"] for lab, a, u, _m, k in d["by_cond"]]))
    L.append("")
    br = d["by_rno"]
    hi, lo = br.in1.idxmax(), br.in1.idxmin()
    uhi = br.up.idxmax()
    L.append(f"- R別: イン最高 {hi}R {pct(br.in1[hi])}%、最低 {lo}R {pct(br.in1[lo])}%、万舟が多いのは {uhi}R {pct(br.up[uhi])}%"
             f"（各 n≈{int(br.n.median()):,}）")
    L.append("")
    # 当日条件
    L.append("## 当日条件の影響（参考。前夜には分からないので選定には使わない）")
    L.append("")
    nw = dict((lab, a) for lab, a, _ in n["wind"])
    L.append(table(["条件", "イン1着", "全国", "n"],
                   [[lab, pct(a), pct(nw[lab]), f"{k:,}"] for lab, a, k in d["wind"] if k > 0]))
    L.append("")
    L.append("追い／向かいの向きは展示タイムから推定（notes/analysis/venue_course.md 2-0）。n が100未満は参考程度。")
    L.append("")
    # 回収率
    L.append("## 回収率（定番の買い方、1点100円）")
    L.append("")
    rows = []
    for k, nm, _, _ in STRATS:
        a, b = d["roi"][k], n["roi"][k]
        rows.append([nm, r3(a["d"]), f"**{r3(a['v'])}**", r3(b["v"]), f"{a['d_wm'][0]}/{a['d_wm'][1]}・{a['v_wm'][0]}/{a['v_wm'][1]}"])
    a, b = d["roi"]["all120"], n["roi"]["all120"]
    rows.append(["3連単 全通り(120点)", r3(a["d"]), f"**{r3(a['v'])}**", r3(b["v"]), f"{a['d_wm'][0]}/{a['d_wm'][1]}・{a['v_wm'][0]}/{a['v_wm'][1]}"])
    L.append(table(["買い方", "発見", "検証", "全国検証", "勝ち月 発見・検証"], rows))
    L.append("")
    L.append(f"検証期間の単勝1の標準誤差は約 ±{d['roi']['t1']['v_se']:.3f}、3連単1-2-3 は約 ±{d['roi']['s123']['v_se']:.3f}。"
             "「3連単 全通り」は配当の厚さの目安（全国並みより低い＝荒れても配当が安い）。")
    L.append("")
    # 難易度
    L.append("## 難易度（予想しやすさ）")
    L.append("")
    L.append(f"総合 **{dj['rank']}位/24**（{dj['label']}、1位が一番やさしい）。内訳（各指標の全国順位）:")
    L.append("")
    L.append(table(["指標", "値", "順位"], [
        [f"最多出目（{d['top1_comb']}）の出現率", f"{pct(d['top1'])}%", f"{dj['top1_rk']}"],
        ["3連単出目のエントロピー", f"{d['entropy']:.2f} bit", f"{dj['entropy_rk']}"],
        ["1号艇の1着率", f"{pct(d['l1win'])}%", f"{dj['l1win_rk']}"],
        ["人気通り（単勝<200円）", f"{pct(d['fav'])}%", f"{dj['fav_rk']}"],
    ]))
    L.append("")
    # 仮説
    hyps, _, _, _ = hypotheses(j, R, V, book)
    L.append("## 前夜に使えそうな傾向（仮説・未採用）")
    L.append("")
    for i, h in enumerate(hyps, 1):
        L.append(f"{i}. {h}")
    L.append("")
    L.append("いずれも発見用・検証用の両方で同じ向きのものだけ。ただし「○○は1号艇の単勝が損益分岐に近い／買わない」型は、"
             "24場で試した数からみて偶然とほぼ同じ数しか出ていない（README「各カルテの見方」）ので、いちばん弱い根拠として読む。"
             "採用するには、1本に決めて前向き（2026-11以降）に確かめる。")
    L.append("")
    L.append("[一覧へ戻る](README.md)")
    return "\n".join(L) + "\n"


def readme(R, V, D, verdicts, hypcounts):
    n = V[NAT]
    L = ["# 場ごとのカルテ（24場）", ""]
    L.append("期間 2023-11〜2026-10（発見用 2023-11〜2025-10／検証用 2025-11〜2026-10）、"
             f"{n['n']:,}レース。`python scripts/analysis/venue_cards.py` で作り直す。回収率は確定配当、1点100円。")
    L.append("")
    L.append("**判定は仮説であって採用ではない。** 採用は CLAUDE.md の手順どおり、事前に1本に決めて前向きに確かめてから。"
             f"そもそも**全場・全定番の買い方で回収率1.0を超えるものはない**（一番高い全国の単勝1でも検証 {r3(n['roi']['t1']['v'])}）。"
             "「狙う」は「損が小さく、絞り込みの土台にしやすい場」という意味。")
    L.append("")
    L.append("## 一覧（難易度順。1位が一番やさしい）")
    L.append("")
    rows = []
    for j in D.sort_values("rank").index:
        d = V[j]
        rows.append([f"[{vn(j)}]({j}_{vn(j)}.md)", pct(d["in1"]), pct(d["upset"]), f"{int(d['med']):,}",
                     f"{D.loc[j, 'rank']}", r2(d["roi"]["t1"]["d"]) + "/" + r2(d["roi"]["t1"]["v"]),
                     f"**{verdicts[j][0]}**"])
    rows.append(["全国", pct(n["in1"]), pct(n["upset"]), f"{int(n['med']):,}", "—",
                 r2(n["roi"]["t1"]["d"]) + "/" + r2(n["roi"]["t1"]["v"]), ""])
    L.append(table(["場", "イン1着%", "万舟%", "3単中央値", "難易度", "単勝1 発見/検証", "判定"], rows))
    L.append("")
    for lab in ["狙う", "買わない"]:
        js = [j for j in JCDS if verdicts[j][0] == lab]
        L.append(f"## {lab}場")
        L.append("")
        if not js:
            L.append("- なし")
        for j in js:
            v, rg, rb, hc = verdicts[j]
            rs_ = rg if lab == "狙う" else rb
            extra = "；荒れるのに配当が薄い" if (lab == "買わない" and hc) else ""
            L.append(f"- **{vn(j)}**: " + "; ".join(r[2] for r in rs_[:2]) + extra)
        L.append("")
    L.append("## 判定の付け方")
    L.append("")
    L.append(f"- 定番の買い方5つ（単勝1・2連単1-2・2連単1-全・3連単1-2-3・3連単1-全-全）について、場の回収率と全国の差を"
             f"発見用・検証用それぞれで出す。**両期間とも差が {EDGE:+.2f} 以上**で、3年通算の差の z 値が {EDGE_Z} 以上のものだけを「意味のある差」とした（下側も同様）。")
    L.append("- **狙う**: 1号艇の本線（単勝1・2連単1-2・3連単1-2-3）のどれかで、両期間とも全国より上の差がある。かつ下側の差がない。")
    L.append("- **買わない**: 単勝1（本線で一番ぶれが小さい）が両期間とも全国より下。または本線のうち2つ以上が下。または1つが下で、かつ「難しい」側（難易度17位以下）"
             "で3連単全通りの回収率が両期間とも全国より低い（荒れるのに配当が見合っていない）。")
    L.append("- **様子見**: それ以外。")
    L.append(f"- 試した組み合わせは 24場×5買い方×上下。偶然で1〜2場が引っかかる程度の数なので、根拠は両期間の一致と z 値で絞っている。")
    L.append("")
    L.append("## 難易度の計算方法")
    L.append("")
    L.append("3年通算で、場ごとに次の4指標を出し、24場の中で標準化（平均0・標準偏差1、堅いほど＋になる向き）して平均した。その順位が難易度（1位が一番予想しやすい）。1〜8位「やさしい」、9〜16位「ふつう」、17〜24位「難しい」。")
    L.append("")
    L.append("1. **最多出目の出現率**: 3連単で一番多い出目（どの場も 1-2-3）が出る割合。高いほど堅い。")
    L.append("2. **3連単出目のエントロピー**: 120通りの出目の出現分布の −Σp·log2(p)（ビット）。低いほど出目が偏っていて読みやすい。最大は log2(120)=6.91。")
    L.append("3. **1号艇の1着率**（枠番基準）。")
    L.append(f"4. **人気通りの決着率**: 単勝の払戻が{FAV_PAY}円未満（オッズ2倍未満の艇が勝った）レースの割合で近似。")
    fc = V["_fav_check"]
    if fc:
        L.append(f"   - 近似の確認: 単勝オッズが取れている {fc['from']}〜{fc['to']} の {fc['n']:,} レースで、場ごとの「1番人気の勝率」"
                 f"（全国 {pct(fc['fav_won_nat'])}%）とこの近似（同 {pct(fc['cheap_nat'])}%）の相関は {fc['corr']:.2f}。")
    L.append("")
    L.append("4指標は互いに強く相関している（イン1着率が高い場ほど全部がやさしい側に動く）。つまり難易度は、ほぼ「インの強さ」の言い換え。"
             "**やさしい場ほど配当も安い**ので、難易度が高い＝儲からない、ではない。")
    L.append("")
    L.append("## 各カルテの見方")
    L.append("")
    L.append("- 出目の「全国より多い／少ない」は、全国の出現率が1%前後以上の出目のうち、発見用・検証用の両方で同じ向きにずれ、2期間を合わせた z 値が |2| 以上のもの。")
    L.append(f"- 仮説は、前夜に分かる条件（R番号・R帯・種別・1号艇の級・1号艇が全国勝率1位か）で、単勝1が両期間とも {HYP_GOOD} 以上、または両期間とも {HYP_BAD} 以下のもの"
             f"（最低 発見 {HYP_MIN_D}・検証 {HYP_MIN_V} レース）と、2連単の出やすい出目で回収率も両期間で全国より上のもの。"
             f"各場で 25 前後の条件を試しているので、24場全体では偶然の当たりも混じる（「両期間で一致」は1条件あたり数%の確率で偶然起きる）。")
    L.append(f"  試したセル {hypcounts['cells']}。単勝1が両期間 ≥{HYP_GOOD} … **{hypcounts['good']}**（偶然でも約 {hypcounts['exp_good']:.0f} 出る）、"
             f"両期間 ≤{HYP_BAD} … **{hypcounts['bad']}**（偶然でも約 {hypcounts['exp_bad']:.0f}）。"
             "偶然の目安は「その条件に場の平均以上の効果はない」と仮定し、各セルの標準誤差から両期間とも基準を越える確率を足したもの。"
             "観測が目安とほぼ同じなら、カルテの単勝1の仮説は大半が偶然と見るべき。")
    L.append("- 返還の扱いは notes/analysis/roi_scan.md と同じ（欠場艇を含む買い目は返還、position が空で ST がある艇は外れ扱い）。")
    L.append("- 当日条件（風・波）は参考。前夜（8時の購入）には分からないので判定・仮説には使っていない。")
    L.append("")
    return "\n".join(L) + "\n"


def main():
    t0 = time.time()
    races, void, pay, ent, res, odds = load_all()
    R = build_race_table(races, pay, ent, res)
    # 風の向き（venue_course と同じ推定）
    rcw = R[["hd", "jcd", "rno"]].copy()
    rcw["wd"], rcw["ws"], rcw["month"] = R.wind_direction, R.wind_speed, R.hd.str[4:6]
    ex = res[["hd", "jcd", "rno"]].copy()
    ex["ex"] = pd.to_numeric(res.ex, errors="coerce")
    orient = vc.wind_orientation(ex, rcw)
    R["rel"] = vc.rel_wind(rcw, orient)["rel"].to_numpy()
    book = rs.Book(len(races), pay, void)
    V = compute(R, book, odds)
    D = difficulty(V)
    verdicts = {j: verdict(j, V, D) for j in JCDS}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    hc = {"good": 0, "bad": 0, "exp_good": 0.0, "exp_bad": 0.0, "cells": 0}
    for j in JCDS:
        _, g, b, e = hypotheses(j, R, V, book)
        hc["good"] += len(g); hc["bad"] += len(b)
        hc["exp_good"] += e["good"]; hc["exp_bad"] += e["bad"]; hc["cells"] += e["cells"]
        (OUT_DIR / f"{j}_{vn(j)}.md").write_text(card(j, R, V, D, book, verdicts), encoding="utf-8")
    (OUT_DIR / "README.md").write_text(readme(R, V, D, verdicts, hc), encoding="utf-8")
    print(f"[venue_cards] wrote {OUT_DIR} ({time.time()-t0:.0f}s)", file=sys.stderr)
    for j in D.sort_values("rank").index:
        print(j, vn(j), D.loc[j, "rank"], verdicts[j][0], round(V[j]["roi"]["t1"]["d"], 3), round(V[j]["roi"]["t1"]["v"], 3))


if __name__ == "__main__":
    main()
