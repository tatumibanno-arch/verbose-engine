"""前夜に分かる条件 × 定番の買い方 の回収率を総当たりで調べる（払戻データだけで計算）。

使い方:
    python scripts/analysis/roi_scan.py                # 結果を標準出力
    python scripts/analysis/roi_scan.py --out PATH.md  # 結果の表を Markdown で書き出す

方針（詳細は notes/analysis/roi_scan.md）:
  * 回収率 = 払戻合計 ÷ 購入額合計。1点100円。払戻テーブルにない買い目は外れ（払戻0）。
  * 返還: results で position が NULL かつ st_timing も NULL の艇 = 欠場とみなし、
    その艇を含む買い目は返還（購入しなかったのと同じ: 購入額0・払戻0）。
    position が NULL でも st_timing がある艇（F・転覆・失格など区別不能）は外れ扱い（保守的）。
    感度として「NULL は全部返還」の基準線も出す。
  * 3連単の払戻が無いレース（中止など）は対象外。
  * 期間: 発見用 2023-11-01〜2025-10-31 / 検証用 2025-11-01〜2026-10-31。
  * 評価の主軸は暦月の回収率（月の払戻合計 ÷ 購入合計 > 1.0 を「勝ち月」）。
  * DB は読み取り専用で開く。書き込みはしない。
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DB_URI = f"file:{ROOT / 'data' / 'boatrace.db'}?mode=ro"

DISC = ("20231101", "20251031")
VAL = ("20251101", "20261031")
MONTHS = [f"{y}{m:02d}" for y, m in
          [(2023, 11), (2023, 12)] + [(2024, m) for m in range(1, 13)] + [(2025, m) for m in range(1, 13)]
          + [(2026, m) for m in range(1, 11)]]
NM = len(MONTHS)          # 36
N_DISC_M = 24             # 先頭24か月が発見用、残り12か月が検証用

# 閾値（発見用で候補にする最低条件）
MIN_RACES = 300
MIN_HITS = 30
TOP_K = 40                # 発見用から検証へ回す候補数
BOOT = 1000               # ブートストラップ回数（日単位ブロック）
SEED = 20261102

LANES = [1, 2, 3, 4, 5, 6]


def load_adopt():
    """採用の合格ライン（config/rules.yaml の evaluation.adopt_*）。検証用12か月に当てる。"""
    import yaml
    ev = yaml.safe_load((ROOT / "config" / "rules.yaml").read_text(encoding="utf-8")).get("evaluation", {})
    return float(ev.get("adopt_min_total_roi", 1.0)), int(ev.get("adopt_min_win_months", 8)), \
        int(ev.get("adopt_window_months", 12))


ADOPT_ROI, ADOPT_WIN, ADOPT_WINDOW = load_adopt()


# ---------------------------------------------------------------- 読み込み
def connect():
    return sqlite3.connect(DB_URI, uri=True, timeout=60)


def race_cat(name: str | None) -> str | None:
    if not name:
        return None
    if "準優" in name:
        return "準優勝戦"
    if "優勝" in name:
        return "優勝戦"
    if "ドリーム" in name:
        return "ドリーム"
    if "予選" in name:
        return "予選特" if ("特" in name) else "予選"
    if "一般" in name:
        return "一般特" if ("特" in name) else "一般"
    if any(k in name for k in ("選抜", "特選", "特賞")):
        return "選抜特選"
    return "その他(愛称戦)"


def load(con):
    t0 = time.time()
    races = pd.read_sql(
        "select hd, jcd, rno, race_name, weather, wind_direction, wind_speed, wave_height from races "
        "where hd between ? and ?", con, params=(DISC[0], VAL[1]))
    st_races = pd.read_sql(
        "select distinct hd, jcd, rno from payouts where ticket_type='sanrensho' and hd between ? and ?",
        con, params=(DISC[0], VAL[1]))
    races = races.merge(st_races, on=["hd", "jcd", "rno"]).sort_values(["hd", "jcd", "rno"]).reset_index(drop=True)
    races["rid"] = np.arange(len(races))
    key = races.set_index(["hd", "jcd", "rno"])["rid"]

    res = pd.read_sql(
        "select hd, jcd, rno, lane, position, json_extract(data,'$.st_timing') st from results "
        "where position is null and hd between ? and ?", con, params=(DISC[0], VAL[1]))
    res = res.join(key, on=["hd", "jcd", "rno"], how="inner")
    void_strict = np.zeros((len(races), 6), bool)    # 欠場（返還）
    void_all = np.zeros((len(races), 6), bool)       # NULL は全部返還（感度用）
    r = res["rid"].to_numpy(); l = res["lane"].to_numpy().astype(int) - 1
    ok = (l >= 0) & (l < 6)
    void_all[r[ok], l[ok]] = True
    s = res["st"].isna().to_numpy() & ok
    void_strict[r[s], l[s]] = True

    pay = pd.read_sql("select hd, jcd, rno, ticket_type tt, combination comb, payout from payouts "
                      "where hd between ? and ?", con, params=(DISC[0], VAL[1]))
    pay = pay.join(key, on=["hd", "jcd", "rno"], how="inner")

    ent = pd.read_sql(
        "select hd, jcd, rno, lane, racer_no, grade, "
        "json_extract(data,'$.national_win_rate') nwr, json_extract(data,'$.local_win_rate') lwr, "
        "json_extract(data,'$.motor_2ren_rate') m2, json_extract(data,'$.boat_2ren_rate') b2 "
        "from entries where hd between ? and ?", con, params=(DISC[0], VAL[1]))
    ent = ent.join(key, on=["hd", "jcd", "rno"], how="inner")
    # 番組表と成績の選手が一致しないレースは番組表特徴を使わない（場コード取り違え等への保険）
    rr = pd.read_sql("select hd, jcd, rno, lane, cast(json_extract(data,'$.racer_number') as int) rn from results "
                     "where hd between ? and ?", con, params=(DISC[0], VAL[1]))
    chk = ent[["hd", "jcd", "rno", "lane", "racer_no", "rid"]].merge(rr, on=["hd", "jcd", "rno", "lane"], how="left")
    bad = chk.loc[chk.rn.notna() & (pd.to_numeric(chk.racer_no, errors="coerce") != chk.rn), "rid"].unique()
    if len(bad):
        print(f"[load] 番組表と成績の選手が不一致のレース {len(bad)} 件 → 番組表特徴から除外", file=sys.stderr)
    ent = ent[~ent.rid.isin(bad)]
    print(f"[load] races={len(races)} payouts={len(pay)} entries={len(ent)} ({time.time()-t0:.0f}s)", file=sys.stderr)
    return races, void_strict, void_all, pay, ent


# ---------------------------------------------------------------- 払戻行列
def combos(tt):
    if tt in ("tansho", "fukusho"):
        return [str(a) for a in LANES]
    if tt == "nirensho":
        return [f"{a}-{b}" for a, b in itertools.permutations(LANES, 2)]
    if tt in ("nirenfuku", "wide"):
        return [f"{a}-{b}" for a, b in itertools.combinations(LANES, 2)]
    if tt == "sanrensho":
        return [f"{a}-{b}-{c}" for a, b, c in itertools.permutations(LANES, 3)]
    if tt == "sanrenfuku":
        return [f"{a}-{b}-{c}" for a, b, c in itertools.combinations(LANES, 3)]
    raise KeyError(tt)


TTS = ["tansho", "fukusho", "nirensho", "nirenfuku", "wide", "sanrensho", "sanrenfuku"]
TT_JA = {"tansho": "単勝", "fukusho": "複勝", "nirensho": "2連単", "nirenfuku": "2連複", "wide": "拡連複",
         "sanrensho": "3連単", "sanrenfuku": "3連複"}


class Book:
    """券種ごとに レース×買い目 の払戻行列（100円あたり÷100 = 倍率）と返還行列を持つ。"""

    def __init__(self, nr, pay, void):
        self.nr = nr
        self.P, self.V, self.idx = {}, {}, {}
        for tt in TTS:
            cs = combos(tt)
            ix = {c: i for i, c in enumerate(cs)}
            self.idx[tt] = ix
            M = np.zeros((nr, len(cs)), np.float32)
            p = pay[pay.tt == tt]
            ci = p["comb"].map(ix)
            good = ci.notna()
            np.add.at(M, (p["rid"].to_numpy()[good], ci[good].astype(int).to_numpy()),
                      p["payout"].to_numpy()[good] / 100.0)
            self.P[tt] = M
            lanes_in = np.array([[int(x) - 1 for x in c.split("-")] for c in cs])
            Vm = np.zeros((nr, len(cs)), bool)
            for k in range(lanes_in.shape[1]):
                Vm |= void[:, lanes_in[:, k]]
            self.V[tt] = Vm

    def static(self, tt, cs):
        """固定の買い目集合を全レースで買う → (cost, ret)。単位は 100円。"""
        ii = [self.idx[tt][c] for c in cs]
        live = ~self.V[tt][:, ii]
        cost = live.sum(1).astype(np.float32)
        ret = (self.P[tt][:, ii] * live).sum(1)
        return cost, ret

    def dynamic(self, tt, ci_list):
        """レースごとに変わる買い目（買い目index配列のリスト、-1 は買わない）。"""
        cost = np.zeros(self.nr, np.float32); ret = np.zeros(self.nr, np.float32)
        ar = np.arange(self.nr)
        for ci in ci_list:
            ok = ci >= 0
            c = np.where(ok, ci, 0)
            live = ok & ~self.V[tt][ar, c]
            cost += live
            ret += np.where(live, self.P[tt][ar, c], 0)
        return cost, ret


# ---------------------------------------------------------------- 番組表の特徴（前夜に分かる）
def wide_entries(nr, ent):
    W = {}
    for col in ["nwr", "lwr", "m2", "b2"]:
        a = np.full((nr, 6), np.nan)
        e = ent[ent.lane.between(1, 6)]
        a[e.rid.to_numpy(), e.lane.to_numpy().astype(int) - 1] = pd.to_numeric(e[col], errors="coerce").to_numpy()
        W[col] = a
    g = np.full((nr, 6), None, object)
    e = ent[ent.lane.between(1, 6)]
    g[e.rid.to_numpy(), e.lane.to_numpy().astype(int) - 1] = e.grade.to_numpy()
    W["grade"] = g
    has = ~np.isnan(W["nwr"]).any(1)
    return W, has


def argmax_lane(a, lanes=None):
    """最大値の艇（同値は内側）。lanes で対象艇を絞る。NaN 行は -1。"""
    b = a.copy()
    if lanes is not None:
        mask = np.ones(6, bool); mask[[x - 1 for x in lanes]] = False
        b[:, mask] = -np.inf
    out = np.nanargmax(np.where(np.isnan(b), -np.inf, b), 1)
    bad = np.isnan(a).all(1)
    return np.where(bad, -1, out)  # 0-based lane


def build_strategies(book, W, has):
    S = {}
    # --- 固定の買い目
    for k in LANES:
        S[f"単勝 {k}"] = ("tansho", [str(k)])
        S[f"複勝 {k}"] = ("fukusho", [str(k)])
    S["2連単 1-2"] = ("nirensho", ["1-2"])
    S["2連単 1-3"] = ("nirensho", ["1-3"])
    S["2連単 1-23"] = ("nirensho", ["1-2", "1-3"])
    S["2連単 1-全"] = ("nirensho", [f"1-{b}" for b in range(2, 7)])
    S["2連単 2-1"] = ("nirensho", ["2-1"])
    S["2連単 BOX123"] = ("nirensho", [f"{a}-{b}" for a, b in itertools.permutations([1, 2, 3], 2)])
    S["2連複 1=2"] = ("nirenfuku", ["1-2"])
    S["2連複 1=3"] = ("nirenfuku", ["1-3"])
    S["2連複 1=全"] = ("nirenfuku", [f"1-{b}" for b in range(2, 7)])
    S["拡連複 1=2"] = ("wide", ["1-2"])
    S["拡連複 1=3"] = ("wide", ["1-3"])
    S["拡連複 1=4"] = ("wide", ["1-4"])
    S["3連単 1-2-3"] = ("sanrensho", ["1-2-3"])
    S["3連単 1-23-23"] = ("sanrensho", ["1-2-3", "1-3-2"])
    S["3連単 1-234-234"] = ("sanrensho", [f"1-{b}-{c}" for b, c in itertools.permutations([2, 3, 4], 2)])
    S["3連単 1-2345-2345"] = ("sanrensho", [f"1-{b}-{c}" for b, c in itertools.permutations([2, 3, 4, 5], 2)])
    S["3連単 1-全-全"] = ("sanrensho", [f"1-{b}-{c}" for b, c in itertools.permutations(range(2, 7), 2)])
    S["3連単 1-2-全"] = ("sanrensho", [f"1-2-{c}" for c in range(3, 7)])
    S["3連単 1-3-全"] = ("sanrensho", [f"1-3-{c}" for c in [2, 4, 5, 6]])
    S["3連単 1-4-全"] = ("sanrensho", [f"1-4-{c}" for c in [2, 3, 5, 6]])
    S["3連単 BOX123"] = ("sanrensho", [f"{a}-{b}-{c}" for a, b, c in itertools.permutations([1, 2, 3])])
    S["3連単 2-1-全"] = ("sanrensho", [f"2-1-{c}" for c in [3, 4, 5, 6]])
    for h in [2, 3, 4, 5, 6]:
        S[f"3連単 {h}-全-全"] = ("sanrensho", [f"{h}-{b}-{c}" for b, c in itertools.permutations([x for x in LANES if x != h], 2)])
    S["3連複 1=2=3"] = ("sanrenfuku", ["1-2-3"])
    S["3連複 1=2=全"] = ("sanrenfuku", [f"1-2-{c}" for c in range(3, 7)])
    S["3連複 1=全=全"] = ("sanrenfuku", [f"1-{b}-{c}" for b, c in itertools.combinations(range(2, 7), 2)])
    S["3連複 BOX1234"] = ("sanrenfuku", [f"{a}-{b}-{c}" for a, b, c in itertools.combinations([1, 2, 3, 4], 3)])
    # 全通り（各券種の「何も考えない」基準）
    for tt in TTS:
        S[f"全通り {TT_JA[tt]}"] = (tt, combos(tt))

    out = {name: book.static(tt, cs) for name, (tt, cs) in S.items()}

    # --- 番組表で決まる買い目（entries がないレースは買わない）
    def L(a):  # 0-based lane -> 1-based, -1 そのまま
        return np.where(a >= 0, a + 1, -1)
    nwr = W["nwr"]; m2 = W["m2"]
    top = L(argmax_lane(nwr)); top_m = L(argmax_lane(m2)); top_l = L(argmax_lane(W["lwr"]))
    top_out = L(argmax_lane(nwr, [4, 5, 6]))
    top_oth = L(argmax_lane(nwr, [2, 3, 4, 5, 6]))
    order = np.argsort(-np.where(np.isnan(nwr), -np.inf, nwr), 1, kind="stable") + 1  # 勝率順の艇
    order[~has] = -1

    def ci(tt, *lanes):
        ix = book.idx[tt]
        sep = "-"
        res = np.full(book.nr, -1)
        good = np.all([x > 0 for x in lanes], 0) & has
        if tt in ("nirenfuku", "wide", "sanrenfuku"):
            keys = [sep.join(map(str, sorted(t))) for t in zip(*[x.tolist() for x in lanes])]
        else:
            keys = [sep.join(map(str, t)) for t in zip(*[x.tolist() for x in lanes])]
        vals = np.array([ix.get(k, -1) for k in keys])
        return np.where(good, vals, -1)

    one = np.ones(book.nr, int)
    D = {
        "単勝 勝率1位": ("tansho", [ci("tansho", top)]),
        "複勝 勝率1位": ("fukusho", [ci("fukusho", top)]),
        "単勝 当地勝率1位": ("tansho", [ci("tansho", top_l)]),
        "単勝 モーター1位": ("tansho", [ci("tansho", top_m)]),
        "複勝 モーター1位": ("fukusho", [ci("fukusho", top_m)]),
        "単勝 外枠(4-6)勝率1位": ("tansho", [ci("tansho", top_out)]),
        "複勝 外枠(4-6)勝率1位": ("fukusho", [ci("fukusho", top_out)]),
        "2連単 勝率1位-2位": ("nirensho", [ci("nirensho", order[:, 0], order[:, 1])]),
        "2連単 1-他艇勝率1位": ("nirensho", [ci("nirensho", one, top_oth)]),
        "2連複 1=他艇勝率1位": ("nirenfuku", [ci("nirenfuku", one, top_oth)]),
        "拡連複 1=外枠勝率1位": ("wide", [ci("wide", one, top_out)]),
        "3連単 勝率順1-2-3": ("sanrensho", [ci("sanrensho", order[:, 0], order[:, 1], order[:, 2])]),
        "3連複 勝率上位3艇": ("sanrenfuku", [ci("sanrenfuku", order[:, 0], order[:, 1], order[:, 2])]),
    }
    # 1号艇以外の勝率上位2艇を2・3着に（1-a-b, 1-b-a）
    oth = np.where(has[:, None], order, -1)
    o2 = np.array([[x for x in row if x != 1][:2] if row[0] > 0 else [-1, -1] for row in oth])
    D["3連単 1-他上位2艇-他上位2艇"] = ("sanrensho", [ci("sanrensho", one, o2[:, 0], o2[:, 1]),
                                                ci("sanrensho", one, o2[:, 1], o2[:, 0])])
    D["3連単 外枠勝率1位-全-全"] = ("sanrensho", [
        ci("sanrensho", top_out, np.full(book.nr, b), np.full(book.nr, c))
        for b, c in itertools.permutations(LANES, 2)] )
    # 上の「外枠-全-全」は top_out と重なる b,c は ci で -1 になる
    for name, (tt, lst) in D.items():
        if name == "3連単 外枠勝率1位-全-全":
            lst = [np.where((top_out > 0) & (top_out != b) & (top_out != c), x, -1)
                   for x, (b, c) in zip(lst, itertools.permutations(LANES, 2))]
        out[name] = book.dynamic(tt, lst)
    return out


# ---------------------------------------------------------------- 条件（特徴量）
def cut(x, edges, labels):
    c = pd.cut(pd.Series(x), edges, labels=labels, right=False)
    return c.astype(object).where(c.notna(), None).to_numpy()


def build_features(races, W, has):
    F_pre, F_day = {}, {}
    hd = races.hd
    F_pre["場"] = races.jcd.to_numpy()
    F_pre["R"] = races.rno.astype(int).map(lambda r: f"{r:02d}R").to_numpy()
    F_pre["R帯"] = races.rno.astype(int).map(lambda r: "1-4R" if r <= 4 else ("5-8R" if r <= 8 else "9-12R")).to_numpy()
    F_pre["種別"] = races.race_name.map(race_cat).to_numpy()
    F_pre["月"] = hd.str[4:6].to_numpy()
    F_pre["季節"] = hd.str[4:6].astype(int).map(lambda m: {12: "冬", 1: "冬", 2: "冬", 3: "春", 4: "春", 5: "春",
                                                      6: "夏", 7: "夏", 8: "夏"}.get(m, "秋")).to_numpy()
    F_pre["曜日"] = pd.to_datetime(hd).dt.dayofweek.map(lambda d: "土日" if d >= 5 else "平日").to_numpy()

    nwr = W["nwr"]; g = W["grade"]
    def mask(a):
        return np.where(has, a, None)
    F_pre["1号艇級"] = mask(g[:, 0])
    F_pre["1号艇勝率"] = mask(cut(nwr[:, 0], [-1, 5, 6, 7, 8, 99], ["<5", "5-6", "6-7", "7-8", "8+"]))
    lw = W["lwr"][:, 0]
    F_pre["1号艇当地勝率"] = mask(cut(np.where(lw > 0, lw, np.nan), [-1, 5, 6, 7, 99], ["<5", "5-6", "6-7", "7+"]))
    diff = nwr[:, 0] - np.nanmax(nwr[:, 1:], 1)
    F_pre["1号艇-他最高 勝率差"] = mask(cut(diff, [-99, -1, 0, 1, 2, 99], ["<-1", "-1~0", "0~1", "1~2", "2+"]))
    a1 = (g == "A1").sum(1).astype(object)
    a1 = np.where(a1 >= 3, "3+", a1.astype(str))
    F_pre["A1の数"] = mask(a1)
    top = argmax_lane(nwr) + 1
    F_pre["勝率1位の艇"] = mask(np.where(top > 0, top.astype(str), None))
    out_gap = np.nanmax(nwr[:, 3:], 1) - nwr[:, 0]
    F_pre["外枠最高-1号艇 勝率差"] = mask(cut(out_gap, [-99, -1, 0, 99], ["<-1(1号艇優位)", "-1~0", "0+(外が上)"]))
    m1 = W["m2"][:, 0]
    F_pre["1号艇モーター2連率"] = mask(cut(np.where(m1 > 0, m1, np.nan), [-1, 30, 40, 50, 101], ["<30", "30-40", "40-50", "50+"]))
    topm = argmax_lane(W["m2"]) + 1
    F_pre["モーター1位の艇"] = mask(np.where(topm > 0, topm.astype(str), None))
    oth_mean = np.nanmean(nwr[:, 1:], 1)
    F_pre["2-6号艇平均勝率"] = mask(cut(oth_mean, [-1, 4.5, 5.5, 6.5, 99], ["<4.5", "4.5-5.5", "5.5-6.5", "6.5+"]))

    # 当日にならないと分からない情報（参考枠）
    F_day["天候"] = races.weather.to_numpy()
    F_day["風速"] = cut(races.wind_speed.to_numpy().astype(float), [-1, 2, 4, 6, 99], ["0-1m", "2-3m", "4-5m", "6m+"])
    F_day["波高"] = cut(races.wave_height.to_numpy().astype(float), [-1, 3, 6, 999], ["0-2cm", "3-5cm", "6cm+"])
    return F_pre, F_day


# ---------------------------------------------------------------- 集計
def stats_from(cost, ret, ret2):
    roi = np.divide(ret, cost, out=np.full_like(ret, np.nan), where=cost > 0)
    return roi


def scan(feature_sets, F, strategies, month_idx, min_races, min_hits):
    """各 (特徴の組, 値, 買い方) について 発見用/検証用 の集計と月別回収率を返す。"""
    rows = []
    n_cells_tested = 0
    for fs in feature_sets:
        cols = [F[f] for f in fs]
        valid = np.all([np.array([v is not None for v in c]) for c in cols], 0) & (month_idx >= 0)
        if len(cols) == 1:
            codes, uniq = pd.factorize(pd.Series(cols[0]).where(valid, None))
            labels = [str(u) for u in uniq]
        else:
            key = pd.Series(["|".join(map(str, t)) if ok else None for t, ok in zip(zip(*cols), valid)])
            codes, uniq = pd.factorize(key)
            labels = list(uniq)
        G = len(labels)
        if G == 0:
            continue
        ok = codes >= 0
        k = codes[ok] * NM + month_idx[ok]
        for sname, (cost, ret) in strategies.items():
            c = cost[ok]; r = ret[ok]
            act = (c > 0).astype(np.float64)
            C = np.bincount(k, c, G * NM).reshape(G, NM)
            R = np.bincount(k, r, G * NM).reshape(G, NM)
            R2 = np.bincount(k, r.astype(np.float64) ** 2, G * NM).reshape(G, NM)
            C2 = np.bincount(k, c.astype(np.float64) ** 2, G * NM).reshape(G, NM)
            CR = np.bincount(k, c.astype(np.float64) * r, G * NM).reshape(G, NM)
            N = np.bincount(k, act, G * NM).reshape(G, NM)
            H = np.bincount(k, (r > 0).astype(np.float64), G * NM).reshape(G, NM)
            d = slice(0, N_DISC_M); v = slice(N_DISC_M, NM)
            nd = N[:, d].sum(1); hd_ = H[:, d].sum(1)
            elig = (nd >= min_races) & (hd_ >= min_hits)
            n_cells_tested += int(elig.sum())
            for gi in np.nonzero(elig)[0]:
                rec = {"features": "×".join(fs), "value": labels[gi], "strategy": sname}
                for tag, sl in (("d", d), ("v", v)):
                    cs, rs = C[gi, sl], R[gi, sl]
                    n = N[gi, sl].sum(); csum = cs.sum(); rsum = rs.sum()
                    roi = rsum / csum if csum > 0 else np.nan
                    # デルタ法の標準誤差（レース単位の比推定）
                    if n > 1 and csum > 0:
                        cbar = csum / n
                        var = (R2[gi, sl].sum() - 2 * roi * CR[gi, sl].sum() + roi ** 2 * C2[gi, sl].sum()) / (n - 1)
                        se = math.sqrt(max(var, 0) / n) / cbar
                    else:
                        se = np.nan
                    m_on = cs > 0
                    mroi = rs[m_on] / cs[m_on]
                    rec.update({f"{tag}_races": int(n), f"{tag}_hits": int(H[gi, sl].sum()),
                                f"{tag}_bet": float(csum), f"{tag}_ret": float(rsum), f"{tag}_roi": roi, f"{tag}_se": se,
                                f"{tag}_months": int(m_on.sum()), f"{tag}_win_m": int((mroi > 1.0).sum()),
                                f"{tag}_med_m": float(np.median(mroi)) if len(mroi) else np.nan,
                                f"{tag}_worst_m": float(mroi.min()) if len(mroi) else np.nan})
                    if tag == "v":
                        rec["v_monthly"] = [round(x, 2) if c_ > 0 else None for x, c_ in
                                            zip(np.divide(rs, cs, out=np.zeros_like(rs), where=cs > 0), cs)]
                rows.append(rec)
    return pd.DataFrame(rows), n_cells_tested


def day_bootstrap(hd, cost, ret, mask, B, rng):
    """日単位ブロックのブートストラップで回収率の 95% 区間。"""
    df = pd.DataFrame({"hd": hd[mask], "c": cost[mask], "r": ret[mask]}).groupby("hd").sum()
    c = df.c.to_numpy(); r = df.r.to_numpy()
    if len(c) < 5:
        return (np.nan, np.nan)
    idx = rng.integers(0, len(c), (B, len(c)))
    rois = r[idx].sum(1) / c[idx].sum(1)
    return tuple(np.percentile(rois, [2.5, 97.5]))


def cell_mask(F, features, value, n):
    fs = features.split("×"); vals = value.split("|") if len(fs) > 1 else [value]
    m = np.ones(n, bool)
    for f, v in zip(fs, vals):
        m &= np.array([str(x) == v if x is not None else False for x in F[f]])
    return m


# ---------------------------------------------------------------- 出力
def fmt_table(df, cols, headers):
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", help="結果表を書き出す Markdown のパス（省略時は標準出力）")
    ap.add_argument("--csv", help="全セルの集計を CSV で書き出す（任意）")
    args = ap.parse_args()

    con = connect()
    races, void_strict, void_all, pay, ent = load(con)
    nr = len(races)
    hd = races.hd.to_numpy()
    mi = {m: i for i, m in enumerate(MONTHS)}
    month_idx = races.hd.str[:6].map(mi).fillna(-1).astype(int).to_numpy()
    W, has = wide_entries(nr, ent)
    t0 = time.time()
    book = Book(nr, pay, void_strict)
    strategies = build_strategies(book, W, has)
    print(f"[strategies] {len(strategies)} ({time.time()-t0:.0f}s)", file=sys.stderr)
    F_pre, F_day = build_features(races, W, has)

    out = []
    P = out.append
    is_d = (month_idx >= 0) & (month_idx < N_DISC_M); is_v = month_idx >= N_DISC_M
    P(f"# roi_scan 結果（自動生成 {time.strftime('%Y-%m-%d %H:%M')}）\n")
    P(f"- 対象レース: 発見用 {is_d.sum()} / 検証用 {is_v.sum()}（3連単払戻があるレース）")
    P(f"- 番組表あり: 発見用 {(has & is_d).sum()} / 検証用 {(has & is_v).sum()}")
    P(f"- 欠場（返還）艇を含むレース: {void_strict.any(1).sum()}、position NULL を含むレース: {void_all.any(1).sum()}\n")

    # ---- 基準線
    book_all = Book(nr, pay, void_all)
    base = []
    strat_all = build_strategies(book_all, W, has)
    for name, (cost, ret) in strategies.items():
        rec = {"買い方": name}
        for tag, m in (("発見", is_d), ("検証", is_v)):
            c, r = cost[m], ret[m]
            rec[f"{tag}R"] = int((c > 0).sum())
            rec[f"{tag}的中率"] = f"{(r[c > 0] > 0).mean():.3f}" if (c > 0).any() else "-"
            rec[f"{tag}回収率"] = f"{r.sum() / c.sum():.3f}" if c.sum() > 0 else "-"
            if tag == "検証":
                mm = pd.DataFrame({"m": month_idx[m], "c": c, "r": r}).groupby("m").sum()
                mm = mm[mm.c > 0]; mr = mm.r / mm.c
                rec["検証勝ち月"] = f"{(mr > 1).sum()}/{len(mr)}"
                rec["検証月中央値"] = f"{mr.median():.3f}"
                rec["検証最悪月"] = f"{mr.min():.3f}"
        ca, ra = strat_all[name]
        rec["全返還扱い(検証)"] = f"{ra[is_v].sum() / ca[is_v].sum():.3f}" if ca[is_v].sum() > 0 else "-"
        base.append(rec)
    base = pd.DataFrame(base)
    P("## 基準線（全レースで買った場合）\n")
    P(fmt_table(base, list(base.columns), list(base.columns)))
    P("")

    # ---- 総当たり（前夜）
    pre_names = list(F_pre)
    redundant = {frozenset(["R", "R帯"]), frozenset(["月", "季節"])}
    fsets = [(f,) for f in pre_names] + [p for p in itertools.combinations(pre_names, 2) if frozenset(p) not in redundant]
    t0 = time.time()
    res, ntested = scan(fsets, F_pre, strategies, month_idx, MIN_RACES, MIN_HITS)
    print(f"[scan pre] rows={len(res)} tested={ntested} ({time.time()-t0:.0f}s)", file=sys.stderr)
    if args.csv:
        res.drop(columns=["v_monthly"]).to_csv(args.csv, index=False)

    res["d_z"] = (res.d_roi - 1) / res.d_se
    res["v_z"] = (res.v_roi - 1) / res.v_se
    # 片側 p<0.05/N の z 閾値（正規近似）
    from statistics import NormalDist
    bonf_z = NormalDist().inv_cdf(1 - 0.05 / max(ntested, 1))

    P("## 総当たりの規模\n")
    P(f"- 条件の組: 単独 {len(pre_names)} 特徴 + 2特徴の組 {len(fsets) - len(pre_names)}（計 {len(fsets)} 通りの切り口）")
    P(f"- 買い方: {len(strategies)} 通り")
    P(f"- 発見用で最低サンプル（{MIN_RACES}レース・{MIN_HITS}的中）を満たした (条件値×買い方) セル: **{ntested}**")
    P(f"- Bonferroni（片側 α=0.05/{ntested}）の z 閾値: {bonf_z:.2f}")
    nd1 = (res.d_roi > 1).sum()
    P(f"- 発見用で回収率>1.0 のセル: {nd1}（{nd1 / len(res):.1%}）、うち z>{bonf_z:.2f}: {(res.d_z > bonf_z).sum()}")
    sub = res[res.d_roi > 1]
    P(f"- 発見用で>1.0 だったセルの検証用回収率: 中央値 {sub.v_roi.median():.3f}、>1.0 は {(sub.v_roi > 1).mean():.1%}")
    ctl = res[(res.d_roi > 0.7) & (res.d_roi < 0.8)]
    P(f"- 対照: 発見用 0.7〜0.8 のセルの検証用 >1.0 割合 {(ctl.v_roi > 1).mean():.1%}")
    P(f"- 発見用回収率と検証用回収率の相関（全セル, Spearman）: {res[['d_roi', 'v_roi']].corr('spearman').iloc[0, 1]:.3f}\n")
    for lo, hi in [(0, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0), (1.0, 1.1), (1.1, 1.3), (1.3, 99)]:
        s = res[(res.d_roi >= lo) & (res.d_roi < hi)]
        if len(s):
            P(f"  - 発見用 {lo}〜{hi}: {len(s)} セル → 検証用 中央値 {s.v_roi.median():.3f} / 加重 {s.v_ret.sum() / s.v_bet.sum():.3f} / >1.0 {(s.v_roi > 1).mean():.1%}")
    P("")

    passed = (res.v_roi > ADOPT_ROI) & (res.v_win_m >= ADOPT_WIN)
    P(f"- 採用の合格ライン（検証用{ADOPT_WINDOW}か月で通算>{ADOPT_ROI} かつ 勝ち月>={ADOPT_WIN}）: "
      f"全セル中 {int(passed.sum())}、うち発見用でも>1.0: {int((passed & (res.d_roi > 1)).sum())}\n")
    rng = np.random.default_rng(SEED)

    def detail(df, Fdict, k):
        rows = []
        for _, r in df.head(k).iterrows():
            m = cell_mask(Fdict, r.features, r.value, nr)
            cost, ret = strategies[r.strategy]
            lo_d, hi_d = day_bootstrap(hd, cost, ret, m & is_d, BOOT, rng)
            lo_v, hi_v = day_bootstrap(hd, cost, ret, m & is_v, BOOT, rng)
            rows.append({
                "条件": f"{r.features}={r.value}", "買い方": r.strategy,
                "発見R": r.d_races, "発見的中率": f"{r.d_hits / max(r.d_races, 1):.3f}",
                "発見回収率[95%]": f"{r.d_roi:.3f} [{lo_d:.2f},{hi_d:.2f}]",
                "発見勝ち月": f"{r.d_win_m}/{r.d_months}",
                "検証R": r.v_races, "検証的中率": f"{r.v_hits / max(r.v_races, 1):.3f}",
                "検証回収率[95%]": f"{r.v_roi:.3f} [{lo_v:.2f},{hi_v:.2f}]",
                "検証勝ち月": f"{r.v_win_m}/{r.v_months}",
                "検証月中央値": f"{r.v_med_m:.2f}", "検証最悪月": f"{r.v_worst_m:.2f}",
                "合格": "○" if (r.v_roi > ADOPT_ROI and r.v_win_m >= ADOPT_WIN) else "×",
                "検証月別": " ".join("-" if x is None else f"{x:.2f}" for x in r.v_monthly),
            })
        return pd.DataFrame(rows)

    # 候補: 発見用の z（=偶然では説明しにくさ）上位。重複を減らすため買い方×条件で並べる
    cand = res[res.d_roi > 1].sort_values("d_z", ascending=False)
    P(f"## 発見用 z 上位 {TOP_K}（→ 検証用での成績）\n")
    tab = detail(cand, F_pre, TOP_K)
    P(fmt_table(tab, list(tab.columns), list(tab.columns)))
    P("")
    cand2 = res[(res.d_roi > 1) & (res.d_races >= 1000)].sort_values("d_roi", ascending=False)
    P(f"## 発見用 回収率上位（1000レース以上）{TOP_K}（→ 検証用）\n")
    tab = detail(cand2, F_pre, TOP_K)
    P(fmt_table(tab, list(tab.columns), list(tab.columns)))
    P("")
    # 両期間で残ったもの
    both = res[(res.d_roi > 1) & (res.v_roi > 1) & (res.v_races >= 150)].copy()
    both["score"] = np.minimum(both.d_z, both.v_z)
    both = both.sort_values("score", ascending=False)
    P(f"## 発見用・検証用ともに回収率>1.0（検証150レース以上）: {len(both)} セル。min(z) 上位\n")
    tab = detail(both, F_pre, TOP_K)
    P(fmt_table(tab, list(tab.columns), list(tab.columns)))
    P("")
    # 検証で勝ち月が多いもの（主評価）
    vm = res[(res.d_roi > 1) & (res.v_months >= 10) & (res.v_races >= 150)].sort_values(
        ["v_win_m", "v_med_m"], ascending=False)
    P("## 発見用>1.0 のうち、検証用の勝ち月数が多い順（主評価）\n")
    tab = detail(vm, F_pre, 25)
    P(fmt_table(tab, list(tab.columns), list(tab.columns)))
    P("")
    pp = res[(res.d_roi > 1) & (res.v_roi > ADOPT_ROI) & (res.v_win_m >= ADOPT_WIN)].sort_values("v_roi", ascending=False)
    P(f"## 合格ライン到達 かつ 発見用>1.0 のセル（{len(pp)}）\n")
    if len(pp):
        tab = detail(pp, F_pre, 50)
        P(fmt_table(tab, list(tab.columns), list(tab.columns)))
    P("")
    # 偶然だった例
    gone = res[(res.d_roi > 1.3) & (res.d_races >= 500)].sort_values("d_roi", ascending=False)
    gone = gone[gone.v_roi < 0.85]
    P("## 検証で消えた例（発見用>1.3・500レース以上 → 検証用<0.85）\n")
    tab = detail(gone, F_pre, 15)
    P(fmt_table(tab, list(tab.columns), list(tab.columns)))
    P("")

    # ---- 当日情報（参考）
    F_mix = dict(F_pre); F_mix.update(F_day)
    day_sets = [(f,) for f in F_day] + [(a, b) for a in F_day for b in ["場", "1号艇級", "1号艇-他最高 勝率差", "R帯"]] \
        + [("風速", "波高")]
    res_d, nt_d = scan(day_sets, F_mix, strategies, month_idx, MIN_RACES, MIN_HITS)
    res_d["d_z"] = (res_d.d_roi - 1) / res_d.d_se
    P(f"## 当日情報（天候・風速・波高）を使った場合の参考（{nt_d} セル試行）\n")
    single = res_d[res_d.features.isin(list(F_day))]
    keyS = ["単勝 1", "3連単 1-全-全", "全通り 3連単", "単勝 4", "単勝 5", "3連単 1-2-3", "2連単 1-全"]
    s = single[single.strategy.isin(keyS)].copy()
    s["発見"] = s.d_roi.map("{:.3f}".format); s["検証"] = s.v_roi.map("{:.3f}".format)
    s["検証R"] = s.v_races; s["検証勝ち月"] = s.v_win_m.astype(str) + "/" + s.v_months.astype(str)
    P(fmt_table(s.sort_values(["features", "value", "strategy"]), ["features", "value", "strategy", "発見", "検証", "検証R", "検証勝ち月"],
                ["当日特徴", "値", "買い方", "発見回収率", "検証回収率", "検証R", "検証勝ち月"]))
    P("")
    both_d = res_d[(res_d.d_roi > 1) & (res_d.v_roi > 1) & (res_d.v_races >= 150)].copy()
    both_d["v_z"] = (both_d.v_roi - 1) / both_d.v_se
    both_d["score"] = np.minimum(both_d.d_z, both_d.v_z)
    P(f"### 当日情報込みで両期間>1.0 のセル（{len(both_d)}）上位\n")
    tab = detail(both_d.sort_values("score", ascending=False), F_mix, 15)
    P(fmt_table(tab, list(tab.columns), list(tab.columns)))
    P("")

    text = "\n".join(out)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        print(text)


if __name__ == "__main__":
    main()
