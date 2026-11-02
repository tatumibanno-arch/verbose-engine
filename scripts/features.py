"""前夜（番組表が出た時点）に使える情報だけで作る、1艇1行の特徴量テーブル。

使い方:
    python scripts/features.py --from 20251101 --to 20261031 --out /path/features.parquet
    python scripts/features.py --leak-check 20260315,20260610,20261001
    python scripts/features.py --report /path/features.parquet   # 単変量の1着との関係

関数:
    build_features(con, hd_from, hd_to, stats_lag_days=0) -> DataFrame

約束（リーク対策）:
    - レース日 hd の特徴量は「hd より前の日」の results と、
      「hd より前（+lag）に算出期間が終わった」racer_stats と、
      hd 当日までの entries（番組表は前夜に出る）だけから作る。
    - 当日のレース結果・展示タイム・進入・ST・天候・オッズ・直前情報は使わない。
      同じ日の前半レースの結果も使わない（前夜に作る前提なので）。
    - 列名が y_ で始まるものはラベル（当日の結果）。特徴量として使ってはいけない。

詳しくは notes/analysis/features.md。
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "boatrace.db"

GRADE_MAP = {"A1": 4, "A2": 3, "B1": 2, "B2": 1}

# ---- パラメータ（特徴量の定義。ルールではないので rules.yaml には置かない） ----
RACER_WINDOWS = (30, 90, 180)       # 選手の直近成績の窓（日）
COURSE_WINDOW = 365                 # コース別・当地・枠なり率の窓（日）
BASE_WINDOWS = (60, 365)            # 場×枠番の基礎確率の窓（日）
EXPECT_WINDOW = 365                 # モーター残差の期待値テーブルの窓（日）
MOTOR_MAX_DAYS = 400                # モーター期の上限（検出漏れの保険）
MOTOR_RESET_ZERO_SHARE = 0.8        # 番組表の motor_2ren_rate==0 の割合がこれ以上 → 入れ替え日
MOTOR_RESET_MIN_GAP = 180           # 入れ替え日どうしの最小間隔（日）
SHRINK_K = 10.0                     # コース別率の縮小推定の強さ（仮想レース数）

_K = 1 << 20  # 合成キー: key * _K + day


def connect_ro(path: str | Path = DB_PATH) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=60)


def _day(hd: pd.Series) -> np.ndarray:
    """'YYYYMMDD' → 1970-01-01 からの日数（int64）"""
    return pd.to_datetime(hd, format="%Y%m%d").values.astype("datetime64[D]").astype(np.int64)


# --------------------------------------------------------------------------
# 窓付き集計（過去のみ）
# --------------------------------------------------------------------------
class _Hist:
    """履歴行 (key, day, values) を持ち、問い合わせ (key, day) に対して
    [day - window, day - 1] の合計を返す。day 当日は含まない。"""

    def __init__(self, key: np.ndarray, day: np.ndarray, vals: dict[str, np.ndarray]):
        comp = key.astype(np.int64) * _K + day.astype(np.int64)
        order = np.argsort(comp, kind="stable")
        self.comp = comp[order]
        self.day = day.astype(np.int64)[order]
        self.names = list(vals)
        m = np.column_stack([np.asarray(vals[n], dtype=np.float64)[order] for n in self.names])
        self.cum = np.vstack([np.zeros((1, m.shape[1])), np.cumsum(m, axis=0)])

    def query(self, qkey, qday, window=None, start_day=None) -> pd.DataFrame:
        qkey = np.asarray(qkey, dtype=np.int64)
        qday = np.asarray(qday, dtype=np.int64)
        hi = np.searchsorted(self.comp, qkey * _K + qday, "left")
        if window is None:
            lo_day = np.zeros_like(qday)
        else:
            lo_day = qday - window
        if start_day is not None:
            lo_day = np.maximum(lo_day, np.asarray(start_day, dtype=np.int64))
        lo = np.searchsorted(self.comp, qkey * _K + lo_day, "left")
        lo = np.minimum(lo, hi)
        s = self.cum[hi] - self.cum[lo]
        return pd.DataFrame(s, columns=self.names)

    def last_day(self, qkey, qday) -> np.ndarray:
        """key の行のうち qday より前で最も新しい日（無ければ NaN）"""
        qkey = np.asarray(qkey, dtype=np.int64)
        hi = np.searchsorted(self.comp, qkey * _K + np.asarray(qday, dtype=np.int64), "left")
        idx = hi - 1
        ok = (idx >= 0) & (self.comp[np.maximum(idx, 0)] // _K == qkey)
        out = np.full(len(qkey), np.nan)
        out[ok] = self.day[idx[ok]]
        return out


def _rate(num, den):
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


# --------------------------------------------------------------------------
# 読み込み
# --------------------------------------------------------------------------
_ENTRY_JSON = [
    "national_win_rate", "national_2ren_rate", "local_win_rate", "local_2ren_rate",
    "motor_2ren_rate", "boat_2ren_rate", "age", "weight", "series_results",
    "early_hayami", "distance",
]


def load_targets(con, hd_from: str, hd_to: str) -> pd.DataFrame:
    cols = ", ".join(f"json_extract(data,'$.{c}') AS {c}" for c in _ENTRY_JSON)
    q = f"""SELECT hd, jcd, rno, lane, racer_no, grade, motor_no, boat_no, close_time,
                   race_name, {cols}
            FROM entries WHERE hd BETWEEN ? AND ?"""
    df = pd.read_sql(q, con, params=(hd_from, hd_to))
    df["racer"] = pd.to_numeric(df["racer_no"], errors="coerce").fillna(-1).astype(np.int64)
    df["jcd_i"] = df["jcd"].astype(int)
    df["day"] = _day(df["hd"])
    return df.sort_values(["hd", "jcd", "rno", "lane"]).reset_index(drop=True)


def venue_fix_map(con, hd_lo: str, hd_hi_excl: str) -> pd.DataFrame:
    """results / payouts で唐津(23)が津(09)として入っている取り込みバグの補正表。
    jcd='09' のレースごとに、6艇の (hd, rno, lane, 選手) を番組表の 09 / 23 と突き合わせ、
    多数決で本当の場を決める。返り値: hd, rno, jcd_fixed（'09' か '23'）。
    番組表は前夜に出るので、レース日以前の番組表を使うのはリークではない。"""
    r = pd.read_sql("""SELECT hd, rno, lane, json_extract(data,'$.racer_number') AS racer
                       FROM results WHERE jcd='09' AND hd >= ? AND hd < ?""", con, params=(hd_lo, hd_hi_excl))
    e = pd.read_sql("""SELECT hd, jcd AS jcd_fixed, rno, lane, CAST(racer_no AS int) AS racer
                       FROM entries WHERE jcd IN ('09','23') AND hd >= ? AND hd < ?""",
                    con, params=(hd_lo, hd_hi_excl))
    if r.empty:
        return pd.DataFrame(columns=["hd", "rno", "jcd_fixed"])
    r["racer"] = pd.to_numeric(r["racer"], errors="coerce")
    m = r.merge(e, on=["hd", "rno", "lane", "racer"], how="inner")
    c = m.groupby(["hd", "rno", "jcd_fixed"]).size().rename("k").reset_index()
    c = c.sort_values("k", ascending=False).drop_duplicates(["hd", "rno"])
    return c[["hd", "rno", "jcd_fixed"]]


def _apply_venue_fix(df: pd.DataFrame, fix: pd.DataFrame) -> pd.DataFrame:
    if fix.empty or not (df["jcd"] == "09").any():
        return df
    df = df.merge(fix, on=["hd", "rno"], how="left")
    sel = (df["jcd"] == "09") & df["jcd_fixed"].notna()
    df.loc[sel, "jcd"] = df.loc[sel, "jcd_fixed"]
    return df.drop(columns=["jcd_fixed"])


def load_results(con, hd_before: str) -> pd.DataFrame:
    """hd_before より前の全成績（1艇1行）。唐津/津の取り違えを補正済み。"""
    q = """SELECT hd, jcd, rno, lane, position,
                  json_extract(data,'$.racer_number') AS racer,
                  json_extract(data,'$.motor_number') AS motor,
                  json_extract(data,'$.exhibition_time') AS ex,
                  json_extract(data,'$.entrance_position') AS course,
                  json_extract(data,'$.st_timing') AS st
           FROM results WHERE hd < ?"""
    h = pd.read_sql(q, con, params=(hd_before,))
    h = _apply_venue_fix(h, venue_fix_map(con, "00000000", hd_before))
    h["day"] = _day(h["hd"])
    h["jcd_i"] = h["jcd"].astype(int)
    for c in ("racer", "motor", "course"):
        h[c] = pd.to_numeric(h[c], errors="coerce").fillna(-1).astype(np.int64)
    for c in ("position", "ex", "st"):
        h[c] = pd.to_numeric(h[c], errors="coerce")
    h.loc[h["ex"] <= 0, "ex"] = np.nan
    return h


_RS_TOP = ["win_rate", "top2_rate", "avg_st", "ability", "ability_prev", "starts",
           "first_count", "second_count", "final_count", "champion_count", "training_term"]
_RS_C = ["starts", "pos1", "pos2", "pos3", "top2_rate", "avg_st", "avg_start_rank", "F", "L0", "K0"]


def load_racer_stats(con, hd_before: str) -> pd.DataFrame:
    cols = [f"json_extract(data,'$.{c}') AS rs_{c}" for c in _RS_TOP]
    cols += [f"json_extract(data,'$.c{n}_{c}') AS c{n}_{c}" for n in range(1, 7) for c in _RS_C]
    cols += ["json_extract(data,'$.grade') AS rs_grade", "json_extract(data,'$.grade_prev1') AS rs_grade_prev1"]
    q = f"SELECT hd, racer_no, {', '.join(cols)} FROM racer_stats WHERE hd < ?"
    rs = pd.read_sql(q, con, params=(hd_before,))
    rs["racer"] = pd.to_numeric(rs["racer_no"], errors="coerce").fillna(-1).astype(np.int64)
    rs["rs_day"] = _day(rs["hd"])
    for c in rs.columns:
        if c.startswith(("rs_", "c")) and c not in ("rs_grade", "rs_grade_prev1", "rs_day"):
            rs[c] = pd.to_numeric(rs[c], errors="coerce")
    return rs.drop(columns=["hd", "racer_no"])


def motor_boundaries(con, hd_max: str) -> dict[int, np.ndarray]:
    """場ごとのモーター入れ替え日（日数）。番組表の motor_2ren_rate が場の8割以上で 0 になった
    最初の日を入れ替え日とする（hd_max 以前の番組表だけを見る＝前夜に分かる）。"""
    q = """SELECT hd, jcd,
                  SUM(json_extract(data,'$.motor_2ren_rate') = 0) AS z,
                  SUM(json_extract(data,'$.motor_2ren_rate') IS NOT NULL) AS n
           FROM entries WHERE hd <= ? GROUP BY hd, jcd"""
    df = pd.read_sql(q, con, params=(hd_max,))
    df["day"] = _day(df["hd"])
    df["hi"] = (df["z"] / df["n"].replace(0, np.nan)) >= MOTOR_RESET_ZERO_SHARE
    out: dict[int, np.ndarray] = {}
    for jcd, g in df.sort_values("day").groupby("jcd"):
        prev = g["hi"].shift(1, fill_value=False).astype(bool)
        cand = g.loc[g["hi"] & ~prev, "day"].tolist()
        keep: list[int] = []
        for d in cand:
            if not keep or d - keep[-1] >= MOTOR_RESET_MIN_GAP:
                keep.append(int(d))
        out[int(jcd)] = np.array(keep, dtype=np.int64)
    return out


def _epoch(jcd_i: np.ndarray, day: np.ndarray, bounds: dict[int, np.ndarray]):
    """(epoch 番号, epoch 開始日) を返す。開始日が分からない最初の期は -1。"""
    ep = np.zeros(len(day), dtype=np.int64)
    start = np.full(len(day), -1, dtype=np.int64)
    for j in np.unique(jcd_i):
        m = jcd_i == j
        b = bounds.get(int(j), np.array([], dtype=np.int64))
        k = np.searchsorted(b, day[m], "right")  # day 以前の入れ替え回数
        ep[m] = k
        st = np.full(m.sum(), -1, dtype=np.int64)
        st[k > 0] = b[k[k > 0] - 1]
        start[m] = st
    return ep, start


def _attach_rs(df: pd.DataFrame, rs: pd.DataFrame, lag: int) -> pd.DataFrame:
    """各行に、day より前（rs_day + lag < day）に算出期間が終わった最新の期別成績を付ける。"""
    left = df[["racer", "day"]].copy()
    left["_i"] = np.arange(len(df))
    left = left.sort_values("day")
    right = rs.copy()
    right["key_day"] = right["rs_day"] + lag
    right = right.sort_values("key_day")
    m = pd.merge_asof(left, right, left_on="day", right_on="key_day", by="racer",
                      allow_exact_matches=False, direction="backward")
    return m.sort_values("_i").drop(columns=["_i", "racer", "day", "key_day"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# 本体
# --------------------------------------------------------------------------
def _parse_series(s: pd.Series) -> pd.DataFrame:
    """今節成績（例 '5 336' / '433 S6'）。数字=着順、F/L/K/S=事故・欠場。"""
    s = s.fillna("").astype(str)
    chars = s.str.replace(" ", "", regex=False)
    n = chars.str.len()
    digits = chars.str.replace(r"[^1-6]", "", regex=True)
    nd = digits.str.len()
    dsum = digits.apply(lambda x: sum(int(c) for c in x)) if len(s) else pd.Series(dtype=float)
    last = digits.str[-1:].replace("", np.nan).astype(float)
    return pd.DataFrame({
        "ser_n": n,
        "ser_avg_pos": _rate(dsum, nd),
        "ser_win": _rate(digits.str.count("1"), nd),
        "ser_top3": _rate(digits.str.count("[123]"), nd),
        "ser_last_pos": last,
        "ser_F": chars.str.count("F"),
        "ser_accident": chars.str.count("[LKS]"),
    })


def build_features(con, hd_from: str, hd_to: str, stats_lag_days: int = 0,
                   with_labels: bool = True, verbose: bool = False) -> pd.DataFrame:
    t0 = time.time()

    def log(msg):
        if verbose:
            print(f"[{time.time() - t0:6.1f}s] {msg}", file=sys.stderr)

    T = load_targets(con, hd_from, hd_to)
    if T.empty:
        return T
    H = load_results(con, hd_to)          # hd_to より前の全成績（各行の参照はさらに day 未満に絞る）
    rs = load_racer_stats(con, hd_to)
    bounds = motor_boundaries(con, hd_to)
    log(f"loaded targets={len(T)} hist={len(H)} racer_stats={len(rs)}")

    out = T[["hd", "jcd", "rno", "lane", "racer_no", "motor_no", "boat_no"]].copy()
    tday = T["day"].values
    lane = T["lane"].values.astype(np.int64)

    # ---- 1. 番組表 ------------------------------------------------------
    out["grade"] = T["grade"].map(GRADE_MAP)
    for c in ["national_win_rate", "national_2ren_rate", "local_win_rate", "local_2ren_rate",
              "motor_2ren_rate", "boat_2ren_rate", "age", "weight"]:
        out[c] = pd.to_numeric(T[c], errors="coerce")
    out["local_win_rate_missing"] = (out["local_win_rate"].fillna(0) == 0).astype(int)
    out.loc[out["local_win_rate_missing"] == 1, ["local_win_rate", "local_2ren_rate"]] = np.nan
    out["local_minus_national"] = out["local_win_rate"] - out["national_win_rate"]
    out["has_hayami"] = T["early_hayami"].notna().astype(int)
    out["distance"] = pd.to_numeric(T["distance"], errors="coerce")
    rn = T["race_name"].fillna("")
    out["r_fixed_course"] = rn.str.contains("進入固定").astype(int)
    out["r_final"] = (rn.str.contains("優勝戦") & ~rn.str.contains("準優")).astype(int)
    out["r_semifinal"] = rn.str.contains("準優").astype(int)
    out["r_special"] = rn.str.contains("ドリーム|選抜|特選|特賞|記念").astype(int)
    out["r_qualify"] = rn.str.contains("予選").astype(int)
    ct = T["close_time"].fillna("")
    out["close_hour"] = pd.to_numeric(ct.str[:2], errors="coerce")
    out = pd.concat([out, _parse_series(T["series_results"])], axis=1)
    log("entries done")

    # ---- 2. 期別成績（算出期間の最終日 < レース日 - lag） ------------------
    R = _attach_rs(T, rs, stats_lag_days)
    for c in _RS_TOP:
        out[f"rs_{c}"] = R[f"rs_{c}"].values
    out.loc[out["rs_starts"].fillna(0) == 0, ["rs_avg_st", "rs_win_rate", "rs_top2_rate"]] = np.nan
    out["rs_grade"] = R["rs_grade"].map(GRADE_MAP).values
    out["rs_grade_prev1"] = R["rs_grade_prev1"].map(GRADE_MAP).values
    out["rs_age_days"] = tday - R["rs_day"].values  # 期別成績の古さ
    idx = np.clip(lane, 1, 6)

    def pick(field):
        mat = np.column_stack([R[f"c{n}_{field}"].values for n in range(1, 7)]).astype(float)
        return mat[np.arange(len(mat)), idx - 1], mat

    cst, cst_all = pick("starts")
    p1, _ = pick("pos1"); p2, _ = pick("pos2"); p3, _ = pick("pos3")
    out["rs_c_starts"] = cst
    out["rs_c_win"] = _rate(p1, cst)
    out["rs_c_top2"] = _rate(p1 + p2, cst)
    out["rs_c_top3"] = _rate(p1 + p2 + p3, cst)
    out["rs_c_avg_st"], _ = pick("avg_st")
    out["rs_c_avg_start_rank"], _ = pick("avg_start_rank")
    out.loc[out["rs_c_starts"].fillna(0) == 0, ["rs_c_avg_st", "rs_c_avg_start_rank"]] = np.nan
    _, Fm = pick("F"); _, Lm = pick("L0"); _, Km = pick("K0")
    out["rs_F"] = np.nansum(Fm, axis=1)
    out["rs_L"] = np.nansum(Lm, axis=1)
    out["rs_K"] = np.nansum(Km, axis=1)
    out.loc[R["rs_day"].isna().values, ["rs_F", "rs_L", "rs_K"]] = np.nan
    out["rs_course_share"] = _rate(cst, np.nansum(cst_all, axis=1))  # その枠のコースに入る割合
    out = out.copy()
    log("racer_stats done")

    # ---- 3. 選手の直近成績（results, day 未満） -----------------------------
    H = H.sort_values(["day", "jcd_i", "rno", "lane"]).reset_index(drop=True)
    pos = H["position"].values
    st = H["st"].values
    ran = (~np.isnan(pos)) | (~np.isnan(st))
    fin = ~np.isnan(pos)
    grp = H.groupby(["day", "jcd_i", "rno"], sort=False)
    ex_mean = grp["ex"].transform("mean").values
    st_rank = grp["st"].rank(method="average").values
    base_vals = {
        "n": ran.astype(float),
        "fin": fin.astype(float),
        "win": (pos == 1).astype(float),
        "top2": (pos <= 2).astype(float),
        "top3": (pos <= 3).astype(float),
        "pos": np.nan_to_num(pos),
        "acc": (ran & ~fin).astype(float),
        "st_n": (~np.isnan(st)).astype(float),
        "st": np.nan_to_num(st),
        "st2": np.nan_to_num(st) ** 2,
        "st_rank": np.nan_to_num(st_rank),
        "ex_n": (~np.isnan(H["ex"].values - ex_mean)).astype(float),
        "ex_res": np.nan_to_num(H["ex"].values - ex_mean),
    }
    hr = _Hist(H["racer"].values, H["day"].values, base_vals)
    rk = T["racer"].values
    for w in RACER_WINDOWS:
        S = hr.query(rk, tday, w)
        p = f"r{w}_"
        out[p + "n"] = S["n"].values
        out[p + "win"] = _rate(S["win"], S["n"])
        out[p + "top2"] = _rate(S["top2"], S["n"])
        out[p + "top3"] = _rate(S["top3"], S["n"])
        out[p + "avg_pos"] = _rate(S["pos"], S["fin"])
        out[p + "acc"] = _rate(S["acc"], S["n"])
        out[p + "avg_st"] = _rate(S["st"], S["st_n"])
        out[p + "st_sd"] = np.sqrt(np.maximum(_rate(S["st2"], S["st_n"]) - _rate(S["st"], S["st_n"]) ** 2, 0))
        out[p + "st_rank"] = _rate(S["st_rank"], S["st_n"])
        out[p + "ex_res"] = _rate(S["ex_res"], S["ex_n"])
    last = hr.last_day(rk, tday)
    out["days_since_last_race"] = tday - last

    # コース別（実際の進入コース基準）の直近成績。枠番=コースと仮定して lane のコースを引く
    hc = _Hist(H["racer"].values * 10 + H["course"].values, H["day"].values, base_vals)
    S = hc.query(rk * 10 + lane, tday, COURSE_WINDOW)
    out["rc_n"] = S["n"].values
    out["rc_win"] = _rate(S["win"], S["n"])
    out["rc_top2"] = _rate(S["top2"], S["n"])
    out["rc_top3"] = _rate(S["top3"], S["n"])
    out["rc_avg_st"] = _rate(S["st"], S["st_n"])
    out["rc_st_rank"] = _rate(S["st_rank"], S["st_n"])

    # 当地（同じ場）の直近成績
    hj = _Hist(H["racer"].values * 100 + H["jcd_i"].values, H["day"].values, base_vals)
    S = hj.query(rk * 100 + T["jcd_i"].values, tday, COURSE_WINDOW)
    out["rl_n"] = S["n"].values
    out["rl_win"] = _rate(S["win"], S["n"])
    out["rl_top3"] = _rate(S["top3"], S["n"])

    # 枠なり率・前付け傾向: この選手がこの枠番に入ったとき、何コースから出たか
    cvals = {
        "n": (H["course"].values > 0).astype(float),
        "same": (H["course"].values == H["lane"].values).astype(float),
        "inward": np.where(H["course"].values > 0, H["lane"].values - H["course"].values, 0).astype(float),
    }
    hl = _Hist(H["racer"].values * 10 + H["lane"].values, H["day"].values, cvals)
    S = hl.query(rk * 10 + lane, tday, COURSE_WINDOW)
    out["rlane_n"] = S["n"].values
    out["rlane_keep"] = _rate(S["same"], S["n"])        # 枠なりで入った割合
    out["rlane_inward"] = _rate(S["inward"], S["n"])    # 平均で何コース内に入ったか（+ が前付け）
    S = _Hist(H["racer"].values, H["day"].values, cvals).query(rk, tday, COURSE_WINDOW)
    out["r_inward_all"] = _rate(S["inward"], S["n"])
    out = out.copy()
    log("racer recent done")

    # ---- 5a. 場×枠番の基礎確率 -------------------------------------------
    hb = _Hist(H["jcd_i"].values * 10 + H["lane"].values, H["day"].values, base_vals)
    for w in BASE_WINDOWS:
        S = hb.query(T["jcd_i"].values * 10 + lane, tday, w)
        out[f"vl{w}_win"] = _rate(S["win"], S["n"])
        out[f"vl{w}_top2"] = _rate(S["top2"], S["n"])
        out[f"vl{w}_top3"] = _rate(S["top3"], S["n"])
    # 場の1コース逃げ率（直近60日）＝水面の荒れ具合の代理
    S = hb.query(T["jcd_i"].values * 10 + 1, tday, 60)
    out["v_lane1_win60"] = _rate(S["win"], S["n"])
    # 縮小推定したコース別1着率（選手のコース別直近 → 場×枠の基礎確率へ寄せる）
    prior = out["vl365_win"].fillna(out["vl365_win"].mean()).values
    out["rc_win_shrunk"] = (np.nan_to_num(out["rc_win"].values * out["rc_n"].values) + SHRINK_K * prior) / (
        out["rc_n"].values + SHRINK_K)
    rsw = np.nan_to_num(out["rs_c_win"].values * out["rs_c_starts"].values)
    out["rs_c_win_shrunk"] = (rsw + SHRINK_K * prior) / (np.nan_to_num(out["rs_c_starts"].values) + SHRINK_K)
    out = out.copy()
    log("base rates done")

    # ---- 4. モーター（場×入れ替え期×motor_no） ---------------------------
    # 期待3着内率: 枠番 × 選手の期別勝率（0.5刻み）の過去365日の実績（その日より前）
    Hrs = _attach_rs(H[["racer", "day"]], rs, stats_lag_days)
    wrbin = np.clip(np.nan_to_num(Hrs["rs_win_rate"].values, nan=0.0) * 2, 0, 19).astype(np.int64)
    ekey = H["lane"].values * 100 + wrbin
    he = _Hist(ekey, H["day"].values, {"n": base_vals["n"], "top3": base_vals["top3"], "win": base_vals["win"]})
    E = he.query(ekey, H["day"].values, EXPECT_WINDOW)
    lane_prior3 = np.array([np.nan, .75, .62, .55, .47, .37, .25])[H["lane"].values.clip(1, 6)]
    lane_prior1 = np.array([np.nan, .55, .14, .12, .11, .06, .02])[H["lane"].values.clip(1, 6)]
    exp3 = (E["top3"].values + 20 * lane_prior3) / (E["n"].values + 20)
    exp1 = (E["win"].values + 20 * lane_prior1) / (E["n"].values + 20)
    H_ep, _ = _epoch(H["jcd_i"].values, H["day"].values, bounds)
    T_ep, T_start = _epoch(T["jcd_i"].values, tday, bounds)
    mvals = dict(base_vals)
    mvals["res3"] = np.where(ran & ~np.isnan(exp3), base_vals["top3"] - exp3, 0)
    mvals["res1"] = np.where(ran & ~np.isnan(exp1), base_vals["win"] - exp1, 0)
    mkey_h = (H["jcd_i"].values * 100 + H_ep) * 1000 + H["motor"].values.clip(0, 999)
    mkey_t = (T["jcd_i"].values * 100 + T_ep) * 1000 + pd.to_numeric(T["motor_no"], errors="coerce").fillna(0).astype(np.int64).clip(0, 999)
    hm = _Hist(mkey_h, H["day"].values, mvals)
    S = hm.query(mkey_t, tday, MOTOR_MAX_DAYS)
    out["m_n"] = S["n"].values
    out["m_win"] = _rate(S["win"], S["n"])
    out["m_top2"] = _rate(S["top2"], S["n"])
    out["m_top3"] = _rate(S["top3"], S["n"])
    out["m_res3"] = _rate(S["res3"], S["n"])      # 選手の腕と枠を差し引いた 3着内率の上振れ
    out["m_res1"] = _rate(S["res1"], S["n"])
    out["m_ex_res"] = _rate(S["ex_res"], S["ex_n"])  # 展示タイムのレース内平均との差（負が速い）
    # 縮小版（n が少ないと 0 に寄せる）
    out["m_res3_shrunk"] = np.nan_to_num(S["res3"].values) / (S["n"].values + 20)
    out["m_ex_res_shrunk"] = np.nan_to_num(S["ex_res"].values) / (S["ex_n"].values + 20)
    out["motor_epoch_days"] = np.where(T_start >= 0, tday - T_start, np.nan)
    out = out.copy()
    log("motor done")

    # ---- 5b. レース内の相対値 --------------------------------------------
    out = _relative(out, [
        "national_win_rate", "rs_win_rate", "rs_ability", "motor_2ren_rate", "m_res3_shrunk",
        "m_ex_res_shrunk", "r90_top3", "r90_win", "r90_avg_st", "rc_win_shrunk", "rs_c_win_shrunk",
        "rs_c_avg_st", "rs_c_avg_start_rank", "local_win_rate", "grade",
    ])
    # レースの格: A1 の数、1号艇の強さ
    out = out.copy()
    race_key = out["hd"] + out["jcd"] + out["rno"].astype(str).str.zfill(2)
    out["race_n_a1"] = (out["grade"] == 4).groupby(race_key).transform("sum").values
    out["race_n_boats"] = out.groupby(race_key)["lane"].transform("count").values
    log("relative done")

    if with_labels:
        out = _attach_labels(con, out)
    log("done")
    return out


def _relative(out: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    """レース内の順位・平均差・他艇最大との差・1号艇との差・シェア。値が大きいほど良いとは限らない
    （ST は小さいほど良い）ので、順位は「大きい順」で付け、意味は名前で判断する。"""
    race_key = out["hd"] + out["jcd"] + out["rno"].astype(str).str.zfill(2)
    codes, uniq = pd.factorize(race_key)
    li = out["lane"].values.clip(1, 6) - 1
    new = {}
    for f in feats:
        M = np.full((len(uniq), 6), np.nan)
        M[codes, li] = out[f].values.astype(float)
        v = M[codes, li]
        valid = ~np.isnan(M)
        cnt = valid.sum(1)
        with np.errstate(invalid="ignore"):
            gt = (M[codes] > v[:, None]).sum(1)
        rank = np.where(np.isnan(v), np.nan, 1 + gt)
        mean = np.where(cnt > 0, np.nansum(M, 1) / np.maximum(cnt, 1), np.nan)[codes]
        # 自分以外の最大
        omax = np.full(len(v), np.nan)
        for j in range(6):
            Mj = M.copy()
            Mj[:, j] = np.nan
            with np.errstate(all="ignore"):
                mx = np.where(np.isnan(Mj).all(1), np.nan, np.nanmax(np.nan_to_num(Mj, nan=-np.inf), 1))
            sel = li == j
            omax[sel] = mx[codes[sel]]
        new[f"{f}__rank"] = rank
        new[f"{f}__dmean"] = v - mean
        new[f"{f}__dmax_other"] = v - omax
        new[f"{f}__dlane1"] = v - M[codes, 0]
    for f in ["national_win_rate", "rs_win_rate"]:
        M = np.full((len(uniq), 6), np.nan)
        M[codes, li] = out[f].values.astype(float)
        tot = np.nansum(np.clip(M, 0, None), 1)[codes]
        new[f"{f}__share"] = _rate(np.clip(out[f].values, 0, None), tot)
    return pd.concat([out, pd.DataFrame(new, index=out.index)], axis=1)


def _attach_labels(con, out: pd.DataFrame) -> pd.DataFrame:
    """ラベル（当日の結果）。y_ で始まる列は特徴量に使わないこと。"""
    hd_from, hd_to = out["hd"].min(), out["hd"].max()
    y = pd.read_sql("""SELECT hd, jcd, rno, lane, position AS y_pos,
                              json_extract(data,'$.entrance_position') AS y_course
                       FROM results WHERE hd BETWEEN ? AND ?""", con, params=(hd_from, hd_to))
    tan = pd.read_sql("""SELECT hd, jcd, rno, CAST(combination AS int) AS lane, payout AS y_tansho_payout
                         FROM payouts WHERE ticket_type='tansho' AND hd BETWEEN ? AND ?""",
                      con, params=(hd_from, hd_to))
    fix = venue_fix_map(con, hd_from, str(int(hd_to) + 1))
    y = _apply_venue_fix(y, fix)
    tan = _apply_venue_fix(tan, fix)
    out = out.merge(y, on=["hd", "jcd", "rno", "lane"], how="left")
    out = out.copy()
    out = out.merge(tan.drop_duplicates(["hd", "jcd", "rno", "lane"]), on=["hd", "jcd", "rno", "lane"], how="left")
    out["y_has_result"] = out.groupby(["hd", "jcd", "rno"])["y_pos"].transform("count").gt(0)
    out["y_win"] = (out["y_pos"] == 1).astype(int)
    out["y_top2"] = (out["y_pos"] <= 2).astype(int)
    out["y_top3"] = (out["y_pos"] <= 3).astype(int)
    return out


# --------------------------------------------------------------------------
# リーク検査
# --------------------------------------------------------------------------
def make_truncated_db(src: Path, dst: Path, hd: str) -> None:
    """hd の前夜に手に入るはずのデータだけを持つ DB を作る。
    entries: hd 以前 / results・payouts・races: hd より前 / racer_stats: hd より前"""
    if dst.exists():
        dst.unlink()
    con = sqlite3.connect(f"file:{dst}", uri=True)
    con.execute(f"ATTACH DATABASE 'file:{src}?mode=ro' AS src")
    for t, cond in [("entries", "hd <= ?"), ("results", "hd < ?"), ("payouts", "hd < ?"),
                    ("racer_stats", "hd < ?")]:
        sql = con.execute("SELECT sql FROM src.sqlite_master WHERE name=?", (t,)).fetchone()[0]
        con.execute(sql)
        con.execute(f"INSERT INTO main.{t} SELECT * FROM src.{t} WHERE {cond}", (hd,))
    con.commit()
    con.close()


def leak_check(days: list[str], workdir: Path, db: Path = DB_PATH, stats_lag_days: int = 0) -> pd.DataFrame:
    workdir.mkdir(parents=True, exist_ok=True)
    full = connect_ro(db)
    # 期間をまとめて作った場合とも一致するか（期間の後ろの日のデータが混ざらないか）
    d_lo, d_hi = min(days), max(days)
    ranged = build_features(full, d_lo, d_hi, stats_lag_days, with_labels=False)
    rows = []
    for d in days:
        a = build_features(full, d, d, stats_lag_days, with_labels=False)
        tmp = workdir / f"trunc_{d}.db"
        make_truncated_db(db, tmp, d)
        tcon = sqlite3.connect(f"file:{tmp}?mode=ro", uri=True)
        b = build_features(tcon, d, d, stats_lag_days, with_labels=False)
        tcon.close()
        tmp.unlink()
        r = ranged[ranged["hd"] == d].reset_index(drop=True)
        cols = [c for c in a.columns if not c.startswith("y_")]
        n_bad, n_bad_r, bad = 0, 0, []
        for c in cols:
            for other, tag in ((b, "trunc"), (r, "range")):
                x, y = a[c], other[c]
                if x.dtype.kind in "fiub":
                    eq = np.isclose(x.astype(float), y.astype(float), equal_nan=True, rtol=1e-9, atol=1e-12)
                else:
                    eq = (x.astype(str) == y.astype(str)).values
                if not eq.all():
                    bad.append(f"{c}({tag}:{(~eq).sum()})")
                    if tag == "trunc":
                        n_bad += 1
                    else:
                        n_bad_r += 1
        rows.append({"hd": d, "rows": len(a), "rows_trunc": len(b), "features": len(cols),
                     "mismatch_trunc": n_bad, "mismatch_range": n_bad_r, "detail": ", ".join(bad[:10])})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 単変量レポート
# --------------------------------------------------------------------------
def _auc(score: np.ndarray, y: np.ndarray) -> float:
    m = ~np.isnan(score)
    s, y = score[m], y[m]
    npos = y.sum()
    nneg = len(y) - npos
    if npos == 0 or nneg == 0:
        return np.nan
    r = pd.Series(s).rank().values
    return (r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def univariate_report(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["y_has_result"]].copy()
    y = df["y_win"].values.astype(int)
    feats = [c for c in df.columns if not c.startswith("y_") and df[c].dtype.kind in "fiub"
             and c not in ("rno", "motor_no", "boat_no")]
    rows = []
    lanes = df["lane"].values
    for c in feats:
        x = df[c].values.astype(float)
        auc = _auc(x, y)
        # 枠番の中での AUC（枠の効果を除いたときの効き）。艇数で重み付け平均
        aucs, ws = [], []
        for l in range(1, 7):
            m = lanes == l
            a = _auc(x[m], y[m])
            if not np.isnan(a):
                aucs.append(a); ws.append(m.sum())
        auc_in = np.average(aucs, weights=ws) if aucs else np.nan
        cover = np.mean(~np.isnan(x))
        # 5分位ごとの1着率
        q = pd.Series(x).rank(pct=True, method="first")
        bins = np.ceil(q * 5).clip(1, 5)
        wr = pd.Series(y).groupby(bins.values).mean()
        rows.append({"feature": c, "coverage": cover, "auc": auc, "auc_within_lane": auc_in,
                     "q1_win": wr.get(1.0, np.nan), "q5_win": wr.get(5.0, np.nan)})
    rep = pd.DataFrame(rows)
    rep["strength"] = (rep["auc_within_lane"] - 0.5).abs()
    return rep.sort_values("strength", ascending=False)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="hd_from")
    ap.add_argument("--to", dest="hd_to")
    ap.add_argument("--out")
    ap.add_argument("--db", default=str(DB_PATH))
    ap.add_argument("--stats-lag-days", type=int, default=0,
                    help="期別成績を算出期間の最終日から何日後に使えるとみなすか（既定0＝翌日から）")
    ap.add_argument("--leak-check", help="カンマ区切りの日付。切り詰めDBで同じ値になるか確かめる")
    ap.add_argument("--workdir", default="/tmp/features_leakcheck")
    ap.add_argument("--report", help="特徴量 Parquet を読み、単変量の1着との関係を表示")
    a = ap.parse_args(argv)

    if a.leak_check:
        res = leak_check(a.leak_check.split(","), Path(a.workdir), Path(a.db), a.stats_lag_days)
        print(res.to_string(index=False))
        return 0 if (res["mismatch_trunc"] == 0).all() and (res["mismatch_range"] == 0).all() else 1
    if a.report:
        df = pd.read_parquet(a.report)
        rep = univariate_report(df)
        pd.set_option("display.width", 200)
        print(rep.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        return 0
    if not (a.hd_from and a.hd_to and a.out):
        ap.error("--from --to --out が必要")
    con = connect_ro(a.db)
    df = build_features(con, a.hd_from, a.hd_to, a.stats_lag_days, verbose=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(a.out, compression="zstd", index=False)
    print(f"wrote {a.out}: {len(df)} rows x {df.shape[1]} cols", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
