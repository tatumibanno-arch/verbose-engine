"""追加の特徴量（研究用）。features.build_features の出力に横付けする 1艇1行のテーブル。

既定の features.py の動作は変えない。使うときだけ build_extra() を呼んで (hd, jcd, rno, lane) で結合する。

使い方:
    python scripts/features_extra.py --from 20251101 --to 20261031 --out /path/extra.parquet
    python scripts/features_extra.py --leak-check 20260315,20261001 --workdir /path/work
    python scripts/features_extra.py --from ... --to ... --out ... --research-sameday   # 研究専用列も付ける

グループ（列名の接頭辞）:
    sr_  今節（同じ場で直近 MEET_WINDOW 日以内の前日まで）の展示タイム順位・ST・進入。前日までのみ
    mm_  モーターの直近（同じ入れ替え期・直近 MOTOR_RECENT_DAYS 日）の展示タイム順位・3着内率
    ce_  進入傾向（前付け）: 選手の前付け率、レース内に前付けする艇がいるか、1号艇がイン取りできる率
    pw_  前日の水面: 同じ場の直近開催日（PREV_MAX_GAP 日以内）の風・波・天候と、その日のイン逃げ率など
    ot_  その他: 地元支部か、ST の最近の傾向、直近の出走密度、内側/外側の艇との ST 差
    sd_  【研究専用】当日のそのレースの天候・風・波（実測）。前夜には分からないので本番の選定に使ってはいけない。
         既定では作らない（--research-sameday / sameday_weather=True のときだけ）。

リーク対策:
    - 履歴は features._Hist で「レース日の0時より前」だけを集計する（当日の前半レースも入らない）。
    - 前日の水面は races の「レース日より前の日」の行だけ。
    - 支部（branch）は番組表の値（前夜に分かる）。
    詳しくは notes/analysis/features_extra.md。
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as F  # noqa: E402

# ---- パラメータ（特徴量の定義。ルールではないので rules.yaml には置かない） ----
MEET_WINDOW = 7            # 今節とみなす窓（同じ場・直近何日）
MOTOR_RECENT_DAYS = 30     # モーターの直近（前の数節）の窓
ENTRY_WINDOWS = (180, 365)  # 前付け率の窓（日）
ST_SHORT, ST_LONG = 30, 365  # ST 傾向の窓
PREV_MAX_GAP = 3           # 「前日の水面」とみなす直近開催日の最大の空き（日）
VENUE_COND_WINDOW = 3      # 場の直近のイン逃げ率などの窓（日）
HIST_DAYS = 420            # 読み込む成績の履歴（hd_from の何日前まで）

# 支部 → 地元の場（支部のある都府県の場）
BRANCH_VENUES = {
    "群馬": ("01",), "埼玉": ("02",), "東京": ("03", "04", "05"), "静岡": ("06",),
    "愛知": ("07", "08"), "三重": ("09",), "福井": ("10",), "滋賀": ("11",), "大阪": ("12",),
    "兵庫": ("13",), "徳島": ("14",), "香川": ("15",), "岡山": ("16",), "広島": ("17",),
    "山口": ("18", "19"), "福岡": ("20", "21", "22"), "佐賀": ("23",), "長崎": ("24",),
}

_rate = F._rate


def _wind_uv(direction, speed):
    """風向コード（1〜16 の16方位、1=北とみなす。17 以上や欠損は無風扱い）と風速 → (u, v)。
    方位の意味は場ごとに水面の向きが違うので、モデルは場コードとの組み合わせで学ぶ前提。"""
    d = pd.to_numeric(pd.Series(direction), errors="coerce").to_numpy(float)
    s = np.nan_to_num(pd.to_numeric(pd.Series(speed), errors="coerce").to_numpy(float))
    ok = (d >= 1) & (d <= 16)
    ang = np.where(ok, (d - 1) * (2 * np.pi / 16), 0.0)
    u = np.where(ok, s * np.sin(ang), 0.0)
    v = np.where(ok, s * np.cos(ang), 0.0)
    return u, v


def _load_targets(con, hd_from, hd_to):
    q = """SELECT hd, jcd, rno, lane, racer_no, motor_no,
                  json_extract(data,'$.branch') AS branch
           FROM entries WHERE hd BETWEEN ? AND ?"""
    T = pd.read_sql(q, con, params=(hd_from, hd_to))
    T["racer"] = pd.to_numeric(T["racer_no"], errors="coerce").fillna(-1).astype(np.int64)
    T["motor"] = pd.to_numeric(T["motor_no"], errors="coerce").fillna(0).astype(np.int64).clip(0, 999)
    T["jcd_i"] = T["jcd"].astype(int)
    T["day"] = F._day(T["hd"])
    return T.sort_values(["hd", "jcd", "rno", "lane"]).reset_index(drop=True)


def _load_races(con, hd_lo, hd_hi_incl):
    q = """SELECT hd, jcd, rno, weather, wind_direction, wind_speed, wave_height
           FROM races WHERE hd >= ? AND hd <= ?"""
    r = pd.read_sql(q, con, params=(hd_lo, hd_hi_incl))
    r["day"] = F._day(r["hd"])
    r["jcd_i"] = r["jcd"].astype(int)
    r["u"], r["v"] = _wind_uv(r["wind_direction"], r["wind_speed"])
    r["wind_speed"] = pd.to_numeric(r["wind_speed"], errors="coerce")
    r["wave_height"] = pd.to_numeric(r["wave_height"], errors="coerce")
    r["rain"] = (r["weather"].fillna("").str.contains("雨|雪")).astype(float)
    return r


def build_extra(con, hd_from: str, hd_to: str, sameday_weather: bool = False,
                verbose: bool = False) -> pd.DataFrame:
    t0 = time.time()

    def log(msg):
        if verbose:
            print(f"[extra {time.time() - t0:6.1f}s] {msg}", file=sys.stderr)

    T = _load_targets(con, hd_from, hd_to)
    if T.empty:
        return T
    h_lo = (pd.Timestamp(hd_from) - pd.Timedelta(days=HIST_DAYS)).strftime("%Y%m%d")
    H = F.load_results(con, hd_to, h_lo)     # hd_to より前（各行の参照はさらに day 未満）
    bounds = F.motor_boundaries(con, hd_to)
    log(f"loaded targets={len(T)} hist={len(H)}")

    out = T[["hd", "jcd", "rno", "lane"]].copy()
    tday = T["day"].to_numpy()
    lane = T["lane"].to_numpy().astype(np.int64)
    rk = T["racer"].to_numpy()
    jc = T["jcd_i"].to_numpy()

    # ---- 履歴側のレース内の値 ----
    H = H.sort_values(["day", "jcd_i", "rno", "lane"]).reset_index(drop=True)
    pos = H["position"].to_numpy()
    st = H["st"].to_numpy()
    ex = H["ex"].to_numpy()
    hc = H["course"].to_numpy()
    hl = H["lane"].to_numpy()
    ran = (~np.isnan(pos)) | (~np.isnan(st))
    grp = H.groupby(["day", "jcd_i", "rno"], sort=False)
    ex_rank = grp["ex"].rank(method="average").to_numpy()      # 1=展示最速
    ex_res = ex - grp["ex"].transform("mean").to_numpy()
    st_rank = grp["st"].rank(method="average").to_numpy()
    has_c = hc > 0
    vals = {
        "n": ran.astype(float),
        "top3": (pos <= 3).astype(float),
        "win": (pos == 1).astype(float),
        "ex_n": (~np.isnan(ex_res)).astype(float),
        "ex_rank": np.nan_to_num(ex_rank),
        "ex_res": np.nan_to_num(ex_res),
        "ex_top2": ((ex_rank <= 2) & ~np.isnan(ex_rank)).astype(float),
        "st_n": (~np.isnan(st)).astype(float),
        "st": np.nan_to_num(st),
        "st_rank": np.nan_to_num(st_rank),
        "c_n": has_c.astype(float),
        "inward": np.where(has_c, hl - hc, 0).astype(float),
        "movein": (has_c & (hc < hl)).astype(float),
        "pushed": (has_c & (hc > hl)).astype(float),
        "c_n_out": (has_c & (hl >= 2)).astype(float),            # 前付けしうる枠（2〜6）で出た回数
        "movein_out": (has_c & (hl >= 2) & (hc < hl)).astype(float),
    }

    # ---- sr_: 今節（同じ場・直近 MEET_WINDOW 日、前日まで） ----
    hkey_rj = H["racer"].to_numpy() * 100 + H["jcd_i"].to_numpy()
    h_rj = F._Hist(hkey_rj, H["day"].to_numpy(), vals)
    S = h_rj.query(rk * 100 + jc, tday, MEET_WINDOW)
    out["sr_n"] = S["n"].to_numpy()
    out["sr_ex_rank"] = _rate(S["ex_rank"], S["ex_n"])
    out["sr_ex_res"] = _rate(S["ex_res"], S["ex_n"])
    out["sr_ex_top2"] = _rate(S["ex_top2"], S["ex_n"])
    out["sr_st"] = _rate(S["st"], S["st_n"])
    out["sr_st_rank"] = _rate(S["st_rank"], S["st_n"])
    out["sr_inward"] = _rate(S["inward"], S["c_n"])
    # 前日（今節の直近の出走日）だけの値
    last = h_rj.last_day(rk * 100 + jc, tday)
    last_ok = ~np.isnan(last) & (tday - np.nan_to_num(last) <= MEET_WINDOW)
    S1 = h_rj.query(rk * 100 + jc, tday, start_day=np.where(last_ok, np.nan_to_num(last), tday).astype(np.int64))
    out["sr_last_ex_rank"] = _rate(S1["ex_rank"], S1["ex_n"])
    out["sr_last_ex_res"] = _rate(S1["ex_res"], S1["ex_n"])
    out["sr_last_st"] = _rate(S1["st"], S1["st_n"])
    # 普段（365日）との差: 今節の足・ST が普段よりどうか
    h_r = F._Hist(H["racer"].to_numpy(), H["day"].to_numpy(), vals)
    SL = h_r.query(rk, tday, ST_LONG)
    out["sr_ex_res_vs_365"] = out["sr_ex_res"].to_numpy() - _rate(SL["ex_res"], SL["ex_n"])
    out["sr_st_vs_365"] = out["sr_st"].to_numpy() - _rate(SL["st"], SL["st_n"])
    log("sr done")

    # ---- mm_: モーターの直近（同じ入れ替え期） ----
    H_ep, _ = F._epoch(H["jcd_i"].to_numpy(), H["day"].to_numpy(), bounds)
    T_ep, T_start = F._epoch(jc, tday, bounds)
    mkey_h = (H["jcd_i"].to_numpy() * 100 + H_ep) * 1000 + H["motor"].to_numpy().clip(0, 999)
    mkey_t = (jc * 100 + T_ep) * 1000 + T["motor"].to_numpy()
    h_m = F._Hist(mkey_h, H["day"].to_numpy(), vals)
    S = h_m.query(mkey_t, tday, MOTOR_RECENT_DAYS)
    out["mm_n"] = S["n"].to_numpy()
    out["mm_ex_rank"] = _rate(S["ex_rank"], S["ex_n"])
    out["mm_ex_top2"] = _rate(S["ex_top2"], S["ex_n"])
    out["mm_top3"] = _rate(S["top3"], S["n"])
    # 縮小版（出走が少ないと中立値に寄せる。展示順位の中立は 3.5）
    out["mm_ex_rank_shrunk"] = (S["ex_rank"].to_numpy() + 10 * 3.5) / (S["ex_n"].to_numpy() + 10)
    log("mm done")

    # ---- ce_: 進入傾向（前付け） ----
    for w in ENTRY_WINDOWS:
        S = h_r.query(rk, tday, w)
        out[f"ce_movein{w}"] = _rate(S["movein_out"], S["c_n_out"])   # 2〜6枠で内に入った割合
        out[f"ce_pushed{w}"] = _rate(S["pushed"], S["c_n"])           # 外に出された割合
    out["ce_n365"] = h_r.query(rk, tday, 365)["c_n_out"].to_numpy()
    # 縮小版（全体の前付け率 ≈ 0.03 へ寄せる）
    S = h_r.query(rk, tday, 365)
    prior = 0.03
    mv = (S["movein_out"].to_numpy() + 20 * prior) / (S["c_n_out"].to_numpy() + 20)
    out["ce_movein_shrunk"] = mv
    # レース単位: 自分より外の艇の最大前付け率（自分のコースを取られる脅威）、レース内最大、1号艇のイン取り率
    race_key = out["hd"] + out["jcd"] + out["rno"].astype(str).str.zfill(2)
    codes, uniq = pd.factorize(race_key)
    li = lane.clip(1, 6) - 1
    M = np.full((len(uniq), 6), np.nan)
    M[codes, li] = mv
    M[:, 0] = np.nan  # 1号艇は前付けしない
    with np.errstate(all="ignore"):
        outer_max = np.full(len(out), np.nan)
        for j in range(6):
            sel = li == j
            if j < 5:
                sub = M[codes[sel]][:, j + 1:]
                outer_max[sel] = np.where(np.isnan(sub).all(1), np.nan, np.nanmax(np.nan_to_num(sub, nan=-1), 1))
        out["ce_outer_max_movein"] = outer_max
        out["ce_race_max_movein"] = np.nanmax(np.nan_to_num(M, nan=-1), 1)[codes]
        out["ce_race_sum_movein"] = np.nansum(M, 1)[codes]
    # 1号艇がこの選手のとき、1コースから出た割合（365日）
    cv = {"n": (hc > 0).astype(float) * (hl == 1), "keep": ((hc == 1) & (hl == 1)).astype(float)}
    S = F._Hist(H["racer"].to_numpy(), H["day"].to_numpy(), cv).query(rk, tday, 365)
    keep1 = (S["keep"].to_numpy() + 10 * 0.97) / (S["n"].to_numpy() + 10)
    K1 = np.full((len(uniq), 6), np.nan)
    K1[codes, li] = keep1
    out["ce_lane1_keep"] = K1[codes, 0]
    # 場ごとの枠なり率（365日、レースの全艇が枠なりだった割合）
    same = ((hc == hl) | (hc <= 0)).astype(float)
    nowaku = grp.ngroup().to_numpy()
    race_same = pd.Series(same).groupby(nowaku).transform("min").to_numpy()
    first_row = ~pd.Series(nowaku).duplicated().to_numpy()
    vv = {"n": first_row.astype(float), "waku": (first_row & (race_same == 1)).astype(float)}
    S = F._Hist(H["jcd_i"].to_numpy(), H["day"].to_numpy(), vv).query(jc, tday, 365)
    out["ce_venue_wakunari365"] = _rate(S["waku"], S["n"])
    log("ce done")

    # ---- pw_: 前日の水面（同じ場の直近開催日、PREV_MAX_GAP 日以内） ----
    races = _load_races(con, h_lo, hd_to)
    vd = (races.groupby(["jcd_i", "day"]).agg(pw_wind=("wind_speed", "mean"), pw_wind_max=("wind_speed", "max"),
                                             pw_wave=("wave_height", "mean"), pw_u=("u", "mean"),
                                             pw_v=("v", "mean"), pw_rain=("rain", "mean"))
          .reset_index().sort_values("day"))
    # 場の直近の結果: 1号艇の1着率・1コース進入率・平均ST（前日まで、VENUE_COND_WINDOW 日）
    l1 = hl == 1
    vc = {"n1": (ran & l1).astype(float), "w1": ((pos == 1) & l1).astype(float),
          "c1": (has_c & l1).astype(float), "k1": ((hc == 1) & l1).astype(float), "n": ran.astype(float),
          "st_n": vals["st_n"], "st": vals["st"],
          "in2": ((pos == 1) & (hl <= 2)).astype(float), "r": ((pos == 1)).astype(float)}
    S = F._Hist(H["jcd_i"].to_numpy(), H["day"].to_numpy(), vc).query(jc, tday, VENUE_COND_WINDOW)
    out["pw_lane1_win"] = _rate(S["w1"], S["n1"])
    out["pw_lane1_keep"] = _rate(S["k1"], S["c1"])
    out["pw_in2_win"] = _rate(S["in2"], S["r"])
    out["pw_venue_st"] = _rate(S["st"], S["st_n"])
    left = pd.DataFrame({"jcd_i": jc, "day": tday, "_i": np.arange(len(out))}).sort_values("day")
    m = pd.merge_asof(left, vd.rename(columns={"day": "pday"}).assign(day=lambda d: d["pday"]),
                      on="day", by="jcd_i", allow_exact_matches=False, direction="backward")
    m = m.sort_values("_i")
    ok = (left.sort_values("_i")["day"].to_numpy() - m["pday"].to_numpy()) <= PREV_MAX_GAP
    for c in ["pw_wind", "pw_wind_max", "pw_wave", "pw_u", "pw_v", "pw_rain"]:
        out[c] = np.where(ok, m[c].to_numpy(), np.nan)
    log("pw done")

    # ---- ot_: その他 ----
    br = T["branch"].fillna("")
    out["ot_home"] = [int(j in BRANCH_VENUES.get(b, ())) for b, j in zip(br, T["jcd"])]
    SS = h_r.query(rk, tday, ST_SHORT)
    st_s = _rate(SS["st"], SS["st_n"])
    st_l = _rate(SL["st"], SL["st_n"])
    out["ot_st_trend"] = st_s - st_l                     # 負＝最近 ST が早くなっている
    out["ot_runs7"] = h_r.query(rk, tday, 7)["n"].to_numpy()
    out["ot_runs30"] = SS["n"].to_numpy()
    # 隣の艇との ST 差（365日の平均ST）。内側の艇より早いと「まくり」、外より遅いと「叩かれる」
    Mst = np.full((len(uniq), 6), np.nan)
    Mst[codes, li] = st_l
    own = Mst[codes, li]
    inner = np.where(li > 0, Mst[codes, np.maximum(li - 1, 0)], np.nan)
    outer = np.where(li < 5, Mst[codes, np.minimum(li + 1, 5)], np.nan)
    out["ot_st_vs_inner"] = own - inner                  # 負＝内側の艇より早い
    out["ot_st_vs_outer"] = own - outer
    # 隣の艇の展示（今節）。内側の艇の今節展示順位
    Mex = np.full((len(uniq), 6), np.nan)
    Mex[codes, li] = out["sr_ex_res"].to_numpy()
    out["ot_inner_sr_ex_res"] = np.where(li > 0, Mex[codes, np.maximum(li - 1, 0)], np.nan)
    log("ot done")

    # ---- レース内の相対値 ----
    out = _relative(out, codes, li, ["sr_ex_res", "sr_ex_rank", "sr_st", "mm_ex_rank_shrunk", "ce_movein_shrunk"])

    # ---- sd_: 【研究専用】当日の実測の天候 ----
    if sameday_weather:
        r = races[(races["hd"] >= hd_from) & (races["hd"] <= hd_to)][
            ["hd", "jcd", "rno", "wind_speed", "wave_height", "u", "v", "rain"]]
        r = r.rename(columns={"wind_speed": "sd_wind", "wave_height": "sd_wave", "u": "sd_u", "v": "sd_v",
                              "rain": "sd_rain"})
        out = out.merge(r.astype({"rno": out["rno"].dtype}), on=["hd", "jcd", "rno"], how="left")
    log("done")
    return out


def _relative(out, codes, li, feats):
    """レース内の相対値（features._relative の簡略版）: 平均との差・1号艇との差・順位（大きい順）。"""
    new = {}
    nr = codes.max() + 1
    for f in feats:
        M = np.full((nr, 6), np.nan)
        M[codes, li] = out[f].to_numpy(float)
        v = M[codes, li]
        with np.errstate(invalid="ignore"):
            mean = np.nanmean(np.where(np.isnan(M).all(1, keepdims=True), 0, M), 1)[codes]
            gt = (M[codes] > v[:, None]).sum(1)
        new[f"{f}__dmean"] = v - mean
        new[f"{f}__dlane1"] = v - M[codes, 0]
        new[f"{f}__rank"] = np.where(np.isnan(v), np.nan, 1 + gt)
    return pd.concat([out, pd.DataFrame(new, index=out.index)], axis=1)


def leak_check(days, workdir: Path, db: Path = F.DB_PATH) -> pd.DataFrame:
    """features.leak_check と同じ考え方: 切り詰めDB（races も hd より前だけ）と全データDB で一致するか。"""
    workdir.mkdir(parents=True, exist_ok=True)
    full = F.connect_ro(db)
    ranged = build_extra(full, min(days), max(days))
    rows = []
    for d in days:
        a = build_extra(full, d, d)
        tmp = workdir / f"trunc_extra_{d}.db"
        if tmp.exists():
            tmp.unlink()
        lo = (pd.Timestamp(d) - pd.Timedelta(days=HIST_DAYS + 10)).strftime("%Y%m%d")
        lo_e = (pd.Timestamp(d) - pd.Timedelta(days=1100)).strftime("%Y%m%d")
        con = sqlite3.connect(f"file:{tmp}", uri=True, timeout=600)
        con.execute(f"ATTACH DATABASE 'file:{db}?mode=ro' AS src")
        for t, cond in [("entries", "hd <= ?"), ("results", "hd < ?"), ("races", "hd < ?")]:
            sql = con.execute("SELECT sql FROM src.sqlite_master WHERE name=?", (t,)).fetchone()[0]
            con.execute(sql)
            # entries はモーター入れ替え検出用に長めに（ENTRY_LOOKBACK 日）。results・races は履歴分だけ
            lo_t = lo_e if t == "entries" else lo
            con.execute(f"INSERT INTO main.{t} SELECT * FROM src.{t} WHERE hd >= ? AND {cond}", (lo_t, d))
        con.commit()
        con.close()
        tcon = sqlite3.connect(f"file:{tmp}?mode=ro", uri=True)
        b = build_extra(tcon, d, d)
        tcon.close()
        tmp.unlink()
        r = ranged[ranged["hd"] == d].reset_index(drop=True)
        bad = []
        for c in a.columns:
            for other, tag in ((b, "trunc"), (r, "range")):
                x, y = a[c], other[c]
                if x.dtype.kind in "fiub":
                    eq = np.isclose(x.astype(float), y.astype(float), equal_nan=True, rtol=1e-9, atol=1e-12)
                else:
                    eq = (x.astype(str).values == y.astype(str).values)
                if not eq.all():
                    bad.append(f"{c}({tag}:{(~eq).sum()})")
        rows.append({"hd": d, "rows": len(a), "rows_trunc": len(b), "cols": a.shape[1],
                     "mismatch": len(bad), "detail": ", ".join(bad[:10])})
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="hd_from")
    ap.add_argument("--to", dest="hd_to")
    ap.add_argument("--out")
    ap.add_argument("--db", default=str(F.DB_PATH))
    ap.add_argument("--research-sameday", action="store_true", help="【研究専用】当日の天候列 sd_* も付ける")
    ap.add_argument("--leak-check")
    ap.add_argument("--workdir", default="/tmp/features_extra_leakcheck")
    a = ap.parse_args(argv)
    if a.leak_check:
        res = leak_check(a.leak_check.split(","), Path(a.workdir), Path(a.db))
        print(res.to_string(index=False))
        return 0 if (res["mismatch"] == 0).all() else 1
    if not (a.hd_from and a.hd_to and a.out):
        ap.error("--from --to --out が必要")
    con = sqlite3.connect(f"file:{a.db}?mode=ro", uri=True, timeout=600)
    df = build_extra(con, a.hd_from, a.hd_to, sameday_weather=a.research_sameday, verbose=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(a.out, compression="zstd", index=False)
    print(f"wrote {a.out}: {len(df)} rows x {df.shape[1]} cols", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
