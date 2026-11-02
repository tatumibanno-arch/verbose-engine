"""場 × コース × 条件 の基礎統計（読み取り専用・再実行可）。

使い方:
  python scripts/analysis/venue_course.py                 # 既定 2023-11-01〜2026-11-01
  python scripts/analysis/venue_course.py --from 20231101 --to 20261101

出力:
  notes/analysis/venue_course.md  … 表は毎回作り直す。
      <!-- MANUAL START --> 〜 <!-- MANUAL END --> の間（手書きの考察・仮説）は再実行しても残す。
  notes/analysis/img/*.png        … matplotlib があれば数枚

DB は読み取り専用で開く（別プロセスが書き込み中でも安全）。
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from common import VENUES  # noqa: E402

DB_URI = f"file:{ROOT / 'data' / 'boatrace.db'}?mode=ro"
OUT_MD = ROOT / "notes" / "analysis" / "venue_course.md"
IMG_DIR = ROOT / "notes" / "analysis" / "img"

MIN_N = 50  # これ未満のセルは「参考」（† 印）
DIR_NAMES = {1: "北", 3: "北東", 5: "東", 7: "南東", 9: "南", 11: "南西", 13: "西", 15: "北西"}


# ---------------------------------------------------------------- load
def load(d_from: str, d_to: str) -> dict[str, pd.DataFrame]:
    con = sqlite3.connect(DB_URI, uri=True, timeout=60)
    p = (d_from, d_to)
    races = pd.read_sql(
        "select hd,jcd,rno,race_name,weather,wind_direction wd,wind_speed ws,wave_height wh "
        "from races where hd between ? and ?", con, params=p)
    res = pd.read_sql(
        "select hd,jcd,rno,lane,position pos,"
        " json_extract(data,'$.entrance_position') course,"
        " json_extract(data,'$.st_timing') st,"
        " json_extract(data,'$.exhibition_time') ex "
        "from results where hd between ? and ?", con, params=p)
    ent = pd.read_sql(
        "select hd,jcd,rno,lane,grade,"
        " json_extract(data,'$.national_win_rate') nwr "
        "from entries where hd between ? and ?", con, params=p)
    pay = pd.read_sql(
        "select hd,jcd,rno,ticket_type tt,combination comb,payout "
        "from payouts where hd between ? and ?", con, params=p)
    con.close()
    # 既知の取り込み不具合: 唐津(23)の成績が津(09)として入っている（2026-11 時点で races/results/payouts に jcd=23 が0件）。
    # 同じ日に両場開催だと片方しか残っていない。番組表(entries)の選手番号と照合して日単位で 23 に付け替える。
    con = sqlite3.connect(DB_URI, uri=True, timeout=60)
    rr = pd.read_sql("select hd,rno,lane,json_extract(data,'$.racer_number') rn from results "
                     "where jcd='09' and hd between ? and ?", con, params=p)
    con.close()
    ent_rn = _entry_racers(p)
    m = rr.merge(ent_rn, on=["hd", "rno", "lane", "rn"], how="inner")
    daymap = m.groupby("hd")["jcd"].agg(lambda s: s.mode()[0])
    to23 = set(daymap[daymap == "23"].index)
    for df in (races, res, pay):
        mask = (df["jcd"] == "09") & df["hd"].isin(to23)
        df.loc[mask, "jcd"] = "23"
    return {"races": races, "res": res, "ent": ent, "pay": pay,
            "remap": {"to23_days": len(to23), "09_days": int((daymap == "09").sum())}}


def _entry_racers(p):
    con = sqlite3.connect(DB_URI, uri=True, timeout=60)
    e = pd.read_sql("select hd,jcd,rno,lane,cast(racer_no as int) rn from entries "
                    "where jcd in ('09','23') and hd between ? and ?", con, params=p)
    con.close()
    return e


# ---------------------------------------------------------------- helpers
K = ["hd", "jcd", "rno"]


def vname(j):
    return VENUES.get(j, j)


def cell(rate, n, digits=1):
    if n == 0 or pd.isna(rate):
        return "-"
    s = f"{rate * 100:.{digits}f}"
    return f"{s}† ({n})" if n < MIN_N else f"{s} ({n})"


def md_table(df: pd.DataFrame, index_name: str | None = None) -> str:
    df = df.copy()
    if index_name is not None:
        df.insert(0, index_name, df.index)
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else str(v) for v in row) + " |")
    return "\n".join(lines)


def rate_table(df, row, col, flag, row_order=None, col_order=None, total=True):
    """df の flag(0/1) を row×col で率と件数に。'率% (n)' の表を返す。"""
    g = df.groupby([row, col])[flag].agg(["mean", "size"])
    rows = row_order or sorted(df[row].dropna().unique())
    cols = col_order or sorted(df[col].dropna().unique())
    out = pd.DataFrame(index=[vname(r) if row == "jcd" else r for r in rows])
    for c in cols:
        vals = []
        for r in rows:
            if (r, c) in g.index:
                m, n = g.loc[(r, c)]
                vals.append(cell(m, int(n)))
            else:
                vals.append("-")
        out[str(c)] = vals
    if total:
        t = df.groupby(row)[flag].agg(["mean", "size"])
        out["全体"] = [cell(t.loc[r, "mean"], int(t.loc[r, "size"])) if r in t.index else "-" for r in rows]
        tc = df.groupby(col)[flag].agg(["mean", "size"])
        allrow = [cell(tc.loc[c, "mean"], int(tc.loc[c, "size"])) if c in tc.index else "-" for c in cols]
        allrow.append(cell(df[flag].mean(), len(df)))
        out.loc["全国"] = allrow
    return out


def race_cat(name) -> str:
    if not isinstance(name, str) or not name:
        return "不明"
    s = unicodedata.normalize("NFKC", name)
    if "準優" in s:
        return "準優勝戦"
    if "優勝" in s:
        return "優勝戦"
    if "ドリーム" in s:
        return "ドリーム"
    if "進入固定" in s:
        return "進入固定"
    if any(k in s for k in ("特選", "特賞", "選抜", "特別", "記者", "選手権")):
        return "特選・選抜系"
    if "予選" in s:
        return "予選"
    if "一般" in s:
        return "一般"
    return "企画・その他"


# ---------------------------------------------------------------- build frames
def build(d):
    races, res, ent, pay = d["races"], d["res"], d["ent"], d["pay"]
    res = res.copy()
    res["course"] = pd.to_numeric(res["course"], errors="coerce")
    res = res[res["course"].between(1, 6)]
    # 完走したレースのみ（1着がいる）
    winners = res[res["pos"] == 1]
    valid = winners.groupby(K).size().rename("nwin").reset_index()
    valid = valid[valid["nwin"] == 1][K]
    res = res.merge(valid, on=K)
    res["w"] = (res["pos"] == 1).astype(int)
    res["p2"] = (res["pos"] == 2).astype(int)
    res["t3"] = res["pos"].between(1, 3).astype(int)

    # レース単位
    rc = races.merge(valid, on=K).copy()
    rc["cat"] = rc["race_name"].map(race_cat)
    rc["month"] = rc["hd"].str[4:6].astype(int)
    wcourse = res[res["w"] == 1][K + ["course", "lane"]].rename(columns={"course": "wc", "lane": "wl"})
    rc = rc.merge(wcourse, on=K, how="left")
    rc["in_win"] = (rc["wc"] == 1).astype(int)
    rc["l1_win"] = (rc["wl"] == 1).astype(int)
    # 進入
    ent_shape = _shape(res)
    rc = rc.merge(ent_shape, on=K, how="left")
    # 番組表（1号艇の級別、他艇の A1 数）
    e = ent.copy()
    l1 = e[e["lane"] == 1][K + ["grade", "nwr"]].rename(columns={"grade": "g1", "nwr": "nwr1"})
    oth = e[e["lane"] != 1].groupby(K).agg(
        n_a1_oth=("grade", lambda s: int((s == "A1").sum())),
        nwr_oth_max=("nwr", "max")).reset_index()
    rc = rc.merge(l1, on=K, how="left").merge(oth, on=K, how="left")
    # 風の相対方向は後で
    # 払戻
    p = pay.copy()
    p3 = p[p["tt"] == "sanrensho"].groupby(K)["payout"].max().rename("p3t").reset_index()
    t1 = p[(p["tt"] == "tansho") & (p["comb"] == "1")].groupby(K)["payout"].sum().rename("tan1").reset_index()
    rc = rc.merge(p3, on=K, how="left").merge(t1, on=K, how="left")
    rc["tan1"] = rc["tan1"].fillna(0)
    rc["man"] = (rc["p3t"] >= 10000).astype(float)
    rc.loc[rc["p3t"].isna(), "man"] = np.nan
    return res, rc, p.merge(valid, on=K)


def _shape(res):
    r = res[["hd", "jcd", "rno", "lane", "course"]].copy()
    r["same"] = (r["course"] == r["lane"]).astype(int)
    r["l1in"] = ((r["lane"] == 1) & (r["course"] == 1)).astype(int)
    r["mz"] = (r["course"] < r["lane"]).astype(int)
    g = r.groupby(K).agg(wakunari=("same", "min"), l1_in=("l1in", "max"),
                         maezuke=("mz", "max"), nboats=("lane", "size")).reset_index()
    return g


# ---------------------------------------------------------------- wind orientation
def wind_orientation(res, rc):
    """展示タイム（バックストレッチ計測）が最も遅くなる風向 = バック向かい風 = ホーム（スタート→1M）追い風、と推定。

    レース平均展示タイムを 場×月 で中心化し、resid ~ ws + ws*cos(θ) + ws*sin(θ) を場ごとに最小二乗。
    θ は風向コード（1,3,..,15 → 16方位の北から時計回り）。
    """
    ex = res[res["ex"] > 0].groupby(K)["ex"].mean().rename("exm").reset_index()
    df = rc[K + ["wd", "ws", "month"]].merge(ex, on=K)
    df = df[df["ws"].notna()]
    df["resid"] = df["exm"] - df.groupby(["jcd", "month"])["exm"].transform("mean")
    out = {}
    for j, g in df.groupby("jcd"):
        g = g.copy()
        th = (g["wd"].fillna(1) - 1) * (2 * np.pi / 16)
        ws = g["ws"].fillna(0)
        X = np.column_stack([np.ones(len(g)), ws, ws * np.cos(th), ws * np.sin(th)])
        beta, *_ = np.linalg.lstsq(X, g["resid"].values, rcond=None)
        amp = float(np.hypot(beta[2], beta[3]))
        phi = float(np.arctan2(beta[3], beta[2]))  # rad, 北=0 時計回り
        code = (phi / (2 * np.pi / 16)) % 16 + 1
        # 各8方位（風速4m以上）の展示偏差
        strong = g[g["ws"] >= 4]
        by = strong.groupby("wd")["resid"].agg(["mean", "size"])
        out[j] = {"phi_code": code, "amp": amp, "by_dir": by, "n": len(g)}
    return out


def code_to_name16(code: float) -> str:
    names16 = ["北", "北北東", "北東", "東北東", "東", "東南東", "南東", "南南東",
               "南", "南南西", "南西", "西南西", "西", "西北西", "北西", "北北西"]
    return names16[int(round(code - 1)) % 16]


def rel_wind(rc, orient):
    def f(row):
        if pd.isna(row["ws"]):
            return np.nan
        if row["ws"] == 0 or pd.isna(row["wd"]):
            return "無風"
        o = orient.get(row["jcd"])
        diff = abs(((row["wd"] - o["phi_code"]) + 8) % 16 - 8)  # 0..8（16方位単位）
        if diff <= 2:
            return "追い"
        if diff >= 6:
            return "向かい"
        return "横"
    rc = rc.copy()
    rc["rel"] = rc.apply(f, axis=1)

    def b(row):
        r = row["rel"]
        if pd.isna(r):
            return np.nan
        if r == "無風" or row["ws"] <= 2:
            return "0-2m"
        sp = "3-4m" if row["ws"] <= 4 else "5m+"
        return f"{r}{sp}"
    rc["windbin"] = rc.apply(b, axis=1)
    return rc


# ---------------------------------------------------------------- sections
def sec1(res, rc):
    out = []
    jorder = sorted(VENUES)
    out.append("## 1. 場別・コース別（進入コース基準）\n")
    out.append(f"対象レース数（1着が確定したレース）: {len(rc):,}。率は%、括弧内は母数（そのコースから出た艇数／レース数）。† は母数{MIN_N}未満で参考。\n")
    for flag, title in (("w", "1着率"), ("p2", "2着率"), ("t3", "3着内率")):
        t = rate_table(res, "jcd", "course", flag, jorder, [1, 2, 3, 4, 5, 6], total=False)
        out.append(f"### 1-{['w','p2','t3'].index(flag)+1}. コース別{title}（%）\n")
        out.append(md_table(t, "場") + "\n")
    # 決まり方
    out.append("### 1-4. 決まり方の傾向（1着コースの構成と、イン逃げ時の2着コース）\n")
    out.append("決まり手そのものはデータに無いので、1着コースで代用（1コース1着≒逃げ、2コース≒差し/まくり、3〜4コース≒まくり/まくり差し、5〜6≒外の展開）。"
               "右側はイン（1コース）1着レースでの2着コース分布。\n")
    rows = []
    sec = res[res["p2"] == 1][K + ["course"]].rename(columns={"course": "c2"})
    rr = rc.merge(sec, on=K, how="left")
    for j in jorder + ["全国"]:
        g = rr if j == "全国" else rr[rr["jcd"] == j]
        n = len(g)
        if n == 0:
            continue
        wc = g["wc"].value_counts(normalize=True)
        gi = g[g["wc"] == 1]
        c2 = gi["c2"].value_counts(normalize=True)
        rows.append([vname(j), n] + [f"{wc.get(c, 0)*100:.1f}" for c in range(1, 7)]
                    + [f"{c2.get(c, 0)*100:.1f}" for c in range(2, 7)])
    t = pd.DataFrame(rows, columns=["場", "レース", "1C勝", "2C勝", "3C勝", "4C勝", "5C勝", "6C勝",
                                    "逃げ時2着2C", "3C", "4C", "5C", "6C"])
    out.append(md_table(t) + "\n")
    # よく出る3連単
    out.append("### 1-5. 3連単の出目（艇番）上位5と占有率\n")
    rows = []
    for j in jorder + ["全国"]:
        g = rc if j == "全国" else rc[rc["jcd"] == j]
        if len(g) == 0:
            continue
        top = TOP3[TOP3["jcd"].isin([j] if j != "全国" else jorder)]["comb"].value_counts()
        tot = top.sum()
        rows.append([vname(j), tot] + [f"{c} {v/tot*100:.1f}%" for c, v in top.head(5).items()])
    out.append(md_table(pd.DataFrame(rows, columns=["場", "レース", "1位", "2位", "3位", "4位", "5位"])) + "\n")
    # 進入
    out.append("### 1-6. 枠番とコースのずれ（進入変化）\n")
    out.append("枠なり率=全艇が枠番どおりのコース。前付け発生率=枠番より内のコースに入った艇が1艇以上いるレース。"
               "1号艇イン率=1号艇が1コースを取った率。1号艇1着率とイン1着率の差がずれの影響。\n")
    rows = []
    for j in jorder + ["全国"]:
        g = rc if j == "全国" else rc[rc["jcd"] == j]
        if len(g) == 0:
            continue
        rows.append([vname(j), len(g), f"{g['wakunari'].mean()*100:.1f}", f"{g['maezuke'].mean()*100:.1f}",
                     f"{g['l1_in'].mean()*100:.1f}", f"{g['l1_win'].mean()*100:.1f}", f"{g['in_win'].mean()*100:.1f}"])
    out.append(md_table(pd.DataFrame(rows, columns=["場", "レース", "枠なり率", "前付け発生率", "1号艇イン率",
                                                    "1号艇1着率", "イン1着率"])) + "\n")
    # 枠番→コース 遷移（全国）
    tr = pd.crosstab(res["lane"], res["course"], normalize="index") * 100
    tr.index = [f"{i}号艇" for i in tr.index]
    tr.columns = [f"{c}C" for c in tr.columns]
    out.append("全国の枠番→進入コース（行%）:\n")
    out.append(md_table(tr.round(2), "枠") + "\n")
    return "\n".join(out)


def sec2(rc, orient):
    out = []
    jorder = sorted(VENUES)
    out.append("## 2. イン（1コース）1着率と条件\n")
    out.append("指標はすべて「1コースに入った艇が1着になった率」（%、括弧内レース数）。† は母数50未満で参考。\n")
    # 風向推定
    out.append("### 2-0. 風向コードの解釈と、場ごとの追い風方向の推定\n")
    out.append(
        "- `wind_direction` は 1,3,5,…,15 の奇数のみ＋NULL。NULL は必ず風速0（無風）。16方位を北=1から時計回りに振った番号の8方位分と解釈した"
        "（1北, 3北東, 5東, 7南東, 9南, 11南西, 13西, 15北西）。値は気象の慣例どおり『風が吹いてくる方角』とみなす。\n"
        "- 場のコースの向きはデータに無いので、**展示タイム**から推定した。展示タイムはバックストレッチ（1M→2M方向）で測るので、"
        "バックで向かい風になる風＝ホームストレッチ（スタート→1M）では追い風、になる。"
        "場×月で中心化したレース平均展示タイムを `風速 + 風速·cosθ + 風速·sinθ` に回帰し、展示が最も遅くなる風向を『追い風の吹いてくる方角』とした。\n"
        "- 分類: 推定方向から±45°以内=追い、135°以上離れる=向かい、それ以外=横。振幅（風速1mあたりの展示タイム差の半分）が小さい場は向きの推定が怪しい。\n")
    rows = []
    for j in jorder:
        o = orient.get(j)
        if not o:
            continue
        by = o["by_dir"]
        def bd(code):
            if code in by.index and by.loc[code, "size"] >= 20:
                return f"{by.loc[code, 'mean']*1000:+.0f}"
            return "-"
        rows.append([vname(j), code_to_name16(o["phi_code"]), f"{o['phi_code']:.1f}", f"{o['amp']*1000:.1f}"]
                    + [bd(c) for c in DIR_NAMES])
    out.append("展示偏差は風速4m以上のレースで、場×月平均との差（1/1000秒、+ が遅い）。\n")
    out.append(md_table(pd.DataFrame(rows, columns=["場", "追い風の方角(推定)", "コード", "振幅(ms/m)"]
                                     + [f"{n}({c})" for c, n in DIR_NAMES.items()])) + "\n")

    wb_order = ["0-2m", "追い3-4m", "追い5m+", "横3-4m", "横5m+", "向かい3-4m", "向かい5m+"]
    out.append("### 2-1. 風（場ごとの相対風向 × 風速）\n")
    out.append(md_table(rate_table(rc, "jcd", "windbin", "in_win", jorder, wb_order), "場") + "\n")
    # 向きの推定の検算: 追い風なら差し(2C)が増えSTが速く、向かい風なら外(5-6C)が増えSTが遅い、はず
    stm = RES_ST.rename("stm").reset_index()
    v = rc.merge(stm, on=K, how="left")
    rows = []
    for wb in wb_order:
        g = v[v["windbin"] == wb]
        rows.append([wb, len(g), f"{g['in_win'].mean()*100:.1f}", f"{(g['wc']==2).mean()*100:.1f}",
                     f"{(g['wc']==3).mean()*100:.1f}", f"{(g['wc']==4).mean()*100:.1f}",
                     f"{(g['wc']>=5).mean()*100:.1f}", f"{g['stm'].mean():.3f}"])
    out.append("向きの推定の検算（全国）: 追い風では差し（2C勝ち）が増えて平均STが速く、向かい風では外（5-6C勝ち）が増えて平均STが遅くなるはず。\n")
    out.append(md_table(pd.DataFrame(rows, columns=["風", "レース", "1C勝%", "2C勝%", "3C勝%", "4C勝%", "5-6C勝%", "平均ST"])) + "\n")
    rc = rc.copy()
    rc["wave"] = pd.cut(rc["wh"], [-1, 2, 5, 9, 100], labels=["0-2cm", "3-5cm", "6-9cm", "10cm+"]).astype(str)
    out.append("### 2-2. 波高\n")
    out.append(md_table(rate_table(rc[rc["wh"].notna()], "jcd", "wave", "in_win", jorder,
                                   ["0-2cm", "3-5cm", "6-9cm", "10cm+"]), "場") + "\n")
    out.append("### 2-3. 天候\n")
    out.append(md_table(rate_table(rc[rc["weather"].notna()], "jcd", "weather", "in_win", jorder, ["晴", "曇", "雨"]), "場") + "\n")
    out.append("### 2-4. 月\n")
    out.append(md_table(rate_table(rc, "jcd", "month", "in_win", jorder, list(range(1, 13))), "場") + "\n")
    out.append("### 2-5. レース番号\n")
    out.append(md_table(rate_table(rc, "jcd", "rno", "in_win", jorder, list(range(1, 13))), "場") + "\n")
    out.append("### 2-6. レース種別（race_name を分類）\n")
    cats = ["予選", "一般", "特選・選抜系", "準優勝戦", "優勝戦", "ドリーム", "進入固定", "企画・その他"]
    out.append("分類: 準優/優勝/ドリーム/進入固定 は名前で、特選・特賞・選抜・特別・記者 を含むもの=特選・選抜系、残りで 予選/一般、"
               "どれでもない場独自の企画レース名（『サンライズ』『ドラドキ』等）=企画・その他。\n")
    out.append(md_table(rate_table(rc, "jcd", "cat", "in_win", jorder, cats), "場") + "\n")
    out.append("### 2-7. 1号艇の級別（番組表 entries）\n")
    sub = rc[rc["g1"].notna()]
    out.append(f"番組表と結果が両方あるレース: {len(sub):,}（期間内全体の{len(sub)/len(rc)*100:.0f}%）。"
               "左はイン1着率、右は1号艇1着率（前夜に買える指標）。\n")
    t1 = rate_table(sub, "jcd", "g1", "in_win", jorder, ["A1", "A2", "B1", "B2"])
    t2 = rate_table(sub, "jcd", "g1", "l1_win", jorder, ["A1", "A2", "B1", "B2"])
    t2.columns = [f"1号艇1着 {c}" for c in t2.columns]
    out.append(md_table(pd.concat([t1, t2], axis=1), "場") + "\n")
    return "\n".join(out), rc


def sec3(res, rc):
    out = ["## 3. スタートタイミング・展示タイム\n"]
    r = res.copy()
    out.append("ST はフライング・出遅れの区別がデータに無い（最小0.00）。F/L 艇は着順 NULL で残るが ST の値の意味が不明なので、ここでは着順のある艇のみ。\n")
    r = r[r["pos"].notna() & r["st"].notna()]
    # 平均ST by venue x course
    g = r.groupby(["jcd", "course"])["st"].mean().unstack()
    g.index = [vname(i) for i in g.index]
    g.loc["全国"] = r.groupby("course")["st"].mean()
    g.columns = [f"{c}C" for c in g.columns]
    out.append("### 3-1. コース別の平均ST（秒）\n")
    out.append(md_table(g.round(3), "場") + "\n")
    # ST帯 × コース の1着率
    r["stb"] = pd.cut(r["st"], [-0.01, 0.09, 0.12, 0.15, 0.18, 1.0],
                      labels=["≤0.09", "0.10-0.12", "0.13-0.15", "0.16-0.18", "≥0.19"]).astype(str)
    out.append("### 3-2. 自艇のST帯 × コース の1着率（全国）\n")
    out.append(md_table(rate_table(r, "stb", "course", "w", ["≤0.09", "0.10-0.12", "0.13-0.15", "0.16-0.18", "≥0.19"],
                                   [1, 2, 3, 4, 5, 6]), "ST帯") + "\n")
    # レース内ST順位
    r["st_rank"] = r.groupby(K)["st"].rank(method="min")
    r["st_diff_in"] = r["st"] - r.groupby(K)["st"].transform(lambda s: s.iloc[0] if len(s) else np.nan)
    out.append("### 3-3. レース内のST順位 × コース の1着率（全国、1=最速、同タイムは上位扱い）\n")
    out.append(md_table(rate_table(r, "st_rank", "course", "w", [1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]), "ST順位") + "\n")
    # インと2コース（隣）とのST差
    stc = r.pivot_table(index=K, columns="course", values="st", aggfunc="first")
    win = rc.set_index(K)["in_win"]
    df = stc.join(win, how="inner")
    df["d12"] = df[2] - df[1]
    df["dmin"] = df[[2, 3, 4, 5, 6]].min(axis=1) - df[1]
    df["dminb"] = pd.cut(df["dmin"], [-1, -0.10, -0.05, -0.001, 0.049, 0.099, 1],
                         labels=["外が0.10以上速い", "外が0.05-0.09速い", "外が0.01-0.04速い", "同〜インが0.04速い",
                                 "インが0.05-0.09速い", "インが0.10以上速い"]).astype(str)
    t = df.groupby("dminb")["in_win"].agg(["mean", "size"])
    order = ["外が0.10以上速い", "外が0.05-0.09速い", "外が0.01-0.04速い", "同〜インが0.04速い",
             "インが0.05-0.09速い", "インが0.10以上速い"]
    t = t.reindex(order)
    out.append("### 3-4. インのSTと外5艇の最速STの差 → イン1着率（全国）\n")
    out.append(md_table(pd.DataFrame({"ST差": order, "イン1着率": [cell(m, int(n)) for m, n in t.values]})) + "\n")
    # 展示順位
    e = res[res["ex"] > 0].copy()
    e["ex_rank"] = e.groupby(K)["ex"].rank(method="min")
    out.append("### 3-5. 展示タイム順位（レース内、1=最速）と着順（全国）\n")
    rows = []
    for k in range(1, 7):
        s = e[e["ex_rank"] == k]
        rows.append([k, len(s), f"{s['w'].mean()*100:.1f}", f"{s['p2'].mean()*100:.1f}", f"{s['t3'].mean()*100:.1f}"])
    out.append(md_table(pd.DataFrame(rows, columns=["展示順位", "艇数", "1着率", "2着率", "3着内率"])) + "\n")
    out.append("### 3-6. 展示順位 × コース の1着率（全国）\n")
    out.append(md_table(rate_table(e, "ex_rank", "course", "w", [1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]), "展示順位") + "\n")
    # 場別: インの展示順位 1位 vs 6位
    e1 = e[e["course"] == 1].copy()
    e1["exg"] = pd.cut(e1["ex_rank"], [0, 1, 3, 5, 6], labels=["1位", "2-3位", "4-5位", "6位"]).astype(str)
    out.append("### 3-7. 場別: インの展示順位別イン1着率\n")
    out.append(md_table(rate_table(e1, "jcd", "exg", "w", sorted(VENUES), ["1位", "2-3位", "4-5位", "6位"]), "場") + "\n")
    return "\n".join(out)


def sec4(rc, pay):
    out = ["## 4. 配当の分布\n"]
    jorder = sorted(VENUES)
    out.append("払戻は100円あたり。平均/中央値は的中組番ごとの行で計算（複勝・ワイドは1レース複数行）。万舟率=3連単払戻≥10,000円のレース比率。\n")
    tts = ["tansho", "fukusho", "nirensho", "nirenfuku", "wide", "sanrensho", "sanrenfuku"]
    tj = {"tansho": "単勝", "fukusho": "複勝", "nirensho": "2連単", "nirenfuku": "2連複", "wide": "ワイド",
          "sanrensho": "3連単", "sanrenfuku": "3連複"}
    g = pay.groupby(["jcd", "tt"])["payout"].agg(["mean", "median"])
    rows = []
    for j in jorder + ["全国"]:
        row = [vname(j)]
        for tt in tts:
            if j == "全国":
                s = pay[pay["tt"] == tt]["payout"]
                m, md = s.mean(), s.median()
            elif (j, tt) in g.index:
                m, md = g.loc[(j, tt)]
            else:
                row.append("-")
                continue
            row.append(f"{m:,.0f} / {md:,.0f}")
        sub = rc if j == "全国" else rc[rc["jcd"] == j]
        row += [f"{sub['man'].mean()*100:.1f}", f"{sub['p3t'].quantile(0.9):,.0f}", int(sub["man"].notna().sum())]
        rows.append(row)
    out.append("### 4-1. 場別・券種別 平均 / 中央値（円）と万舟率\n")
    out.append(md_table(pd.DataFrame(rows, columns=["場"] + [tj[t] for t in tts] + ["万舟率%", "3連単90%点", "レース"])) + "\n")
    out.append("### 4-2. 万舟率 × レース番号\n")
    out.append(md_table(rate_table(rc[rc["man"].notna()], "jcd", "rno", "man", jorder, list(range(1, 13))), "場") + "\n")
    sub = rc[rc["man"].notna() & rc["g1"].notna()]
    out.append("### 4-3. 万舟率 × 1号艇の級別\n")
    out.append(md_table(rate_table(sub, "jcd", "g1", "man", jorder, ["A1", "A2", "B1", "B2"]), "場") + "\n")
    out.append("### 4-4. 万舟率 × 風（相対風向・風速）／波高（全国と場別の要約）\n")
    wb_order = ["0-2m", "追い3-4m", "追い5m+", "横3-4m", "横5m+", "向かい3-4m", "向かい5m+"]
    out.append(md_table(rate_table(rc[rc["man"].notna()], "jcd", "windbin", "man", jorder, wb_order), "場") + "\n")
    out.append("### 4-5. 万舟率 × レース種別\n")
    cats = ["予選", "一般", "特選・選抜系", "準優勝戦", "優勝戦", "ドリーム", "進入固定", "企画・その他"]
    out.append(md_table(rate_table(rc[rc["man"].notna()], "jcd", "cat", "man", jorder, cats), "場") + "\n")
    # 単勝1号艇の回収率
    out.append("### 4-6. 参考: 1号艇の単勝を全部買ったときの回収率（前夜に決められる買い方の基準線）\n")
    out.append("回収率% = 単勝(1)払戻合計 ÷ (100円×レース数)。1号艇の級別ごと。控除率から 75% 前後が基準。\n")
    s = rc[rc["g1"].notna() & rc["p3t"].notna()].copy()
    s["r"] = s["tan1"] / 100.0
    gg = s.groupby(["jcd", "g1"])["r"].agg(["mean", "size"])
    rows = []
    for j in jorder + ["全国"]:
        row = [vname(j)]
        for gr in ["A1", "A2", "B1", "B2"]:
            if j == "全国":
                x = s[s["g1"] == gr]["r"]
                m, n = x.mean(), len(x)
            elif (j, gr) in gg.index:
                m, n = gg.loc[(j, gr)]
            else:
                row.append("-")
                continue
            row.append(cell(m, int(n)))
        x = s if j == "全国" else s[s["jcd"] == j]
        row.append(cell(x["r"].mean(), len(x)))
        rows.append(row)
    out.append(md_table(pd.DataFrame(rows, columns=["場", "A1", "A2", "B1", "B2", "全体"])) + "\n")
    out.append("### 4-7. 参考: 1号艇単勝の回収率 × レース番号（場別）\n")
    out.append("288セルあるので偶然 100% を超えるセルが必ず出る（n≈570 で標準誤差はおよそ ±5〜8 ポイント）。単独のセルを根拠にしないこと。\n")
    s2 = rc[rc["p3t"].notna()].copy()
    s2["r"] = s2["tan1"] / 100.0
    out.append(md_table(rate_table(s2, "jcd", "rno", "r", jorder, list(range(1, 13))), "場") + "\n")
    return "\n".join(out)


def sec5(rc):
    out = ["## 5. 前夜に分かる条件だけでの場別要約（自動集計）\n"]
    out.append("前夜に分かる＝場・レース番号・月・レース種別・番組表（級別、勝率）。風・波・展示・STは当日情報なので除く。\n")
    jorder = sorted(VENUES)
    s = rc[rc["g1"].notna()]
    rows = []
    for j in jorder:
        g = rc[rc["jcd"] == j]
        if len(g) == 0:
            continue
        gs = s[s["jcd"] == j]
        a1 = gs[gs["g1"] == "A1"]
        a1solo = a1[a1["n_a1_oth"] == 0]
        b = gs[gs["g1"].isin(["B1", "B2"])]
        rn = g.groupby("rno")["in_win"].mean()
        mo = g.groupby("month")["in_win"].mean()
        tan_a1solo = (a1solo["tan1"].sum() / (100 * len(a1solo))) if len(a1solo) else np.nan
        rows.append([vname(j), len(g), f"{g['in_win'].mean()*100:.1f}", f"{g['l1_win'].mean()*100:.1f}",
                     cell(a1["l1_win"].mean(), len(a1)), cell(a1solo["l1_win"].mean(), len(a1solo)),
                     cell(tan_a1solo, len(a1solo)), cell(b["l1_win"].mean(), len(b)),
                     f"{int(rn.idxmax())}R {rn.max()*100:.0f} / {int(rn.idxmin())}R {rn.min()*100:.0f}",
                     f"{int(mo.idxmax())}月 {mo.max()*100:.0f} / {int(mo.idxmin())}月 {mo.min()*100:.0f}",
                     f"{g['man'].mean()*100:.1f}"])
    out.append(md_table(pd.DataFrame(rows, columns=[
        "場", "レース", "イン1着%", "1号艇1着%", "1号艇A1→1着%", "1号艇A1&他にA1無し→1着%", "同 単勝1回収率%",
        "1号艇B級→1着%", "イン最強R/最弱R", "イン最強月/最弱月", "万舟率%"])) + "\n")
    # 1号艇 A1 と他艇の A1 数（全国）
    out.append("### 5-2. 1号艇の級別 × 他5艇のA1人数 → 1号艇1着率・単勝回収率（全国）\n")
    s = s.copy()
    s["oa"] = s["n_a1_oth"].clip(upper=2).map({0: "0", 1: "1", 2: "2+"})
    rows = []
    for gr in ["A1", "A2", "B1", "B2"]:
        for oa in ["0", "1", "2+"]:
            x = s[(s["g1"] == gr) & (s["oa"] == oa)]
            if len(x) == 0:
                continue
            rows.append([gr, oa, len(x), f"{x['l1_win'].mean()*100:.1f}", f"{x['tan1'].sum()/(100*len(x))*100:.1f}",
                         f"{x['man'].mean()*100:.1f}"])
    out.append(md_table(pd.DataFrame(rows, columns=["1号艇", "他A1数", "レース", "1号艇1着%", "単勝1回収率%", "万舟率%"])) + "\n")
    # 勝率差
    out.append("### 5-3. 1号艇の全国勝率 − 他5艇の最高勝率 → 1号艇1着率・単勝回収率（全国）\n")
    s["gap"] = s["nwr1"] - s["nwr_oth_max"]
    s["gapb"] = pd.cut(s["gap"], [-10, -1.5, -0.5, 0, 0.5, 1.0, 1.5, 2.5, 10],
                       labels=["≤-1.5", "-1.5〜-0.5", "-0.5〜0", "0〜0.5", "0.5〜1.0", "1.0〜1.5", "1.5〜2.5", ">2.5"]).astype(str)
    rows = []
    for b in ["≤-1.5", "-1.5〜-0.5", "-0.5〜0", "0〜0.5", "0.5〜1.0", "1.0〜1.5", "1.5〜2.5", ">2.5"]:
        x = s[s["gapb"] == b]
        if len(x) == 0:
            continue
        rows.append([b, len(x), f"{x['l1_win'].mean()*100:.1f}", f"{x['tan1'].sum()/(100*len(x))*100:.1f}",
                     f"{x['man'].mean()*100:.1f}"])
    out.append(md_table(pd.DataFrame(rows, columns=["勝率差", "レース", "1号艇1着%", "単勝1回収率%", "万舟率%"])) + "\n")
    return "\n".join(out)


# ---------------------------------------------------------------- figures
def figures(res, rc):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    made = []
    plt.rcParams["font.family"] = ["DejaVu Sans"]
    jorder = sorted(VENUES)
    t = rc.groupby("jcd")["in_win"].mean().reindex(jorder) * 100
    fig, ax = plt.subplots(figsize=(8, 3.2), dpi=90)
    ax.bar(range(len(t)), t.values, color="#3b6ea5")
    ax.axhline(rc["in_win"].mean() * 100, color="#888", lw=1, ls="--")
    ax.set_xticks(range(len(t)))
    ax.set_xticklabels(jorder, fontsize=8)
    ax.set_ylabel("in win %")
    ax.set_title("1-course win rate by venue (jcd)")
    fig.tight_layout()
    p = IMG_DIR / "in_win_by_venue.png"
    fig.savefig(p)
    plt.close(fig)
    made.append(p)
    h = rc.pivot_table(index="jcd", columns="rno", values="in_win", aggfunc="mean").reindex(jorder) * 100
    fig, ax = plt.subplots(figsize=(7, 6), dpi=90)
    im = ax.imshow(h.values, cmap="Blues", vmin=30, vmax=80, aspect="auto")
    ax.set_yticks(range(len(h.index)))
    ax.set_yticklabels(h.index, fontsize=8)
    ax.set_xticks(range(12))
    ax.set_xticklabels(range(1, 13), fontsize=8)
    ax.set_xlabel("race no.")
    ax.set_ylabel("jcd")
    for i in range(h.shape[0]):
        for j in range(h.shape[1]):
            val = h.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=6,
                        color="white" if val > 60 else "black")
    fig.colorbar(im, ax=ax, label="1-course win %")
    ax.set_title("1-course win % by venue x race number")
    fig.tight_layout()
    p = IMG_DIR / "in_win_venue_rno.png"
    fig.savefig(p)
    plt.close(fig)
    made.append(p)
    return made


# ---------------------------------------------------------------- main
TOP3 = None
RES_ST = None


def main():
    global TOP3
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="d_from", default="20231101")
    ap.add_argument("--to", dest="d_to", default="20261101")
    a = ap.parse_args()
    d = load(a.d_from, a.d_to)
    res, rc, pay = build(d)
    TOP3 = pay[pay["tt"] == "sanrensho"][["jcd", "comb"]]
    orient = wind_orientation(res, rc)
    rc = rel_wind(rc, orient)
    global RES_ST
    RES_ST = res.groupby(K)["st"].mean()
    parts = []
    head = (f"# 場 × コース × 条件 の基礎統計\n\n"
            f"期間 {a.d_from}〜{a.d_to}。`python scripts/analysis/venue_course.py --from {a.d_from} --to {a.d_to}` で表を再生成"
            f"（MANUAL ブロックは保持）。率は%、括弧内は母数、† は母数{MIN_N}未満で参考値。\n"
            f"\n**データ補正**: DB では唐津(23)の成績・払戻・レース情報が津(09)の場コードで入っている（取り込み不具合）。"
            f"番組表の選手番号と照合し {d['remap']['to23_days']} 日分を唐津に付け替えた（津のまま {d['remap']['09_days']} 日）。"
            f"両場が同日開催だった日はどちらか一方の成績しか DB に残っていないため、津・唐津はほかの場より件数が少ない。\n")
    parts.append(head)
    s1 = sec1(res, rc)
    s2, rc2 = sec2(rc, orient)
    s3 = sec3(res, rc2)
    s4 = sec4(rc2, pay)
    s5 = sec5(rc2)
    imgs = figures(res, rc2)
    manual = "<!-- MANUAL START -->\n（ここに手書きの要約と仮説）\n<!-- MANUAL END -->"
    if OUT_MD.exists():
        m = re.search(r"<!-- MANUAL START -->.*?<!-- MANUAL END -->", OUT_MD.read_text(encoding="utf-8"), re.S)
        if m:
            manual = m.group(0)
    parts.append(manual)
    parts += [s1, s2, s3, s4, s5]
    if imgs:
        parts.append("## 図\n\n" + "\n".join(f"![{p.stem}](img/{p.name})" for p in imgs))
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n\n".join(parts) + "\n", encoding="utf-8")
    print(f"wrote {OUT_MD} races={len(rc2):,} with_entries={rc2['g1'].notna().sum():,}")


if __name__ == "__main__":
    main()
