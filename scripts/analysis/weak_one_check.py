"""①の展示不調（展示タイム最下位・スタ展最遅）はどれくらい当てになるか（2026-11-18 追加読み b）。

対象: beforeinfo が6艇そろい（展示タイム・スタ展ST）、結果があるレース（2025-11〜）。
- 展示順位6位 = ①の展示タイムが6艇の最大（同タイム含む）
- スタ展最遅 = ①のスタ展STが6艇の最大（同値含む）。F表記はマイナス（早い）、L は遅い(+1.0)扱い
- 気配悪い（番組表＝前夜に分かる情報）: モーター2連率が6艇中下位2以内 / 節間平均着順 4.0 以上（2走以上）
- 単勝回収率は確定配当（締切前のオッズとは違う）
- 市場の見込み = 締切時3連単オッズの 頭X の 1/オッズ 合計 ÷ 全120通りの 1/オッズ 合計（odds がある 〜2026-11-03 のみ）

    python scripts/analysis/weak_one_check.py [出力md]
"""
import sys, json, math, sqlite3
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "data" / "boatrace.db"
FROM, TO = "20251101", "20261117"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else None  # 指定時だけ表を書き出す


def load():
    c = sqlite3.connect(DB)
    bi = pd.read_sql(f"""select hd,jcd,rno,lane,exhibition_time ex,start_st st,
        json_extract(data,'$.start_st_raw') raw from beforeinfo where hd between '{FROM}' and '{TO}'""", c)
    bi.loc[bi.raw.fillna("").str.startswith("L"), "st"] = 1.0
    en = pd.read_sql(f"""select hd,jcd,rno,lane,json_extract(data,'$.motor_2ren_rate') m2,
        json_extract(data,'$.series_results') sr from entries where hd between '{FROM}' and '{TO}'""", c)
    rs = pd.read_sql(f"select hd,jcd,rno,lane,position pos from results where hd between '{FROM}' and '{TO}'", c)
    pt = pd.read_sql(f"""select hd,jcd,rno,combination comb,payout from payouts
        where ticket_type='tansho' and hd between '{FROM}' and '{TO}'""", c)
    od = pd.read_sql(f"""select hd,jcd,rno,substr(combination,1,1) h,sum(1.0/odds) q from odds
        where ticket_type='sanrensho' and odds>0 and hd between '{FROM}' and '{TO}' group by 1,2,3,4""", c)
    return bi, en, rs, pt, od


def series_avg(s):
    d = [int(ch) for ch in str(s or "") if ch in "123456"]
    return np.mean(d) if len(d) >= 2 else np.nan


def build():
    bi, en, rs, pt, od = load()
    k = ["hd", "jcd", "rno"]
    ok = bi.groupby(k).agg(n=("lane", "size"), nex=("ex", "count"), nst=("st", "count")).reset_index()
    ok = ok[(ok.n == 6) & (ok.nex == 6) & (ok.nst == 6)][k]
    bi = bi.merge(ok, on=k)
    g = bi.groupby(k)
    bi["ex_worst"] = bi.ex == g.ex.transform("max")
    bi["ex_best"] = bi.ex == g.ex.transform("min")
    bi["st_slow"] = bi.st == g.st.transform("max")
    bi["st_fast"] = bi.st == g.st.transform("min")
    en["m2"] = pd.to_numeric(en.m2, errors="coerce")
    en["m2rank"] = en.groupby(k).m2.rank(ascending=False, method="min")  # 1=最良
    en["sravg"] = en.sr.map(series_avg)
    bi = bi.merge(en[k + ["lane", "m2rank", "sravg"]], on=k + ["lane"], how="left")
    rs = rs[rs.groupby(k).pos.transform(lambda s: (s == 1).any())]
    bi = bi.merge(rs, on=k + ["lane"], how="inner")
    bi["win"] = bi.pos == 1
    bi["top3"] = bi.pos.between(1, 3)
    pt["lane"] = pd.to_numeric(pt.comb, errors="coerce")
    pt = pt.dropna(subset=["lane"]).astype({"lane": int}).groupby(k + ["lane"]).payout.max().reset_index()
    bi = bi.merge(pt, on=k + ["lane"], how="left")
    bi["payout"] = bi.payout.fillna(0)
    tot = od.groupby(k).q.sum().rename("qs").reset_index()
    od = od.merge(tot, on=k)
    od["mkt"] = od.q / od.qs
    od["lane"] = od.h.astype(int)
    bi = bi.merge(od[k + ["lane", "mkt"]], on=k + ["lane"], how="left")
    # レース単位の横持ち（各艇の1着）
    w = bi.pivot_table(index=k, columns="lane", values="win", aggfunc="first").add_prefix("w")
    one = bi[bi.lane == 1].set_index(k).join(w)
    return bi, one.reset_index()


def wilson(x, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = x / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def pct(x):
    return "-" if pd.isna(x) else f"{x*100:.1f}%"


def row(name, d, lane=1):
    n = len(d)
    if n == 0:
        return f"| {name} | 0 | | | | | | | |"
    wcol = f"w{lane}" if lane != 1 else "win"
    x = int(d[wcol].sum()) if wcol in d else int(d.win.sum())
    lo, hi = wilson(x, n)
    t3 = pct(d.top3.mean()) if lane == 1 else ""
    w2 = pct(d.w2.mean()) if lane == 1 else ""
    w4 = pct(d.w4.mean()) if lane == 1 else ""
    roi = pct(d.payout.sum() / 100 / n)
    m = d.dropna(subset=["mkt"])
    mk = f"{pct(m.mkt.mean())} / {pct(m[wcol].mean())} (n={len(m)})" if len(m) else "-"
    return f"| {name} | {n} | {pct(x/n)} ({pct(lo)}–{pct(hi)}) | {t3} | {w2} | {w4} | {roi} | {mk} |"


HDR = ("| 条件 | レース数 | ①1着率 (95%区間) | ①3着内 | ②1着 | ④1着 | ①単勝回収率 | 市場見込み / 実1着率 (odds期間) |\n"
       "|---|---:|---:|---:|---:|---:|---:|---|")


def table(one):
    ew, ss = one.ex_worst, one.st_slow
    mb = one.m2rank >= 5
    sb = one.sravg >= 4.0
    conds = [
        ("全体", one),
        ("①展示6位", one[ew]),
        ("①スタ展最遅", one[ss]),
        ("①展示6位＋スタ展最遅", one[ew & ss]),
        ("①展示6位＋モーター2連率下位2", one[ew & mb]),
        ("①展示6位＋節間平均4着以下", one[ew & sb]),
        ("①展示6位＋スタ展最遅＋(モーター下位2 or 節間4着以下)", one[ew & ss & (mb | sb)]),
    ]
    return HDR + "\n" + "\n".join(row(nm, d) for nm, d in conds)


def challengers(bi):
    d = bi[(bi.lane >= 2) & bi.ex_best & bi.st_fast]
    lines = ["| 艇 | 件数 | 1着率 (95%区間) | 単勝回収率 | 市場見込み / 実1着率 (odds期間) | 参考: その艇の全体1着率 |",
             "|---|---:|---:|---:|---|---:|"]
    for ln in [2, 3, 4, 5, 6, "2-6計"]:
        s = d if ln == "2-6計" else d[d.lane == ln]
        base = bi[bi.lane >= 2] if ln == "2-6計" else bi[bi.lane == ln]
        n = len(s)
        if n == 0:
            continue
        x = int(s.win.sum()); lo, hi = wilson(x, n)
        m = s.dropna(subset=["mkt"])
        lines.append(f"| {ln} | {n} | {pct(x/n)} ({pct(lo)}–{pct(hi)}) | {pct(s.payout.sum()/100/n)} | "
                     f"{pct(m.mkt.mean())} / {pct(m.win.mean())} (n={len(m)}) | {pct(base.win.mean())} |")
    return "\n".join(lines)


def main():
    bi, one = build()
    out = [f"対象 {one.hd.min()}〜{one.hd.max()} / {len(one)} レース（odds あり {one.mkt.notna().sum()}）\n"]
    for nm, sel in [("全場", None), ("びわこ(11)", "11"), ("丸亀(15)", "15")]:
        o = one if sel is None else one[one.jcd == sel]
        b = bi if sel is None else bi[bi.jcd == sel]
        out += [f"## {nm}", "", table(o), "", f"### {nm} 頭候補（②〜⑥）の展示1位＋スタ展最速", "", challengers(b), ""]
    txt = "\n".join(out)
    if OUT:
        OUT.write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
