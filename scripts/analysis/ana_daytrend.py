"""穴狙い6回目（2026-11-21）: その日の水面傾向（同じ場・同じ日の前のレースの結果）の筋書き W1〜W8。notes/patterns.md に先に固定。

    python scripts/analysis/ana_daytrend.py IN.parquet [シャッフル回数=200]
"""
import sys, itertools
import numpy as np, pandas as pd, duckdb
SPLIT = "20260501"; BAND = (15, 60)
df = pd.read_parquet(sys.argv[1]).reset_index(drop=True); NPERM = int(sys.argv[2]) if len(sys.argv) > 2 else 200
N = len(df); disc = (df.hd < SPLIT).values; c = duckdb.connect(); P = "data/parquet"
d = c.sql(f"""select k.hd, k.jcd, k.rno, k.kimarite, p.combination cb, p.payout from '{P}/kimarite_*.parquet' k
  join '{P}/payouts_*.parquet' p on p.hd=k.hd and cast(p.jcd as int)=k.jcd and p.rno=k.rno and p.ticket_type='sanrensho' where k.hd>='20251101'""").df()
d = d.drop_duplicates(["hd", "jcd", "rno"]).sort_values(["hd", "jcd", "rno"])
d["in1"] = d.cb.str[0] == "1"; d["mk"] = d.kimarite.isin(["まくり", "まくり差し"]); d["s2"] = (d.kimarite == "差し") & (d.cb.str[0] == "2"); d["man"] = d.payout >= 10000; d["one"] = 1
g = d.groupby(["hd", "jcd"])
for a in ("in1", "mk", "s2", "man", "one"):
    d["p_" + a] = g[a].cumsum() - d[a]          # そのレースより前の合計
df = df.merge(d[["hd", "jcd", "rno", "p_in1", "p_mk", "p_s2", "p_man", "p_one"]], on=["hd", "jcd", "rno"], how="left"); assert len(df) == N
n = df.p_one.fillna(0).values; inr = np.where(n > 0, df.p_in1 / np.maximum(n, 1), np.nan); mkr = np.where(n > 0, df.p_mk / np.maximum(n, 1), np.nan)
p1 = df.p_H1.values
C = {"lowin": (n >= 4) & (inr <= .25), "mk": (n >= 4) & (mkr >= .5), "hiin": (n >= 5) & (inr >= .8), "s2": df.p_s2.fillna(0).values >= 2, "man": df.p_man.fillna(0).values >= 2}
print(f"レース {N}。該当数: " + "、".join(f"{k}={int(v.sum())}" for k, v in C.items()))
ALL = ["".join(map(str, x)) for x in itertools.permutations(range(1, 7), 3)]; CI = {x: i for i, x in enumerate(ALL)}
Q = df[["t_" + x for x in ALL]].fillna(0).values; wcb = df.cb.str.replace("-", "").map(CI).values
qw, pay, S = df.qw.values, df.payout.values / 100, df.S.values
def tk(*pats):
    m = np.zeros(120, bool)
    for p in pats:
        for x in ALL:
            if all(a == "x" or a == b for a, b in zip(p, x)): m[CI[x]] = True
    return m
with np.errstate(divide="ignore"): O = np.where(Q > 0, 1 / Q, np.inf)
Om = np.where(O >= 100, O, np.inf); order = np.argsort(Om, axis=1)[:, :10]
ret10 = np.where((O[np.arange(N), wcb] >= 100) & (order == wcb[:, None]).any(1), pay / 10, 0.0)   # 10点・各100円の払戻倍率
def dutch(m):
    sq = Q[:, m].sum(1); hit = m[wcb]
    with np.errstate(all="ignore"): return np.where(hit, qw / sq * pay, 0.0), 1 / sq, sq / S
RET = {"W1 イン率25%以下→頭③④": ("lowin", None, dutch(tk("3xx", "4xx"))), "W2 イン率25%以下＋①人気→2-1,3-1,4-1": ("lowin", p1 >= .5, dutch(tk("21x", "31x", "41x"))),
       "W3 まくり率50%以上→3-4,3-5,4-3,4-5": ("mk", None, dutch(tk("34x", "35x", "43x", "45x"))), "W4 イン率80%以上＋①頭5割以下→頭①": ("hiin", p1 <= .5, dutch(tk("1xx"))),
       "W5 イン率80%以上→1-2,1-3": ("hiin", None, dutch(tk("12x", "13x"))), "W6 2コース差し2回以上→2-1": ("s2", None, dutch(tk("21x"))),
       "W7 万舟2本以上→100倍以上の安い10点": ("man", None, (ret10, np.full(N, np.nan), np.full(N, np.nan))), "W8 イン率25%以下→100倍以上の安い10点": ("lowin", None, (ret10, np.full(N, np.nan), np.full(N, np.nan)))}
cols = ["筋書き", "帯", "n発", "的中発", "見立て発", "実際発", "回収発", "n検", "的中検", "見立て検", "実際検", "回収検", "合成中央"]
def evaluate(perm=None):
    rows = []
    for name, (ck, extra, (ret, co, pm)) in RET.items():
        cond = C[ck] if perm is None else C[ck][perm]
        if extra is not None: cond = cond & extra
        bands = [("全部", cond)] + ([("合成15〜60倍", cond & (co >= BAND[0]) & (co <= BAND[1]))] if not np.isnan(co).all() else [])
        for bn, bm in bands:
            r = []
            for per in (disc, ~disc):
                x = bm & per
                r += [int(x.sum()), int((ret[x] > 0).sum())] + ([np.nanmean(pm[x]) if not np.isnan(pm[x]).all() else np.nan, (ret[x] > 0).mean(), ret[x].mean()] if x.any() else [np.nan] * 3)
            rows.append([name, bn] + r + [np.nanmedian(co[bm]) if bm.any() and not np.isnan(co[bm]).all() else np.nan, bm])
    return rows
rows = evaluate(); t = pd.DataFrame([r[:-1] for r in rows], columns=cols); t["月あたり"] = ((t.n発 + t.n検) / 12).round(0)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 50); print(t.round(3).to_string(index=False))
ok = lambda a, th: int(((a.回収発 >= th) & (a.回収検 >= th) & (a.的中発 >= 15) & (a.的中検 >= 10)).sum())
print(f"\n{len(t)}区分のうち 両期間とも回収1.0以上: {ok(t,1)}、0.9以上: {ok(t,.9)}")
rng = np.random.default_rng(0); a1 = []; a9 = []
for it in range(NPERM):
    perm = np.arange(N)
    for msk in (disc, ~disc):
        idx = np.where(msk)[0]; perm[idx] = rng.permutation(idx)
    t2 = pd.DataFrame([r[:-1] for r in evaluate(perm)], columns=cols); a1.append(ok(t2, 1)); a9.append(ok(t2, .9))
print(f"シャッフル{NPERM}回: 両方1.0以上 平均{np.mean(a1):.2f}、両方0.9以上 平均{np.mean(a9):.2f}")
print("\n== その日のイン率ごとの、①の実際の1着率と市場の見立て（前に4レース以上）==")
b = pd.cut(pd.Series(inr[n >= 4]), [-.01, .25, .4, .6, .8, 1.0])
print(pd.DataFrame({"b": b.values, "実際": (df.wh.values == 1)[n >= 4], "市場": p1[n >= 4], "万舟": (pay >= 100)[n >= 4]}).groupby("b", observed=True).agg(n=("実際", "size"), 実際=("実際", "mean"), 市場=("市場", "mean"), 万舟率=("万舟", "mean")).round(3).to_string())
