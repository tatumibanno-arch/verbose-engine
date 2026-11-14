"""穴狙い4回目（2026-11-14）: 市場の形（締切時の3連単オッズ）から「みんなの予想が外れた時」を狙う筋書き U1〜U12。
筋書きは notes/patterns.md に先に固定。条件にも払戻にも締切時オッズを使う＝実際より甘い。

    python scripts/analysis/ana_market.py IN.parquet [シャッフル回数=200]
"""
import sys, itertools
import numpy as np, pandas as pd, duckdb

SPLIT = "20260501"; BAND = (15, 60)
df = pd.read_parquet(sys.argv[1]).reset_index(drop=True)
NPERM = int(sys.argv[2]) if len(sys.argv) > 2 else 200
N = len(df); disc = (df.hd < SPLIT).values
ALL = ["".join(map(str, c)) for c in itertools.permutations(range(1, 7), 3)]
CI = {c: i for i, c in enumerate(ALL)}
Q = df[["t_" + c for c in ALL]].fillna(0).values
wcb = df.cb.str.replace("-", "").map(CI).values
qw, pay, S = df.qw.values, df.payout.values / 100, df.S.values
pH = {i: df[f"p_H{i}"].values for i in range(1, 7)}


def tk(*pats):
    m = np.zeros(120, bool)
    for p in pats:
        for c in ALL:
            if all(a == "x" or a == b for a, b in zip(p, c)):
                m[CI[c]] = True
    return np.tile(m, (N, 1))


c4 = (pH[4] >= .20) & (pH[1] <= .50); c3 = (pH[3] >= .22) & (pH[1] <= .50); c1 = pH[1] >= .80
order = np.argsort(-Q, axis=1); top3 = order[:, :3]
top3share = np.take_along_axis(Q, top3, axis=1).sum(axis=1) / S
head = np.array([int(c[0]) for c in ALL])
m9 = head[None, :] == head[order[:, 0]][:, None]
np.put_along_axis(m9, top3, False, axis=1)
share41_45 = (df.q_P41.values + df.q_P45.values) / df.q_H4.values
ST = {
    "U1 ④人気→頭⑤": (c4, tk("5xx")),
    "U2 ④人気→②差し 2-145": (c4, tk("21x", "24x", "25x")),
    "U3 ④人気→1-56": (c4, tk("15x", "16x")),
    "U4 ④人気→⑤⑥ 5-6,6-5,5-1,6-1": (c4, tk("56x", "65x", "51x", "61x")),
    "U5 ③人気→頭④": (c3, tk("4xx")),
    "U6 ③人気→1-45": (c3, tk("14x", "15x")),
    "U7 ①鉄板→2-1,3-1": (c1, tk("21x", "31x")),
    "U8 ①鉄板→1-456": (c1, tk("14x", "15x", "16x")),
    "U9 3点集中→同じ頭の残り": (top3share >= .45, m9),
    "U10 ①不人気→頭①": (pH[1] <= .30, tk("1xx")),
    "U11 ②人気→1-34": (pH[2] >= .25, tk("13x", "14x")),
    "U12 ④頭で相手決め打ち→4-236": ((pH[4] >= .15) & (share41_45 >= .65), tk("42x", "43x", "46x")),
}
PRE = {}
for name, (cond, M) in ST.items():
    sq = (Q * M).sum(axis=1); hit = M[np.arange(N), wcb]
    with np.errstate(all="ignore"):
        PRE[name] = (cond, np.where(hit, qw / sq * pay, 0.0), 1 / sq, sq / S)


def evaluate(shuffle=None):
    rows = []
    for name, (cond, ret, co, pm) in PRE.items():
        c = cond if shuffle is None else cond[shuffle]
        for bn, bm in (("全部", c), ("合成15〜60倍", c & (co >= BAND[0]) & (co <= BAND[1]))):
            r = []
            for per in (disc, ~disc):
                x = bm & per
                r += [int(x.sum()), int((ret[x] > 0).sum())] + ([pm[x].mean(), (ret[x] > 0).mean(), ret[x].mean()] if x.any() else [np.nan] * 3)
            rows.append([name, bn] + r + [np.median(co[bm]) if bm.any() else np.nan, bm])
    return rows


cols = ["筋書き", "帯", "n発", "的中発", "見立て発", "実際発", "回収発", "n検", "的中検", "見立て検", "実際検", "回収検", "合成中央"]
rows = evaluate(); t = pd.DataFrame([r[:-1] for r in rows], columns=cols); t["月あたり"] = ((t.n発 + t.n検) / 12).round(0)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 40)
print(t.round(3).to_string(index=False))
ok = lambda a: int(((a.回収発 >= 1) & (a.回収検 >= 1) & (a.的中発 >= 15) & (a.的中検 >= 10)).sum())
ok9 = lambda a: int(((a.回収発 >= .9) & (a.回収検 >= .9) & (a.的中発 >= 15) & (a.的中検 >= 10)).sum())
print(f"\n{len(t)}区分のうち 両期間とも回収1.0以上（的中 発見15・検証10以上）: {ok(t)}、両期間0.9以上: {ok9(t)}")
rng = np.random.default_rng(0); a1 = []; a9 = []
for it in range(NPERM):
    perm = np.arange(N)
    for msk in (disc, ~disc):
        idx = np.where(msk)[0]; perm[idx] = rng.permutation(idx)
    t2 = pd.DataFrame([r[:-1] for r in evaluate(perm)], columns=cols); a1.append(ok(t2)); a9.append(ok9(t2))
print(f"シャッフル{NPERM}回: 両方1.0以上 平均{np.mean(a1):.2f}（最大{max(a1)}）、両方0.9以上 平均{np.mean(a9):.2f}（最大{max(a9)}）")
print("\n== 月ごとの回収（帯=全部）=="); mon = df.hd.str[:6]
for r in rows:
    if r[1] != "全部" or not r[-1].any(): continue
    ret = PRE[r[0]][1]
    g = pd.DataFrame({"m": mon[r[-1]], "r": ret[r[-1]]}).groupby("m").r.agg(["size", "mean"])
    print(r[0], " ".join(f"{m[2:]}:{v:.2f}({int(n)})" for m, n, v in zip(g.index, g["size"], g["mean"])), f" 勝ち月{int((g['mean']>1).sum())}/{len(g)}")

# ---- おまけ: S5・S6 を 2連単・2連複で（各1点、締切時オッズ、払戻は実際の払戻）----
print("\n== おまけ: S5・S6 を2連単・2連複で ==")
c = duckdb.connect(); P = "data/parquet"
f = lambda col: np.asarray(df[col].values, dtype=float)
s5 = (f("p_H1") >= .5) & (f("mo1") <= 25) & ((f("er5") <= 2) | (f("mo5") >= 40))
s6 = (f("gr1") <= 2) & (f("gr6") == 4) & (f("er3") <= 3)
for nm, cond, tt, cb in (("S5 2連単5-1", s5, "nirensho", "5-1"), ("S5 2連複1=5", s5, "nirenfuku", "1-5"), ("S6 2連単3-2", s6, "nirensho", "3-2"), ("S6 2連複2=3", s6, "nirenfuku", "2-3")):
    od = c.sql(f"select hd, cast(jcd as int) jcd, rno, odds from '{P}/odds_*.parquet' where ticket_type='{tt}' and combination='{cb}' and odds>0").df()
    py = c.sql(f"select hd, cast(jcd as int) jcd, rno, payout from '{P}/payouts_*.parquet' where ticket_type='{tt}' and combination='{cb}'").df()
    x = df.loc[np.nan_to_num(cond.astype(float)).astype(bool), ["hd", "jcd", "rno"]].merge(od, on=["hd", "jcd", "rno"]).merge(py, on=["hd", "jcd", "rno"], how="left")
    x["ret"] = x.payout.fillna(0) / 100; x["per"] = np.where(x.hd < SPLIT, "発見", "検証")
    g = x.groupby("per").agg(n=("ret", "size"), 的中=("ret", lambda s: int((s > 0).sum())), 回収=("ret", "mean"), オッズ中央=("odds", "median"))
    print(nm, " / ".join(f"{p}: {int(r.n)}レース 的中{int(r.的中)} 回収{r.回収:.2f} オッズ中央{r.オッズ中央:.1f}" for p, r in g.iterrows()))
