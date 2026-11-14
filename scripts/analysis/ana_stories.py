"""穴狙い2回目（2026-11-14）: 先に notes/patterns.md に固定した筋書き S1〜S11 を確かめる。

    python scripts/analysis/ana_stories.py IN.parquet [シャッフル回数=200] [組=2|3]   （2=S1〜S11、3=T1〜T10）

買い方は合成買い（締切時オッズで配分＝実際より甘い）。帯は「全部」と「合成15〜60倍のレースだけ」。
偶然の目安: 条件に使う列をレース間（同じ期間の中）でシャッフルして同じ手順を回す。
"""
import sys, itertools
import numpy as np, pandas as pd

SPLIT = "20260501"; BAND = (15, 60)
df = pd.read_parquet(sys.argv[1]).reset_index(drop=True)
NPERM = int(sys.argv[2]) if len(sys.argv) > 2 else 200
disc = (df.hd < SPLIT).values
ALL = ["".join(map(str, c)) for c in itertools.permutations(range(1, 7), 3)]
Q = df[["t_" + c for c in ALL]].fillna(0).values
CI = {c: i for i, c in enumerate(ALL)}
wcb = df.cb.str.replace("-", "").map(CI).values
qw, pay, S = df.qw.values, df.payout.values / 100, df.S.values


def tk(*pats):
    """'45x' のような型（x=全）を点の添字に。"""
    out = set()
    for p in pats:
        for c in ALL:
            if all(a == "x" or a == b for a, b in zip(p, c)):
                out.add(CI[c])
    return sorted(out)


def stories(d):
    """名前 → [(条件の真偽, 買い目の添字), ...]（買い目がレースで変わる筋書きは複数に分ける）"""
    g = lambda n, k: np.asarray(d[f"{n}{k}"].values, dtype=float)
    fav = d.p_H1.values >= .5
    w = d.wind.astype(float).values; wd = d.wdir.astype(float).values
    slow = lambda k, a, b: (g("ms", k) - g("ms", k + 1) >= a) & (g("sx", k) - g("sx", k + 1) >= b)
    st = {}
    c1 = fav & slow(3, .03, .05) & (g("er", 4) <= 2)
    st["S1 ③凹み→④攻め→②⑤差し"] = [(c1, tk("21x", "24x", "51x", "54x"))]
    st["S2 ③凹み→④まくり切り"] = [(c1, tk("41x", "45x"))]
    st["S3 ②凹み→③まくり"] = [(fav & slow(2, .03, .05) & (g("er", 3) <= 2), tk("31x", "34x", "43x"))]
    st["S4 ①遅れ→②"] = [(fav & slow(1, .02, .05) & (g("cw", 2) >= 20), tk("21x", "23x"))]
    st["S5 ①機力弱＋⑤良→5-1-234"] = [(fav & (g("mo", 1) <= 25) & ((g("er", 5) <= 2) | (g("mo", 5) >= 40)), tk("512", "513", "514"))]
    st["S6 ①B級⑥A1→3-2"] = [((g("gr", 1) <= 2) & (g("gr", 6) == 4) & (g("er", 3) <= 3), tk("32x"))]
    st["S7 風向5〜8の風4m→3-4,4-3"] = [((w >= 4) & (wd >= 5) & (wd <= 8) & (g("ms", 3) <= g("ms", 2)), tk("34x", "43x"))]
    base8 = fav & (g("cw", 1) < 50) & (g("er", 1) >= 4)
    st["S8 ①弱＋外の展示1位→k-1,k-全-1"] = [(base8 & (g("er", k) == 1) & (g("cw", k) >= th), tk(f"{k}1x", f"{k}x1")) for k, th in ((3, 20), (4, 18), (5, 10))]
    st["S9 前付け深イン→③④"] = [(~d.waku.astype(bool).values & ((g("sc", 5) <= 3) | (g("sc", 6) <= 3)) & (d.p_H1.values >= .45), tk("31x", "34x", "41x", "43x"))]
    b10 = fav & (w >= 5); use3 = g("er", 3) <= g("er", 4)
    st["S10 強風→③か④-1"] = [(b10 & use3 & (g("er", 3) <= 3), tk("31x")), (b10 & ~use3 & (g("er", 4) <= 3), tk("41x"))]
    st["S11 ①③B級＋④A1カド"] = [((g("gr", 1) <= 2) & (g("gr", 3) <= 2) & (g("gr", 4) == 4) & (g("sc", 4) == 4), tk("41x", "45x"))]
    return st


def stories3(d):
    """3回目: 買われすぎの形を外す（T1〜T10）。"""
    g = lambda n, k: np.asarray(d[f"{n}{k}"].values, dtype=float)
    p1 = d.p_H1.values; fav = p1 >= .5
    w = d.wind.astype(float).values
    slow = lambda k, a, b: (g("ms", k) - g("ms", k + 1) >= a) & (g("sx", k) - g("sx", k + 1) >= b)
    perm2 = lambda xs: [f"1{a}{b}" for a in xs for b in xs if a != b]
    st = {}
    c1 = fav & slow(3, .03, .05) & (g("er", 4) <= 2)
    st["T1 ③凹み④好展示→1-235-235"] = [(c1, tk(*perm2("235")))]
    st["T2 ③凹み④好展示→1-2-全"] = [(c1, tk("12x"))]
    st["T3 ②凹み③好展示→1-245-245"] = [(fav & slow(2, .03, .05) & (g("er", 3) <= 2), tk(*perm2("245")))]
    et = np.column_stack([g("et", k) for k in range(1, 7)]); srt = np.sort(et, axis=1)
    lone = srt[:, 1] - srt[:, 0] >= .05
    st["T4 外の展示抜け1位を外す→1-2,1-3"] = [(fav & lone & (et[:, k - 1] == srt[:, 0]),
                                    [i for i in tk("12x", "13x") if str(k) not in ALL[i]]) for k in (4, 5, 6)]
    st["T5 ⑥A1→1-234-234"] = [(fav & (g("gr", 6) == 4), tk(*perm2("234")))]
    st["T6 風5m→1-2,1-3-全"] = [(fav & (w >= 5), tk("12x", "13x"))]
    st["T7 枠なりでない→頭①"] = [(fav & ~d.waku.astype(bool).values, tk("1xx"))]
    st["T8 鉄板①→1-23-234の4点"] = [((p1 >= .7) & (g("gr", 1) == 4) & (g("cw", 1) >= 70) & (g("er", 1) <= 3), tk("123", "132", "124", "134"))]
    ms = np.column_stack([g("ms", k) for k in range(1, 7)])
    with np.errstate(all="ignore"):
        fastest = ms[:, 0] <= np.nanmin(ms[:, 1:], axis=1)
    st["T9 少し疑われた①が展示1位・今節ST最速→頭①"] = [((p1 >= .5) & (p1 <= .65) & (g("er", 1) == 1) & fastest, tk("1xx"))]
    st["T10 ①③B級④A1カド→1-2,2-1-全"] = [((g("gr", 1) <= 2) & (g("gr", 3) <= 2) & (g("gr", 4) == 4) & (g("sc", 4) == 4), tk("12x", "21x"))]
    return st


def evaluate(st):
    rows = []
    for name, parts in st.items():
        m = np.zeros(len(df), bool); sq = np.zeros(len(df)); hit = np.zeros(len(df), bool)
        for cond, idx in parts:
            cond = np.nan_to_num(cond.astype(float)).astype(bool) & ~m
            m |= cond
            sq[cond] = Q[cond][:, idx].sum(axis=1)
            hit[cond] = np.isin(wcb[cond], idx)
        with np.errstate(divide="ignore", invalid="ignore"):
            ret = np.where(hit & m, qw / sq * pay, 0.0); co = 1 / sq
        for bn, bm in (("全部", m), ("合成15〜60倍", m & (co >= BAND[0]) & (co <= BAND[1]))):
            r = []
            for per in (disc, ~disc):
                x = bm & per
                r += [int(x.sum()), int((ret[x] > 0).sum()), (sq[x] / S[x]).mean() if x.any() else np.nan,
                      (ret[x] > 0).mean() if x.any() else np.nan, ret[x].mean() if x.any() else np.nan]
            rows.append([name, bn] + r + [np.median(co[bm]) if bm.any() else np.nan, ret, bm])
    return rows


STF = stories3 if (len(sys.argv) > 3 and sys.argv[3] == "3") else stories
rows = evaluate(STF(df))
cols = ["筋書き", "帯", "n発", "的中発", "見立て発", "実際発", "回収発", "n検", "的中検", "見立て検", "実際検", "回収検", "合成中央"]
t = pd.DataFrame([r[:-2] for r in rows], columns=cols)
t["月あたり"] = ((t.n発 + t.n検) / 12).round(0)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 40)
print(t.round(3).to_string(index=False))
ok = lambda a: int(((a.回収発 >= 1) & (a.回収検 >= 1) & (a.的中発 >= 15) & (a.的中検 >= 10)).sum())
print(f"\n{len(t)}区分のうち 両期間とも回収1.0以上（的中 発見15・検証10以上）: {ok(t)}、両期間0.9以上: {int(((t.回収発>=.9)&(t.回収検>=.9)).sum())}")
feat = [c for c in df.columns if c[:2] in ("et", "er", "sc", "sx", "ms", "mo", "gr", "nw", "cw", "cs") and c[2:].isdigit()] + ["wind", "wdir", "wave", "waku"]
rng = np.random.default_rng(0); c1 = []; c9 = []
for it in range(NPERM):
    perm = np.arange(len(df))
    for msk in (disc, ~disc):
        idx = np.where(msk)[0]; perm[idx] = rng.permutation(idx)
    d2 = df.copy(); d2[feat] = df[feat].iloc[perm].reset_index(drop=True)
    t2 = pd.DataFrame([r[:-2] for r in evaluate(STF(d2))], columns=cols)
    c1.append(ok(t2)); c9.append(int(((t2.回収発 >= .9) & (t2.回収検 >= .9)).sum()))
print(f"シャッフル{NPERM}回: 両方1.0以上 平均{np.mean(c1):.2f}（最大{max(c1)}）、両方0.9以上 平均{np.mean(c9):.2f}（最大{max(c9)}）")
print("\n== 月ごとの回収（帯=全部）==")
mon = df.hd.str[:6]
for r in rows:
    if r[1] != "全部": continue
    g = pd.DataFrame({"m": mon[r[-1]], "r": r[-2][r[-1]]}).groupby("m").r.agg(["size", "mean"])
    print(r[0], " ".join(f"{m[2:]}:{v:.2f}({int(n)})" for m, n, v in zip(g.index, g["size"], g["mean"])), f" 勝ち月{int((g['mean']>1).sum())}/{len(g)}")
