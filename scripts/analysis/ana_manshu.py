"""万舟（3連単の払戻1万円以上＝100倍以上）が出やすいレースと、取り方の当たりやすさ・回収（2026-11-21）。

    python scripts/analysis/ana_manshu.py IN.parquet      （IN は ana_prep.py の表。2025-11〜2026-10）

締切時オッズで買い目を選ぶ計算＝実際より甘い。発見=2025-11〜2026-04、検証=2026-05〜10。
"""
import sys, itertools
import numpy as np, pandas as pd
sys.path.insert(0, "scripts"); from common import VENUES
df = pd.read_parquet(sys.argv[1]).reset_index(drop=True); N = len(df)
ALL = ["".join(map(str, x)) for x in itertools.permutations(range(1, 7), 3)]; CI = {x: i for i, x in enumerate(ALL)}
Q = df[["t_" + x for x in ALL]].fillna(0).values
with np.errstate(divide="ignore"): O = np.where(Q > 0, 1 / Q, np.inf)
wcb = df.cb.str.replace("-", "").map(CI).values; pay = df.payout.values / 100
man = pay >= 100; per = np.where(df.hd < "20260501", "発見", "検証")
pd.set_option("display.width", 220)
print(f"レース {N}。万舟率 全体 {man.mean():.3f}（発見 {man[per=='発見'].mean():.3f} / 検証 {man[per=='検証'].mean():.3f}）。払戻の中央値 {np.median(pay)*100:.0f}円")
pm = (Q * (O >= 100)).sum(1) / Q.sum(1)   # 市場が見ている「100倍以上の目が来る確率」
df["man"] = man; df["pm"] = pm; df["per"] = per
def tab(key, lab):
    g = df.groupby([key, "per"], observed=True).man.agg(["size", "mean"]).unstack("per"); g.columns = [f"{a}_{b}" for a, b in g.columns]
    print(f"\n== {lab} =="); print(g.round(3).to_string())
df["①頭シェア(締切時)"] = pd.cut(df.p_H1, [0, .2, .3, .4, .5, .6, .7, .8, 1]); tab("①頭シェア(締切時)", "締切時の①頭シェアごとの万舟率")
df["①の1コース1着率(前夜)"] = pd.cut(df.cw1, [-1, 20, 30, 40, 50, 60, 70, 101]); tab("①の1コース1着率(前夜)", "前夜に分かる: ①の1コース期別1着率ごとの万舟率")
df["①の級"] = df.gr1.map({4: "A1", 3: "A2", 2: "B1", 1: "B2"}); tab("①の級", "前夜に分かる: ①の級ごとの万舟率")
df["A1の数"] = sum((df[f"gr{k}"] == 4).astype(int) for k in range(2, 7)).clip(upper=3); tab("A1の数", "前夜: ②〜⑥のA1の人数")
df["風"] = pd.cut(df.wind.astype(float), [-1, 1, 3, 5, 7, 99]); tab("風", "当日: 風速(m)")
g = df.groupby("jcd").man.agg(["size", "mean"]).sort_values("mean"); g.index = [VENUES[f"{i:02d}"] for i in g.index]; print("\n== 場ごとの万舟率 =="); print(g.round(3).T.to_string())
df["前夜の荒れ度"] = ((df.cw1.fillna(45) < 40).astype(int) + (df.gr1 <= 2).astype(int) + (df["A1の数"] >= 2).astype(int)); tab("前夜の荒れ度", "前夜の荒れ度（①の1コース1着率40%未満・①がB級・外にA1が2人以上、当てはまる数）")

# ---- 取り方: 100倍以上の目のうち、安い順に K 点（締切時オッズ）----
print("\n== 取り方: 100倍以上の目を安い順に K 点、各100円（締切時オッズで選ぶ＝甘い）==")
Om = np.where(O >= 100, O, np.inf); order = np.argsort(Om, axis=1)
rank_w = np.array([np.where(order[i] == wcb[i])[0][0] for i in range(N)]); wman = O[np.arange(N), wcb] >= 100
rows = []
for lab, cond in (("全レース", np.ones(N, bool)), ("①頭シェア5割以下", df.p_H1.values <= .5), ("①頭シェア3割以下", df.p_H1.values <= .3), ("前夜の荒れ度2以上", df["前夜の荒れ度"].values >= 2), ("前夜の荒れ度3", df["前夜の荒れ度"].values >= 3)):
    for K in (5, 10, 20, 40):
        hit = cond & wman & (rank_w < K)
        for p in ("発見", "検証"):
            m = cond & (per == p); h = hit & (per == p)
            rows.append((lab, K, p, int(m.sum()), h.sum() / m.sum(), pay[h].sum() / (m.sum() * K), np.median(pay[h]) * 100 if h.any() else 0))
t = pd.DataFrame(rows, columns=["レース", "点数", "期間", "レース数", "万舟的中率", "回収", "当たりの払戻中央値"])
print(t.pivot_table(index=["レース", "点数"], columns="期間", values=["レース数", "万舟的中率", "回収", "当たりの払戻中央値"], sort=False).round(3).to_string())
# ---- 万舟の中身 ----
print("\n== 万舟になったレースの頭（1着艇）の内訳 ==")
print((pd.Series(df.wh[man]).value_counts(normalize=True).sort_index() * 100).round(1).to_string())
print("万舟のうち①が3着以内:", round(float(np.mean([("1" in ALL[i]) for i in wcb[man]])), 3), " 払戻1万〜2万円の割合:", round(float(((pay >= 100) & (pay < 200)).sum() / man.sum()), 3))
