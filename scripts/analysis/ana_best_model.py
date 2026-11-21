"""「データを全部入れた見立て」でどこまで行けるかの上限確認（2026-11-21）。
市場の頭シェア＋直前情報（展示タイム・スタ展ST・進入）＋今節ST＋枠別1着率＋モーター＋級＋風 で頭の確率を学習（発見期間）し、
検証期間で「見立て÷市場」の帯ごとに、その艇の頭20点を合成買いした回収を見る。締切時オッズ＝実際より甘い。

    python scripts/analysis/ana_best_model.py IN.parquet
"""
import sys
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
df = pd.read_parquet(sys.argv[1]).reset_index(drop=True); disc = (df.hd < "20260501").values
rows = []
for k in range(1, 7):
    d = pd.DataFrame({"race": np.arange(len(df)), "lane": k, "mp": df[f"p_H{k}"], "et": df[f"et{k}"], "er": df[f"er{k}"], "sx": df[f"sx{k}"], "sc": df[f"sc{k}"],
                      "ms": df[f"ms{k}"], "cw": df[f"cw{k}"], "mo": df[f"mo{k}"], "gr": df[f"gr{k}"], "nw": df[f"nw{k}"], "wind": df.wind.astype(float), "wdir": df.wdir.astype(float),
                      "jcd": df.jcd, "y": (df.wh == k).astype(int), "ret": df[f"r_H{k}"], "disc": disc})
    for n in ("et", "sx", "ms", "mo", "nw"):
        allv = df[[f"{n}{j}" for j in range(1, 7)]]; d[n + "_d"] = df[f"{n}{k}"] - allv.mean(axis=1); d[n + "_min"] = df[f"{n}{k}"] - allv.min(axis=1)
    if k < 6: d["sx_out"] = df[f"sx{k}"] - df[f"sx{k+1}"]
    if k > 1: d["sx_in"] = df[f"sx{k}"] - df[f"sx{k-1}"]
    rows.append(d)
b = pd.concat(rows, ignore_index=True); b["lmp"] = np.log(b.mp.clip(1e-4))
F = [c for c in b.columns if c not in ("race", "y", "ret", "disc", "mp")]
m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0, random_state=0).fit(b.loc[b.disc, F], b.loc[b.disc, "y"])
b["p"] = m.predict_proba(b[F])[:, 1]; b["p"] = b.p / b.groupby("race").p.transform("sum")
v = b[~b.disc].copy(); v["ratio"] = v.p / v.mp
ll = lambda p, y: -np.mean(y * np.log(p.clip(1e-9)) + (1 - y) * np.log((1 - p).clip(1e-9)))
print(f"検証期間の当てやすさ（小さいほど良い）: 見立て {ll(v.p, v.y):.5f} / 市場 {ll(v.mp, v.y):.5f}")
v["帯"] = pd.cut(v.ratio, [0, .7, .85, .95, 1.05, 1.15, 1.3, 1.5, 99], labels=["〜0.7", "0.7〜0.85", "0.85〜0.95", "0.95〜1.05", "1.05〜1.15", "1.15〜1.3", "1.3〜1.5", "1.5〜"])
pd.set_option("display.width", 200)
print(v.groupby("帯", observed=True).agg(艇数=("y", "size"), 見立て=("p", "mean"), 市場=("mp", "mean"), 実際=("y", "mean"), 回収=("ret", "mean")).round(3).to_string())
