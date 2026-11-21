"""1点ごとの「データから見た確率」とオッズを比べ、割安な点を買い続けたらどうなるか（2026-11-21、運用者の案）。
例: ①が堅いレースで 1-2-3 が5倍、同じくらい来る 1-4-2 が20倍、のような「並びの値付けのずれ」を拾えるか。

    python scripts/analysis/ana_combo_model.py IN.parquet

全120点×全レースを1行ずつにして、その点が当たる確率を学習（発見=2025-11〜2026-04）。材料は、市場（その点のオッズ、頭・2着・3着の艇の人気、
2着までの売れ方、並びの売れ方の素直さ）と、3艇それぞれの直前情報・成績（展示順位、スタ展ST、今節ST、枠別1着率、全国勝率、モーター、級）。
検証（2026-05〜10）で「見立て÷市場」の帯ごとに、各点100円で買った回収を見る。締切時オッズ＝実際より甘い。
"""
import sys, itertools
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
df = pd.read_parquet(sys.argv[1]).reset_index(drop=True); N = len(df); disc = (df.hd < "20260501").values
ALL = [x for x in itertools.permutations(range(1, 7), 3)]; names = ["".join(map(str, x)) for x in ALL]
Q = df[["t_" + n for n in names]].fillna(0).values.astype(np.float32); S = Q.sum(1, keepdims=True)
A = np.array(ALL) - 1                                   # 120×3
w = df.cb.str.replace("-", "").map({n: i for i, n in enumerate(names)}).values; pay = (df.payout.values / 100).astype(np.float32)
pH = np.stack([Q[:, A[:, 0] == k].sum(1) for k in range(6)], 1) / S       # 頭シェア N×6
pIn3 = np.stack([Q[:, (A == k).any(1)].sum(1) for k in range(6)], 1) / S  # 3着以内シェア
pair = {}
for a in range(6):
    for b in range(6):
        if a != b: pair[(a, b)] = Q[:, (A[:, 0] == a) & (A[:, 1] == b)].sum(1) / S[:, 0]
per_boat = {n: df[[f"{n}{k}" for k in range(1, 7)]].values.astype(np.float32) for n in ("er", "sx", "ms", "cw", "nw", "mo", "gr")}
cols = {}
mp = (Q / S); cols["lmp"] = np.log(np.clip(mp, 1e-6, 1))
for j, nm in enumerate(("a", "b", "c")):
    cols["lane_" + nm] = np.tile(A[:, j][None, :], (N, 1)).astype(np.float32)
    cols["pH_" + nm] = pH[:, A[:, j]]; cols["in3_" + nm] = pIn3[:, A[:, j]]
    for n, v in per_boat.items(): cols[f"{n}_{nm}"] = v[:, A[:, j]]
pab = np.stack([pair[(a, b)] for a, b, c in A], 1)
cols["p_ab"] = pab; cols["b_given_a"] = pab / np.clip(cols["pH_a"], 1e-6, 1); cols["c_given_ab"] = mp / np.clip(pab, 1e-6, 1)
hv = cols["pH_a"] * cols["pH_b"] / np.clip(1 - cols["pH_a"], 1e-3, 1) * cols["pH_c"] / np.clip(1 - cols["pH_a"] - cols["pH_b"], 1e-3, 1)
cols["lharv"] = np.log(np.clip(hv, 1e-7, 1)); cols["dev"] = cols["lmp"] - cols["lharv"]     # 並びの売れ方が頭シェアから見てどれだけ厚いか
F = list(cols); X = np.stack([cols[f].ravel() for f in F], 1).astype(np.float32)
y = np.zeros((N, 120), np.int8); y[np.arange(N), w] = 1; y = y.ravel()
dm = np.repeat(disc, 120); okm = np.repeat(Q.sum(1) > 0, 120) & (mp.ravel() > 0)
rng = np.random.default_rng(0); tr = np.where(dm & okm)[0]; tr = rng.choice(tr, min(len(tr), 2_000_000), replace=False)
m = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.06, max_leaf_nodes=63, min_samples_leaf=200, l2_regularization=1.0, random_state=0).fit(X[tr], y[tr])
va = np.where(~dm & okm)[0]
p = m.predict_proba(X[va])[:, 1]; race = va // 120
p = p / np.bincount(race, weights=p, minlength=N)[race]                  # レース内で合計1に
v = pd.DataFrame({"race": race, "combo": va % 120, "p": p, "mp": mp.ravel()[va], "odds": 1 / Q.ravel()[va], "y": y[va]})
v["ret"] = v.y * pay[v.race.values]; v["ratio"] = v.p / v.mp; v["head"] = A[v.combo.values, 0] + 1
ll = lambda q: -np.mean(np.log(np.clip(q[v.y == 1], 1e-9, 1)))
print(f"検証期間の当てやすさ（小さいほど良い）: 見立て {ll(v.p.values):.4f} / 市場 {ll(v.mp.values):.4f}")
pd.set_option("display.width", 220)
v["帯"] = pd.cut(v.ratio, [0, .7, .85, .95, 1.05, 1.15, 1.3, 1.5, 2, 99], labels=["〜0.7", "0.7〜0.85", "0.85〜0.95", "0.95〜1.05", "1.05〜1.15", "1.15〜1.3", "1.3〜1.5", "1.5〜2", "2〜"])
agg = dict(点数=("y", "size"), 見立て=("p", "mean"), 市場=("mp", "mean"), 実際=("y", "mean"), 回収=("ret", "mean"))
print("\n== 全部の点: 見立て÷市場 の帯ごと =="); print(v.groupby("帯", observed=True).agg(**agg).round(4).to_string())
for lab, x in (("オッズ5〜20倍", v[(v.odds >= 5) & (v.odds < 20)]), ("オッズ15〜60倍", v[(v.odds >= 15) & (v.odds < 60)]), ("頭①でオッズ15〜60倍", v[(v.head == 1) & (v.odds >= 15) & (v.odds < 60)]), ("オッズ60〜300倍", v[(v.odds >= 60) & (v.odds < 300)])):
    print(f"\n== {lab} =="); print(x.groupby("帯", observed=True).agg(**agg).round(4).to_string())
mon = df.hd.str[:6].values[v.race.values]; v["m"] = mon
for th in (1.15, 1.3):
    x = v[(v.ratio >= th) & (v.odds >= 15) & (v.odds < 60)]
    g = x.groupby("m").ret.agg(["size", "mean"])
    print(f"\n割安（比{th}以上）でオッズ15〜60倍の点を全部買う: {len(x)}点 的中{int(x.y.sum())} 回収{x.ret.mean():.3f}  月別 " + " ".join(f"{i[4:]}月:{r['mean']:.2f}({int(r['size'])})" for i, r in g.iterrows()))
imp = sorted(zip(F, getattr(m, "feature_importances_", [0] * len(F))), key=lambda t: -t[1])[:8] if hasattr(m, "feature_importances_") else []
