"""「舟足特化型」の買い方（上位2艇を軸・最下位を切る・18点マルチ・20倍未満は消し・均等買い）を、手元の数字で代用して確かめる（2026-11-21）。
記事の舟足評価は予想家の目。ここでは代用として (a) 展示タイム順 (b) モーター2連率順 (c) 両方の順位の合計 を使う。締切時オッズで20倍未満を消す＝実際（5分前）より甘い。

    python scripts/analysis/ana_funaashi.py IN.parquet
"""
import sys, itertools
import numpy as np, pandas as pd
df = pd.read_parquet(sys.argv[1]).reset_index(drop=True); N = len(df); per = np.where(df.hd < "20260501", "発見", "検証")
ALL = ["".join(map(str, x)) for x in itertools.permutations(range(1, 7), 3)]; CI = {x: i for i, x in enumerate(ALL)}
Q = df[["t_" + x for x in ALL]].fillna(0).values
with np.errstate(divide="ignore"): O = np.where(Q > 0, 1 / Q, np.inf)
wcb = df.cb.str.replace("-", "").map(CI).values; pay = df.payout.values / 100
A = np.array([[int(ch) for ch in x] for x in ALL])  # 120×3
et = df[[f"et{k}" for k in range(1, 7)]].values; mo = df[[f"mo{k}" for k in range(1, 7)]].values
rk = lambda v, asc: pd.DataFrame(v if asc else -v).rank(axis=1, method="first").values
sc = {"展示タイム順": rk(et, True), "モーター2連率順": rk(mo, False)}; sc["展示＋モーター"] = rk(sc["展示タイム順"] + sc["モーター2連率順"] + 0.01 * sc["展示タイム順"], True)
rng = np.random.default_rng(0); sc["でたらめ（比較用）"] = rk(rng.random((N, 6)), True)
ok = ~np.isnan(et).any(1) & ~np.isnan(mo).any(1)
rows = []
for name, r in sc.items():
    top = r <= 2; worst = r == 6                                  # N×6
    ntop = top[:, A - 1].sum(2) if False else np.stack([top[np.arange(N), A[j, 0] - 1].astype(int) + top[np.arange(N), A[j, 1] - 1] + top[np.arange(N), A[j, 2] - 1] for j in range(120)], 1)
    hasw = np.stack([worst[np.arange(N), A[j, 0] - 1] | worst[np.arange(N), A[j, 1] - 1] | worst[np.arange(N), A[j, 2] - 1] for j in range(120)], 1)
    M18 = (ntop == 2) & ~hasw                                     # 上位2艇を両方含み、最下位を含まない＝18点
    for cut in (0, 20, 40):
        M = M18 & (O >= cut); npts = M.sum(1); hit = M[np.arange(N), wcb]
        for p in ("発見", "検証"):
            x = ok & (per == p)
            rows.append((name, f"{cut}倍未満を消す" if cut else "18点全部", p, int(x.sum()), npts[x].mean(), hit[x].mean(), pay[x & hit].sum() / npts[x].sum(), np.median(pay[x & hit]) * 100))
t = pd.DataFrame(rows, columns=["評価の代用", "買い方", "期間", "レース", "平均点数", "的中率", "回収", "当たりの中央値(円)"])
pd.set_option("display.width", 220); print(t.round(3).to_string(index=False))
