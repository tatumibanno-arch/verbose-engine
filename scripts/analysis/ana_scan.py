"""穴狙いパターン探し 1回目（2026-11-14）。ana_prep.py の表を使う。

    python scripts/analysis/ana_scan.py IN.parquet [並べ替え回数=20]

やること: レース前に分かる情報の条件（約120通り）× 形（頭6通り＋ i-j-全 30通り）× オッズ帯2通り（全部／合成15〜60倍）
ごとに、締切時オッズの見立て確率・実際の的中率・合成買いの回収を、発見用（2025-11〜2026-04）と検証用（2026-05〜10）で出す。
「発見で回収1.0以上（的中20本以上）→ 検証でも1.0以上（的中10本以上）」を再現とみなす。
偶然で通る数の目安は、条件の行をレース間でシャッフルして同じ手順を回して数える（条件と結果の関係だけを壊す）。
注意: 回収は締切時オッズで配分した合成買い＝実際に買う時点より後の情報を使っており、実際より甘い。
"""
import sys, itertools
import numpy as np, pandas as pd

SPLIT = "20260501"
MIN_HIT_D, MIN_HIT_V = 20, 10
BAND = (15, 60)
df = pd.read_parquet(sys.argv[1]).reset_index(drop=True)
NPERM = int(sys.argv[2]) if len(sys.argv) > 2 else 20
disc = (df.hd < SPLIT).values
shapes = [f"H{i}" for i in range(1, 7)] + [f"P{i}{j}" for i, j in itertools.permutations(range(1, 7), 2)]


def conditions(d):
    """レース前に分かる情報だけで作る条件。名前→真偽の配列。"""
    C = {}
    g = lambda n, k: d[f"{n}{k}"].values
    for k in range(1, 6):  # k が遅い → 外隣 k+1 が攻めやすい
        dm = g("ms", k) - g("ms", k + 1); dx = g("sx", k) - g("sx", k + 1)
        C[f"今節ST {k}が{k+1}より.03遅い"] = dm >= .03
        C[f"今節ST {k}が{k+1}より.05遅い"] = dm >= .05
        C[f"スタ展 {k}が{k+1}より.05遅い"] = dx >= .05
        C[f"スタ展 {k}が{k+1}より.10遅い"] = dx >= .10
        C[f"今節.02かつスタ展.05 {k}が{k+1}より遅い"] = (dm >= .02) & (dx >= .05)
    et = np.column_stack([g("et", k) for k in range(1, 7)])
    srt = np.sort(et, axis=1)
    for k in range(1, 7):
        C[f"展示1位が{k}"] = g("er", k) == 1
        C[f"展示で{k}が.05以上抜けて1位"] = (et[:, k - 1] == srt[:, 0]) & (srt[:, 1] - srt[:, 0] >= .05)
    C["①の展示が5〜6位"] = g("er", 1) >= 5
    C["①のモーター25%以下"] = g("mo", 1) <= 25
    for k in range(2, 7):
        C[f"{k}のモーター45%以上で①は32%以下"] = (g("mo", k) >= 45) & (g("mo", 1) <= 32)
        C[f"①がB級で{k}がA1"] = (g("gr", 1) <= 2) & (g("gr", k) == 4)
    for k, th in zip(range(2, 7), (20, 20, 18, 10, 6)):
        C[f"{k}のその枠の1着率{th}%以上"] = g("cw", k) >= th
    C["①の1コース1着率40%未満"] = g("cw", 1) < 40
    w = d.wind.astype(float).values; wd = d.wdir.astype(float).values
    C["風5m以上"] = w >= 5
    for a, b in ((1, 4), (5, 8), (9, 12), (13, 16)):
        C[f"風4m以上で風向{a}〜{b}"] = (w >= 4) & (wd >= a) & (wd <= b)
    C["波5cm以上"] = d.wave.values >= 5
    C["風1m以下"] = w <= 1
    C["枠なりでない"] = ~d.waku.astype(bool).values
    for k in (4, 5, 6):
        C[f"{k}がスタ展で内に動いた"] = g("sc", k) < k
    fav = d.p_H1.values >= .5
    for n in list(C):
        C[n + " ＋①人気(市場5割以上)"] = C[n] & fav
    return {n: np.nan_to_num(np.asarray(v, dtype=float)).astype(bool) for n, v in C.items()}


R = {s: df["r_" + s].values for s in shapes}
PM = {s: df["p_" + s].values for s in shapes}
CO = {s: 1 / df["q_" + s].values for s in shapes}
INB = {s: (CO[s] >= BAND[0]) & (CO[s] <= BAND[1]) for s in shapes}


def scan(C, full=True):
    rows = []
    for cn, m in C.items():
        for s in shapes:
            for bn, bm in (("全部", None), ("合成15〜60倍", INB[s])):
                mm = m if bm is None else (m & bm)
                a = mm & disc; b = mm & ~disc
                ra, rb = R[s][a], R[s][b]
                ha, hb = int((ra > 0).sum()), int((rb > 0).sum())
                if ha < MIN_HIT_D:
                    continue
                roa = ra.mean(); rob = rb.mean() if len(rb) else np.nan
                if full:
                    rows.append((cn, s, bn, len(ra), ha, PM[s][a].mean(), ha / len(ra), roa, len(rb), hb,
                                 PM[s][b].mean() if len(rb) else np.nan, hb / max(len(rb), 1), rob, CO[s][mm].mean() if False else np.median(CO[s][mm])))
                else:
                    rows.append((roa, hb, rob))
    return rows


C = conditions(df)
rows = scan(C)
t = pd.DataFrame(rows, columns=["条件", "形", "帯", "n発", "的中発", "見立て発", "実際発", "回収発", "n検", "的中検", "見立て検", "実際検", "回収検", "合成中央"])
print(f"条件 {len(C)} 通り × 形 {len(shapes)} × 帯 2 = {len(C)*len(shapes)*2} 区分。うち発見用で的中{MIN_HIT_D}本以上: {len(t)}")
sel = t[t.回収発 >= 1.0]
rep = sel[(sel.回収検 >= 1.0) & (sel.的中検 >= MIN_HIT_V)]
print(f"発見で回収1.0以上: {len(sel)} → 検証でも1.0以上（的中{MIN_HIT_V}本以上）: {len(rep)}")
print(f"参考: 発見0.9以上 {int((t.回収発>=.9).sum())} → 検証も0.9以上 {int(((t.回収発>=.9)&(t.回収検>=.9)).sum())}")
print("全区分の回収の中央値 発見/検証:", round(t.回収発.median(), 3), round(t.回収検.median(), 3),
      " 発見と検証の相関:", round(np.corrcoef(t.回収発, t.回収検.fillna(0))[0, 1], 3))

# 偶然で通る数: 条件の行をレース間（同じ期間の中）でシャッフル
rng = np.random.default_rng(0)
feat = [c for c in df.columns if c[:2] in ("et", "er", "sc", "sx", "ms", "mo", "gr", "nw", "cw", "cs") and c[2:].isdigit()] + ["wind", "wdir", "wave", "waku"]
cnt = []
for it in range(NPERM):
    perm = np.arange(len(df))
    for msk in (disc, ~disc):
        idx = np.where(msk)[0]; perm[idx] = rng.permutation(idx)
    d2 = df.copy(); d2[feat] = df[feat].iloc[perm].reset_index(drop=True)
    r2 = scan(conditions(d2), full=False)
    a = np.array(r2, dtype=float)
    s1 = a[:, 0] >= 1.0
    cnt.append((int(s1.sum()), int((s1 & (a[:, 2] >= 1.0) & (a[:, 1] >= MIN_HIT_V)).sum())))
cnt = np.array(cnt)
print(f"シャッフル{NPERM}回: 発見で1.0以上 平均{cnt[:,0].mean():.1f}（{cnt[:,0].min()}〜{cnt[:,0].max()}）、両方通過 平均{cnt[:,1].mean():.1f}（{cnt[:,1].min()}〜{cnt[:,1].max()}）")

pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500); pd.set_option("display.max_colwidth", 60)
fmt = lambda x: x.round(3)
print("\n== 再現した区分（発見・検証とも回収1.0以上）=="); print(fmt(rep.sort_values("回収検", ascending=False)).to_string(index=False))
print("\n== 発見で回収1.0以上だが検証で崩れた区分（上位40）=="); print(fmt(sel[~sel.index.isin(rep.index)].sort_values("回収発", ascending=False).head(40)).to_string(index=False))
t["ae発"] = t.実際発 / t.見立て発; t["ae検"] = t.実際検 / t.見立て検
big = t[(t.帯 == "全部") & (t.的中発 >= 60) & (t.的中検 >= 40)]
print("\n== 見立てより多く来る形（全部の帯、的中 発見60本・検証40本以上、実際÷見立てが両期間とも1.10以上）==")
print(fmt(big[(big.ae発 >= 1.10) & (big.ae検 >= 1.10)].sort_values("ae検", ascending=False)).to_string(index=False))
print("\n== 見立てより来ない形（両期間とも0.85以下）==")
print(fmt(big[(big.ae発 <= .85) & (big.ae検 <= .85)].sort_values("ae検")).to_string(index=False))
t.to_csv(sys.argv[1].replace(".parquet", "_scan.csv"), index=False)
