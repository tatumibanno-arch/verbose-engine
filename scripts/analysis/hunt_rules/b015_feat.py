# b015: 1着確率モデル（オッズ以外の列で学習）。学習は 2020-01〜2025-02（B期間より前）
import numpy as np, pandas as pd, lightgbm as lgb
from pathlib import Path
D = Path("/home/user/hunt"); MP = D / "rules" / "b015_model.txt"
PB = ["gr","nw","n2","lw","l2","mo","bo","age","wt","sex","cw","ct","cst","cn","ast","abil","ms","mf","mn","mm","mmn","et","er"]
RACE = ["jcd","rno","day_no","distance","wind","wave"]
def cols(): return ["hd","jcd","rno"] + RACE[2:] + [f"{c}{k}" for c in PB for k in range(1,7)]
def long(df):
    rows = []
    for k in range(1,7):
        x = pd.DataFrame({c: pd.to_numeric(df[f"{c}{k}"], errors="coerce").values for c in PB})
        x["lane"] = k
        for c in RACE: x[c] = pd.to_numeric(df[c], errors="coerce").values
        x["rid"] = np.arange(len(df)); rows.append(x)
    L = pd.concat(rows, ignore_index=True)
    for c in ["nw","abil","et","cw","mo","ast","n2"]:
        g = L.groupby("rid")[c]; L[c+"_d"] = L[c] - g.transform("mean"); L[c+"_r"] = g.rank(ascending=(c in ("et","ast")))
    return L
FE = None
def feats(L): return [c for c in L.columns if c not in ("rid",)]
def train():
    a = pd.read_parquet(D/"A_disc_races.parquet", columns=cols()+["w1"]); a = a[a.hd >= "20200101"]
    b = pd.read_parquet(D/"A_valid_races.parquet", columns=cols()+["w1"]); b = b[b.hd < "20250301"]
    t = pd.concat([a,b], ignore_index=True); L = long(t)
    y = (L.lane.values == pd.to_numeric(t.w1, errors="coerce").values[L.rid.values]).astype(int)
    m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=200, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=0, verbose=-1)
    m.fit(L[feats(L)], y, categorical_feature=["jcd"]); m.booster_.save_model(str(MP)); return m.booster_
def winprob(df):
    bst = lgb.Booster(model_file=str(MP)) if MP.exists() else train()
    L = long(df); p = bst.predict(L[feats(L)])
    P = np.zeros((len(df),6)); P[L.rid.values, L.lane.values-1] = p
    return P / P.sum(1, keepdims=True)
