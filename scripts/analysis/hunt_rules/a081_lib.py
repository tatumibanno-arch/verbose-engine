# a081: 3連複の1点買い用の機械学習（艇ごとの1着/3着内モデル→Plackett-Luceで3連複確率）
import numpy as np, pandas as pd, itertools
BF = ["gr","nw","n2","lw","l2","mo","bo","age","wt","sex","cw","ct","cst","cn","ast","abil","ms","mf","mn","mm","mmn"]
TRIOS = list(itertools.combinations(range(1,7),3))
def long_feats(df, use_ex=False):
    bf = BF + (["et","er"] if use_ex else [])
    n = len(df); rows = []
    race = df[["jcd","rno","distance","day_no"]].to_numpy(float)
    rel_cols = ["nw","abil","cw","ct","mo","cst","n2"]+(["et"] if use_ex else [])
    mats = {c: df[[f"{c}{k}" for k in range(1,7)]].to_numpy(float) for c in bf}
    for k in range(6):
        X = {c: mats[c][:,k] for c in bf}
        X["lane"] = np.full(n, k+1.0)
        for c in rel_cols:
            m = mats[c]; X[f"{c}_d"] = m[:,k]-np.nanmean(m,axis=1)
            X[f"{c}_r"] = (m > m[:,[k]]).sum(1) if c not in ("cst","et") else (m < m[:,[k]]).sum(1)
        X["abil_in"] = mats["abil"][:,0]; X["abil_max_o"] = np.nanmax(np.delete(mats["abil"],k,axis=1),axis=1)
        X["cw1"] = mats["cw"][:,0]; X["cst1"] = mats["cst"][:,0]
        for i,c in enumerate(["jcd","rno","distance","day_no"]): X[c]=race[:,i]
        rows.append(pd.DataFrame(X))
    return rows  # list of 6 frames aligned with df rows
def stack(rows): return pd.concat(rows, ignore_index=True)
def trio_probs(pw):
    # pw: n x 6 win probabilities (normalized). Plackett-Luce prob of each unordered trio being top3
    pw = pw/pw.sum(1,keepdims=True); n=len(pw); out=np.zeros((n,20))
    for t,(a,b,c) in enumerate(TRIOS):
        s=0
        for x,y,z in itertools.permutations((a-1,b-1,c-1)):
            px=pw[:,x]; py=pw[:,y]/(1-px); pz=pw[:,z]/(1-px-pw[:,y]); s=s+px*py*pz
        out[:,t]=s
    return out
