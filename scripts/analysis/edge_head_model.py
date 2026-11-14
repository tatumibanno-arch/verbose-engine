"""「市場＋直前情報」で1着を当て直すモデル（2026-11-14）。市場に勝てるレースを厳選できるかの検証。
各艇の特徴: 市場の頭シェア（締切時3連単オッズから）、今の3連単モデルの頭確率、展示タイム（レース平均との差・順位）、スタ展ST、進入コース、チルト、体重、部品交換、風・波、場。
目的: その艇が1着か。学習＝2025-11〜2026-06、検証＝2026-07〜2026-11（学習に使わない）。
賭け方（評価）: 「艇iの頭の20点を 1/オッズ に比例して買う」＝どれが来ても同じ払戻 R=1/Σ(1/odds)。モデル確率×R（期待値）が高いものだけ買ったときの回収を帯ごとに見る。
注意: 締切時オッズを特徴と払戻の両方に使っているので、実際（締切10分前のオッズで判断）より甘い。"""
import sqlite3, glob, itertools, json, numpy as np, pandas as pd, lightgbm as lgb
W="/tmp/claude-0/-home-user-verbose-engine/968008a0-6ecf-5641-a3ff-006912b41722/scratchpad/trifecta/pred"
c=sqlite3.connect("data/boatrace.db"); F="20251101"
def key(d): d["jcd"]=d.jcd.astype(int); d["hd"]=d.hd.astype(str); return d
o=key(pd.read_sql(f"select hd,jcd,rno,combination as cb,odds from odds where ticket_type='sanrensho' and odds>0 and hd>='{F}'",c))
o["lane"]=o.cb.str[0].astype(int); o["q"]=1/o.odds
g=o.groupby(["hd","jcd","rno","lane"]).q.sum().rename("qs").reset_index(); g["tot"]=g.groupby(["hd","jcd","rno"]).qs.transform("sum")
g["m"]=g.qs/g.tot; g["R"]=1/g.qs
b=key(pd.read_sql(f"select hd,jcd,rno,lane,exhibition_time as et,tilt,start_course as sc,start_st as st,data from beforeinfo where hd>='{F}'",c))
dj=b.data.map(json.loads); b["weight"]=dj.map(lambda x:x.get("weight")); b["parts"]=dj.map(lambda x:len(x.get("parts_changed") or [])); b["stF"]=dj.map(lambda x:str(x.get("start_st_raw") or "").startswith("F")).astype(int)
b["wind"]=dj.map(lambda x:x.get("wind_speed")); b["wdir"]=dj.map(lambda x:x.get("wind_direction")); b["wave"]=dj.map(lambda x:x.get("wave_height")); b=b.drop(columns="data")
k=["hd","jcd","rno"]
b["et_d"]=b.et-b.groupby(k).et.transform("mean"); b["et_r"]=b.groupby(k).et.rank(method="min"); b["st_d"]=b.st-b.groupby(k).st.transform("mean"); b["st_r"]=b.groupby(k).st.rank(method="min")
b["moved"]=(b.sc!=b.lane).astype(float); b["any_moved"]=b.groupby(k).moved.transform("max")
r=key(pd.read_sql(f"select hd,jcd,rno,lane,position from results where hd>='{F}'",c)); r["win"]=(pd.to_numeric(r.position,errors="coerce")==1).astype(int)
PERMS=list(itertools.permutations(range(1,7),3)); rows=[]
for f in sorted(glob.glob(f"{W}/pred_*.npz")):
    z=np.load(f); hd=z["hd"].astype(str)
    if max(hd)<F: continue
    H=np.zeros((len(hd),6))
    for j,p in enumerate(PERMS): H[:,p[0]-1]+=z["P"][:,j]
    for l in range(6): rows.append(pd.DataFrame(dict(hd=hd,jcd=z["jcd"].astype(int),rno=z["rno"].astype(int),lane=l+1,pm=H[:,l])))
pm=pd.concat(rows)
d=g.merge(b,on=k+["lane"]).merge(r[k+["lane","win"]],on=k+["lane"]).merge(pm,on=k+["lane"],how="left")
d=d[d.groupby(k).win.transform("sum")==1]
d["lm"]=np.log(d.m.clip(1e-4)); d["lpm"]=np.log(d.pm.clip(1e-4)); d["gap"]=d.lpm-d.lm
feats=["lane","lm","lpm","gap","et_d","et_r","st","st_d","st_r","stF","sc","moved","any_moved","tilt","weight","parts","wind","wdir","wave","jcd","rno"]
tr=d[d.hd<"20260701"]; va=d[d.hd>="20260701"].copy()
print("学習",tr.groupby(k).ngroups,"R / 検証",va.groupby(k).ngroups,"R")
def fit(fs,seed=0):
    m=lgb.LGBMClassifier(n_estimators=400,learning_rate=0.03,num_leaves=31,min_child_samples=200,subsample=0.8,subsample_freq=1,colsample_bytree=0.8,reg_lambda=5,verbose=-1,random_state=seed)
    m.fit(tr[fs],tr.win); return m
def ll(p,y): p=np.clip(p,1e-6,1-1e-6); return -np.mean(y*np.log(p)+(1-y)*np.log(1-p))
va["p_mkt"]=va.m
print(f"検証の対数損失: 市場だけ {ll(va.m,va.win):.5f}")
for name,fs in [("市場のみ(補正)",["lane","lm","jcd"]),("市場＋今のモデル",["lane","lm","lpm","gap","jcd","rno"]),("市場＋直前情報",[x for x in feats if x not in("lpm","gap")]),("全部",feats),("市場なし（今のモデル＋直前情報）",[x for x in feats if x not in("lm","gap")])]:
    m=fit(fs); p=m.predict_proba(va[fs])[:,1]; va["p_"+name]=p; s=va.groupby(k)["p_"+name].transform("sum"); va["p_"+name]=p/s
    print(f"  {name}: {ll(va['p_'+name],va.win):.5f}")
    if name=="全部": print("   重要度",sorted(zip(m.feature_importances_,fs),reverse=True)[:10])
va["m"]=va.hd.str[:6]
def roi(sel): return pd.Series(dict(n=len(sel),的中率=round(sel.win.mean()*100,1),平均倍率=round(sel.R[sel.win==1].mean(),2) if sel.win.any() else np.nan,回収=round((sel.win*sel.R).mean(),3)))
for name in ["市場＋直前情報","全部","市場なし（今のモデル＋直前情報）"]:
    va["ev"]=va["p_"+name]*va.R; va["evb"]=pd.cut(va.ev,[0,.7,.8,.9,1.0,1.1,1.2,1.4,9])
    print(f"== {name}: 期待値（モデル確率×払戻倍率）の帯ごと。検証期間 =="); print(va.groupby("evb",observed=True).apply(roi).to_string())
    s=va[va.ev>=1.1]; print("期待値1.1以上を月ごと"); print(s.groupby("m").apply(roi).to_string())
    print("期待値1.1以上を倍率の帯ごと"); print(s.groupby(pd.cut(s.R,[0,2,4,8,20,1e9]),observed=True).apply(roi).to_string())
    print("期待値1.1以上を艇ごと"); print(s.groupby("lane").apply(roi).to_string())
va[k+["lane","m","R","win"]+[x for x in va.columns if x.startswith("p_")]].to_csv("/tmp/claude-0/-home-user-verbose-engine/968008a0-6ecf-5641-a3ff-006912b41722/scratchpad/edge_va.csv",index=False)
