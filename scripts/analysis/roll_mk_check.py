"""転がしの夜の検証（2026-11-13）: オッズから見た的中率で候補を絞ると1段の成績がどう変わるか。
締切時オッズで組むので実際より甘い（実際は締切10分前のオッズで組み、当たり目は締切までに下がりやすい）。
全レース対象（モデル①1着の範囲・モデル的中40%の条件を外す）。配分は余裕なし（今のオッズで2倍・ガミなし・20点まで・残高1,000円）。
区分: オッズから見た的中率の帯 × モデル条件（なし／①1着0.6〜0.8かつモデル的中40%以上）× 進入（枠なり／動いた）。
使い方: python scripts/analysis/roll_mk_check.py [FROM=20260501] [SPLIT=20260801]"""
import sys, glob, math, itertools, sqlite3, numpy as np, pandas as pd
W="/tmp/claude-0/-home-user-verbose-engine/968008a0-6ecf-5641-a3ff-006912b41722/scratchpad/trifecta/pred"
PERMS=["-".join(map(str,p)) for p in itertools.permutations(range(1,7),3)]
B=1000; T=2.0; MAXP=20
FROM=sys.argv[1] if len(sys.argv)>1 else "20260501"; SPLIT=sys.argv[2] if len(sys.argv)>2 else "20260801"
def plan(probs, odds):
    items=sorted([(c,probs[c],odds[c]) for c in odds if c in probs and odds[c]>0],key=lambda x:-x[1]*x[2])[:40]
    U=B//100; best=[[-1]*(U+1) for _ in range(MAXP+1)]; best[0][0]=0; ch={}
    for c,p,o in items:
        cost=math.ceil(T*B/o/100-1e-9)
        if cost>U: continue
        for k in range(MAXP-1,-1,-1):
            for u in range(U-cost,-1,-1):
                if best[k][u]>=0 and best[k][u]+p>best[k+1][u+cost]:
                    best[k+1][u+cost]=best[k][u]+p; ch[(k+1,u+cost)]=ch.get((k,u),())+((c,cost*100),)
    bk=max(((best[k][u],k,u) for k in range(MAXP+1) for u in range(U+1) if best[k][u]>0),default=None)
    if not bk: return None,0
    sel=list(ch[(bk[1],bk[2])]); left=B-sum(a for _,a in sel); i=0
    while left>=100 and sel:
        sel.sort(key=lambda x:-probs[x[0]]); c,a=sel[i%len(sel)]; sel[i%len(sel)]=(c,a+100); left-=100; i+=1
    return dict(sel),bk[0]
c=sqlite3.connect("data/boatrace.db")
def key(df): df["jcd"]=df.jcd.astype(str).str.zfill(2); df["hd"]=df.hd.astype(str); return df
od=key(pd.read_sql(f"select hd,jcd,rno,combination,odds from odds where ticket_type='sanrensho' and hd>='{FROM}'",c))
OD={k:dict(zip(g.combination,g.odds)) for k,g in od.groupby(["hd","jcd","rno"])}
pay=key(pd.read_sql(f"select hd,jcd,rno,combination,payout from payouts where ticket_type='sanrensho' and hd>='{FROM}'",c))
PAY={(r.hd,r.jcd,r.rno):(r.combination,r.payout) for r in pay.itertuples()}
bi=key(pd.read_sql(f"select hd,jcd,rno,lane,start_course from beforeinfo where hd>='{FROM}'",c))
WAKU={k:bool((g.start_course==g.lane).all()) and len(g)==6 for k,g in bi.dropna(subset=["start_course"]).groupby(["hd","jcd","rno"])}
NIGHT={"07","12","17","19","20","24","01"}  # 蒲郡・住之江・丸亀(※)・下関・若松・大村・桐生
rows=[]
for f in sorted(glob.glob(f"{W}/pred_*.npz")):
    z=np.load(f); hd=z["hd"].astype(str)
    if max(hd)<FROM: continue
    for i in range(len(hd)):
        k=(hd[i],str(z["jcd"][i]).zfill(2),int(z["rno"][i]))
        if k[0]<FROM or k not in OD or k not in PAY: continue
        P=z["P"][i]; probs={PERMS[j]:float(P[j]) for j in range(120)}
        p1=sum(v for cc,v in probs.items() if cc[0]=="1"); res,pv=PAY[k]; oo=OD[k]
        tot=sum(1/v for v in oo.values() if v>0)
        mp={cc:(1/v)/tot for cc,v in oo.items() if v>0}; m1=sum(v for cc,v in mp.items() if cc[0]=="1")
        sel,hit=plan(probs,oo)
        if not sel: continue
        mk=sum(mp.get(cc,0) for cc in sel)
        rows.append(dict(hd=k[0],jcd=k[1],hit=res in sel,mult=sel.get(res,0)*pv/100/B,model=hit,mk=mk,pts=len(sel),p1=p1,m1=m1,waku=WAKU.get(k)))
d=pd.DataFrame(rows); d["per"]=np.where(d.hd<SPLIT,"古い","新しい")
def agg(g): return pd.Series(dict(段=len(g),的中率=round(g.hit.mean()*100,1),当たり倍率=round(g[g.hit].mult.mean(),2) if g.hit.any() else float("nan"),回収=round(g.mult.mean(),3),モデル=round(g.model.mean()*100,1),オッズ的中=round(g.mk.mean()*100,1),点数=round(g.pts.mean(),1)))
print(f"期間 {d.hd.min()}〜{d.hd.max()}（古い＝{SPLIT}より前） 全{len(d)}R")
d["mkb"]=pd.cut(d.mk,[0,.2,.25,.3,.35,.4,.45,1])
print("== A) 全レース（モデルの条件なし）: オッズから見た的中率の帯 × 期間 =="); print(d.groupby(["mkb","per"],observed=True).apply(agg).to_string())
print("== B) 帯だけ（期間まとめ） =="); print(d.groupby("mkb",observed=True).apply(agg).to_string())
print("== C) オッズ的中35%以上 × 進入 × 期間 =="); print(d[d.mk>=.35].dropna(subset=["waku"]).groupby(["waku","per"]).apply(agg).to_string())
print("== D) オッズ的中35%以上 × モデル的中40%以上か × 期間 =="); d["m40"]=d.model>=.4; print(d[d.mk>=.35].groupby(["m40","per"]).apply(agg).to_string())
print("== E) オッズ的中35%以上 × 市場の①1着シェア × 期間 =="); d["m1b"]=pd.cut(d.m1,[0,.5,.65,.8,1]); print(d[d.mk>=.35].groupby(["m1b","per"],observed=True).apply(agg).to_string())
m=d[d.mk>=.35].copy(); m["m"]=m.hd.str[:6]
print("== F) オッズ的中35%以上: 月ごと =="); print(m.groupby("m").apply(agg).to_string())
print(f"1日あたりの段数 {len(m)/d.hd.nunique():.1f}")
