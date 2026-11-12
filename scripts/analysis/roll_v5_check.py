"""転がし v5 の夜の検証（2026-11-12）。締切時オッズで組むので実際より甘い。
1) 配分の比較: 余裕25%（SHRINK 0.75）と余裕なし（1.0）で、1段の的中率・当たった時の倍率・払戻÷投入。
2) スタ展の進入が枠なりかどうかで分けた成績（v5.1 の裏づけ）。
3) ①1着のとき、2着が各コースになる割合のモデル予測と実際（2着④を低く見ていないか）。
対象: ウォークフォワード予測と締切時3連単オッズがある期間、モデル①1着0.6〜0.8、モデル的中40%以上。
使い方: python scripts/analysis/roll_v5_check.py [FROM=20260501] [SPLIT=20260801]"""
import sys, glob, math, itertools, sqlite3, numpy as np, pandas as pd
W="/tmp/claude-0/-home-user-verbose-engine/968008a0-6ecf-5641-a3ff-006912b41722/scratchpad/trifecta/pred"
PERMS=["-".join(map(str,p)) for p in itertools.permutations(range(1,7),3)]
B=1000; T=2.0; MAXP=20
FROM=sys.argv[1] if len(sys.argv)>1 else "20260501"; SPLIT=sys.argv[2] if len(sys.argv)>2 else "20260801"
def plan(probs, odds, SH):
    items=sorted([(c,probs[c],odds[c]) for c in odds if c in probs and odds[c]>0],key=lambda x:-x[1]*x[2])[:40]
    U=B//100; best=[[-1]*(U+1) for _ in range(MAXP+1)]; best[0][0]=0; ch={}
    for c,p,o in items:
        cost=math.ceil(T*B/(o*SH)/100-1e-9)
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
rows=[]; sec=[]
for f in sorted(glob.glob(f"{W}/pred_*.npz")):
    z=np.load(f); hd=z["hd"].astype(str)
    if max(hd)<FROM: continue
    for i in range(len(hd)):
        k=(hd[i],str(z["jcd"][i]).zfill(2),int(z["rno"][i]))
        if k[0]<FROM or k not in OD or k not in PAY: continue
        P=z["P"][i]; probs={PERMS[j]:float(P[j]) for j in range(120)}
        p1=sum(v for cc,v in probs.items() if cc[0]=="1"); res,pv=PAY[k]
        if len(res)==5 and res[0]=="1":
            sec.append(dict(hd=k[0],act=int(res[2]),waku=WAKU.get(k),**{f"p{s}":sum(v for cc,v in probs.items() if cc.startswith(f"1-{s}"))/max(p1,1e-9) for s in range(2,7)}))
        if not (0.6<=p1<=0.8): continue
        oo=OD[k]; tot=sum(1/v for v in oo.values() if v>0)
        for SH in (0.75,1.0):
            sel,hit=plan(probs,oo,SH)
            if not sel or hit<0.40: continue
            mk=sum((1/oo[cc])/tot for cc in sel if oo.get(cc))
            rows.append(dict(hd=k[0],sh=SH,hit=res in sel,mult=sel.get(res,0)*pv/100/B,model=hit,mk=mk,pts=len(sel),waku=WAKU.get(k)))
d=pd.DataFrame(rows); d["per"]=np.where(d.hd<SPLIT,"古い","新しい")
def agg(g): return pd.Series(dict(段=len(g),的中率=round(g.hit.mean()*100,1),当たり倍率=round(g[g.hit].mult.mean(),2),回収=round(g.mult.mean(),3),モデル=round(g.model.mean()*100,1),オッズ的中=round(g.mk.mean()*100,1),点数=round(g.pts.mean(),1)))
print(f"期間 {d.hd.min()}〜{d.hd.max()}（古い＝{SPLIT}より前）")
print("== 1) 配分の比較 =="); print(d.groupby(["sh","per"]).apply(agg).to_string())
print(d.groupby("sh").apply(agg).to_string())
print("== 2) 進入（スタ展）枠なり / 動いた =="); print(d.dropna(subset=["waku"]).groupby(["sh","waku","per"]).apply(agg).to_string())
print(d.dropna(subset=["waku"]).groupby(["sh","waku"]).apply(agg).to_string())
print("== 余裕なしで、オッズから見た的中率の帯ごと ==")
d["mkb"]=pd.cut(d.mk,[0,.2,.25,.3,.35,.4,1]); print(d[d.sh==1.0].groupby(["mkb","per"],observed=True).apply(agg).to_string())
s=pd.DataFrame(sec); print(f"== 3) ①1着のときの2着（{len(s)}R）: モデル予測% / 実際% ==")
for nm,g in [("全部",s),("枠なり",s[s.waku==True]),("進入が動いた",s[s.waku==False])]:
    print(nm,len(g)," ".join(f"{x}号艇 {g[f'p{x}'].mean()*100:.1f}/{(g.act==x).mean()*100:.1f}" for x in range(2,7)))
