"""転がし v3 の1段の過去検証（締切時オッズで組むので実際より甘い）。
対象: ウォークフォワード予測があり、締切時3連単オッズがある期間。モデル①1着確率0.6〜0.8。
配分: roll_one と同じ DP（残高1,000円、2倍・ガミなし、20点まで）を SHRINK 倍したオッズで組む。
的中なら払戻=賭け金×確定払戻（締切時オッズ≒確定オッズ）。見送りライン: モデル的中40%未満。"""
import sys, glob, math, itertools, sqlite3, numpy as np, pandas as pd
W="/tmp/claude-0/-home-user-verbose-engine/968008a0-6ecf-5641-a3ff-006912b41722/scratchpad/trifecta/pred"
PERMS=["-".join(map(str,p)) for p in itertools.permutations(range(1,7),3)]
SHRINK=float(sys.argv[1]) if len(sys.argv)>1 else 0.75; B=1000; T=2.0; MAXP=20; FROM=sys.argv[2] if len(sys.argv)>2 else "20260501"
def plan(probs, odds):
    items=sorted([(c,probs[c],odds[c]) for c in odds if c in probs and odds[c]>0],key=lambda x:-x[1]*x[2])[:40]
    U=B//100; best=[[-1]*(U+1) for _ in range(MAXP+1)]; best[0][0]=0; ch={}
    for c,p,o in items:
        cost=math.ceil(T*B/(o*SHRINK)/100-1e-9)
        if cost>U: continue
        for k in range(MAXP-1,-1,-1):
            for u in range(U-cost,-1,-1):
                if best[k][u]>=0 and best[k][u]+p>best[k+1][u+cost]:
                    best[k+1][u+cost]=best[k][u]+p; ch[(k+1,u+cost)]=ch.get((k,u),())+((c,cost*100),)
    bk=max(((best[k][u],k,u) for k in range(MAXP+1) for u in range(U+1) if best[k][u]>0),default=None)
    if not bk: return None,0
    sel=list(ch[(bk[1],bk[2])]); spent=sum(a for _,a in sel); left=B-spent
    i=0
    while left>=100 and sel:  # 余りを確率の高い目へ
        sel.sort(key=lambda x:-probs[x[0]]); c,a=sel[i%len(sel)]; sel[i%len(sel)]=(c,a+100); left-=100; i+=1
    return dict(sel),bk[0]
c=sqlite3.connect("data/boatrace.db")
od=pd.read_sql(f"select hd,jcd,rno,combination,odds from odds where ticket_type='sanrensho' and hd>='{FROM}'",c)
od["jcd"]=od.jcd.astype(str).str.zfill(2); od["hd"]=od.hd.astype(str)
OD={k:dict(zip(g.combination,g.odds)) for k,g in od.groupby(["hd","jcd","rno"])}
pay=pd.read_sql(f"select hd,jcd,rno,combination,payout from payouts where ticket_type='sanrensho' and hd>='{FROM}'",c)
pay["jcd"]=pay.jcd.astype(str).str.zfill(2); pay["hd"]=pay.hd.astype(str)
PAY={(r.hd,r.jcd,r.rno):(r.combination,r.payout) for r in pay.itertuples()}
rows=[]; sec2=[]
for f in sorted(glob.glob(f"{W}/pred_*.npz")):
    z=np.load(f); hd=z["hd"].astype(str)
    if max(hd)<FROM: continue
    for i in range(len(hd)):
        k=(hd[i],str(z["jcd"][i]).zfill(2),int(z["rno"][i]))
        if k not in OD or k not in PAY: continue
        P=z["P"][i]; probs={PERMS[j]:float(P[j]) for j in range(120)}
        p1=sum(v for cc,v in probs.items() if cc[0]=="1")
        res,pv=PAY[k]
        if res[0]=="1": sec2.append((sum(v for cc,v in probs.items() if cc.startswith("1-2"))/max(p1,1e-9), res[2]=="2"))
        if not (0.6<=p1<=0.8): continue
        sel,hit=plan(probs,OD[k])
        if not sel or hit<0.40: continue
        ret=sel.get(res,0)*pv/100
        oo=OD[k]; tot=sum(1/v for v in oo.values() if v>0); mk=sum((1/oo[c])/tot for c in sel if oo.get(c))
        rows.append(dict(hd=k[0],hit=res in sel,mult=ret/B,model=hit,mk=mk,p1=p1))
d=pd.DataFrame(rows); d["m"]=d.hd.str[:6]
print(f"SHRINK={SHRINK} 期間 {d.hd.min()}〜{d.hd.max()} 段数 {len(d)}  的中率 {d.hit.mean()*100:.1f}%  モデル的中の平均 {d.model.mean()*100:.1f}%")
print(f"当たった時の倍率 平均 {d[d.hit].mult.mean():.2f}（最小 {d[d.hit].mult.min():.2f}）  1段の期待値（払戻÷投入）{d.mult.mean():.3f}")
print(f"2段連続の成功率（独立とみなす）{(d.hit.mean())**2*100:.1f}%")
print(d.groupby("m").agg(段=("hit","size"),的中=("hit","mean"),回収=("mult","mean")).round(3).to_string())
d["mkb"]=pd.cut(d.mk,[0,.2,.25,.3,.35,1]); print(d.groupby("mkb",observed=True).agg(段=("hit","size"),的中=("hit","mean"),回収=("mult","mean"),モデル=("model","mean")).round(3).to_string())
s2=pd.DataFrame(sec2,columns=["pred2","act2"])
print(f"①1着のとき、モデルの「2着②」の予測割合 {s2.pred2.mean()*100:.1f}% / 実際に2着② {s2.act2.mean()*100:.1f}%（{len(s2)}R）")
