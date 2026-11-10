# 使い方: python roll_scan.py [倍率=2.0]  全レースで「ガミなし・倍率以上・20点まで」の最大的中率の組を出す（日付は ROLL_DATE か今日）
import sys, itertools, math, requests, numpy as np, pandas as pd, sqlite3
sys.path.insert(0,"scripts")
from scrape_web import fetch, parse_odds3t, parse_odds2tf
from common import VENUES
from concurrent.futures import ThreadPoolExecutor
import os, datetime as _dt; D=os.environ.get("ROLL_DATE") or _dt.datetime.now(_dt.timezone(_dt.timedelta(hours=9))).strftime("%Y%m%d")
T=float(sys.argv[1]) if len(sys.argv)>1 else 2.0
BANKS=[1000,2000,4000]
PERMS=np.array(list(itertools.permutations(range(6),3)))
z=np.load(f"/tmp/claude-0/-home-user-verbose-engine/968008a0-6ecf-5641-a3ff-006912b41722/scratchpad/trifecta/pred/today_{D}.npz")
con=sqlite3.connect("data/boatrace.db")
close=dict(((str(j).zfill(2),int(r)),c) for j,r,c in con.execute("select distinct jcd,rno,close_time from entries where hd=?",(D,)))
s=requests.Session()
def plan(probs, odds, B, maxpts=20):
    # 100円単位。各目の払戻 >= T*B を守る最小賭け金 = ceil(T*B/odds/100)*100。予算B内で当たる確率最大（DP）
    items=[(c,probs[c],odds[c]) for c in odds if c in probs and odds[c]>0]
    items=sorted(items,key=lambda x:-x[1]*x[2])[:40]
    U=B//100; NEG=-1
    best=[[NEG]*(U+1) for _ in range(maxpts+1)]; best[0][0]=0; choice={}
    for idx,(c,p,o) in enumerate(items):
        cost=math.ceil(T*B/o/100-1e-9)
        if cost>U: continue
        for k in range(maxpts-1,-1,-1):
            for u in range(U-cost,-1,-1):
                if best[k][u]>=0 and best[k][u]+p>best[k+1][u+cost]:
                    best[k+1][u+cost]=best[k][u]+p; choice[(k+1,u+cost)]=choice.get((k,u),())+((c,cost*100,o),)
    bk=max(((best[k][u],k,u) for k in range(maxpts+1) for u in range(U+1) if best[k][u]>0),default=None)
    if not bk: return None
    sel=choice[(bk[1],bk[2])]
    return dict(hit=bk[0], pts=len(sel), spend=sum(x[1] for x in sel), minret=min(x[1]*x[2] for x in sel), combos=" ".join(f"{c}:{a}円({o:.1f})" for c,a,o in sorted(sel,key=lambda x:-probs[x[0]])))
def work(i):
    j=str(z["jcd"][i]).zfill(2); r=int(z["rno"][i]); P=z["P"][i]
    p3={f"{a+1}-{b+1}-{c+1}":float(P[k]) for k,(a,b,c) in enumerate(PERMS)}
    p2={}
    for k,v in p3.items(): p2[k[:3]]=p2.get(k[:3],0)+v
    try: h=fetch("odds2tf",D,j,r,s); o2={c:o for tt,c,o,om in parse_odds2tf(h) if tt=="nirensho" and o} if h else {}
    except Exception: o2={}
    try: h=fetch("odds3t",D,j,r,s); o3={c:o for tt,c,o,om in parse_odds3t(h) if o} if h else {}
    except Exception: o3={}
    head=[float(sum(P[k] for k,(a,b,c) in enumerate(PERMS) if a==l)) for l in range(6)]
    out=[]
    for name,pp,oo in [("2連単",p2,o2),("3連単",p3,o3)]:
        if not oo: continue
        for B in BANKS:
            pl=plan(pp,oo,B)
            if pl: out.append(dict(venue=VENUES.get(j,j),jcd=j,rno=r,close=close.get((j,r),""),kind=name,bank=B,p_head1=round(head[0],3),top_head=int(np.argmax(head))+1,**pl))
    return out
with ThreadPoolExecutor(8) as ex: res=[x for l in ex.map(work, range(len(z["jcd"]))) for x in l]
df=pd.DataFrame(res); df.to_csv(f"/tmp/claude-0/-home-user-verbose-engine/968008a0-6ecf-5641-a3ff-006912b41722/scratchpad/roll_scan_{D}_T{T}.csv",index=False)
df=df[df.close>="10:00"]
for kind in ("2連単","3連単"):
    for B in BANKS:
        d=df[(df.kind==kind)&(df.bank==B)].sort_values("hit",ascending=False).head(8)
        print(f"\n=== {kind} 資金{B}円 合成{T}倍・ガミなし・20点まで")
        for x in d.itertuples(): print(f"{x.close} {x.venue}{x.rno}R {x.pts}点 {x.spend}円 的中{x.hit*100:.0f}% 最低払戻{x.minret:.0f}円 | {x.combos}")
