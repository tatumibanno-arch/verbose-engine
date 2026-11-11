# 使い方: python roll_one.py <jcd> <rno> <残高> [倍率=2.0] [最大点数=20]
# 1レースについて、今のオッズ×モデル確率で「各目の払戻 >= 倍率×残高（ガミなし）」を守り、100円単位・残高内で
# 当たる確率が最大になる組み合わせを2連単と3連単の両方で出す。
import sys, math, itertools, requests, numpy as np
sys.path.insert(0,"/home/user/verbose-engine/scripts")
from scrape_web import fetch, parse_odds3t, parse_odds2tf
import os, datetime as _dt; D=os.environ.get("ROLL_DATE") or _dt.datetime.now(_dt.timezone(_dt.timedelta(hours=9))).strftime("%Y%m%d")
SHRINK=float(os.environ.get("ROLL_SHRINK","1.0"))  # v3: 0.75（今のオッズが締切までに25%下がっても倍率を守る）
j=sys.argv[1].zfill(2); r=int(sys.argv[2]); B=int(sys.argv[3]); T=float(sys.argv[4]) if len(sys.argv)>4 else 2.0; MAXP=int(sys.argv[5]) if len(sys.argv)>5 else 20
PERMS=np.array(list(itertools.permutations(range(6),3)))
z=np.load(f"/tmp/claude-0/-home-user-verbose-engine/968008a0-6ecf-5641-a3ff-006912b41722/scratchpad/trifecta/pred/today_{D}.npz")
i=[k for k in range(len(z["jcd"])) if str(z["jcd"][k]).zfill(2)==j and int(z["rno"][k])==r][0]; P=z["P"][i]
p3={f"{a+1}-{b+1}-{c+1}":float(P[k]) for k,(a,b,c) in enumerate(PERMS)}; p2={}
for k,v in p3.items(): p2[k[:3]]=p2.get(k[:3],0)+v
s=requests.Session()
o2={c:o for tt,c,o,om in parse_odds2tf(fetch("odds2tf",D,j,r,s)) if tt=="nirensho" and o}
o3={c:o for tt,c,o,om in parse_odds3t(fetch("odds3t",D,j,r,s)) if o}
def plan(probs, odds):
    items=sorted([(c,probs[c],odds[c]) for c in odds if c in probs],key=lambda x:-x[1]*x[2])[:40]
    U=B//100; best=[[-1]*(U+1) for _ in range(MAXP+1)]; best[0][0]=0; ch={}
    for c,p,o in items:
        cost=math.ceil(T*B/(o*SHRINK)/100-1e-9)  # 締切までの下がり分を見込む
        if cost>U: continue
        for k in range(MAXP-1,-1,-1):
            for u in range(U-cost,-1,-1):
                if best[k][u]>=0 and best[k][u]+p>best[k+1][u+cost]:
                    best[k+1][u+cost]=best[k][u]+p; ch[(k+1,u+cost)]=ch.get((k,u),())+((c,cost*100,o,p),)
    bk=max(((best[k][u],k,u) for k in range(MAXP+1) for u in range(U+1) if best[k][u]>0),default=None)
    if not bk: return None
    sel=sorted(ch[(bk[1],bk[2])],key=lambda x:-x[3]); spend=sum(x[1] for x in sel)
    # 余りは確率の高い目に足す（払戻はさらに増えるだけでガミにはならない）
    left=B-spend; sel=[list(x) for x in sel]; k=0
    while left>=100 and sel: sel[k%len(sel)][1]+=100; left-=100; k+=1
    return bk[0], sel
for name,pp,oo in [("2連単",p2,o2),("3連単",p3,o3)]:
    res=plan(pp,oo)
    if not res: print(f"{name}: 条件を満たす組み合わせなし"); continue
    hit,sel=res
    tot=sum(1/v for v in oo.values()); mk=sum((1/oo[x[0]])/tot for x in sel)
    print(f"{name}: {len(sel)}点 合計{sum(x[1] for x in sel)}円 モデル的中{hit*100:.0f}% オッズから見た的中{mk*100:.0f}% 最低払戻{min(x[1]*x[2] for x in sel):.0f}円（今のオッズ）/ {min(x[1]*x[2]*SHRINK for x in sel):.0f}円（{SHRINK:.0%}に下がった場合）")
    for c,a,o,p in sel: print(f"  {c} {a}円 ({o:.1f}倍→{a*o:.0f}円) 確率{p*100:.1f}%")
