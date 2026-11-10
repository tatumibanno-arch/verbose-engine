# 使い方: python roll_pre.py <jcd> <rno> <目,目,...>  展示・スタ展・気象・市場の頭シェア・指定した目のオッズとモデル確率（日付は ROLL_DATE か今日）
import sys, json, itertools, requests, numpy as np
sys.path.insert(0,"/home/user/verbose-engine/scripts")
from scrape_web import fetch, parse_beforeinfo, parse_odds3t
import os, datetime as _dt; D=os.environ.get("ROLL_DATE") or _dt.datetime.now(_dt.timezone(_dt.timedelta(hours=9))).strftime("%Y%m%d")
jcd, rno = sys.argv[1].zfill(2), int(sys.argv[2])
s=requests.Session()
h=fetch("beforeinfo",D,jcd,rno,s); bi=parse_beforeinfo(h) if h else []
ex=[(r["lane"],r["exhibition_time"]) for r in bi if r.get("exhibition_time")]
rank={l:i+1 for i,(l,t) in enumerate(sorted(ex,key=lambda x:x[1]))}
for r in bi:
    print(f"{r['lane']} 展示{r.get('exhibition_time')}({rank.get(r['lane'],'-')}位) スタ展{r.get('start_st_raw')} 進入{r.get('start_course')} チルト{r.get('tilt')} 交換{r.get('parts_changed')} 体重{r.get('weight')}")
if bi: print("天候",bi[0].get("weather"),"風",bi[0].get("wind_speed"),"m 向",bi[0].get("wind_direction"),"波",bi[0].get("wave_height"))
h2=fetch("odds3t",D,jcd,rno,s); o={c:od for tt,c,od,om in parse_odds3t(h2) if od} if h2 else {}
PERMS=np.array(list(itertools.permutations(range(6),3)))
z=np.load(f"/tmp/claude-0/-home-user-verbose-engine/968008a0-6ecf-5641-a3ff-006912b41722/scratchpad/trifecta/pred/today_{D}.npz")
idx={(str(j).zfill(2),int(r)):i for i,(j,r) in enumerate(zip(z["jcd"],z["rno"]))}
P=z["P"][idx[(jcd,rno)]]; pk={f"{a+1}-{b+1}-{c+1}":P[k] for k,(a,b,c) in enumerate(PERMS)}
tot=sum(1/v for v in o.values()) if o else 1
print("頭シェア", {k: round(sum(1/v for c,v in o.items() if c[0]==k)/tot*100) for k in "123456"})
json.dump({"bi":bi,"o":o,"p":{k:float(v) for k,v in pk.items()}},open(f"/tmp/claude-0/-home-user-verbose-engine/968008a0-6ecf-5641-a3ff-006912b41722/scratchpad/pre_{jcd}_{rno}.json","w"),ensure_ascii=False)
for c in sys.argv[3].split(","):
    print(c, o.get(c), f"{pk.get(c,0)*100:.1f}%")
