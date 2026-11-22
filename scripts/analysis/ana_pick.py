import sqlite3, json, sys, itertools, numpy as np, pandas as pd
sys.path.insert(0,'scripts'); from common import VENUES
D='20261122'; S='/tmp/claude-0/-home-user-verbose-engine/968008a0-6ecf-5641-a3ff-006912b41722/scratchpad'
z=np.load(f'{S}/trifecta/pred/today_{D}.npz',allow_pickle=True)
perms=list(itertools.permutations(range(1,7),3))
P=z['P']; H=np.zeros((len(P),6))
for i,p in enumerate(perms): H[:,p[0]-1]+=P[:,i]
m=pd.DataFrame(H,columns=[f'h{k}' for k in range(1,7)]); m['jcd']=[f"{int(x):02d}" for x in z['jcd']]; m['rno']=z['rno'].astype(int)
c=sqlite3.connect('data/boatrace.db')
e=pd.read_sql(f"select jcd,rno,lane,racer_no,racer_name,grade,motor_no,close_time,race_name,data from entries where hd='{D}'",c)
j=e.data.apply(json.loads)
for k in ['national_win_rate','local_win_rate','motor_2ren_rate','series_results','age']: e[k]=j.apply(lambda x:x.get(k))
st=pd.read_sql("select racer_no,data from racer_stats where hd='20260430'",c); sj=st.data.apply(json.loads)
for k in range(1,7): st[f'cw{k}']=sj.apply(lambda x:x.get(f'c{k}_win_rate')); st[f'ct{k}']=sj.apply(lambda x:x.get(f'c{k}_top2_rate')); st[f'cs{k}']=sj.apply(lambda x:x.get(f'c{k}_avg_st'))
e=e.merge(st.drop(columns='data'),on='racer_no',how='left')
e['cw']=[getattr(r,f'cw{r.lane}') for r in e.itertuples()]; e['ct']=[getattr(r,f'ct{r.lane}') for r in e.itertuples()]; e['cs']=[getattr(r,f'cs{r.lane}') for r in e.itertuples()]
# 今節（前日まで連続開催）: 選手の着順・ST、モーターの着順
r=pd.read_sql(f"select r.hd,r.jcd,r.rno,r.lane,r.position, json_extract(r.data,'$.racer_number') racer, json_extract(r.data,'$.motor_number') motor, json_extract(r.data,'$.st_timing') st, json_extract(r.data,'$.entrance_position') ent, json_extract(r.data,'$.exhibition_time') et from results r where r.hd between '20261114' and '20261121'",c)
days={jc:sorted(g.hd.unique()) for jc,g in r.groupby('jcd')}
def meet_days(jc):
    ds=days.get(jc,[]); out=[]; cur=pd.Timestamp(D)
    for d in reversed(ds):
        if (cur-pd.Timestamp(d)).days==1: out.append(d); cur=pd.Timestamp(d)
        else: break
    return out
e['racer']=e.racer_no.astype(int)
rows=[]
for (jc,rn),g in e.groupby(['jcd','rno']):
    md=meet_days(jc); rr=r[(r.jcd==jc)&r.hd.isin(md)]
    info=[]
    for x in g.sort_values('lane').itertuples():
        a=rr[rr.racer==x.racer].sort_values(['hd','rno']); mo=rr[rr.motor==int(x.motor_no)] if str(x.motor_no).isdigit() else rr.iloc[0:0]
        info.append(dict(lane=x.lane,name=x.racer_name.replace('　','').replace(' ',''),gr=x.grade,nw=x.national_win_rate,lw=x.local_win_rate,mo=x.motor_2ren_rate,cw=x.cw,ct=x.ct,cs=x.cs,
            meet=''.join(str(int(p)) if pd.notna(p) else '失' for p in a.position), mst=round(a.st.astype(float).mean(),2) if len(a) else None, met=round(a.et.astype(float).mean(),2) if len(a) else None))
    rows.append(dict(jcd=jc,rno=rn,close=g.close_time.iloc[0],name=g.race_name.iloc[0],day=len(md)+1,info=info))
R=pd.DataFrame(rows).merge(m,on=['jcd','rno'])
def risk(i):
    a=i[0]; n=0
    n+= a['gr'] in ('B1','B2'); n+= (a['cw'] or 0)<40; n+= (a['mo'] or 0)<30; n+= (a['cs'] or 0)>=0.18
    return int(n)
R['risk']=R['info'].apply(risk); R['best']=R[[f'h{k}' for k in range(2,7)]].max(axis=1); R['bl']=R[[f'h{k}' for k in range(2,7)]].values.argmax(axis=1)+2
R.to_pickle(f'{S}/ana_pick_{D}.pkl')
now=sys.argv[1]
c2=R[(R.close>now)].sort_values('h1')
print('全',len(R),'これから',len(c2),' ①頭モデル<35%:',(c2.h1<.35).sum(),' 不安3つ以上:',(c2.risk>=3).sum())
for x in c2.head(22).itertuples():
    print(f"{x.close} {VENUES[x.jcd]}{x.rno}R {x.name[:8]} {x.day}日目 ①頭{x.h1:.0%} 不安{x.risk} 最有力{x.bl}({x.best:.0%}) | "+' '.join(f"{i['lane']}{i['gr']}{i['nw']}" for i in x.info))
