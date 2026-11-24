"""展示の進入（前付け・枠なり崩れ）とチルトで、3連単の歪みがあるか。オッズ不使用。事前に決めた仮説だけ。"""
import sqlite3, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
c=sqlite3.connect('data/boatrace.db')
b=pd.read_sql("select hd,jcd,rno,lane,tilt,start_course sc from beforeinfo",c)
p=pd.read_sql("select hd,jcd,rno,combination res,payout from payouts where ticket_type in ('sanrensho')",c)
if p.empty: print(pd.read_sql("select distinct ticket_type from payouts",c)); raise SystemExit
for d in (b,p): d['hd']=d.hd.astype(str); d['jcd']=d.jcd.astype(str).str.zfill(2); d['rno']=d.rno.astype(int)
key=['hd','jcd','rno']; cut='20260201'
P=b.pivot_table(index=key,columns='lane',values=['tilt','sc']); P.columns=[f'{a}{c}' for a,c in P.columns]; P=P.reset_index().merge(p.drop_duplicates(key),on=key)
P=P[P[[f'sc{i}' for i in range(1,7)]].notna().all(1)]; P['ym']=P.hd.str[:6]
wak=np.all([P[f'sc{i}']==i for i in range(1,7)],axis=0); print('全',len(P),'枠なり',wak.mean().round(3))
tried=0
def rep(nm,m,cbs):
    global tried; tried+=1; s=P[m]; 
    if isinstance(cbs,str): cbs=[cbs]
    pay=sum(np.where(s.res==cb,s.payout,0) for cb in cbs); k=len(cbs); a=(s.hd<cut).values
    if len(s)<100: print(nm,len(s),'少ない'); return
    h=np.sort(pay[pay>0])[::-1]; mo=pd.DataFrame({'ym':s.ym,'p':pay}).groupby('ym').p.agg(['size','sum'])
    print(f'{nm} | {len(s)} 的中{(pay>0).mean():.3f} 回収{pay.sum()/100/k/len(s):.3f} 前{pay[a].sum()/100/k/max(a.sum(),1):.3f} 後{pay[~a].sum()/100/k/max((~a).sum(),1):.3f} 上位3除{(pay.sum()-h[:3].sum())/100/k/len(s):.3f} 勝月{(mo["sum"]/100/k/mo["size"]>1).sum()}/{len(mo)}')
rep('枠なり 1-2-3',wak,'1-2-3'); rep('枠なり崩れ 1-2-3',~wak,'1-2-3')
# 1号艇が1コースを取れない（前付けされた）
no1=P.sc1!=1; rep('①が1コースでない → 1着①外し 2-3-4以外?: 1-2-3',no1,'1-2-3')
# 1コースに入った艇
P['in1']=sum(np.where(P[f'sc{i}']==1,i,0) for i in range(1,7))
for k in (2,3,4,5,6): rep(f'{k}号艇が1コース → {k}-1-2',P.in1==k,f'{k}-1-2')
# 外の艇が内に入る（6号艇がコース4以内）
for k in (5,6):
    m=P[f'sc{k}']<=3; rep(f'{k}号艇が3コース以内に前付け → 1-2-3',m,'1-2-3'); rep(f'{k}号艇が3コース以内 → 1-{k}-2',m,f'1-{k}-2')
# 深い①（①が1コースだが他が前付けで6艇以外…）略
# チルト: 外枠で+1以上（伸び型）
for k in (4,5,6):
    m=P[f'tilt{k}']>=1.0; rep(f'{k}号艇チルト+1以上 → 1-{k}-2',m,f'1-{k}-2'); rep(f'{k}号艇チルト+1以上 → {k}-1-2',m,f'{k}-1-2'); rep(f'{k}号艇チルト+1以上 → 1-2-3（①の味方?）',m,'1-2-3')
m=P.tilt1>=0.5; rep('①チルト+0.5以上 → 2-1-3',m,'2-1-3'); rep('①チルト+0.5以上 → 1-2-3',m,'1-2-3')
print('試した数',tried)
