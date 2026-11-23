"""選手の「型」（握る＝まくり型、差し型など）で、隣の艇の来やすさがオッズに入っているか（運用者の例: ①が強く②が握る選手なら④が来やすい）。
型 = レース日の前月までの12か月で、その選手が2〜6コースから勝ったときの決まり手の割合。決まり手は公式の成績ファイル（kimarite_*.parquet）。
見方: ①が勝ったレースで、各艇が2着・3着以内に入る売れ方（3連単）と実際の差。2025-04〜、前半〜2026-01／後半2026-02〜。締切時オッズ。"""
import sqlite3, sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0,'scripts/analysis'); import overbought as ob
key=['hd','jcd','rno']; c=sqlite3.connect('data/boatrace.db'); cut='20260201'
K=pd.concat([pd.read_parquet(f'data/parquet/kimarite_{y}.parquet') for y in (2023,2024,2025,2026)]); K['jcd']=K.jcd.astype(int).astype(str).str.zfill(2); K['rno']=K.rno.astype(int)
r=pd.read_sql("select r.hd,r.jcd,r.rno,r.lane,r.position, cast(json_extract(r.data,'$.entrance_position') as int) crs, e.racer_no from results r join entries e using(hd,jcd,rno,lane) where r.hd>='20230101'",c)
r['racer_no']=r.racer_no.astype(int); r=r.merge(K[key+['kimarite']],on=key,how='left'); r['ym']=r.hd.str[:6]
out=r[r.crs.between(2,6)]; w=out[out.position==1]
m=out.groupby(['racer_no','ym']).size().rename('st').to_frame()
for kk,nm in (('まくり','mk'),('まくり差し','ms'),('差し','sa')): m[nm]=w[w.kimarite==kk].groupby(['racer_no','ym']).size()
m['win']=w.groupby(['racer_no','ym']).size(); m=m.fillna(0).reset_index()
months=sorted(r.ym.unique()); rows=[]
for i,ym in enumerate(months):
    if i<12: continue
    prev=months[i-12:i]; s=m[m.ym.isin(prev)].groupby('racer_no')[['st','mk','ms','sa','win']].sum(); s['ym']=ym; rows.append(s.reset_index())
S=pd.concat(rows); S['mk_rate']=(S.mk+S.ms)/(S.st+10); S['sa_rate']=S.sa/(S.st+10); S['agg']=(S.mk+S.ms)/(S.win+3)
e=pd.read_sql("select hd,jcd,rno,lane,racer_no from entries where hd>='20250401'",c); e['racer_no']=e.racer_no.astype(int); e['ym']=e.hd.str[:6]
e=e.merge(S[['racer_no','ym','mk_rate','sa_rate','agg']],on=['racer_no','ym'],how='left')
q=lambda col: e[col].quantile([1/3,2/3]).values
e['mk_t']=pd.cut(e.mk_rate,[-1,*q('mk_rate'),9],labels=['まくり少','まくり中','まくり多']); e['sa_t']=pd.cut(e.sa_rate,[-1,*q('sa_rate'),9],labels=['差し少','差し中','差し多'])
R,L,oc=ob.load(); W=L[L.win1&(L.p1>=.4)].copy()
nb=e[key+['lane','mk_t','sa_t']]
def table(nm,target,col,nlane,stat='in3',mk='mk3'):
    x=W[W.lane==target].merge(nb[nb.lane==nlane][key+[col]],on=key,how='inner')
    t=x.groupby(col,observed=True).agg(n=(stat,'size'),m=(mk,'mean'),a=(stat,'mean')); t['差']=t.a-t.m
    for h,qq in (('前半',x[x.hd<cut]),('後半',x[x.hd>=cut])): u=qq.groupby(col,observed=True).agg(m=(mk,'mean'),a=(stat,'mean')); t[h]=u.a-u.m
    t.columns=['件数','売れ方','実際','差','前半の差','後半の差']; print(f'\n[{nm}]'); print(t.round(3).to_string())
table('①が勝ったとき、④が2着 ｜ ②のまくり型','4','mk_t',2,'in2','mk2') if False else None
for tgt,nl in ((4,2),(3,2),(4,3),(5,4),(5,3),(6,5)):
    table(f'①が勝ったとき、{tgt}号艇が3着以内 ｜ {nl}号艇のまくり型',tgt,'mk_t',nl)
    table(f'①が勝ったとき、{tgt}号艇が2着 ｜ {nl}号艇のまくり型',tgt,'mk_t',nl,'in2','mk2')
for tgt,nl in ((3,2),(4,3),(2,3)):
    table(f'①が勝ったとき、{tgt}号艇が3着以内 ｜ {nl}号艇の差し型',tgt,'sa_t',nl)
