"""潮（気象庁の推算潮位、scripts/fetch_tide.py の出力）がオッズに入っているか。海水・汽水の場だけ（淡水プールの住之江・尼崎・唐津は除く）。
潮位は締切時刻の時の値。場ごとに3つに分ける（低い・中・高い）。上げ／下げは前後1時間の差。2025-04〜2026-11、締切時オッズ。"""
import sqlite3, sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0,'scripts/analysis'); sys.path.insert(0,'scripts'); import overbought as ob; from common import VENUES
key=['hd','jcd','rno']; c=sqlite3.connect('data/boatrace.db')
T=pd.read_parquet(ob.D+'tide.parquet'); T=T[~T.jcd.isin(['12','13','23'])].sort_values(['jcd','hd','hour'])
T['nxt']=T.groupby('jcd')['level'].shift(-1); T['prv']=T.groupby('jcd')['level'].shift(1); T['slope']=(T.nxt-T.prv)/2
e=pd.read_sql("select distinct hd,jcd,rno,close_time from entries where hd>='20250401'",c); e['hour']=e.close_time.str[:2].astype(float)
R,L,oc=ob.load(); A=R[key+['res','tot']+oc].copy(); A['p1']=(1/A[oc]).sum(1)/A.tot; A['win1']=(A.res.str[0]=='1').astype(float)
A=A.merge(e,on=key).merge(T[['jcd','hd','hour','level','slope','day_range']],on=['jcd','hd','hour'])
A['lv']=A.groupby('jcd')['level'].transform(lambda s:pd.qcut(s,3,labels=['低い','中','高い'])); A['dir']=np.where(A.slope>=8,'上げ',np.where(A.slope<=-8,'下げ','止まり')); A['rg']=A.groupby('jcd').day_range.transform(lambda s:pd.qcut(s,3,labels=['小潮寄り','中','大潮寄り']))
base=A.win1.mean()-A.p1.mean(); print('対象レース',len(A),'場',A.jcd.nunique(),'全体の差',round(base,3))
def show(nm,b,s=A):
    s=s.assign(b=b); t=s.groupby('b',observed=True).agg(n=('win1','size'),mk=('p1','mean'),ac=('win1','mean')); t['差']=t.ac-t.mk
    for nm2,q in (('前半',s[s.hd<'20260401']),('後半',s[s.hd>='20260401'])): u=q.groupby('b',observed=True).agg(mk=('p1','mean'),ac=('win1','mean')); t[nm2]=u.ac-u.mk
    t.columns=['件数','売れ方','実際','差','前半の差','後半の差']; print('\n[1号艇] '+nm); print(t.round(3).to_string())
show('潮位（場ごとに3分割）',A.lv); show('上げ・下げ',A.dir); show('潮の大きさ',A.rg); show('潮位×上げ下げ',A.lv.astype(str)+'・'+A.dir)
print('\n場ごと: 1号艇の1着率（潮位 低い／中／高い）と、売れ方との差')
rows=[]
for j,s in A.groupby('jcd'):
    r={'場':VENUES[j],'件数':len(s)}
    for l in ('低い','中','高い'):
        q=s[s.lv==l]; r[f'{l} 1着率']=round(q.win1.mean(),3); r[f'{l} 差']=round(q.win1.mean()-q.p1.mean(),3)
    for l in ('低い','高い'):
        for nm2,q in (('前',s[(s.lv==l)&(s.hd<'20260401')]),('後',s[(s.lv==l)&(s.hd>='20260401')])): r[f'{l}差{nm2}']=round(q.win1.mean()-q.p1.mean(),3)
    rows.append(r)
print(pd.DataFrame(rows).set_index('場').to_string())
