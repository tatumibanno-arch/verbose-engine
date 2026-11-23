"""モーターの「気配」（2連率ではなく、前節と今節でそのモーターが乗り手の力以上に走ったか）を作り、オッズに入っているかを見る。
気配 = （選手の勝率と枠から見込まれる着順 − 実際の着順）の平均。プラスなら乗り手の力以上に走っている。
前節 = 同じ場・同じモーターの、今の節より前でいちばん近い節（60日以内・4走以上）。今節 = 今の節の前日まで（2走以上）。
発見=〜2026-03、検証=2026-04〜。締切時オッズ（締切前とは違う）。"""
import sqlite3, sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0,'scripts/analysis'); import overbought as ob
D=ob.D; key=['hd','jcd','rno']; c=sqlite3.connect('data/boatrace.db')
e=pd.read_sql("select hd,jcd,rno,lane,motor_no,json_extract(data,'$.national_win_rate') nw,json_extract(data,'$.motor_2ren_rate') mo from entries where hd>='20241001'",c)
r=pd.read_sql("select hd,jcd,rno,lane,position from results where hd>='20241001'",c)
e['nw']=pd.to_numeric(e.nw,errors='coerce'); e['mo']=pd.to_numeric(e.mo,errors='coerce')
r=r.merge(e,on=key+['lane']); r['pos']=pd.to_numeric(r.position,errors='coerce'); r=r.dropna(subset=['pos','nw','motor_no'])
X=np.column_stack([np.ones(len(r)),r.nw]+[(r.lane==k).astype(float) for k in range(2,7)]); beta=np.linalg.lstsq(X,r.pos.values,rcond=None)[0]; r['res']=X@beta-r.pos
dd=r[['jcd','hd']].drop_duplicates().sort_values(['jcd','hd']); dd['d']=pd.to_datetime(dd.hd); dd['meet']=(dd.groupby('jcd').d.diff().dt.days!=1).groupby(dd.jcd).cumsum()
ms=dd.groupby(['jcd','meet']).d.min().rename('mstart').reset_index(); r=r.merge(dd[['jcd','hd','meet']],on=['jcd','hd'])
# 節ごとのモーター成績
g=r.groupby(['jcd','motor_no','meet']).res.agg(['mean','count']).reset_index().merge(ms,on=['jcd','meet']).sort_values(['jcd','motor_no','meet'])
g['pm']=g.groupby(['jcd','motor_no'])['mean'].shift(1); g['pn']=g.groupby(['jcd','motor_no'])['count'].shift(1); g['pstart']=g.groupby(['jcd','motor_no']).mstart.shift(1)
g.loc[(g.pn<4)|((g.mstart-g.pstart).dt.days>60),'pm']=np.nan
# 今節（前日まで）
h=r.groupby(['jcd','motor_no','meet','hd']).res.agg(['sum','count']).reset_index().sort_values('hd'); k2=['jcd','motor_no','meet']
h['cs']=h.groupby(k2)['sum'].cumsum()-h['sum']; h['cc']=h.groupby(k2)['count'].cumsum()-h['count']; h['cm']=np.where(h.cc>=2,h.cs/h.cc,np.nan)
ed=e.merge(dd[['jcd','hd','meet']],on=['jcd','hd'],how='left').merge(g[k2+['pm']],on=k2,how='left').merge(h[k2+['hd','cm']],on=k2+['hd'],how='left')
ed['form']=ed[['pm','cm']].mean(axis=1); F=ed[key+['lane','pm','cm','form','mo']]
for col in ('pm','cm','form','mo'): F[col+'_rk']=F.groupby(key)[col].rank(ascending=False,method='min')
F.to_parquet(D+'motor_form.parquet'); print('気配が付いた艇の割合 前節',F.pm.notna().mean().round(2),'今節',F.cm.notna().mean().round(2))
print('前節の気配と今節の気配の相関',F[['pm','cm']].corr().iloc[0,1].round(3),' 2連率との相関 前節',F[['pm','mo']].corr().iloc[0,1].round(3),'今節',F[['cm','mo']].corr().iloc[0,1].round(3))
# (1) オッズに入っているか: 相手②〜⑥、①が勝ったレース
R,L,oc=ob.load(); L=L.merge(F.drop(columns='mo'),on=key+['lane'],how='left'); W=L[L.win1&(L.p1>=.4)]
def cal(nm,col,bins,labels):
    s=W.dropna(subset=[col]); s=s.assign(b=pd.cut(s[col],bins,labels=labels))
    for per,q in (('発見',s[s.hd<'20260401']),('検証',s[s.hd>='20260401'])):
        t=q.groupby('b',observed=True).agg(n=('in3','size'),売れ方=('mk3','mean'),実際=('in3','mean')); t['差']=(t.実際-t.売れ方).round(3); print(f'[{nm}] {per}'); print(t.round(3).T.to_string())
cal('前節の気配（着順の上振れ）','pm',[-9,-.6,-.2,.2,.6,9],['大きく悪い','悪い','ふつう','良い','大きく良い'])
cal('今節の気配','cm',[-9,-.8,-.3,.3,.8,9],['大きく悪い','悪い','ふつう','良い','大きく良い'])
cal('レース内の気配の順位（前節＋今節）','form_rk',[0,1,2,4,5,6],['1位','2位','3〜4位','5位','6位'])
cal('（比較）モーター2連率のレース内順位','mo_rk',[0,1,2,4,5,6],['1位','2位','3〜4位','5位','6位'])
# ①について
A=R[key+['res','tot']+oc].copy(); A['p1']=(1/A[oc]).sum(1)/A.tot; A['win1']=A.res.str[0]=='1'; A=A.merge(F[F.lane==1],on=key,how='left')
for nm,col,bins,labels in (('①の前節の気配','pm',[-9,-.6,-.2,.2,.6,9],['大きく悪い','悪い','ふつう','良い','大きく良い']),('①の今節の気配','cm',[-9,-.8,-.3,.3,.8,9],['大きく悪い','悪い','ふつう','良い','大きく良い'])):
    s=A.dropna(subset=[col]); t=s.assign(b=pd.cut(s[col],bins,labels=labels)).groupby('b',observed=True).agg(n=('win1','size'),売れ方1=('p1','mean'),実際の1着率=('win1','mean')); t['差']=(t.実際の1着率-t.売れ方1).round(3); print(f'[{nm}]'); print(t.round(3).T.to_string())
