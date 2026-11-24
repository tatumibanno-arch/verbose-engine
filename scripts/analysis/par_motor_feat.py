"""モーター直近調子の特徴量を作ってparquetに保存（過去日のみ・同日は含めない）。"""
import sqlite3, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
D='/tmp/claude-0/-home-user-verbose-engine/968008a0-6ecf-5641-a3ff-006912b41722/scratchpad/pm/'
c=sqlite3.connect('data/boatrace.db')
e=pd.read_sql("select hd,jcd,rno,lane,racer_no,motor_no,json_extract(data,'$.national_win_rate') nw from entries where hd>='20240601'",c)
r=pd.read_sql("select hd,jcd,rno,lane,position pos from results where hd>='20240601'",c)
L=e.merge(r,on=['hd','jcd','rno','lane'],how='left')
L['pos']=pd.to_numeric(L.pos,errors='coerce'); L['nw']=pd.to_numeric(L.nw,errors='coerce')
L=L[L.pos.notna()&L.motor_no.notna()].copy()
# 期待着順: (枠, 勝率帯) ごとの平均着順 / 2連対率。残差=実際-期待
L['nb']=L.nw.round().clip(2,8)
g=L.groupby(['lane','nb']); L['ep']=g.pos.transform('mean'); L['e2']=g.pos.transform(lambda s:(s<=2).mean())
L['res']=L.pos-L.ep; L['t2']=(L.pos<=2).astype(float); L['t2r']=L.t2-L.e2; L['w']=(L.pos==1).astype(float)
L['d']=pd.to_datetime(L.hd).map(pd.Timestamp.toordinal)
L['gid']=L.groupby(['jcd','motor_no']).ngroup()
L=L.sort_values(['gid','hd','rno','lane']).reset_index(drop=True)
comp=L.gid.values.astype(np.int64)*100000+L.d.values
gs=np.searchsorted(L.gid.values,L.gid.values,'left')  # group start
hi=np.searchsorted(comp,comp - L.d.values + L.d.values,'left')  # races with same gid and date<d
hi=np.searchsorted(comp,comp,'left')
def win(days,n):
    lo=np.searchsorted(comp,comp-days,'left'); lo=np.maximum(lo,hi-n) if n else lo
    return lo
out={}
for nm,days,n in (('m10',150,10),('m20',150,20),('meet',6,None)):
    lo=win(days,n); cnt=hi-lo
    for col in ('pos','res','t2','t2r'):
        cs=np.concatenate([[0],np.cumsum(L[col].values)])
        out[f'{nm}_{col}']=np.where(cnt>0,(cs[hi]-cs[lo])/np.maximum(cnt,1),np.nan)
    out[f'{nm}_n']=cnt
F=pd.concat([L[['hd','jcd','rno','lane','racer_no','motor_no','pos']],pd.DataFrame(out)],axis=1)
F.to_parquet(D+'feat.parquet'); print(F.shape); print(F[F.hd>='20250601'].describe().T.to_string())
