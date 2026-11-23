"""「似たレースを過去から引っ張ってきて出目を見る」（運用者の案）。似たレース＝6艇の級・勝率・枠別成績・モーター・展示順位と風が近いレース。
過去（2025-04〜12）から似たレースk件の出目の割合を「見立て」にし、確認期間（2026-01〜11）で 見立て×オッズ が高い組を買う。締切時オッズ（締切前とは違う）。"""
import sqlite3, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from sklearn.neighbors import NearestNeighbors
D='/tmp/claude-0/-home-user-verbose-engine/968008a0-6ecf-5641-a3ff-006912b41722/scratchpad/n2/'; key=['hd','jcd','rno']; c=sqlite3.connect('data/boatrace.db')
E=pd.read_parquet(D+'ob_ent.parquet'); E['g']=E.grade.map({'A1':4,'A2':3,'B1':2,'B2':1})
for q in ('nw','mo'): E[q]=pd.to_numeric(E[q],errors='coerce')
E['c1']=(E.cw.fillna(0)+1)/(E.cn.fillna(0)+5); E['c3r']=(E.c3.fillna(0)+2)/(E.cn.fillna(0)+5)
F=E.pivot_table(index=key,columns='lane',values=['g','nw','mo','c1','c3r','er'])
F.columns=[f'{a}{b}' for a,b in F.columns]; F=F.dropna().reset_index()
b=pd.read_sql("select hd,jcd,rno,max(cast(json_extract(data,'$.wind_speed') as real)) ws from beforeinfo where hd>='20250401' group by hd,jcd,rno",c); F=F.merge(b,on=key,how='left'); F['ws']=F.ws.fillna(F.ws.median())
p=pd.read_sql("select hd,jcd,rno,combination res,payout from payouts where ticket_type='sanrensho' and hd>='20250401'",c).drop_duplicates(key); F=F.merge(p,on=key)
cols=[c_ for c_ in F.columns if c_ not in key+['res','payout']]; w={'g':1.0,'nw':1.0,'c1':1.0,'c3r':0.7,'mo':0.5,'er':0.7}
Z=(F[cols]-F[cols].mean())/F[cols].std(); 
for c_ in cols: Z[c_]*=w.get(c_.rstrip('123456'),0.5)
tr=F.hd<'20260101'; te=~tr; print('過去（学習）',tr.sum(),'確認',te.sum())
combos=[f'{a}-{b}-{c3}' for a in range(1,7) for b in range(1,7) for c3 in range(1,7) if len({a,b,c3})==3]; ci={x:i for i,x in enumerate(combos)}
Y=np.zeros((tr.sum(),120),dtype=np.float32); rtr=F[tr].res.values
for i,x in enumerate(rtr):
    if x in ci: Y[i,ci[x]]=1
O=pd.read_parquet(D+'cheap100.parquet'); O=O[O.hd>='20260101']; O['pay']=np.where(O.hit==1,O.payout,0)
R=lambda s: s.pay.sum()/(100*len(s)) if len(s) else np.nan
for k in (100,300):
    nn=NearestNeighbors(n_neighbors=k).fit(Z[tr].values); _,idx=nn.kneighbors(Z[te].values)
    P=Y[idx].mean(1)  # 似たレースk件の出目の割合
    T=F[te][key].reset_index(drop=True); pr=pd.DataFrame(P,columns=combos); pr[key]=T
    L=pr.melt(id_vars=key,var_name='cb',value_name='pk'); X=O.merge(L,on=key+['cb']); X['edge']=X.pk*X.odds
    print(f'\n=== k={k}（似たレース{k}件）。似たレースの出目の割合と、売れ方の比べ: 相関 {np.corrcoef(X.pk,X.q)[0,1]:.3f}')
    for th in (1.0,1.2,1.5,2.0):
        s=X[X.edge>=th]; mo=s.groupby(s.hd.str[:6]).apply(R); print(f'  見立て×オッズ{th}以上 | 点 {len(s)} 的中 {s.hit.mean():.3f} 平均オッズ {s.odds.mean():.1f} 回収 {R(s):.3f} 勝ち月 {(mo>1).sum()}/{len(mo)}')
    for lo,hi in ((0,10),(10,30),(30,100)):
        s=X[(X.edge>=1.3)&(X.odds>=lo)&(X.odds<hi)]; print(f'  見立て×オッズ1.3以上かつ{lo}〜{hi}倍 | 点 {len(s)} 回収 {R(s):.3f}')
    s=X.loc[X.groupby(key).edge.idxmax()]; print(f'  各レースで見立て×オッズ最大の1点 | 点 {len(s)} 回収 {R(s):.3f}')
