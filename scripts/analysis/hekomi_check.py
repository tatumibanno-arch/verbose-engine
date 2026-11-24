"""①凹み・②一周1位 → 2-3-4 の確認。上位除外・月別・周辺の買い目・偽の世界。"""
import numpy as np, pandas as pd, sqlite3, warnings; warnings.filterwarnings('ignore')
exec(open('scripts/analysis/hekomi.py').read().split("print('全レース'")[0])
m=(P.r_st1>=5)&(P.r_lap1>=4)&(P.r_lap2==1); s=P[m].copy(); s['pay']=np.where(s.res=='2-3-4',s.payout,0)
h=s[s.pay>0].sort_values('pay',ascending=False); print('的中',len(h)); print(h[['hd','jcd','rno','pay']].to_string(index=False))
for k in (0,1,3,5): print(f'上位{k}除外 回収 {(s.pay.sum()-h.pay.head(k).sum())/100/len(s):.3f}')
s['ym']=s.hd.astype(str).str[:6]; g=s.groupby('ym').agg(n=('pay','size'),r=('pay','sum')); g['roi']=g.r/100/g.n; print(g.roi.round(2).to_dict()); print('勝ち月',(g.roi>1).sum(),'/',len(g))
for cb in ('2-3-4','2-4-3','2-3-5','2-4-5','2-3-1','2-4-1','2-1-4'):
    pay=np.where(s.res==cb,s.payout,0); print(cb,f'{pay.sum()/100/len(s):.3f}', (pay>0).sum())
# 偽の世界: 3連単の確定オッズから市場確率→結果を引き直す（その組の当たり確率の分布）
con=sqlite3.connect('data/boatrace.db')
o=pd.read_sql("select hd,jcd,rno,combination,odds from odds where ticket_type='3t' or ticket_type='trifecta'",con)
print('odds行',len(o), o.head(2).to_dict('records') if len(o) else '')
