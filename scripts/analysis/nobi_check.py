"""6伸び・①強 → 1-6-2 の確認。"""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
exec(open('scripts/analysis/nobi.py').read().split("print('直線ありのレース'")[0])
nb=(P.r_cho6==1)&(P.r_st6<=2)&(P.r_lap1<=2); s=P[nb].copy(); s['pay']=np.where(s.res=='1-6-2',s.payout,0)
h=s.pay[s.pay>0].sort_values(ascending=False); print('的中',len(h),'上位',h.head(6).tolist(),'中央値',h.median())
for k in (0,1,3,5): print(f'上位{k}除外 {(s.pay.sum()-h.head(k).sum())/100/len(s):.3f}')
s['ym']=s.hd.astype(str).str[:6]; g=s.groupby('ym').pay.agg(['size','sum']); g['roi']=g['sum']/100/g['size']; print(g.roi.round(2).to_dict()); print('勝ち月',(g.roi>1).sum(),'/',len(g))
for cb in ['1-6-2','1-6-3','1-6-4','1-6-5','1-2-6','1-3-6','6-1-2']:
    pay=np.where(s.res==cb,s.payout,0); print(cb,f'{pay.sum()/100/len(s):.3f}',(pay>0).sum())
# 条件をずらす（頑健さ）
for nm,m in (('6直線1位だけ ①強',(P.r_cho6==1)&(P.r_lap1<=2)),('6直線2位内・スタ展2内 ①強',(P.r_cho6<=2)&(P.r_st6<=2)&(P.r_lap1<=2)),('6スタ展1位 ①強',(P.r_st6==1)&(P.r_lap1<=2)),('6直線1位・スタ展3内 ①強',(P.r_cho6==1)&(P.r_st6<=3)&(P.r_lap1<=2)),('6伸び ①一周1位',(P.r_cho6==1)&(P.r_st6<=2)&(P.r_lap1==1)),('①強だけ(比較)',P.r_lap1<=2)):
    t=P[m]; pay=np.where(t.res=='1-6-2',t.payout,0); a=t.hd.values<cut
    print(nm,len(t),f'{pay.sum()/100/len(t):.3f} 前半{pay[a].sum()/100/a.sum():.3f} 後半{pay[~a].sum()/100/(~a).sum():.3f}')
# 偽の世界の代わり: 同じ条件で全期間（2025-06前は直線なし）→ 場別の偏り
print(s.groupby('jcd').pay.agg(['size','sum']).assign(roi=lambda d:d['sum']/100/d['size']).sort_values('size',ascending=False).head(8).round(2).to_string())
