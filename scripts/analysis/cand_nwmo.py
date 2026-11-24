"""候補: ②③が一周3位内・①が勝率1位・①がモーター2連率1位 → 1-2-3 の月別。"""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
exec(open('scripts/analysis/one_pick.py').read().split("run('公式展示だけ'")[0])
P=X.pivot_table(index=key,columns='lane',values=['r_lap','r_nw','r_mo']); P.columns=[f'{a}{b}' for a,b in P.columns]; P=P.reset_index().merge(p,on=key)
s=P[(P.r_lap2<=3)&(P.r_lap3<=3)&(P.r_nw1==1)&(P.r_mo1==1)].copy()
s['pay']=np.where(s.res=='1-2-3',s.payout,0); s['ym']=s.hd.astype(str).str[:6]
g=s.groupby('ym').agg(n=('pay','size'),hit=('pay',lambda x:(x>0).sum()),pay=('pay','sum'),big=('pay','max'))
g['roi']=g.pay/100/g.n; g['p100']=g.pay-g.n*100; g['cum']=g.p100.cumsum()
for ym,r in g.iterrows(): print(f"{ym[:4]}/{ym[4:]} {int(r.n)} {int(r.hit)} {r.roi:.2f} {int(r.p100):+} {int(r.cum):+} {int(r.big)}")
print(len(s), s.pay.sum()/100/len(s), (s.pay>0).mean(), s.pay[s.pay>0].median())
