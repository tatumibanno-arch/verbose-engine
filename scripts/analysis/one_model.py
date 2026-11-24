"""①頭の1点を、オッズを使わずに学習で選ぶ。候補は 1-a-b の20通り。材料は各艇の7つの順位（一周・まわり足・直線・公式展示・全国勝率・モーター・枠の2連率）とスタ展順位。
学習 2025-06〜2025-12、確かめ 2026-01〜2026-11（学習に使っていない）。各レースで当たる見込みが最大の1点を、見込みが高いときだけ買う。払戻は確定。"""
import numpy as np, pandas as pd, warnings, itertools; warnings.filterwarnings('ignore')
from sklearn.ensemble import HistGradientBoostingClassifier
exec(open('scripts/analysis/one_pick.py').read().split("run('公式展示だけ'")[0])
cols=['r_lap','r_maw','r_cho','r_ex','r_nw','r_mo','r_c2','r_st']; X['sc']=X[cols[:7]].sum(1)
F=X.pivot_table(index=key,columns='lane',values=cols+['sc']); F.columns=[f'{a}_{b}' for a,b in F.columns]; F=F.reset_index().merge(p,on=key)
rows=[]
for a,b_ in itertools.permutations(range(2,7),2):
    d=F[key+['res','payout']].copy(); d['cb']=f'1-{a}-{b_}'; d['a']=a; d['b']=b_
    for cname in cols+['sc']:
        d[f'one_{cname}']=F[f'{cname}_1']; d[f'a_{cname}']=F[f'{cname}_{a}']; d[f'b_{cname}']=F[f'{cname}_{b_}']
    oth=[k for k in range(2,7) if k not in (a,b_)]; d['rest_best_sc']=F[[f'sc_{k}' for k in oth]].min(1); rows.append(d)
D=pd.concat(rows,ignore_index=True); D['hit']=(D.res==D.cb).astype(int); D['pay']=np.where(D.hit==1,D.payout,0)
feat=[c_ for c_ in D.columns if c_.startswith(('one_','a_','b_'))]+['a','b','rest_best_sc']
tr=D[D.hd<'20260101']; te=D[D.hd>='20260101'].copy()
m=HistGradientBoostingClassifier(max_iter=300,learning_rate=.05,max_leaf_nodes=15,min_samples_leaf=300,l2_regularization=1.0,random_state=0).fit(tr[feat].fillna(-1),tr.hit)
te['pr']=m.predict_proba(te[feat].fillna(-1))[:,1]; best=te.loc[te.groupby(key).pr.idxmax()]
R=lambda s: s.pay.sum()/(100*len(s)) if len(s) else np.nan
print('確かめ期間のレース',len(best),'全部買う',round(R(best),3))
for th in (0.06,0.08,0.10,0.12,0.15,0.18):
    s=best[best.pr>=th]; mo=s.groupby(s.hd.str[:6]).apply(R) if len(s) else pd.Series(dtype=float)
    print(f'見込み{th:.2f}以上 | レース {len(s)} 的中 {s.hit.mean() if len(s) else 0:.3f} 見込みの平均 {s.pr.mean() if len(s) else 0:.3f} 回収 {R(s):.3f} 勝ち月 {(mo>1).sum()}/{len(mo)}')
q=best.pr.quantile([.5,.8,.9,.95,.98]).values; lo=-1
for lab,hi in zip(['下位50%','50〜80%','80〜90%','90〜95%','95〜98%'],q): s=best[(best.pr>lo)&(best.pr<=hi)]; print(f'  {lab}: {len(s)} 回収 {R(s):.3f}'); lo=hi
s=best[best.pr>lo]; print(f'  上位2%: {len(s)} 回収 {R(s):.3f} 的中 {s.hit.mean():.3f}')
