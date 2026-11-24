"""外の「伸び」艇（4〜6号艇で直線1位・スタ展2番手以内）がいるとき。オッズ不使用。試した数を数える。"""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
exec(open('scripts/analysis/one_pick.py').read().split("run('公式展示だけ'")[0])
P=X.pivot_table(index=key,columns='lane',values=['r_lap','r_st','r_cho','r_maw','r_ex','r_nw']); P.columns=[f'{a}{b}' for a,b in P.columns]
P=P.reset_index().merge(p,on=key); P=P[P[[f'r_cho{i}' for i in range(1,7)]].notna().sum(1)>=5]
print('直線ありのレース',len(P)); tried=0
def rep(nm,m,cb):
    global tried; tried+=1; s=P[m]; pay=np.where(s.res.values==cb,s.payout.values,0); a=s.hd.values<cut
    if len(s)<80: print(nm,len(s),'少ない'); return
    print(f'{nm} | {len(s)} 的中{(pay>0).mean():.3f} 回収{pay.sum()/100/len(s):.3f} 前半{pay[a].sum()/100/max(a.sum(),1):.3f} 後半{pay[~a].sum()/100/max((~a).sum(),1):.3f}')
for k in (4,5,6):
    nb=(P[f'r_cho{k}']==1)&(P[f'r_st{k}']<=2); s1=P.r_lap1<=2
    rep(f'{k}伸び → {k}-1-2',nb,f'{k}-1-2'); rep(f'{k}伸び → 1-{k}-2',nb,f'1-{k}-2'); rep(f'{k}伸び ①強 → 1-{k}-2',nb&s1,f'1-{k}-2')
    rep(f'{k}伸び ①強 → 1-2-{k}',nb&s1,f'1-2-{k}'); rep(f'{k}伸び ①強 → 1-{k}-3',nb&s1,f'1-{k}-3')
    rep(f'{k}伸び ①弱(一周4位〜) → {k}-1-2',nb&(P.r_lap1>=4),f'{k}-1-2')
    rep(f'{k}伸び ①弱 → {k}-2-1',nb&(P.r_lap1>=4),f'{k}-2-1')
    rep(f'{k}伸び+まわり足3位内 → 1-{k}-2',nb&(P[f'r_maw{k}']<=3),f'1-{k}-2')
print('試した数',tried)
