"""「凹み」の形: ある艇がスタ展・一周とも悪いとき、その外の艇が攻めて2着に来るか。オッズは使わない。試した数を数える。"""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
exec(open('scripts/analysis/one_pick.py').read().split("run('公式展示だけ'")[0])
P=X.pivot_table(index=key,columns='lane',values=['lap','st','r_lap','r_st','r_ex','r_nw','r_mo','r_maw']); P.columns=[f'{a}{b}' for a,b in P.columns]
P=P.reset_index().merge(p,on=key); P=P[P[[f'r_lap{i}' for i in range(1,7)]].notna().sum(1)>=5]
print('全レース',len(P)); tried=0
def rep(nm,m,cb):
    global tried; tried+=1; s=P[m]; c=cb if isinstance(cb,str) else cb[m.values]
    pay=np.where(s.res.values==c,s.payout.values,0); a=s.hd.values<cut
    if len(s)<80: print(f'{nm} | {len(s)} 少ない'); return
    print(f'{nm} | {len(s)} 的中{(pay>0).mean():.3f} 回収{pay.sum()/100/len(s):.3f} 前半{pay[a].sum()/100/max(a.sum(),1):.3f} 後半{pay[~a].sum()/100/max((~a).sum(),1):.3f}')
strong1=(P.r_lap1<=2)&(P.r_st1<=3)
for k in (2,3):
    hk=(P[f'r_st{k}']>=5)&(P[f'r_lap{k}']>=4)
    o=k+1; o2=k+2
    third=np.where(P[f'r_lap{o2}']<P[f'r_lap{6 if k==3 else 2}'] if False else P[f'r_lap{o2}']<=3, f'{o2}', '2' if k==3 else '3')
    rep(f'{k}凹み・①強い → 1-{o}-{o2}',strong1&hk,f'1-{o}-{o2}')
    rep(f'{k}凹み・①強い・{o}一周3位内 → 1-{o}-{o2}',strong1&hk&(P[f'r_lap{o}']<=3),f'1-{o}-{o2}')
    rep(f'{k}凹み・①強い・{o}スタ展2位内 → 1-{o}-{o2}',strong1&hk&(P[f'r_st{o}']<=2),f'1-{o}-{o2}')
    rep(f'{k}凹み・①強い・{o}一周3位内 → {o}-1-{o2}',strong1&hk&(P[f'r_lap{o}']<=3),f'{o}-1-{o2}')
    # ①相手に残った内側
    other='3' if k==2 else '2'
    rep(f'{k}凹み・①強い → 1-{o}-{other}',strong1&hk,f'1-{o}-{other}')
    rep(f'{k}凹み・①強い・{o}一周3位内 → 1-{o}-{other}',strong1&hk&(P[f'r_lap{o}']<=3),f'1-{o}-{other}')
# ①凹み: ①がスタ展・一周とも悪い → ②③頭
h1=(P.r_st1>=5)&(P.r_lap1>=4)
rep('①凹み → 2-3-1',h1,'2-3-1'); rep('①凹み → 3-2-1',h1,'3-2-1'); rep('①凹み・②一周1位 → 2-1-3',h1&(P.r_lap2==1),'2-1-3')
rep('①凹み・②一周1位 → 2-3-4',h1&(P.r_lap2==1),'2-3-4'); rep('①凹み・③一周1位 → 3-4-2',h1&(P.r_lap3==1),'3-4-2'); rep('①凹み・③一周1位 → 3-2-4',h1&(P.r_lap3==1),'3-2-4')
print('試した数',tried)
