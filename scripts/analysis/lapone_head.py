"""知らせ1と同じレース（①②③が一周1〜3位、3位と4位の差0.4秒以上）で、①が負ける形を拾えるか。結果の内訳と、②や③の頭の1点を条件つきで足す。オッズは使わない。"""
import sqlite3, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
exec(open('scripts/analysis/one_pick.py').read().split("run('公式展示だけ'")[0])
Lp=X[key+['lane','lap']].dropna().sort_values(key+['lap','lane']); Lp['o']=Lp.groupby(key).cumcount()+1
W=Lp.pivot_table(index=key,columns='o',values=['lane','lap'],aggfunc='first'); W.columns=[f'{a}{b}' for a,b in W.columns]; W=W.reset_index()
P=X.pivot_table(index=key,columns='lane',values=['et','st','r_st','r_ex','r_maw','r_nw','r_mo','r_c2']); P.columns=[f'{a}{b}' for a,b in P.columns]; P=P.reset_index()
W=W[(W.lane1==1)&W[['lane2','lane3']].isin([2,3]).all(1)&((W.lap4-W.lap3)>=0.4)].merge(P,on=key).merge(p,on=key)
print('レース',len(W)); print('結果の内訳',W.res.value_counts(normalize=True).head(8).round(3).to_dict())
pk=np.where(W.et2<=W.et3,'1-2-3','1-3-2'); W['base']=np.where(W.res==pk,W.payout,0)
def rep(nm,s,cb):
    pay=np.where(s.res==cb,s.payout,0); print(f'{nm} | レース {len(s)} 的中 {(pay>0).mean():.3f} 回収 {pay.sum()/(100*len(s)):.3f} 前半 {pay[s.hd<cut].sum()/(100*max((s.hd<cut).sum(),1)):.3f} 後半 {pay[s.hd>=cut].sum()/(100*max((s.hd>=cut).sum(),1)):.3f}')
rep('2-1-3 を全部',W,'2-1-3'); rep('3-1-2 を全部',W,'3-1-2')
for nm,m in (('②のスタ展が①より速い',W.st2<W.st1),('②のスタ展が①より0.05以上速い',W.st2<=W.st1-0.05),('①のスタ展が5〜6番手',W.r_st1>=5),('②の展示が①より速い',W.et2<W.et1),('②の一周が①と0.1秒以内',(np.where(W.lane2==2,W.lap2,W.lap3)-W.lap1)<=0.1),('②の勝率が①より上',W.r_nw2<W.r_nw1),('②のまわり足が1位',W.r_maw2==1)):
    s=W[m]; rep(f'2-1-3 ｜ {nm}',s,'2-1-3'); rep(f'   （同じレースの知らせ1の1点）',s,np.where(s.et2<=s.et3,'1-2-3','1-3-2'))
