"""知らせ1/2/3 の発火レース内で、伸び艇（4/5/6）の有無で成績が変わるか。事前仮説12本。オッズ不使用。"""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
src=open('scripts/analysis/three_signals.py').read().split("R=lambda s,col")[0]
exec(src)   # S (sig, cb, res, payout, ...) と X, p, cut, key
P=X.pivot_table(index=key,columns='lane',values=['r_lap','r_st','r_cho']); P.columns=[f'{a}{b}' for a,b in P.columns]; P=P.reset_index()
ok=P[[f'r_cho{i}' for i in range(1,7)]].notna().sum(1)>=5
P=P[ok]
for k in (4,5,6): P[f'nb{k}']=(P[f'r_cho{k}']==1)&(P[f'r_st{k}']<=2)
P['nb6c']=(P.r_cho6==1)
S=S.merge(P[key+['nb4','nb5','nb6','nb6c']],on=key)   # 直線データのある発火のみ
S['ym']=S.hd.str[:6]
def rep(nm,df,tick):
    """df: レース行, tick: 行ごとの買い目リスト関数"""
    global n_try; n_try+=1
    if len(df)<30: print(f'{nm} | n={len(df)} 少ない'); return
    stake=[];pay=[]
    for r in df.itertuples():
        t=tick(r); stake.append(100*len(t)); pay.append(r.payout if r.res in t else 0)
    stake=np.array(stake);pay=np.array(pay,float);a=df.hd.values<cut
    roi=lambda m:pay[m].sum()/stake[m].sum() if stake[m].sum() else np.nan
    h=np.sort(pay[pay>0])[::-1]; tot=pay.sum()
    mo=pd.DataFrame({'ym':df.ym.values,'p':pay,'s':stake}).groupby('ym').sum(); mr=mo.p/mo.s
    print(f'{nm} | n={len(df)} 点/R={stake.mean()/100:.1f} 的中{(pay>0).mean():.3f} ROI {tot/stake.sum():.3f} 前{roi(a):.3f} 後{roi(~a):.3f} 上位1除{(tot-h[:1].sum())/stake.sum():.3f} 上位3除{(tot-h[:3].sum())/stake.sum():.3f} 勝月{(mr>1).sum()}/{len(mr)}')
n_try=0
sp=lambda r:[r.cb]
x6=lambda r:[f'1-6-{x}' for x in (2,3,4,5)]
Sd=S.drop_duplicates(key+['cb'])
for sg in ['知らせ1','知らせ2','知らせ3','全部']:
    D=Sd if sg=='全部' else Sd[Sd.sig==sg]
    print('=====',sg,'発火',len(D),'(直線あり)')
    rep('H1a 知らせ買い目 伸び6なし',D[~D.nb6],sp); rep('H1b 知らせ買い目 6伸びあり',D[D.nb6],sp)
    rep('H2 6伸び: 1-6-x(2-5)4点に置換',D[D.nb6],x6)
    rep('H3 6伸び: 知らせ+1-6-2 追加',D[D.nb6],lambda r:list({r.cb,'1-6-2'}))
    rep('H4 6伸び: 1-6-2 のみ',D[D.nb6],lambda r:['1-6-2'])
    rep('H5 6伸び: 1-2-6,1-3-6',D[D.nb6],lambda r:['1-2-6','1-3-6'])
    rep('H6a 5伸びなし',D[~D.nb5],sp); rep('H6b 5伸びあり',D[D.nb5],sp)
    rep('H7a 4伸びなし',D[~D.nb4],sp); rep('H7b 4伸びあり',D[D.nb4],sp)
    rep('H8 4/5/6いずれか伸び→1-x-y(x=伸び艇)',D[D.nb4|D.nb5|D.nb6],lambda r:[f'1-{k}-{y}' for k in (4,5,6) if getattr(r,f'nb{k}') for y in (2,3,4,5,6) if y!=k])
    rep('H10 6直線1位のみ: 1-6-x',D[D.nb6c],x6)
    rep('H12 6伸び: 知らせ+1-6-x(重複除く)',D[D.nb6],lambda r:list(set([r.cb]+x6(r))))
# H11 参照: 全レース 6伸び&①lap<=2 の 1-6-x と、知らせ外
allr=X[key].drop_duplicates().merge(P,on=key).merge(p,on=key); allr['ym']=allr.hd.str[:6]
base=allr[allr.nb6&(allr.r_lap1<=2)]
print('===== 参照'); rep('H11a 全レース 6伸び&①一周2位内 1-6-x',base,x6)
fired=Sd[key].drop_duplicates()
nf=base.merge(fired,on=key,how='left',indicator=True); rep('H11b 同条件のうち知らせ発火あり',nf[nf._merge=='both'],x6); rep('H11c 同条件のうち知らせ発火なし',nf[nf._merge=='left_only'],x6)
print('試した数(rep)',n_try)
