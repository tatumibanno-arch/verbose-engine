"""選手スタイル（過去365日・当日より前、results の進入別成績）× ①の強さ × 公式展示順位。オッズ不使用。仮説15本を事前に固定。"""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
exec(open('scripts/analysis/one_pick.py').read().split("run('公式展示だけ'")[0])
S='/tmp/claude-0/-home-user-verbose-engine/968008a0-6ecf-5641-a3ff-006912b41722/scratchpad/'
R=pd.read_parquet(S+'res.parquet'); R['rn']=R.rn.astype(int); R['cs']=R.cs.astype(int)
# race-level: course1 boat's position
c1=R[R.cs==1][key+['pos']].rename(columns={'pos':'p1'}).drop_duplicates(key)
R=R.merge(c1,on=key,how='left')
W=R[R.pos==1]
R['d']=pd.to_datetime(R.hd)
R['w']=(R.pos==1).astype(int)
R['w_mk']=((R.pos==1)&((R.p1>=3)|R.p1.isna())).astype(int)   # 勝って①が3着以下=まくり系の代理
R['w_sa']=((R.pos==1)&(R.p1==2)).astype(int)                  # 勝って①が2着=差し系の代理
R['t2']=(R.pos<=2).astype(int)
feats={}
for k in (1,2,3,4):
    s=R[R.cs==k].groupby(['rn','d']).agg(n=('w','size'),w=('w','sum'),mk=('w_mk','sum'),sa=('w_sa','sum'),t2=('t2','sum')).reset_index().sort_values('d')
    for q in ('n','w','mk','sa','t2'): s['c'+q]=s.groupby('rn')[q].cumsum()
    feats[k]=s
def hist(k,rn_d):  # rn_d: DataFrame rn,d  -> 365日の合計
    s=feats[k]; out=rn_d.copy().sort_values('d')
    cols=['cn','cw','cmk','csa','ct2']
    a=pd.merge_asof(out,s[['rn','d']+cols].sort_values('d'),on='d',by='rn',allow_exact_matches=False)
    b=pd.merge_asof(out.assign(d=out.d-pd.Timedelta(days=365)),s[['rn','d']+cols].sort_values('d'),on='d',by='rn',allow_exact_matches=False)
    r=out[['rn','d']].copy()
    for q in cols: r[q[1:]+str(k)]=(a[q].fillna(0).values-b[q].fillna(0).values)
    return r
# レース×艇番の選手
L=R[R.lane.isin([1,2,3,4])][key+['lane','rn','d']].drop_duplicates(key+['lane'])
Z=None
for k in (1,2,3,4):
    l=L[L.lane==k][key+['rn','d']]; h=hist(k,l[['rn','d']].drop_duplicates())
    l=l.merge(h,on=['rn','d'],how='left'); l=l.drop(columns=['rn','d'])
    Z=l if Z is None else Z.merge(l,on=key,how='outer')
P=X.pivot_table(index=key,columns='lane',values=['r_ex','r_lap']); P.columns=[f'{a}{b}' for a,b in P.columns]; P=P.reset_index()
P=P.merge(Z,on=key,how='inner').merge(p,on=key)
P=P[P.hd>='20250601']
# 率
esc=P.w1/P.n1.where(P.n1>0); P['esc']=np.where(P.n1>=15,esc,np.nan)
P['mk2']=np.where(P.n2>=10,P.mk2/P.n2.where(P.n2>0),np.nan); P['sa2']=np.where(P.n2>=10,P.sa2/P.n2.where(P.n2>0),np.nan)
P['w3']=np.where(P.n3>=10,P.w3/P.n3.where(P.n3>0),np.nan); P['w4']=np.where(P.n4>=10,P.w4/P.n4.where(P.n4>0),np.nan)
print('レース',len(P)); print(P[['esc','mk2','sa2','w3','w4']].describe(percentiles=[.25,.5,.75,.9]).round(3).to_string())
q=lambda c,x: P[c].quantile(x)
S1=P.esc>=q('esc',.67); W1=P.esc<=q('esc',.33)
M2=P.mk2>=q('mk2',.75); A2=P.sa2>=q('sa2',.75); M3=P.w3>=q('w3',.75); K4=P.w4>=q('w4',.75)
print('閾値 esc67%',round(q('esc',.67),3),'esc33%',round(q('esc',.33),3),'mk2 75%',round(q('mk2',.75),3),'sa2 75%',round(q('sa2',.75),3),'w3 75%',round(q('w3',.75),3),'w4 75%',round(q('w4',.75),3))
H=[('H1 ①強&②まくり型 1-4-2',S1&M2,'1-4-2'),('H2 ①強&②まくり型 1-4-3',S1&M2,'1-4-3'),
('H3 ①強&②まくり&④カド 1-4-2',S1&M2&K4,'1-4-2'),('H4 ①強&②まくり&④カド 1-4-3',S1&M2&K4,'1-4-3'),
('H5 ①強&②差し型 1-2-3',S1&A2,'1-2-3'),('H6 ①強&②差し型 1-2-4',S1&A2,'1-2-4'),
('H7 ①強&③まくり型 1-3-2',S1&M3,'1-3-2'),('H8 ①強&③まくり型 1-3-4',S1&M3,'1-3-4'),
('H9 ①弱&②まくり型 2-1-3',W1&M2,'2-1-3'),('H10 ①弱&③まくり型 3-1-2',W1&M3,'3-1-2'),
('H11 ①弱&④カド 4-1-2',W1&K4,'4-1-2'),
('H12 ①強&②まくり&展示④≤2 1-4-2',S1&M2&(P.r_ex4<=2),'1-4-2'),
('H13 ①強&③まくり&展示③≤2 1-3-2',S1&M3&(P.r_ex3<=2),'1-3-2'),
('H14 ①弱&④カド&展示④1位 4-1-2',W1&K4&(P.r_ex4<=1.5),'4-1-2'),
('H15 ①強&②差し&展示②≤2 1-2-3',S1&A2&(P.r_ex2<=2),'1-2-3')]
print('仮説数',len(H))
def stat(m,cb):
    s=P[m]; pay=np.where(s.res.values==cb,s.payout.values,0); n=len(s)
    if n<30: return None
    a=s.hd.values<cut; pc=np.sort(pay)[::-1]; roi=lambda v,k: v.sum()/100/k if k else np.nan
    mo=pd.Series(pay).groupby(s.hd.str[:6].values).sum()/(100*s.groupby(s.hd.str[:6]).size())
    return n,(pay>0).mean(),roi(pay,n),roi(pay[a],a.sum()),roi(pay[~a],(~a).sum()),(pc[1:].sum()/100/n),(pc[3:].sum()/100/n),f'{(mo>1).sum()}/{len(mo)}',(pay>0).sum()
print('名前 | n 的中率 ROI 前半 後半 除top1 除top3 勝月 的中数')
for nm,m,cb in H:
    r=stat(m.fillna(False),cb); 
    if r is None: print(nm,'n<30'); continue
    b=stat(pd.Series(True,index=P.index),cb)
    print(f'{nm} | n{r[0]} 的中{r[1]:.3f} ROI{r[2]:.3f} 前{r[3]:.3f} 後{r[4]:.3f} 除1{r[5]:.3f} 除3{r[6]:.3f} 勝月{r[7]} 的中{r[8]} || 同券全レースROI{b[2]:.3f}')
