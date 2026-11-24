import pandas as pd, numpy as np
exec(open('scripts/analysis/one_pick.py').read().split("run('公式展示だけ'")[0])
W=pd.read_pickle('/tmp/claude-0/-home-user-verbose-engine/968008a0-6ecf-5641-a3ff-006912b41722/scratchpad/wind.pkl')
P=X.pivot_table(index=key,columns='lane',values=['r_lap','r_st','r_ex']); P.columns=[f'{a}{b}' for a,b in P.columns]
P=P.reset_index().merge(p,on=key).merge(W,on=key); P=P[P.ws.notna()]
P['w1']=P.res.str[0]=='1'
# venue wind phase from first half only (ws>=3): fit w1 ~ cos(theta-phi), 16-compass codes 1..16
ph={}
f=P[(P.hd<cut)&(P.ws>=3)&(P.wd<=16)]
for j,g in f.groupby('jcd'):
    th=(g.wd.values-1)*2*np.pi/16; y=g.w1.values-g.w1.mean()
    a=(y*np.cos(th)).sum(); b=(y*np.sin(th)).sum(); ph[j]=(np.arctan2(b,a),np.hypot(a,b)/len(g))
def zone(r):
    if r.wd>16 or r.jcd not in ph: return 'calm'
    d=((r.wd-1)*2*np.pi/16-ph[r.jcd][0]+np.pi)%(2*np.pi)-np.pi
    return 'tail' if abs(d)<=np.pi/4+1e-9 else ('head' if abs(d)>=3*np.pi/4-1e-9 else 'side')
P['z']=P.apply(zone,axis=1)
for z,g in P[P.ws>=3].groupby(['z',P.hd<cut]): print(z,len(g),'①1着率',round(g.w1.mean(),3))
tried=0
def rep(nm,m,tick):
    global tried; tried+=1; s=P[m]; 
    if len(s)<100: print(nm,len(s),'少'); return
    tk=tick if isinstance(tick,list) else [tick]
    pay=np.isin(s.res.values,tk)*s.payout.values; cost=len(tk)
    a=(s.hd<cut).values; mo=s.hd.astype(str).str[:6].values
    def r(x,n): return x.sum()/100/(n*cost) if n else float('nan')
    srt=np.sort(pay)[::-1]; t1=pay.sum()-srt[:1].sum(); t3=pay.sum()-srt[:3].sum()
    m_=pd.DataFrame({'m':mo,'p':pay}).groupby('m').p.agg(['sum','count']); wm=((m_['sum']/100/(m_['count']*cost))>1).sum()
    print(f'{nm} | n{len(s)} hit{(pay>0).mean():.3f} ROI{r(pay,len(s)):.3f} 前{r(pay[a],a.sum()):.3f} 後{r(pay[~a],(~a).sum()):.3f} -top1 {t1/100/(len(s)*cost):.3f} -top3 {t3/100/(len(s)*cost):.3f} 勝月{wm}/{len(m_)}')
hi=P.ws>=4; t=P.z=='tail'; h=P.z=='head'
rep('H1 強追い風 1-2-3',hi&t,'1-2-3')
rep('H2 強追い風 1-2-3,1-3-2',hi&t,['1-2-3','1-3-2'])
rep('H3 強向い風 2-1-3',hi&h,'2-1-3')
rep('H4 強向い風&①lap4位下 2-1-3',hi&h&(P.r_lap1>=4),'2-1-3')
rep('H5 強向い風&②lap1位 2-1-3',hi&h&(P.r_lap2==1),'2-1-3')
rep('H6 強向い風&①lap1位 1-2-3',hi&h&(P.r_lap1==1),'1-2-3')
rep('H7 強向い風&①lap1位 1-3-2',hi&h&(P.r_lap1==1),'1-3-2')
rep('H8 強追い風&①lap4位下 1-2-3',hi&t&(P.r_lap1>=4),'1-2-3')
rep('H9 強向い風&④st1位 4-1-2',hi&h&(P.r_st4==1),'4-1-2')
rep('H10 波5cm以上 2-1-3',P.wave>=5,'2-1-3')
rep('H11 波5cm以上&①lap1位 1-2-3',(P.wave>=5)&(P.r_lap1==1),'1-2-3')
rep('H12 無風波1以下&①lap1位 1-2-3',(P.ws<=1)&(P.wave<=1)&(P.r_lap1==1),'1-2-3')
rep('H13 風6m以上 2-1-3',P.ws>=6,'2-1-3')
rep('H14 強追い風&③st1位 1-3-2',hi&t&(P.r_st3==1),'1-3-2')
rep('H15 波4以上&強風4以上&①lap4位下 2-1-3',(P.wave>=4)&hi&(P.r_lap1>=4),'2-1-3')
print('tried',tried)
