"""3連単の「オッズ帯ごとの回収率」と、1番人気の組を条件で絞ったときの回収率。
注意: 締切時オッズで帯を決める（締切前とは違う）。払戻は確定値。"""
import sqlite3, sys, numpy as np, pandas as pd
D='/tmp/claude-0/-home-user-verbose-engine/968008a0-6ecf-5641-a3ff-006912b41722/scratchpad/n2/'
c=sqlite3.connect('data/boatrace.db')
bins=[0,3,4,5,6,8,10,15,20,30,50,100,200,500,1e9]
if sys.argv[1]=='build':
    agg=[]; fav=[]
    p=pd.read_sql("select hd,jcd,rno,combination res,payout from payouts where ticket_type='sanrensho' and hd>='20250301'",c).drop_duplicates(['hd','jcd','rno'])
    for m in pd.period_range('2025-03','2026-11',freq='M'):
        a=m.strftime('%Y%m01'); b=m.strftime('%Y%m31')
        o=pd.read_sql(f"select hd,jcd,rno,combination cb,odds from odds where ticket_type='sanrensho' and hd between '{a}' and '{b}' and odds>0",c)
        if o.empty: continue
        o=o.merge(p,on=['hd','jcd','rno']); o['n']=o.groupby(['hd','jcd','rno']).cb.transform('size'); o=o[o.n>=110]
        o['hit']=o.cb==o.res; o['bk']=pd.cut(o.odds,bins,labels=[f'{i:02d}:{int(bins[i])}〜{int(min(bins[i+1],99999))}' for i in range(len(bins)-1)]).astype(str); o['rk']=o.groupby(['hd','jcd','rno']).odds.rank(method='first')
        g=o.groupby('bk',observed=True).agg(n=('hit','size'),pay=('payout',lambda s:0)); g['pay']=o[o.hit].groupby('bk',observed=True).payout.sum(); g['ym']=str(m); agg.append(g.reset_index())
        o['inv']=1/o.odds; tot=o.groupby(['hd','jcd','rno']).inv.sum().rename('tot'); h1=o[o.cb.str[0]=='1'].groupby(['hd','jcd','rno']).inv.sum().rename('h1')
        f=o[o.rk<=3][['hd','jcd','rno','cb','odds','rk','hit','payout']].merge(tot.reset_index(),on=['hd','jcd','rno']).merge(h1.reset_index(),on=['hd','jcd','rno'],how='left'); fav.append(f); print(m,flush=True)
    pd.concat(agg).to_parquet(D+'fav_bk.parquet'); pd.concat(fav).to_parquet(D+'fav_top.parquet')
else:
    g=pd.read_parquet(D+'fav_bk.parquet'); t=g.groupby('bk',observed=True)[['n','pay']].sum(); t['回収率']=(t.pay/(t.n*100)).round(3)
    for nm,s in (('前半',g[g.ym<'2026-04']),('後半',g[g.ym>='2026-04'])):
        x=s.groupby('bk',observed=True)[['n','pay']].sum(); t[nm]=(x.pay/(x.n*100)).round(3)
    print('オッズ帯ごとの回収率（3連単、全組を100円ずつ）'); print(t[['n','回収率','前半','後半']].to_string())
    f=pd.read_parquet(D+'fav_top.parquet'); f['p1']=f.h1/f.tot; f['pay']=np.where(f.hit,f.payout,0)
    b=pd.read_sql("select hd,jcd,rno,lane,exhibition_time et,start_course sc from beforeinfo where hd>='20250401'",c); b['et']=pd.to_numeric(b.et,errors='coerce'); b['er']=b.groupby(['hd','jcd','rno']).et.rank(method='min')
    b['waku']=b.sc==b.lane; w=b.groupby(['hd','jcd','rno']).waku.all().rename('waku').reset_index()
    for k in (1,2,3):
        f=f.merge(b[b.lane==k][['hd','jcd','rno','er']].rename(columns={'er':f'er{k}'}),on=['hd','jcd','rno'],how='left')
    f=f.merge(w,on=['hd','jcd','rno'],how='left')
    def t2(nm,s):
        o=[nm,'n',len(s),'的中',round(s.hit.mean(),3),'平均オッズ',round(s.odds.mean(),1),'回収',round(s.pay.sum()/(100*len(s)),3)]
        for h,q in (('前半',s[s.hd<'20260401']),('後半',s[s.hd>='20260401'])): o+=[h,len(q),round(q.pay.sum()/(100*max(len(q),1)),3)]
        print(*o)
    f1=f[f.rk==1]; print('\n1番人気の組だけ買う'); t2('全部',f1)
    for lo,hi in ((0,3),(3,4),(4,5),(5,7),(7,10),(10,99)): t2(f'1番人気のオッズ {lo}〜{hi}',f1[(f1.odds>=lo)&(f1.odds<hi)])
    for lo,hi in ((0,.4),(.4,.6),(.6,.7),(.7,.8),(.8,1)): t2(f'①頭の売れ方 {lo}〜{hi}',f1[(f1.p1>=lo)&(f1.p1<hi)])
    a=f1[f1.cb.str[0]=='1']; t2('1番人気が①頭',a); t2('　＋①展示1位',a[a.er1==1]); t2('　＋①展示1〜2位＆枠なり',a[(a.er1<=2)&(a.waku==True)]); t2('　＋①展示5〜6位',a[a.er1>=5])
    t2('　＋①頭80%以上＆①展示1位',a[(a.p1>=.8)&(a.er1==1)]); t2('　＋①頭80%以上＆オッズ4倍未満',a[(a.p1>=.8)&(a.odds<4)])
    a['er2nd']=[getattr(r,f"er{r.cb[2]}",np.nan) if r.cb[2] in '23' else np.nan for r in a.itertuples()]
    t2('　＋2着の艇（②か③）も展示2位以内',a[(a.er1<=2)&(a.er2nd<=2)])
    print('\n1〜3番人気の3点'); t2('全部',f); t2('①頭80%以上',f[f.p1>=.8]); t2('①頭80%以上＆①展示1位',f[(f.p1>=.8)&(f.er1==1)])
    jc=f1.groupby('jcd').apply(lambda s:pd.Series({'n':len(s),'roi':round(s.pay.sum()/(100*len(s)),3),'前半':round(s[s.hd<'20260401'].pay.sum()/(100*max((s.hd<'20260401').sum(),1)),3),'後半':round(s[s.hd>='20260401'].pay.sum()/(100*max((s.hd>='20260401').sum(),1)),3)})); print('\n場ごとの1番人気'); print(jc.sort_values('roi',ascending=False).head(6).to_string())
