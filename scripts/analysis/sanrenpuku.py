"""3連複の回収率。(1) 22年分の確定払戻で、組ごと・条件ごと。(2) 2025-03〜は3連単の締切時オッズから3連複の売れ方を推定し、
売れ方の帯ごとに実際の3連複払戻で回収率を出す（3連複のオッズそのものは持っていない。締切時オッズは締切前とは違う）。"""
import sqlite3, sys, itertools, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
c=sqlite3.connect('data/boatrace.db'); key=['hd','jcd','rno']
p=pd.read_sql("select hd,jcd,rno,combination cb,payout from payouts where ticket_type='sanrenfuku' and hd>='20050101'",c).drop_duplicates(key)
p['s']=p.cb.str.replace('=','-').str.split('-').apply(lambda x:''.join(sorted(x)))
if sys.argv[1]=='long':
    n=len(p); print('3連複 レース',n,'平均払戻',round(p.payout.mean()),'中央値',p.payout.median())
    g=p.groupby('s').payout.agg(['size','sum']); g['出現率']=(g['size']/n).round(4); g['回収率']=(g['sum']/(100*n)).round(3); print(g.sort_values('出現率',ascending=False)[['出現率','回収率']].T.to_string())
    e=pd.read_sql("select hd,jcd,rno,lane,grade from entries where hd>='20050101'",c); e['A']=e.grade.isin(['A1','A2']).astype(int)
    w=e.pivot_table(index=key,columns='lane',values='A'); w.columns=[f'a{k}' for k in w.columns]; d=p.merge(w.reset_index(),on=key).dropna()
    def rep(nm,s,combo): print(f'{nm} | レース {len(s)} 的中 {(s.s==combo).mean():.3f} 回収 {s[s.s==combo].payout.sum()/(100*len(s)):.3f} | 2019年〜 {s[(s.s==combo)&(s.hd>="20190101")].payout.sum()/(100*max((s.hd>="20190101").sum(),1)):.3f}')
    rep('1=2=3 全レース',d,'123'); rep('1=2=3 ①②③がA級・④⑤⑥がB級',d[(d.a1+d.a2+d.a3==3)&(d.a4+d.a5+d.a6==0)],'123'); rep('1=2=3 ①②③がA級',d[(d.a1+d.a2+d.a3==3)],'123')
    rep('1=2=4 ①②④がA級・他B級',d[(d.a1+d.a2+d.a4==3)&(d.a3+d.a5+d.a6==0)],'124'); rep('1=3=4 ①③④がA級・他B級',d[(d.a1+d.a3+d.a4==3)&(d.a2+d.a5+d.a6==0)],'134')
    d['na']=d[[f'a{k}' for k in range(1,7)]].sum(1); x=d[d.na==3].copy(); x['pick']=x.apply(lambda r:''.join(str(k) for k in range(1,7) if r[f'a{k}']==1),axis=1)
    print(f'A級がちょうど3人のレースで、その3人の3連複 | レース {len(x)} 的中 {(x.s==x.pick).mean():.3f} 回収 {x[x.s==x.pick].payout.sum()/(100*len(x)):.3f} | 2019年〜 {x[(x.s==x.pick)&(x.hd>="20190101")].payout.sum()/(100*(x.hd>="20190101").sum()):.3f}')
else:
    rows=[]
    for m in pd.period_range('2025-03','2026-11',freq='M'):
        a=m.strftime('%Y%m01'); b=m.strftime('%Y%m31')
        o=pd.read_sql(f"select hd,jcd,rno,combination cb,odds from odds where ticket_type='sanrensho' and hd between '{a}' and '{b}' and odds>0",c)
        if o.empty: continue
        o['inv']=1/o.odds; g=o.groupby(key).inv; o['tot']=g.transform('sum'); o['n']=g.transform('size'); o=o[o.n==120]; o['s']=o.cb.str.split('-').apply(lambda x:''.join(sorted(x)))
        f=o.groupby(key+['s']).agg(q=('inv','sum'),tot=('tot','first')).reset_index(); f['q']=f.q/f.tot; f['rk']=f.groupby(key).q.rank(ascending=False,method='first'); rows.append(f); print(m,flush=True)
    F=pd.concat(rows).merge(p[key+['s','payout']].rename(columns={'s':'res'}),on=key); F['pay']=np.where(F.s==F.res,F.payout,0)
    F[F.q>=.2].to_parquet('/tmp/claude-0/-home-user-verbose-engine/968008a0-6ecf-5641-a3ff-006912b41722/scratchpad/n2/s3f.parquet')
    R=lambda s: round(s.pay.sum()/(100*len(s)),3) if len(s) else np.nan
    def rep(nm,s): print(f'{nm} | 点 {len(s)} 的中 {(s.pay>0).mean():.3f} 回収 {R(s)} 前半 {R(s[s.hd<"20260201"])} 後半 {R(s[s.hd>="20260201"])}')
    print('3連単の売れ方から見た3連複の確率の帯ごと（全20組）')
    for lo,hi in ((.6,1),(.5,.6),(.4,.5),(.3,.4),(.2,.3),(.1,.2),(.05,.1),(.02,.05),(0,.02)): rep(f'確率 {lo}〜{hi}',F[(F.q>=lo)&(F.q<hi)])
    print('人気順'); 
    for k in (1,2,3): rep(f'{k}番人気',F[F.rk==k])
    rep('1番人気で確率50%以上',F[(F.rk==1)&(F.q>=.5)]); rep('1番人気が1=2=3で確率40%以上',F[(F.rk==1)&(F.s=='123')&(F.q>=.4)]); rep('上位2組で確率70%以上のレースの上位2組',F[(F.rk<=2)&(F.groupby(key).q.transform(lambda x:x.nlargest(2).sum())>=.7)])
