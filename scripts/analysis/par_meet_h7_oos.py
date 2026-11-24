"""par_meet_h7_oos: H7固定定義を過去期間で検証（再調整なし）。final 予選日, ①gap<=-8, |②gap|<=6, 2-1-3 & 2-1-4。"""
import sqlite3, json, sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
a,z=sys.argv[1],sys.argv[2]; key=['hd','jcd','rno']
c=sqlite3.connect('data/boatrace.db')
e=pd.read_sql(f"select hd,jcd,rno,lane,racer_no,race_name,data from entries where hd>='{a}' and hd<='{z}'",c)
e['sr']=e.pop('data').map(lambda s: json.loads(s).get('series_results') or '')
print('rows',len(e),'sr非空率',round((e.sr!='').mean(),3))
P={'1':10,'2':8,'3':6,'4':4,'5':2,'6':1}
e['pts']=e.sr.map(lambda s: sum(P.get(ch,0) for ch in s)); e['d']=pd.to_datetime(e.hd)
sf=e[e.race_name.str.contains('準優')].groupby(['jcd','hd']).racer_no.nunique().rename('nsf').reset_index()
sf['d']=pd.to_datetime(sf.hd)-pd.Timedelta(days=1); e=e.merge(sf[['jcd','d','nsf']],on=['jcd','d'],how='left')
e['last']=e.nsf.notna()&e.race_name.str.contains('予選'); e=e[e['last']]
L=e.sort_values(key).drop_duplicates(['hd','jcd','racer_no'])
cu=L.groupby(['hd','jcd']).apply(lambda g: np.sort(g.pts.values)[::-1][min(int(g.nsf.iloc[0]),len(g))-1]).rename('cut').reset_index()
e=e.merge(cu,on=['hd','jcd']); e['gap']=e.pts-e.cut
w=e.pivot_table(index=key,columns='lane',values='gap',aggfunc='first').reset_index()
p=pd.read_sql(f"select hd,jcd,rno,combination res,payout from payouts where ticket_type='sanrensho' and hd>='{a}' and hd<='{z}'",c).drop_duplicates(key)
R=w.merge(p,on=key); print('final-day races',len(R))
def rep(nm,m):
    s=R[m]; pay=np.where(s.res=='2-1-3',s.payout,0)+np.where(s.res=='2-1-4',s.payout,0); st=200.0
    if len(s)==0: print(nm,'n=0'); return
    o=np.sort(pay)[::-1]; tot=pay.sum()
    mo=pd.DataFrame({'m':s.hd.str[:6].values,'p':pay}).groupby('m').p.agg(['sum','size']); mr=mo['sum']/(200*mo['size'])
    yr=pd.DataFrame({'y':s.hd.str[:4].values,'p':pay}).groupby('y').p.agg(['sum','size']); yr=(yr['sum']/(200*yr['size'])).round(3).to_dict()
    n=len(s); print(f'{nm}: n={n} hit={(pay>0).mean():.4f} ROI={tot/(st*n):.3f} 年別={yr} ex1={(tot-o[:1].sum())/(st*n):.3f} ex3={(tot-o[:3].sum())/(st*n):.3f} ex5={(tot-o[:5].sum())/(st*n):.3f} 勝ち月={(mr>1).sum()}/{len(mr)}')
rep('H7固定',((R[1]<=-8)&(R[2].abs()<=6)).values)
rep('基準:最終日全レース',np.ones(len(R),bool))
rep('参考:①圏外のみ',(R[1]<=-8).values)
