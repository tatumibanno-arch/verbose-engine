import sqlite3,pandas as pd
con=sqlite3.connect('data/boatrace.db')
b=pd.read_sql("select hd,jcd,rno,lane,exhibition_time ex,start_course c from beforeinfo where exhibition_time>6 and exhibition_time<7.5",con)
r=pd.read_sql("select hd,jcd,rno,lane,position from results where hd>='20250101'",con)
p=pd.read_sql("select hd,jcd,rno,combination,payout from payouts where ticket_type='tansho' and hd>='20250101'",con)
d=b.merge(r,on=['hd','jcd','rno','lane'])
d['position']=pd.to_numeric(d.position,errors='coerce')
g=d.groupby(['hd','jcd','rno'])
d=d[g.lane.transform('count')==6]
d['rk']=d.groupby(['hd','jcd','rno']).ex.rank(method='min')
d['win']=d.position==1; d['top3']=d.position<=3
p['lane']=pd.to_numeric(p.combination,errors='coerce')
d=d.merge(p[['hd','jcd','rno','lane','payout']],on=['hd','jcd','rno','lane'],how='left'); d['pay']=d.payout.fillna(0)
print('races',d.groupby(['hd','jcd','rno']).ngroups, d.hd.min(), d.hd.max())
print('\n展示順位ごと（全艇）'); print(d.groupby('rk').agg(n=('win','size'),win=('win','mean'),top3=('top3','mean'),tansho_roi=('pay',lambda x:x.mean()/100)).round(3))
print('\n枠×展示順位の1着率'); print(d[d.rk.isin([1,2,3,4,5,6])].pivot_table(index='lane',columns='rk',values='win',aggfunc='mean').round(3))
print('\n枠×展示順位の3連対率'); print(d.pivot_table(index='lane',columns='rk',values='top3',aggfunc='mean').round(3))
print('\n枠×展示順位の単勝回収率'); print(d.pivot_table(index='lane',columns='rk',values='pay',aggfunc='mean').div(100).round(2))
d['m']=d.jcd=='15'
print('\n丸亀だけ 展示順位ごと'); print(d[d.m].groupby('rk').agg(n=('win','size'),win=('win','mean'),top3=('top3','mean')).round(3))
