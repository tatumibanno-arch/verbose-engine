import sqlite3, pandas as pd, numpy as np
c=sqlite3.connect('data/boatrace.db')
F='20240101'
e=pd.read_sql(f"select e.hd,e.jcd,e.rno,e.lane,e.motor_no,r.position from entries e join results r using(hd,jcd,rno,lane) where e.hd>='{F}'",c)
p=pd.read_sql(f"select hd,jcd,rno,ticket_type,combination,payout from payouts where hd>='{F}' and ticket_type in ('tansho','fukusho')",c)
print(len(e),len(p),p.ticket_type.value_counts().to_dict())
e['pos']=pd.to_numeric(e.position,errors='coerce')
e['d']=pd.to_datetime(e.hd)
# 節: 同じ場で日付が連続する塊
days=e[['jcd','d']].drop_duplicates().sort_values(['jcd','d'])
days['new']=(days.groupby('jcd').d.diff().dt.days!=1).astype(int)
days['meet']=days.groupby('jcd').new.cumsum()
days['day_no']=days.groupby(['jcd','meet']).cumcount()+1
e=e.merge(days[['jcd','d','meet','day_no']],on=['jcd','d'])
e=e.sort_values(['jcd','meet','motor_no','d','rno'])
g=e.groupby(['jcd','meet','motor_no'])
e['top2']=(e.pos<=2).astype(float); e['win']=(e.pos==1).astype(float)
# 前日までの節間成績（当日分は含めない）
daily=e.groupby(['jcd','meet','motor_no','day_no']).agg(n=('top2','size'),t2=('top2','sum'),w=('win','sum')).reset_index().sort_values(['jcd','meet','motor_no','day_no'])
gg=daily.groupby(['jcd','meet','motor_no'])
daily['pn']=gg.n.cumsum()-daily.n; daily['pt2']=gg.t2.cumsum()-daily.t2; daily['pw']=gg.w.cumsum()-daily.w
e=e.merge(daily[['jcd','meet','motor_no','day_no','pn','pt2','pw']],on=['jcd','meet','motor_no','day_no'])
t=p[p.ticket_type=='tansho'].copy(); t['lane']=pd.to_numeric(t.combination,errors='coerce')
e=e.merge(t[['hd','jcd','rno','lane','payout']],on=['hd','jcd','rno','lane'],how='left')
e['ret']=e.payout.fillna(0)/100
has=e.merge(t[['hd','jcd','rno']].drop_duplicates(),on=['hd','jcd','rno'])
def show(name,m):
    x=has[m]; 
    if len(x)==0: print(name,'0'); return
    x=x.assign(ym=x.hd.str[:6]); mo=x.groupby('ym').ret.mean()
    print(f"{name}: {len(x)}走 1着{x.win.mean():.1%} 単勝回収{x.ret.mean():.2f} 勝ち月{(mo>1).sum()}/{len(mo)}")
has['r2']=has.pt2/has.pn.where(has.pn>0)
show('全体',has.pn>=0)
for lo in [3,4]:
  for th in [.67,.8,1.0]:
    show(f'節間{lo}走以上・2連率{th:.0%}以上',(has.pn>=lo)&(has.r2>=th))
    show(f'  うち2〜6号艇',(has.pn>=lo)&(has.r2>=th)&(has.lane>=2))
    show(f'  うち4〜6号艇',(has.pn>=lo)&(has.r2>=th)&(has.lane>=4))
    show(f'  うち2日目のみ・2〜6号艇',(has.pn>=lo)&(has.r2>=th)&(has.lane>=2)&(has.day_no==2))
show('節間3走以上・2連率0%（逆）',(has.pn>=3)&(has.r2==0))
show('  うち1号艇',(has.pn>=3)&(has.r2==0)&(has.lane==1))
show('節間2連率67%以上の1号艇',(has.pn>=3)&(has.r2>=.67)&(has.lane==1))
