"""①が逃げて②が3着以内に来ない形（1-345-345 の6点）を、締切時オッズ・展示で検証する。
注意: オッズは締切時（確定に近い）。締切前に見えるオッズとは違う。
使い方: python scripts/analysis/in_nige_no2.py build|report"""
import sqlite3, sys, json, itertools, numpy as np, pandas as pd
OUT='/tmp/claude-0/-home-user-verbose-engine/968008a0-6ecf-5641-a3ff-006912b41722/scratchpad/n2/races.parquet'
SIX=[f"1-{a}-{b}" for a,b in itertools.permutations([3,4,5],2)]
def build():
    c=sqlite3.connect('data/boatrace.db'); rows=[]
    months=pd.period_range('2025-04','2026-11',freq='M')
    for m in months:
        a=m.strftime('%Y%m01'); b=m.strftime('%Y%m31')
        o=pd.read_sql(f"select hd,jcd,rno,combination,odds from odds where ticket_type='sanrensho' and hd between '{a}' and '{b}' and odds>0",c)
        if o.empty: continue
        o['inv']=1/o.odds; o['h']=o.combination.str[0]; o['has2']=o.combination.str[2:].str.contains('2')
        g=o.groupby(['hd','jcd','rno'])
        r=pd.DataFrame({'tot':g.inv.sum(),'n':g.size()})
        r['p1']=o[o.h=='1'].groupby(['hd','jcd','rno']).inv.sum()/r.tot
        r['p1_2']=o[(o.h=='1')&o.has2].groupby(['hd','jcd','rno']).inv.sum()/r.tot
        r['p2h']=o[o.h=='2'].groupby(['hd','jcd','rno']).inv.sum()/r.tot
        s=o[o.combination.isin(SIX)].groupby(['hd','jcd','rno']).inv.agg(['sum','count']); r['inv6']=s['sum']; r['n6']=s['count']
        rows.append(r.reset_index()); print(m,len(r),flush=True)
    R=pd.concat(rows); R=R[(R.n>=110)&(R.n6==6)]
    p=pd.read_sql("select hd,jcd,rno,combination res,payout from payouts where ticket_type='sanrensho' and hd>='20250401'",c).drop_duplicates(['hd','jcd','rno'])
    R=R.merge(p,on=['hd','jcd','rno'])
    b=pd.read_sql("select hd,jcd,rno,lane,exhibition_time et,start_course sc,start_st st,json_extract(data,'$.parts_changed') parts from beforeinfo where hd>='20250401'",c)
    b['et']=pd.to_numeric(b.et,errors='coerce'); b['er']=b.groupby(['hd','jcd','rno']).et.rank(method='min')
    b['st']=pd.to_numeric(b.st,errors='coerce'); b['strk']=b.groupby(['hd','jcd','rno']).st.rank(method='min')
    b['waku']=(b.sc==b.lane)
    for k in (1,2,3):
        x=b[b.lane==k][['hd','jcd','rno','et','er','st','strk','parts']].rename(columns=lambda q:q if q in('hd','jcd','rno') else f'{q}{k}'); R=R.merge(x,on=['hd','jcd','rno'],how='left')
    R=R.merge(b.groupby(['hd','jcd','rno']).waku.all().rename('wakunari').reset_index(),on=['hd','jcd','rno'],how='left')
    e=pd.read_sql("select hd,jcd,rno,lane,grade,json_extract(data,'$.national_win_rate') nw,json_extract(data,'$.motor_2ren_rate') mo from entries where hd>='20250401'",c)
    for k in (1,2):
        x=e[e.lane==k][['hd','jcd','rno','grade','nw','mo']].rename(columns={'grade':f'g{k}','nw':f'nw{k}','mo':f'mo{k}'}); R=R.merge(x,on=['hd','jcd','rno'],how='left')
    R.to_parquet(OUT); print(len(R))
def report():
    d=pd.read_parquet(OUT); d['syn']=1/d.inv6; d['hit']=d.res.isin(SIX); d['win1']=d.res.str[0]=='1'; d['two_in']=d.res.str[2:].str.contains('2')
    d['pay']=np.where(d.hit,d.payout,0); d['ym']=d.hd.str[:6]
    d['mk6']=d.inv6/d.tot   # 市場が見ている6点の確率（控除前換算）
    d['no2_given1_mkt']=1-d.p1_2/d.p1
    print('レース',len(d),'期間',d.hd.min(),d.hd.max())
    print('①1着',d.win1.mean().round(3),'①1着のとき②が3着以内',d[d.win1].two_in.mean().round(3),'6点的中',d.hit.mean().round(3),'市場換算',d.mk6.mean().round(3),'均等買い回収率',(d.pay.sum()/(600*len(d))).round(3))
    def t(name,key,sub=None):
        s=d if sub is None else sub; g=s.groupby(key,observed=True).agg(n=('hit','size'),win1=('win1','mean'),hit=('hit','mean'),mkt=('mk6','mean'),syn=('syn','median'),pay=('pay','sum')); g['roi']=g.pay/(600*g.n); print('\n'+name); print(g.drop(columns='pay').round(3).to_string())
    t('合成オッズ帯',pd.cut(d.syn,[0,3,4,5,6,8,10,15,25,999]))
    t('①頭の売れ方',pd.cut(d.p1,[0,.3,.4,.5,.6,.7,1]))
    t('②頭の売れ方',pd.cut(d.p2h,[0,.05,.1,.15,.2,.3,1]))
    t('②の展示順位','er2'); t('②のスタ展順位','strk2'); t('②級','g2'); t('①の展示順位','er1')
    t('②部品交換',d.parts2.fillna('[]').str.len()>2)
    t('②モーター',pd.cut(pd.to_numeric(d.mo2,errors='coerce'),[0,25,30,35,40,100]))
    return d
if __name__=='__main__':
    build() if sys.argv[1]=='build' else report()
