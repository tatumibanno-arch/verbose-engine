"""3連単を1レース1点だけ買う（運用者「1から1点」）。買い目はオッズを見ずに決める（一周タイムの順位・艇番だけ）ので後出しなし。払戻は確定。
2025-06〜2026-11、前半〜2026-01／後半2026-02〜。"""
import sqlite3, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
key=['hd','jcd','rno']; c=sqlite3.connect('data/boatrace.db'); cut='20260201'
o=pd.read_sql("select hd,jcd,rno,lane,lap,mawari,chokusen from oriten where hd>='20250601'",c); o['lr']=o.groupby(key).lap.rank(method='first')
w=o.pivot_table(index=key,columns='lr',values='lane').reset_index(); w.columns=key+[f'L{int(c_)}' for c_ in w.columns[3:]]
w=w.dropna(subset=['L1','L2','L3'])
for k in ('L1','L2','L3','L4'): w[k]=w[k].astype('Int64')
b=pd.read_sql("select hd,jcd,rno,max(cast(json_extract(data,'$.wind_speed') as real)) ws,max(cast(json_extract(data,'$.wave_height') as real)) wv from beforeinfo where hd>='20250601' group by 1,2,3",c)
e=pd.read_sql("select hd,jcd,rno,lane,grade from entries where hd>='20250601' and lane<=3",c).pivot_table(index=key,columns='lane',values='grade',aggfunc='first').reset_index(); e.columns=key+['g1','g2','g3']
p=pd.read_sql("select hd,jcd,rno,combination res,payout from payouts where ticket_type='sanrensho' and hd>='20250601'",c).drop_duplicates(key)
d=w.merge(p,on=key).merge(b,on=key,how='left').merge(e,on=key,how='left')
R=lambda s: s.pay.sum()/(100*len(s)) if len(s) else np.nan
def rep(nm,s,combo):
    s=s.assign(pay=np.where(s.res==combo(s),s.payout,0)); mo=s.groupby(s.hd.str[:6]).apply(R)
    print(f'{nm} | レース {len(s)} 的中 {(s.pay>0).mean():.3f} 当たりの平均 {s[s.pay>0].payout.mean():.0f}円 回収 {R(s):.3f} 前半 {R(s[s.hd<cut]):.3f} 後半 {R(s[s.hd>=cut]):.3f} 勝ち月 {(mo>1).sum()}/{len(mo)}')
fx=lambda a,b_,c_: (lambda s: a+'-'+b_+'-'+c_) 
print('== 決め打ち（全レース）')
for cb in ('1-2-3','1-3-2','1-2-4','1-3-4','1-4-2'): rep(f'いつも {cb}',d,lambda s,cb=cb: cb)
lapc=lambda s: s.L1.astype(str)+'-'+s.L2.astype(str)+'-'+s.L3.astype(str)
print('== 一周タイムの順に1点（1着=一周1位、2着=2位、3着=3位）')
rep('全レース',d,lapc); A=d[d.L1==1]; rep('①が一周1位のレースだけ',A,lapc)
rep('　＋風3m以下',A[A.ws<=3],lapc); rep('　＋波2cm以下',A[A.wv<=2],lapc); rep('　＋2・3位が②③',A[A[['L2','L3']].isin([2,3]).all(1)],lapc)
rep('　＋2・3位に⑤⑥がいる',A[(A.L2>=5)|(A.L3>=5)],lapc); rep('　＋①がA1',A[A.g1=='A1'],lapc); rep('　＋①がB級',A[A.g1.isin(['B1','B2'])],lapc)
print('== ①頭で、2・3着を一周の順に（①が一周1位でなくても）')
o2=lambda s: '1-'+np.where(s.L1==1,s.L2,s.L1).astype(str)+'-'+np.where(s.L1==1,s.L3,np.where(s.L2==1,s.L3,s.L2)).astype(str)
rep('全レース',d,o2); rep('①が一周2位のとき',d[d.L2==1],o2); rep('①が一周3位のとき',d[d.L3==1],o2)
