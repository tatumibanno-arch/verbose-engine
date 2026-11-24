"""「一周タイムで上位がはっきり抜けている」レースを、オッズを使わずに選ぶ形の広げ方（t3-lap-one-v1 の仲間探し）。
買い目も一周タイム・公式展示・艇番だけで決める（後出しなし）。払戻は確定。2025-06〜2026-11、前半〜2026-01／後半2026-02〜。"""
import sqlite3, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
key=['hd','jcd','rno']; c=sqlite3.connect('data/boatrace.db'); cut='20260201'
o=pd.read_sql("select hd,jcd,rno,lane,lap from oriten where hd>='20250601' and lap is not null",c)
ex=pd.read_sql("select hd,jcd,rno,lane,exhibition_time et from beforeinfo where hd>='20250601'",c)
o=o.merge(ex,on=key+['lane'],how='left').sort_values(key+['lap','lane'])
o['r']=o.groupby(key).cumcount()+1; n=o.groupby(key).size().rename('n')
W=o.pivot_table(index=key,columns='r',values=['lane','lap','et'],aggfunc='first'); W.columns=[f'{a}{b}' for a,b in W.columns]; W=W.join(n).reset_index(); W=W[W.n>=4]
for k in (1,2,3,4,5): W[f'lane{k}']=W[f'lane{k}'].astype('Int64') if f'lane{k}' in W else pd.NA
W['g12']=W.lap2-W.lap1; W['g23']=W.lap3-W.lap2; W['g34']=W.lap4-W.lap3
P={t:pd.read_sql(f"select hd,jcd,rno,combination res,payout from payouts where ticket_type='{t}' and hd>='20250601'",c).drop_duplicates(key).rename(columns={'res':f'r_{t}','payout':f'p_{t}'}) for t in ('sanrensho','sanrenfuku','nirensho','nirenfuku','tansho')}
for t,df in P.items(): W=W.merge(df,on=key,how='left')
R=lambda s,col: s[col].sum()/(100*len(s)) if len(s) else np.nan
def rep(nm,s,t,combo):
    s=s.dropna(subset=[f'r_{t}']).copy(); cb=combo(s); res=s[f'r_{t}'].str.replace('=','-')
    if t in ('sanrenfuku','nirenfuku'): res=res.apply(lambda x:'-'.join(sorted(x.split('-')))); cb=cb.apply(lambda x:'-'.join(sorted(x.split('-'))))
    s['pay']=np.where(res==cb,s[f'p_{t}'],0); mo=s.groupby(s.hd.str[:6]).apply(lambda q:R(q,'pay'))
    print(f'{nm} | レース {len(s)} 的中 {(s.pay>0).mean():.3f} 回収 {R(s,"pay"):.3f} 前半 {R(s[s.hd<cut],"pay"):.3f} 後半 {R(s[s.hd>=cut],"pay"):.3f} 勝ち月 {(mo>1).sum()}/{len(mo)}')
L=lambda a: a.astype(str)
t3_ex=lambda s: L(s.lane1)+'-'+np.where(s.et2<=s.et3,L(s.lane2),L(s.lane3))+'-'+np.where(s.et2<=s.et3,L(s.lane3),L(s.lane2))
t3_lap=lambda s: L(s.lane1)+'-'+L(s.lane2)+'-'+L(s.lane3)
print('== A 3連単1点: 一周上位3艇が4位から離れている（艇番は問わない）、頭=一周1位、2・3着=公式展示順')
for th in (0.3,0.4,0.5):
    s=W[W.g34>=th]; rep(f'3位と4位の差{th}秒以上 全部',s,'sanrensho',t3_ex); rep(f'  うち頭が①',s[s.lane1==1],'sanrensho',t3_ex); rep(f'  うち頭が①以外',s[s.lane1!=1],'sanrensho',t3_ex); rep(f'  うち①が上位3艇に入っていない',s[~((s.lane1==1)|(s.lane2==1)|(s.lane3==1))],'sanrensho',t3_ex)
print('== B 3連複1点: 一周上位3艇のボックス1点（4位から離れている）')
for th in (0.3,0.4,0.5):
    s=W[W.g34>=th]; rep(f'3位と4位の差{th}秒以上',s,'sanrenfuku',t3_lap); rep(f'  うち①②③',s[s[['lane1','lane2','lane3']].isin([1,2,3]).all(1)],'sanrenfuku',t3_lap)
print('== C 2連単・2連複1点: 一周上位2艇が3位から離れている')
t2_ex=lambda s: L(s.lane1)+'-'+L(s.lane2); 
for th in (0.3,0.4,0.5):
    s=W[W.g23>=th]; rep(f'2位と3位の差{th}秒以上 2連単（一周1位-2位）',s,'nirensho',t2_ex); rep(f'  2連複',s,'nirenfuku',t2_ex); rep(f'  うち①②で2連単',s[s[['lane1','lane2']].isin([1,2]).all(1)],'nirensho',t2_ex)
print('== D 単勝1点: 一周1位が2位から離れている')
for th in (0.3,0.4,0.5):
    s=W[W.g12>=th]; rep(f'1位と2位の差{th}秒以上 単勝（一周1位の艇）',s,'tansho',lambda q:L(q.lane1)); rep(f'  うち①',s[s.lane1==1],'tansho',lambda q:L(q.lane1)); rep(f'  うち①以外',s[s.lane1!=1],'tansho',lambda q:L(q.lane1))
print('\n== E ①頭で、2・3着の組を状況で変える（①が一周1位、2・3着=一周2・3位の艇、順は公式展示）。2・3着の組ごと')
s0=W[(W.lane1==1)]
for th in (0.0,0.3,0.4):
    s=s0[s0.g34>=th]; s=s.assign(pair=s.apply(lambda r:''.join(sorted([str(r.lane2),str(r.lane3)])),axis=1))
    print(f'-- 3位と4位の差{th}秒以上')
    for pr,g in s.groupby('pair'):
        if len(g)>=150: rep(f'  2・3着が {pr[0]}・{pr[1]}',g,'sanrensho',t3_ex)
