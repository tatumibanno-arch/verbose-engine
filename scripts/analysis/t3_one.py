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
print('\n== ①②③が一周1〜3位のレースで、1点の選び方と条件を足す（オッズは使わない）')
o['mr']=o.groupby(key).mawari.rank(method='first'); o['cr']=o.groupby(key).chokusen.rank(method='first')
m=o.sort_values('lap').groupby(key).lap.apply(lambda x: pd.Series({'g12':x.iloc[1]-x.iloc[0],'g23':x.iloc[2]-x.iloc[1],'g34':x.iloc[3]-x.iloc[2] if len(x)>3 else np.nan})).unstack().reset_index()
mw=o.pivot_table(index=key,columns='lane',values='mr').reset_index(); mw.columns=key+[f'M{int(x)}' for x in mw.columns[3:]]
ex=pd.read_sql("select hd,jcd,rno,lane,exhibition_time et from beforeinfo where hd>='20250601'",c); ex['er']=ex.groupby(key).et.rank(method='first'); ew=ex.pivot_table(index=key,columns='lane',values='er').reset_index(); ew.columns=key+[f'E{int(x)}' for x in ew.columns[3:]]
B=d[(d.L1==1)&d[['L2','L3']].isin([2,3]).all(1)].merge(m,on=key,how='left').merge(mw,on=key,how='left').merge(ew,on=key,how='left')
rep('土台（一周の順に1点）',B,lapc)
rep('2・3着の順をまわり足で決める',B,lambda s:'1-'+np.where(s.M2<s.M3,'2','3')+'-'+np.where(s.M2<s.M3,'3','2'))
rep('2・3着の順を公式展示で決める',B,lambda s:'1-'+np.where(s.E2<s.E3,'2','3')+'-'+np.where(s.E2<s.E3,'3','2'))
rep('いつも1-2-3（このレースだけ）',B,lambda s:'1-2-3'); rep('いつも1-3-2（このレースだけ）',B,lambda s:'1-3-2')
for lo,hi in ((0,.2),(.2,.5),(.5,9)): rep(f'  ①と一周2位の差 {lo}〜{hi}秒',B[(B.g12>=lo)&(B.g12<hi)],lapc)
for lo,hi in ((0,.2),(.2,.5),(.5,9)): rep(f'  一周3位と4位の差 {lo}〜{hi}秒（④以下が離れている）',B[(B.g34>=lo)&(B.g34<hi)],lapc)
rep('  ①がまわり足も1位',B[B.M1==1],lapc); rep('  ①が公式展示も1位',B[B.E1==1],lapc); rep('  ①②③が公式展示も1〜3位',B[B[['E1','E2','E3']].max(1)<=3],lapc)
rep('  ①②③がまわり足も1〜3位',B[B[['M1','M2','M3']].max(1)<=3],lapc); rep('  波2cm以下',B[B.wv<=2],lapc); rep('  風3m以下',B[B.ws<=3],lapc)
rep('  ②か③がB級',B[B.g2.isin(['B1','B2'])|B.g3.isin(['B1','B2'])],lapc); rep('  ②③ともA級',B[B.g2.isin(['A1','A2'])&B.g3.isin(['A1','A2'])],lapc)
print('\n== 公式展示で2・3着の順を決める1点 × ④以下が離れている')
exo=lambda s:'1-'+np.where(s.E2<s.E3,'2','3')+'-'+np.where(s.E2<s.E3,'3','2')
for th in (0.3,0.4,0.5): rep(f'  一周3位と4位の差 {th}秒以上',B[B.g34>=th],exo)
rep('  ①と一周2位の差0.2秒未満',B[B.g12<.2],exo); rep('  ①と一周2位の差0.2秒未満＆3位と4位の差0.3秒以上',B[(B.g12<.2)&(B.g34>=.3)],exo)
print('\n== 0.4秒以上の形が1日に何レースあるか')
Q=B[B.g34>=.4]; days=pd.Series(sorted(d.hd.unique())); cnt=Q.groupby('hd').size().reindex(days,fill_value=0)
print('日数',len(cnt),'1日あたり平均',round(cnt.mean(),2),' 0レースの日',(cnt==0).mean().round(3),' 1レース',(cnt==1).mean().round(3),' 2レース',(cnt==2).mean().round(3),' 3レース以上',(cnt>=3).mean().round(3),' 最多',cnt.max())
import sys; sys.path.insert(0,'scripts'); from common import VENUES
print('場ごとの件数（多い順）', {VENUES[j]:n for j,n in Q.jcd.value_counts().head(10).items()})
print('出ない場（オリジナル展示なし）', sorted(VENUES[j] for j in set(VENUES)-set(d.jcd.unique())))
print('\n== 確認: 2026-11 の当てはめ（lap_one.py と件数が合うか）')
Qn=Q[Q.hd>='20261101']; rep('11月',Qn,exo); print(Qn.groupby(Qn.hd.str[:6]).size().to_dict() if len(Qn) else 0)
mo=Q.assign(pay=np.where(Q.res==exo(Q),Q.payout,0)); print((mo.groupby(mo.hd.str[:6]).pay.sum()/(100*mo.groupby(mo.hd.str[:6]).size())).round(2).to_dict())
print('\n== 該当レース（0.4秒以上）の結果の内訳と、点数を足したとき')
Q2=Q.copy(); Q2['pick']=exo(Q2); Q2['other']=np.where(Q2.pick=='1-2-3','1-3-2','1-2-3')
r=Q2.res
print('件数',len(Q2),'| 買い目の1点が的中',round((r==Q2.pick).mean(),3),'| もう片方（1-2-3/1-3-2の逆）',round((r==Q2.other).mean(),3),
      '| ①頭で④⑤⑥が2・3着に入った',round((r.str[0]=='1').mean()-((r==Q2.pick)|(r==Q2.other)).mean(),3),'| ①が負けた',round((r.str[0]!='1').mean(),3))
def roi(sel): 
    tot=0; pay=0
    for cb in sel:
        tot+=len(Q2); pay+=np.where(r==(Q2[cb] if cb in Q2 else cb),Q2.payout,0).sum()
    return round(pay/(100*tot),3)
print('1点（今の形）',roi(['pick']),' 2点（1-2-3と1-3-2）',roi(['pick','other']))
for extra in (['1-2-4','1-3-4'],['1-2-4','1-3-4','1-4-2','1-4-3']):
    print('1点＋',extra,roi(['pick']+extra))
