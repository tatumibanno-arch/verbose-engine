"""①頭の1点（1-○-○）を、一周タイムの差（t3-lap-one-v1）以外の決め方で選ぶ。オッズは使わない（後出しなし）。払戻は確定。
艇ごとの材料: 一周・まわり足・直線・公式展示の順位、スタ展の順位、全国勝率、モーター2連率、その枠の2連率。
「点数」= 材料の順位の合計（小さいほど良い）。2・3着=①以外で点数の良い2艇、順は公式展示。条件: ①が点数1位、選んだ2艇と4番手の点数の差。
2025-06〜2026-11、前半〜2026-01／後半2026-02〜。"""
import sqlite3, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
key=['hd','jcd','rno']; c=sqlite3.connect('data/boatrace.db'); cut='20260201'
D='/tmp/claude-0/-home-user-verbose-engine/968008a0-6ecf-5641-a3ff-006912b41722/scratchpad/n2/'
o=pd.read_sql("select hd,jcd,rno,lane,lap,mawari,chokusen from oriten where hd>='20250601'",c)
b=pd.read_sql("select hd,jcd,rno,lane,exhibition_time et,start_st st from beforeinfo where hd>='20250601'",c)
for q in ('et','st'): b[q]=pd.to_numeric(b[q],errors='coerce')
E=pd.read_parquet(D+'ob_ent.parquet')[key+['lane','nw','mo','cn','c2']]
for q in ('nw','mo'): E[q]=pd.to_numeric(E[q],errors='coerce')
E['c2r']=(E.c2.fillna(0)+2)/(E.cn.fillna(0)+5)
X=o.merge(b,on=key+['lane'],how='left').merge(E,on=key+['lane'],how='left')
X=X[X.groupby(key).lap.transform('count')>=5]
rk=lambda col,asc=True: X.groupby(key)[col].rank(ascending=asc,method='average')
X['r_lap']=rk('lap'); X['r_maw']=rk('mawari'); X['r_cho']=rk('chokusen'); X['r_ex']=rk('et'); X['r_st']=rk('st'); X['r_nw']=rk('nw',False); X['r_mo']=rk('mo',False); X['r_c2']=rk('c2r',False)
for q in ('r_cho','r_st','r_maw'): X[q]=X[q].fillna(3.5)
p=pd.read_sql("select hd,jcd,rno,combination res,payout from payouts where ticket_type='sanrensho' and hd>='20250601'",c).drop_duplicates(key)
R=lambda s: s.pay.sum()/(100*len(s)) if len(s) else np.nan
def run(nm,cols,gaps):
    Y=X.copy(); Y['sc']=Y[cols].sum(1); Y=Y.sort_values(key+['sc','lane'])
    Y['ord']=Y.groupby(key).cumcount()+1
    top=Y[Y.ord<=4].pivot_table(index=key,columns='ord',values=['lane','sc','et'],aggfunc='first'); top.columns=[f'{a}{b}' for a,b in top.columns]; top=top.reset_index()
    top=top[top.lane1==1].merge(p,on=key)
    a,b_=top.lane2.astype(int).astype(str),top.lane3.astype(int).astype(str); first=top.et2<=top.et3
    top['cb']='1-'+np.where(first,a,b_)+'-'+np.where(first,b_,a); top['pay']=np.where(top.res==top.cb,top.payout,0); top['gap']=top.sc4-top.sc3
    print(f'--- 点数={nm}')
    for g in gaps:
        s=top[top.gap>=g]; mo=s.groupby(s.hd.str[:6]).apply(R)
        print(f'  4番手との差{g}以上 | レース {len(s)} 的中 {(s.pay>0).mean():.3f} 回収 {R(s):.3f} 前半 {R(s[s.hd<cut]):.3f} 後半 {R(s[s.hd>=cut]):.3f} 勝ち月 {(mo>1).sum()}/{len(mo)}')
run('公式展示だけ',['r_ex'],[0,1,2])
run('まわり足だけ',['r_maw'],[0,1,2])
run('一周＋公式展示',['r_lap','r_ex'],[0,2,3,4])
run('一周＋まわり足＋直線',['r_lap','r_maw','r_cho'],[0,3,4,5])
run('一周＋まわり足＋直線＋公式展示',['r_lap','r_maw','r_cho','r_ex'],[0,4,6,8])
run('足（4つ）＋スタ展',['r_lap','r_maw','r_cho','r_ex','r_st'],[0,4,6,8])
run('足（4つ）＋勝率＋モーター＋枠の2連率',['r_lap','r_maw','r_cho','r_ex','r_nw','r_mo','r_c2'],[0,5,8,11])
run('勝率＋モーター＋枠の2連率（足を使わない＝前夜に分かる）',['r_nw','r_mo','r_c2'],[0,3,5,7])
print('\n== 足（4つ）＋勝率＋モーター＋枠の2連率 を細かく。t3-lap-one-v1 と重なるか')
import sys; sys.path.insert(0,'scripts')
run('足（4つ）＋勝率＋モーター＋枠の2連率',['r_lap','r_maw','r_cho','r_ex','r_nw','r_mo','r_c2'],[7,8,9,10,11,12,13])
print('\n== 差9以上の中身: 2・3着の組、t3-lap-one-v1 との重なり')
cols=['r_lap','r_maw','r_cho','r_ex','r_nw','r_mo','r_c2']; Y=X.copy(); Y['sc']=Y[cols].sum(1); Y=Y.sort_values(key+['sc','lane']); Y['ord']=Y.groupby(key).cumcount()+1
top=Y[Y.ord<=4].pivot_table(index=key,columns='ord',values=['lane','sc','et'],aggfunc='first'); top.columns=[f'{a}{b}' for a,b in top.columns]; top=top.reset_index(); top=top[top.lane1==1].merge(p,on=key)
a,b_=top.lane2.astype(int).astype(str),top.lane3.astype(int).astype(str); first=top.et2<=top.et3
top['cb']='1-'+np.where(first,a,b_)+'-'+np.where(first,b_,a); top['pay']=np.where(top.res==top.cb,top.payout,0); top['gap']=top.sc4-top.sc3
top['pair']=[''.join(sorted([x,y])) for x,y in zip(a,b_)]
# lap-one 該当
lo=X[['hd','jcd','rno','lane','lap']].dropna().sort_values(key+['lap','lane']); lo['r']=lo.groupby(key).cumcount()+1
lw=lo.pivot_table(index=key,columns='r',values=['lane','lap'],aggfunc='first'); lw.columns=[f'{a_}{b2}' for a_,b2 in lw.columns]; lw=lw.reset_index()
lw['lapone']=(lw.lane1==1)&lw[['lane2','lane3']].isin([2,3]).all(1)&((lw.lap4-lw.lap3)>=0.4)
top=top.merge(lw[key+['lapone']],on=key,how='left'); top['lapone']=top.lapone.fillna(False)
for g in (9,10):
    s=top[top.gap>=g]; print(f'差{g}以上 {len(s)}レース 組の内訳',s.pair.value_counts().head(6).to_dict(),' t3-lap-one と重なる',int(s.lapone.sum()))
    t=s[~s.lapone]; mo=t.groupby(t.hd.str[:6]).apply(R); print(f'  重ならない分 | レース {len(t)} 的中 {(t.pay>0).mean():.3f} 回収 {R(t):.3f} 前半 {R(t[t.hd<cut]):.3f} 後半 {R(t[t.hd>=cut]):.3f} 勝ち月 {(mo>1).sum()}/{len(mo)}')
    for pr in ('23',):
        u=s[s.pair==pr]; print(f'  ②③の組だけ | {len(u)} 回収 {R(u):.3f}  ②③以外 | {len(s)-len(u)} 回収 {R(s[s.pair!=pr]):.3f} 前半 {R(s[(s.pair!=pr)&(s.hd<cut)]):.3f} 後半 {R(s[(s.pair!=pr)&(s.hd>=cut)]):.3f}')
days=pd.Series(sorted(X.hd.unique())); cnt=top[top.gap>=9].groupby('hd').size().reindex(days,fill_value=0); print('差9以上 1日平均',round(cnt.mean(),2),'0の日',round((cnt==0).mean(),3))
print('\n== 差10以上: 1点（公式展示順）と、逆順1点、2点（両方）')
s=top[top.gap>=10].copy(); rev='1-'+np.where(first.loc[s.index],b_.loc[s.index],a.loc[s.index])+'-'+np.where(first.loc[s.index],a.loc[s.index],b_.loc[s.index])
pr=np.where(s.res==rev,s.payout,0)
print('1点（公式展示順）',round(s.pay.sum()/(100*len(s)),3),' 逆順1点',round(pr.sum()/(100*len(s)),3),' 2点（両方）',round((s.pay.sum()+pr.sum())/(200*len(s)),3),' 的中: 1点',round((s.pay>0).mean(),3),'2点',round(((s.pay>0)|(pr>0)).mean(),3))
