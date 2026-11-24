"""「3艇で決まる度合い」の点数を、材料の組み合わせを変えて作り直す。①が点数1位、上位3艇と4番手の差で絞り、①頭3連単1点（2・3着は公式展示順）。オッズは使わない。
オリジナル展示を使わない点数なら、江戸川やオリジナル展示が出ないレースにも使える（2025-04〜）。"""
import sqlite3, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
key=['hd','jcd','rno']; c=sqlite3.connect('data/boatrace.db'); cut='20260101'
D='/tmp/claude-0/-home-user-verbose-engine/968008a0-6ecf-5641-a3ff-006912b41722/scratchpad/n2/'
b=pd.read_sql("select hd,jcd,rno,lane,exhibition_time et,start_st st,tilt from beforeinfo where hd>='20250401'",c)
for q in ('et','st','tilt'): b[q]=pd.to_numeric(b[q],errors='coerce')
o=pd.read_sql("select hd,jcd,rno,lane,lap,mawari,chokusen from oriten",c)
E=pd.read_parquet(D+'ob_ent.parquet')[key+['lane','nw','mo','cn','c2','cw','grade']]
for q in ('nw','mo'): E[q]=pd.to_numeric(E[q],errors='coerce')
E['c2r']=(E.c2.fillna(0)+2)/(E.cn.fillna(0)+5); E['c1r']=(E.cw.fillna(0)+1)/(E.cn.fillna(0)+5)
X=b.merge(E,on=key+['lane'],how='inner').merge(o,on=key+['lane'],how='left'); X=X[X.groupby(key).et.transform('count')>=5]
rk=lambda col,asc=True: X.groupby(key)[col].rank(ascending=asc,method='average')
X['r_ex']=rk('et'); X['r_st']=rk('st'); X['r_nw']=rk('nw',False); X['r_mo']=rk('mo',False); X['r_c2']=rk('c2r',False); X['r_c1']=rk('c1r',False)
X['r_lap']=rk('lap'); X['r_maw']=rk('mawari').fillna(3.5); X['r_cho']=rk('chokusen').fillna(3.5)
p=pd.read_sql("select hd,jcd,rno,combination res,payout from payouts where ticket_type='sanrensho' and hd>='20250401'",c).drop_duplicates(key)
R=lambda s: s.pay.sum()/(100*len(s)) if len(s) else np.nan
def run(nm,cols,gaps,need_lap=False):
    Y=X.dropna(subset=['lap']) if need_lap else X
    Y=Y.copy(); Y['sc']=Y[cols].sum(1); Y=Y.sort_values(key+['sc','lane']); Y['o']=Y.groupby(key).cumcount()+1
    T=Y[Y.o<=4].pivot_table(index=key,columns='o',values=['lane','sc','et'],aggfunc='first'); T.columns=[f'{a}{b}' for a,b in T.columns]; T=T.reset_index()
    T=T[T.lane1==1].merge(p,on=key); a,b_=T.lane2.astype(int).astype(str),T.lane3.astype(int).astype(str); f=T.et2<=T.et3
    T['cb']='1-'+np.where(f,a,b_)+'-'+np.where(f,b_,a); T['pay']=np.where(T.res==T.cb,T.payout,0); T['gap']=T.sc4-T.sc3
    print(f'--- {nm}（期間 {T.hd.min()}〜）')
    for g in gaps:
        s=T[T.gap>=g]; mo=s.groupby(s.hd.str[:6]).apply(R)
        print(f'  差{g}以上 | レース {len(s)} 的中 {(s.pay>0).mean():.3f} 回収 {R(s):.3f} 前半 {R(s[s.hd<cut]):.3f} 後半 {R(s[s.hd>=cut]):.3f} 勝ち月 {(mo>1).sum()}/{len(mo)}')
run('公式展示＋スタ展＋勝率＋モーター＋枠の2連率（オリジナル展示なし）',['r_ex','r_st','r_nw','r_mo','r_c2'],[4,6,7,8,9])
run('公式展示＋勝率＋モーター＋枠の2連率（オリジナル展示なし）',['r_ex','r_nw','r_mo','r_c2'],[4,5,6,7,8])
run('公式展示＋勝率＋モーター＋枠の1着率・2連率',['r_ex','r_nw','r_mo','r_c2','r_c1'],[5,6,7,8,9])
run('（参考）今の知らせ2と同じ7つ',['r_lap','r_maw','r_cho','r_ex','r_nw','r_mo','r_c2'],[9,10,11],True)
run('7つ＋スタ展',['r_lap','r_maw','r_cho','r_ex','r_nw','r_mo','r_c2','r_st'],[9,10,11,12],True)
run('7つ＋枠の1着率',['r_lap','r_maw','r_cho','r_ex','r_nw','r_mo','r_c2','r_c1'],[9,10,11,12],True)
