"""score_search.py で残った点数の作り方を詳しく見る（月別、大きい払戻を除いた回収、知らせ2との重なり）。"""
import numpy as np, pandas as pd, warnings, sys; warnings.filterwarnings('ignore')
exec(open('scripts/analysis/one_pick.py').read().split("run('公式展示だけ'")[0])
def picks(sub,qn):
    Y=X.copy(); Y['sc']=Y[sub].sum(1); Y=Y.sort_values(key+['sc','lane']); Y['o']=Y.groupby(key).cumcount()+1
    T=Y[Y.o<=4].pivot_table(index=key,columns='o',values=['lane','sc','et'],aggfunc='first'); T.columns=[f'{a}{b}' for a,b in T.columns]; T=T.reset_index()
    T=T[(T.lane1==1)&T.sc4.notna()]; T['gap']=T.sc4-T.sc3; th=T.gap.quantile(qn); T=T[T.gap>=th].merge(p,on=key)
    a,b_=T.lane2.astype(int).astype(str),T.lane3.astype(int).astype(str); f=T.et2<=T.et3; T['cb']='1-'+np.where(f,a,b_)+'-'+np.where(f,b_,a); T['pay']=np.where(T.res==T.cb,T.payout,0); return T,th
S2,_=picks(['r_lap','r_maw','r_cho','r_ex','r_nw','r_mo','r_c2'],0.0); S2=S2[S2.gap>=10]; s2k=set(zip(S2.hd,S2.jcd,S2.rno))
for nm,sub,qn in (('直線＋公式展示＋モーター 上位8%',['r_cho','r_ex','r_mo'],.92),('公式展示＋モーター＋枠2連率 上位4%',['r_ex','r_mo','r_c2'],.96),('一周＋直線＋公式展示＋モーター 上位4%',['r_lap','r_cho','r_ex','r_mo'],.96),('直線＋モーター 上位4%',['r_cho','r_mo'],.96),('一周＋まわり足＋直線＋公式展示＋モーター 上位4%',['r_lap','r_maw','r_cho','r_ex','r_mo'],.96)):
    T,th=picks(sub,qn); h=T[T.pay>0].pay.sort_values(ascending=False); mo=T.groupby(T.hd.str[:6]).apply(R)
    ov=sum(k in s2k for k in zip(T.hd,T.jcd,T.rno))
    print(f'[{nm}] 線（差）{th:.1f}以上 | レース {len(T)} 的中 {(T.pay>0).mean():.3f} 当たり平均 {h.mean():.0f}円 回収 {R(T):.3f} | 上位3本を除くと {(T.pay.sum()-h.head(3).sum())/(100*len(T)):.3f} | 勝ち月 {(mo>1).sum()}/{len(mo)} 最悪 {mo.min():.2f} | 知らせ2と重なる {ov}')
    print('   月別', mo.round(2).to_dict())
