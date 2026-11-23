"""売れすぎの艇Xを外した6点のうち、安い組だけを買う。条件を足して発見(〜2026-03)→検証(2026-04〜)で確かめる。
注意: 締切時オッズ（締切前とは違う）。"""
import itertools, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
D='/tmp/claude-0/-home-user-verbose-engine/968008a0-6ecf-5641-a3ff-006912b41722/scratchpad/n2/'
B=pd.read_parquet(D+'ob2_bets.parquet')
rows=[]
for r in B.itertuples():
    cs=sorted([(getattr(r,f'o{a}{b}'),f'1-{a}-{b}') for a,b in itertools.permutations(r.ls,2)])
    for k,(o,c) in enumerate(cs,1): rows.append((r.Index,k,o,c))
C=pd.DataFrame(rows,columns=['i','rk','odds','cb']).merge(B[['hd','jcd','res','payout','gap','modiff','xer','rer','xmm','rmm','xsr','p1x','er1','X','syn']],left_on='i',right_index=True)
C['hit']=C.cb==C.res; C['pay']=np.where(C.hit,C.payout,0); C['disc']=C.hd<'20260401'; C.to_parquet(D+'ob3.parquet')
def R(s): return round(s.pay.sum()/(100*len(s)),3) if len(s) else np.nan
def t(nm,s):
    print(f'{nm} | 点数 {len(s)} レース {s.i.nunique()} 的中/レース {round(s.groupby("i").hit.max().mean(),3)} 回収 {R(s)} 発見 {R(s[s.disc])} 検証 {R(s[~s.disc])}')
base=C[(C.gap>=.1)&(C.modiff<=-5)&(C.p1x>=.6)]; A=base[base.rer<base.xer]
print('== 土台（売れすぎ幅0.1以上・Xのモーター5以上低い・①頭60%以上）'); t('6点全部',base)
for nm,s in (('安い順1点',base[base.rk==1]),('安い順2点',base[base.rk<=2]),('安い順3点',base[base.rk<=3]),('10倍未満だけ',base[base.odds<10]),('15倍未満だけ',base[base.odds<15]),('20倍未満だけ',base[base.odds<20]),('20倍以上だけ',base[base.odds>=20])): t(nm,s)
print('== 土台＋残り3艇の展示がXより良い'); t('6点全部',A)
for nm,s in (('安い順1点',A[A.rk==1]),('安い順2点',A[A.rk<=2]),('安い順3点',A[A.rk<=3]),('10倍未満だけ',A[A.odds<10]),('15倍未満だけ',A[A.odds<15]),('20倍未満だけ',A[A.odds<20]),('20倍以上だけ',A[A.odds>=20])): t(nm,s)
print('== さらに条件（土台＋残り展示が良い＋15倍未満だけ）')
S=A[A.odds<15]
t('＋①展示1〜2位',S[S.er1<=2]); t('＋①展示1位',S[S.er1==1]); t('＋①頭70%以上',S[S.p1x>=.7]); t('＋①頭80%以上',S[S.p1x>=.8])
t('＋Xの展示5〜6位',S[S.xer>=5]); t('＋Xのモーター10以上低い',S[S.modiff<=-10]); t('＋売れすぎ幅0.2以上',S[S.gap>=.2])
t('＋Xが②',S[S.X==2]); t('＋Xが③',S[S.X==3]); t('＋Xが④⑤⑥',S[S.X>=4])
t('＋Xの節間モーター着順が残りより悪い',S[S.xmm>S.rmm])
t('＋①展示1〜2位＆①頭70%以上',S[(S.er1<=2)&(S.p1x>=.7)]); t('＋①展示1〜2位＆Xの展示5〜6位',S[(S.er1<=2)&(S.xer>=5)])
t('＋①展示1〜2位＆①頭70%以上＆Xの展示5〜6位',S[(S.er1<=2)&(S.p1x>=.7)&(S.xer>=5)])
print('== 発見期間で場ごとに見て、良かった場だけを検証期間で見る')
g=S[S.disc].groupby('jcd').apply(lambda s:pd.Series({'n':len(s),'roi':R(s)})); good=g[(g.n>=60)&(g.roi>=.95)].index.tolist(); print('発見で0.95以上の場',good)
v=S[(~S.disc)&S.jcd.isin(good)]; print('その場だけ・検証',len(v),R(v)); v2=S[(~S.disc)&~S.jcd.isin(good)]; print('それ以外・検証',len(v2),R(v2))
best=S[(S.er1<=2)]; m=best.groupby(best.hd.str[:6]).apply(lambda s:pd.Series({'点数':len(s),'回収':R(s)})); print('== 月別（土台＋残り展示良＋15倍未満＋①展示1〜2位）'); print(m.T.to_string()); print('勝ち月',(m.回収>1).sum(),'/',len(m))
