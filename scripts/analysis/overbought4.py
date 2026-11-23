"""overbought3 のパターンP（土台＋残り展示良＋X展示5〜6位＋15倍未満）をさらに絞る。
発見（〜2026-03）で良かった条件だけを検証（2026-04〜）で見る。締切時オッズ（締切前とは違う）。"""
import numpy as np, pandas as pd, itertools, warnings; warnings.filterwarnings('ignore')
D='/tmp/claude-0/-home-user-verbose-engine/968008a0-6ecf-5641-a3ff-006912b41722/scratchpad/n2/'
C=pd.read_parquet(D+'ob3.parquet'); B=pd.read_parquet(D+'ob2_bets.parquet')[['rsr','xrm','rrm','mo1']]; C=C.merge(B,left_on='i',right_index=True)
P=C[(C.gap>=.1)&(C.modiff<=-5)&(C.p1x>=.6)&(C.rer<C.xer)&(C.xer>=5)&(C.odds<15)].copy()
R=lambda s: round(s.pay.sum()/(100*len(s)),3) if len(s) else np.nan
conds={'①展示1位':P.er1==1,'①展示1〜2位':P.er1<=2,'①展示3位以下':P.er1>=3,'①頭70%以上':P.p1x>=.7,'①頭80%以上':P.p1x>=.8,'①頭60〜70%':P.p1x<.7,
 'Xモーター10以上低い':P.modiff<=-10,'Xモーター15以上低い':P.modiff<=-15,'売れすぎ幅0.15以上':P.gap>=.15,'売れすぎ幅0.2以上':P.gap>=.2,'売れすぎ幅0.3以上':P.gap>=.3,
 'Xが③':P.X==3,'Xが④':P.X==4,'Xが⑤⑥':P.X>=5,'Xが④⑤⑥':P.X>=4,'X展示6位':P.xer==6,'Xスタ展4〜6番手':P.xsr>=4,'Xスタ展1〜3番手':P.xsr<=3,
 '10倍未満':P.odds<10,'8倍未満':P.odds<8,'10〜15倍':P.odds>=10,'安い順1点目':P.rk==1,'安い順2点目まで':P.rk<=2,'①モーター35%以上':P.mo1>=35,'①モーター35%未満':P.mo1<35,
 '6点の合成4倍未満':P.syn<4,'6点の合成4倍以上':P.syn>=4,'残りの展示平均3位以内':P.rer<=3}
print('パターンP',len(P),P.i.nunique(),R(P),R(P[P.disc]),R(P[~P.disc]))
res=[]
for k,m in conds.items():
    s=P[m]; res.append((k,len(s),s.i.nunique(),R(s),R(s[s.disc]),R(s[~s.disc])))
T=pd.DataFrame(res,columns=['条件','点数','レース','全体','発見','検証']); print(T.sort_values('発見',ascending=False).to_string(index=False))
good=[k for k,n,_,_,d,_ in res if d>=1.0 and n>=300]; print('\n発見で1.0以上（300点以上）:',good)
out=[]
for a,b in itertools.combinations(good,2):
    s=P[conds[a]&conds[b]]
    if len(s)>=250: out.append((a+' ＆ '+b,len(s),s.i.nunique(),R(s),R(s[s.disc]),R(s[~s.disc])))
T2=pd.DataFrame(out,columns=['条件','点数','レース','全体','発見','検証']); print(T2.sort_values('発見',ascending=False).head(15).to_string(index=False))
print('\n2条件の組: 発見の上位5つの検証の平均',T2.sort_values('発見',ascending=False).head(5).検証.mean().round(3),' 全ての組の検証の平均',T2.検証.mean().round(3))
T.to_csv(D+'ob4_one.csv',index=False); T2.to_csv(D+'ob4_two.csv',index=False)
