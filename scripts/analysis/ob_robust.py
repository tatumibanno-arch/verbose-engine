"""候補 ob-cheap-v1 の型が偶然かどうかを確かめる。(1) しきい値を動かしたときに回収率がなめらかに変わるか、
(2) 同じ格子（960マス）を「売れ方どおりに当たりを引き直した世界」でも数えて、本物がそれより多く残るか。締切時オッズ。"""
import itertools, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
D='/tmp/claude-0/-home-user-verbose-engine/968008a0-6ecf-5641-a3ff-006912b41722/scratchpad/n2/'
C=pd.read_parquet(D+'ob3.parquet'); B=pd.read_parquet(D+'ob2_bets.parquet'); oc=[c for c in B.columns if c[0]=='o' and len(c)==3 and c[1:].isdigit()]
C=C[C.rer<C.xer].copy(); R=lambda p,n: p/(100*n) if n else np.nan
def one(nm,m): s=C[m]; print(f'  {nm}: 点 {len(s)} 回収 {R(s.pay.sum(),len(s)):.3f} 前半 {R(s[s.disc].pay.sum(),s.disc.sum()):.3f} 後半 {R(s[~s.disc].pay.sum(),(~s.disc).sum()):.3f}')
base=(C.gap>=.1)&(C.modiff<=-5)&(C.p1x>=.7)&(C.odds<15)
print('(1) 他は候補Aのまま、1つだけ動かす'); print(' 外す艇の展示順位')
for k in (2,3,4,5,6): one(f'{k}位',base&(C.xer==k))
print(' オッズの上限')
for lo,hi in ((0,6),(6,10),(10,15),(15,20),(20,30),(30,60)): one(f'{lo}〜{hi}倍',(C.gap>=.1)&(C.modiff<=-5)&(C.p1x>=.7)&(C.xer>=5)&(C.odds>=lo)&(C.odds<hi))
print(' 外す艇のモーター差'); b2=(C.gap>=.1)&(C.p1x>=.7)&(C.xer>=5)&(C.odds<15)
for lo,hi in ((-99,-15),(-15,-10),(-10,-5),(-5,0),(0,5),(5,99)): one(f'{lo}〜{hi}',b2&(C.modiff>=lo)&(C.modiff<hi))
print(' 売れすぎ幅'); b3=(C.modiff<=-5)&(C.p1x>=.7)&(C.xer>=5)&(C.odds<15)
for lo,hi in ((-1,0),(0,.05),(.05,.1),(.1,.15),(.15,.2),(.2,1)): one(f'{lo}〜{hi}',b3&(C.gap>=lo)&(C.gap<hi))
print(' ①頭の売れ方'); b4=(C.gap>=.1)&(C.modiff<=-5)&(C.xer>=5)&(C.odds<15)
for lo,hi in ((.4,.5),(.5,.6),(.6,.7),(.7,.8),(.8,1)): one(f'{lo}〜{hi}',b4&(C.p1x>=lo)&(C.p1x<hi))
# (2) 格子
G=list(itertools.product((0,.05,.1,.15),(0,-5,-10),(1,3,4,5,6),(10,15,20,30),(.5,.6,.7,.8)))
gap=C.gap.values; mo=C.modiff.values; xe=C.xer.values; od=C.odds.values; p1=C.p1x.values; dsc=C.disc.values
masks=[(gap>=a)&(mo<=b)&(xe>=c)&(od<d)&(p1>=e) for a,b,c,d,e in G]
def count(pay):
    n=0; best=[]
    for g,m in zip(G,masks):
        a=m&dsc; b=m&~dsc
        if a.sum()>=400 and b.sum()>=300:
            ra=pay[a].sum()/(100*a.sum()); rb=pay[b].sum()/(100*b.sum())
            if ra>=1 and rb>=1: n+=1; best.append((g,int(m.sum()),round(ra,2),round(rb,2)))
    return n,best
valid=sum(1 for m in masks if (m&dsc).sum()>=400 and (m&~dsc).sum()>=300)
n,best=count(C.pay.values); print(f'\n(2) 格子 {len(G)}マス（うち点数が足りるのは {valid}マス）。本物: 前半・後半とも1.0以上は {n}マス')
rng=np.random.default_rng(1); Bi=B.loc[sorted(C.i.unique())]; inv=1/Bi[oc].values; pr=inv/Bi.tot.values[:,None]; cum=np.cumsum(pr,1); names=np.array([f'1-{c[1]}-{c[2]}' for c in oc]); idx=pd.Series(np.arange(len(Bi)),index=Bi.index); ci=idx.loc[C.i].values; odds_all=Bi[oc].values
res=[]
for rep in range(200):
    u=rng.random(len(Bi)); w=(cum<u[:,None]).sum(1); ok=w<len(oc); wn=np.where(ok,names[np.minimum(w,len(oc)-1)],''); wo=np.where(ok,odds_all[np.arange(len(Bi)),np.minimum(w,len(oc)-1)],0)
    fake=np.where(C.cb.values==wn[ci],wo[ci]*100,0.0); res.append(count(fake)[0])
res=np.array(res); print(f'  でたらめ200回: 平均 {res.mean():.1f}マス、最大 {res.max()}マス、本物（{n}マス）以上になった回数 {int((res>=n).sum())}/200')
print('  本物で残ったマス（売れすぎ幅, モーター差, X展示, オッズ上限, ①頭）上位:'); 
for b in sorted(best,key=lambda t:-t[1])[:12]: print('   ',b)
