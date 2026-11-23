"""素直な型: 条件を「外す艇Xの展示が6位（または5〜6位）」「売れすぎ幅0.1以上」に絞り、線引きを減らす。
3連単（ob3.parquet＝Xを外した6点、①頭40%以上）と、でたらめ世界との比較。締切時オッズ。"""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
D='/tmp/claude-0/-home-user-verbose-engine/968008a0-6ecf-5641-a3ff-006912b41722/scratchpad/n2/'
C=pd.read_parquet(D+'ob3.parquet'); B=pd.read_parquet(D+'ob2_bets.parquet'); oc=[c for c in B.columns if c[0]=='o' and len(c)==3 and c[1:].isdigit()]
def one(nm,m,pay=None):
    s=C[m]; p=s.pay if pay is None else pay[m]
    ym=s.hd.str[:6]; mo=p.groupby(ym).sum()/(100*s.groupby(ym).size()); a=s.disc
    print(f'{nm} | レース {s.i.nunique()} 点 {len(s)} 回収 {p.sum()/(100*len(s)):.3f} 前半 {p[a].sum()/(100*a.sum()):.3f} 後半 {p[~a].sum()/(100*(~a).sum()):.3f} 勝ち月 {(mo>1).sum()}/{len(mo)} 最悪 {mo.min():.2f}')
    return p.sum()/(100*len(s))
print('== 条件2つだけ（売れすぎ幅0.1以上、外す艇の展示順位）。①頭40%以上、6点ぜんぶ')
one('展示6位',(C.gap>=.1)&(C.xer==6)); one('展示5〜6位',(C.gap>=.1)&(C.xer>=5)); one('展示1〜4位（比較）',(C.gap>=.1)&(C.xer<=4))
print('== オッズの線を広く'); 
for cap in (10,15,20,30,50,1e9): one(f'展示6位＆{cap if cap<1e9 else "上限なし"}倍未満',(C.gap>=.1)&(C.xer==6)&(C.odds<cap))
for cap in (15,20,30): one(f'展示5〜6位＆{cap}倍未満',(C.gap>=.1)&(C.xer>=5)&(C.odds<cap))
print('== ①頭の線'); 
for lo in (.4,.5,.6,.7): one(f'展示6位＆30倍未満＆①頭{int(lo*100)}%以上',(C.gap>=.1)&(C.xer==6)&(C.odds<30)&(C.p1x>=lo))
print('== 残りの艇の展示が外す艇より良い（展示6位なら自動で成り立つ）＋外す艇が②③か④⑤⑥か')
one('展示6位＆30倍未満＆Xが②③',(C.gap>=.1)&(C.xer==6)&(C.odds<30)&(C.X<=3)); one('展示6位＆30倍未満＆Xが④⑤⑥',(C.gap>=.1)&(C.xer==6)&(C.odds<30)&(C.X>=4))
# でたらめ世界で「展示6位＆30倍未満」「展示6位＆上限なし」の回収率がどのくらい散らばるか
rng=np.random.default_rng(2); Bi=B.loc[sorted(C.i.unique())]; inv=1/Bi[oc].values; cum=np.cumsum(inv/Bi.tot.values[:,None],1); names=np.array([f'1-{c[1]}-{c[2]}' for c in oc]); idx=pd.Series(np.arange(len(Bi)),index=Bi.index); ci=idx.loc[C.i].values; oa=Bi[oc].values
for nm,m in (('展示6位＆30倍未満',(C.gap>=.1)&(C.xer==6)&(C.odds<30)),('展示6位＆上限なし',(C.gap>=.1)&(C.xer==6)),('展示5〜6位＆30倍未満',(C.gap>=.1)&(C.xer>=5)&(C.odds<30))):
    real=C[m].pay.sum()/(100*m.sum()); mv=m.values; r=[]
    for rep in range(1000):
        u=rng.random(len(Bi)); w=(cum<u[:,None]).sum(1); ok=w<len(oc); w2=np.minimum(w,len(oc)-1); wn=np.where(ok,names[w2],''); wo=np.where(ok,oa[np.arange(len(Bi)),w2],0)
        r.append((np.where(C.cb.values[mv]==wn[ci[mv]],wo[ci[mv]]*100,0.0)).sum()/(100*mv.sum()))
    r=np.array(r); print(f'[{nm}] 本物 {real:.3f} ｜ でたらめ世界1000回: 平均 {r.mean():.3f}、95%が入る範囲 {np.quantile(r,.025):.3f}〜{np.quantile(r,.975):.3f}、本物以上は {int((r>=real).sum())}/1000回')
