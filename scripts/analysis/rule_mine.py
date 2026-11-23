"""条件の総当たり（3連単・締切時100倍未満の全組）。最大3条件の組み合わせを全部試し、
発見（2025-04〜12）で良かったものだけを確認（2026-01〜06）→最後（2026-07〜11）で見る。
偶然の当たりの量を測るため、払戻をレース単位で入れ替えた「でたらめ版」でも同じ数え方をする。締切時オッズ（締切前とは違う）。"""
import sys, itertools, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
D='/tmp/claude-0/-home-user-verbose-engine/968008a0-6ecf-5641-a3ff-006912b41722/scratchpad/n2/'
O=pd.read_parquet(D+'cheap100.parquet'); O['pay']=np.where(O.hit==1,O.payout,0).astype('float32')
P={}
for p,nm in (('a','1着'),('b','2着'),('c','3着')):
    for k in range(1,7): P[f'{nm}={k}号艇']=O[f'{p}_lane']==k
    P[f'{nm}A級']=O[f'{p}_g']>=3; P[f'{nm}B級']=O[f'{p}_g']<=2
    P[f'{nm}モーター+5以上']=O[f'{p}_mod']>=5; P[f'{nm}モーター-5以下']=O[f'{p}_mod']<=-5
    P[f'{nm}展示1〜2位']=O[f'{p}_er']<=2; P[f'{nm}展示5〜6位']=O[f'{p}_er']>=5
    P[f'{nm}スタ展1〜2番手']=O[f'{p}_sr']<=2; P[f'{nm}スタ展5〜6番手']=O[f'{p}_sr']>=5
    P[f'{nm}勝率+1以上']=O[f'{p}_nwd']>=1; P[f'{nm}勝率-1以下']=O[f'{p}_nwd']<=-1
    P[f'{nm}節間着順2.5以内']=O[f'{p}_rm']<=2.5; P[f'{nm}節間着順4.5以上']=O[f'{p}_rm']>=4.5
    P[f'{nm}節間モーター2.5以内']=O[f'{p}_mm']<=2.5; P[f'{nm}節間モーター4.5以上']=O[f'{p}_mm']>=4.5
for lo,hi in ((0,6),(6,10),(10,15),(15,20),(20,30),(30,50),(50,100)): P[f'オッズ{lo}〜{hi}倍']=(O.odds>=lo)&(O.odds<hi)
for lo,hi in ((0,.4),(.4,.6),(.6,.7),(.7,.8),(.8,1.01)): P[f'その頭の売れ方{int(lo*100)}〜{int(hi*100)}%']=(O.ph>=lo)&(O.ph<hi)
P['人気3番以内']=O.rk<=3; P['人気4〜10番']=(O.rk>=4)&(O.rk<=10); P['風5m以上']=O.wind>=5; P['進入変化あり']=O.moved==1; P['1〜4R']=O.rno<=4; P['9〜12R']=O.rno>=9
names=list(P); M=np.column_stack([P[n].fillna(False).values for n in names]); print('組',len(O),'条件',len(names))
per=np.where(O.hd<'20260101',0,np.where(O.hd<'20260701',1,2)); pay=O.pay.values
rid=(O.hd+O.jcd+O.rno.astype(str)).values
def mine(pay,label):
    S=[M[per==k] for k in range(3)]; Y=[pay[per==k] for k in range(3)]; found=[]
    n=len(names); MIN=3000
    for i in range(n):
        mi=S[0][:,i]
        if mi.sum()<MIN: continue
        for j in range(i+1,n):
            mij=mi&S[0][:,j]; c=mij.sum()
            if c<MIN: continue
            r=Y[0][mij].sum()/(100*c)
            if r>=1.03: found.append(((i,j),c,r))
            for k in range(j+1,n):
                m3=mij&S[0][:,k]; c3=m3.sum()
                if c3<MIN: continue
                r3=Y[0][m3].sum()/(100*c3)
                if r3>=1.03: found.append(((i,j,k),c3,r3))
    out=[]
    for idx,c,r in found:
        row=[' ＆ '.join(names[t] for t in idx),c,round(r,3)]
        for k in (1,2):
            m=np.logical_and.reduce([S[k][:,t] for t in idx]); cc=m.sum(); row+=[cc,round(Y[k][m].sum()/(100*cc),3) if cc else np.nan]
        out.append(row)
    T=pd.DataFrame(out,columns=['条件','発見の点数','発見','確認の点数','確認','最後の点数','最後'])
    print(f'\n[{label}] 発見で1.03以上（3,000点以上）: {len(T)}本 → 確認も1.0以上: {(T.確認>=1).sum()}本 → 最後も1.0以上: {((T.確認>=1)&(T.最後>=1)).sum()}本 ｜ 確認の平均 {T.確認.mean():.3f} 最後の平均 {T.最後.mean():.3f}')
    return T
T=mine(pay,'本物'); T.to_csv(D+'mine_real.csv',index=False)
ok=T[(T.確認>=1)&(T.最後>=1)].sort_values('最後の点数',ascending=False); print(ok.head(40).to_string(index=False))
# でたらめ版: 発見期間だけ、レース内の払戻をそのままに「どの組が当たりか」をオッズに比例する確率で引き直す（＝売れ方どおりの世界）
rng=np.random.default_rng(0); d=O[['hd','jcd','rno','inv','odds']].copy(); d['g']=rid
for rep in range(2):
    u=rng.random(len(d)); key=np.log(u)/d.inv.values   # Gumbel風: 1/オッズ に比例して1つ選ぶ（20〜100倍未満に入らない当たりは無視）
    d['k']=key; win=d.groupby('g').k.transform('max')==d.k; scale=rng.random(len(d))  # レースごとに「この中に当たりがある確率」= Σq
    tot=O.groupby(rid).q.transform('sum').values; has=pd.Series(rng.random(len(d))).groupby(rid).transform('first').values<tot
    fake=np.where(win.values&has,(O.odds.values*100*0.75/ (O.inv.groupby(rid).transform('sum').values/O.tot.values) *0+O.odds.values*100),0).astype('float32')
    Tf=mine(fake,f'でたらめ{rep+1}（売れ方どおりに当たりを引き直し、オッズそのまま払い戻す世界）')
