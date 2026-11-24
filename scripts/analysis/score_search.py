"""3連単①頭1点の「点数の作り方」を総当たり（オッズを使わない）。材料8つ（一周・まわり足・直線・公式展示・スタ展・全国勝率・モーター・枠の2連率）の
組み合わせ255通り × 「上位3艇と4番手の差」の線3つ（そのレースの差が上位2%・4%・8%に入る）＝765通り。
探す 2025-06〜12 → 確かめる 2026-01〜06 → 最後 2026-07〜11。偶然の比較: 売れ方（確定の3連単オッズ）どおりに当たりを引き直した世界を3回。"""
import sqlite3, itertools, numpy as np, pandas as pd, warnings, sys; warnings.filterwarnings('ignore')
exec(open('scripts/analysis/one_pick.py').read().split("run('公式展示だけ'")[0])
feats=['r_lap','r_maw','r_cho','r_ex','r_st','r_nw','r_mo','r_c2']
X=X.sort_values(key+['lane']).reset_index(drop=True); X['rid']=X.groupby(key).ngroup()
races=X.drop_duplicates('rid')[key+['rid']].merge(p,on=key,how='left').set_index('rid').sort_index()
nr=len(races); per=np.where(races.hd<'20260101',0,np.where(races.hd<'20260701',1,2))
# 偶然の世界: レースごとに売れ方どおりに当たりの組を引く
o=pd.read_sql("select hd,jcd,rno,combination cb,odds from odds where ticket_type='sanrensho' and hd>='20250601' and odds>0",c).merge(races.reset_index()[key+['rid']],on=key)
o['inv']=1/o.odds; o=o.sort_values(['rid','cb']); o['q']=o.inv/o.groupby('rid').inv.transform('sum'); o['cq']=o.groupby('rid').q.cumsum()
odd=o.set_index(['rid','cb']).odds
fakes=[]
for seed in (1,2,3):
    rng=np.random.default_rng(seed); u=pd.Series(rng.random(nr),index=races.index); oo=o.assign(u=o.rid.map(u)); w=oo[oo.cq>=oo.u].groupby('rid').cb.first(); fakes.append(w.reindex(races.index))
lanes=X.lane.values; et=X.et.values; rid=X.rid.values
res_real=races.res.values; pay_real=races.payout.fillna(0).values
out=[]
for k in range(1,len(feats)+1):
    for sub in itertools.combinations(feats,k):
        sc=X[list(sub)].sum(1).values
        df=pd.DataFrame({'rid':rid,'lane':lanes,'sc':sc,'et':et}).sort_values(['rid','sc','lane']); df['o']=df.groupby('rid').cumcount()+1
        t=df[df.o<=4].pivot(index='rid',columns='o',values=['lane','sc','et'])
        ok=(t[('lane',1)]==1)&t[('sc',4)].notna(); t=t[ok]; gap=(t[('sc',4)]-t[('sc',3)])
        a=t[('lane',2)].astype(int).astype(str); b=t[('lane',3)].astype(int).astype(str); f=t[('et',2)]<=t[('et',3)]
        cb=pd.Series('1-'+np.where(f,a,b)+'-'+np.where(f,b,a),index=t.index)
        for qn in (0.98,0.96,0.92):
            th=gap.quantile(qn); sel=gap[gap>=th].index
            if len(sel)<150: continue
            pc=cb[sel]; rr=races.loc[sel]; hit=(rr.res.values==pc.values); pay=np.where(hit,rr.payout.fillna(0).values,0); pr=per[sel]
            row={'材料':'+'.join(s[2:] for s in sub),'線':f'上位{int(round((1-qn)*100))}%','レース':len(sel)}
            for pi,nm in ((0,'探す'),(1,'確かめる'),(2,'最後')): m=pr==pi; row[nm]=pay[m].sum()/(100*m.sum()) if m.sum() else np.nan
            for fi,fw in enumerate(fakes):
                fh=(fw.loc[sel].values==pc.values); fo=np.array([odd.get((r_,cb_),0) for r_,cb_ in zip(sel,pc.values)])
                fp=np.where(fh,fo*100,0)
                for pi,nm in ((0,'探す'),(1,'確かめる'),(2,'最後')): m=pr==pi; row[f'偶然{fi+1}_{nm}']=fp[m].sum()/(100*m.sum()) if m.sum() else np.nan
            out.append(row)
    print('材料の数',k,'済',len(out),flush=True)
T=pd.DataFrame(out); T.to_csv('/tmp/claude-0/-home-user-verbose-engine/968008a0-6ecf-5641-a3ff-006912b41722/scratchpad/n2/score_search.csv',index=False)
def count(prefix):
    a=T[f'{prefix}探す']>=1.05; b=a&(T[f'{prefix}確かめる']>=1.0); c_=b&(T[f'{prefix}最後']>=1.0); return a.sum(),b.sum(),c_.sum()
print('本物: 探す1.05以上 → 確かめるも1.0以上 → 最後も1.0以上',count(''))
for i in (1,2,3): print(f'偶然{i}:',count(f'偶然{i}_'))
ok=T[(T.探す>=1.05)&(T.確かめる>=1.0)&(T.最後>=1.0)].sort_values('レース',ascending=False); print(ok.head(25)[['材料','線','レース','探す','確かめる','最後']].to_string(index=False))
