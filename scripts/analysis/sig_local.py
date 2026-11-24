"""3つの知らせの発火レースを、①が地元（支部＝場の県）かどうかで分ける。par_aff_build の feat を使う。"""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
exec(open('scripts/analysis/three_signals.py').read().split("R=lambda s,col")[0])
F=pd.read_parquet('/tmp/claude-0/-home-user-verbose-engine/968008a0-6ecf-5641-a3ff-006912b41722/scratchpad/par_aff/feat.parquet')
F['hd']=F.hd.astype(str); F['jcd']=F.jcd.astype(str).str.zfill(2); F['rno']=F.rno.astype(int); F['lane']=F.lane.astype(int)
S['hd']=S.hd.astype(str); S['rno']=S.rno.astype(int); S['jcd']=S.jcd.astype(str).str.zfill(2)
loc=F.pivot_table(index=key,columns='lane',values='local').reset_index(); loc.columns=key+[f'L{c}' for c in loc.columns[3:]]
D=S.drop_duplicates(key+['cb']).merge(loc,on=key,how='left'); D['ym']=D.hd.str[:6]; D['pay']=np.where(D.res==D.cb,D.payout,0)
def rep(nm,s):
    a=s.hd<str(cut); h=s.pay[s.pay>0].sort_values(ascending=False); mo=s.groupby('ym').pay.agg(['size','sum'])
    print(f'{nm} | {len(s)} 的中{(s.pay>0).mean():.3f} 回収{s.pay.sum()/100/max(len(s),1):.3f} 前{s.pay[a].sum()/100/max(a.sum(),1):.3f} 後{s.pay[~a].sum()/100/max((~a).sum(),1):.3f} 上位3除{(s.pay.sum()-h.head(3).sum())/100/max(len(s),1):.3f} 勝月{(mo["sum"]/100/mo["size"]>1).sum()}/{len(mo)}')
rep('①地元',D[D.L1==1]); rep('①地元でない',D[D.L1==0]); rep('不明',D[D.L1.isna()])
nl=D[['L2','L3']].sum(1)
rep('②③に地元あり',D[nl>=1]); rep('②③に地元なし',D[(nl==0)&D.L2.notna()])
