"""堅い3連複（3連単の締切時オッズから見て、その3艇で決まる確率が高い組）に展示を足す。前半=〜2026-01、後半=2026-02〜。
事前に決めた区切りだけ。締切時オッズ（締切前とは違う）。売れ方は3連単からの推定。"""
import sqlite3, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
D='/tmp/claude-0/-home-user-verbose-engine/968008a0-6ecf-5641-a3ff-006912b41722/scratchpad/n2/'; key=['hd','jcd','rno']
F=pd.read_parquet(D+'s3f.parquet'); c=sqlite3.connect('data/boatrace.db')
b=pd.read_sql("select hd,jcd,rno,lane,exhibition_time et from beforeinfo where hd>='20250301'",c); b['et']=pd.to_numeric(b.et,errors='coerce'); b['er']=b.groupby(key).et.rank(method='min')
w=b.pivot_table(index=key,columns='lane',values='er'); w.columns=[f'e{k}' for k in w.columns]; F=F.merge(w.reset_index(),on=key,how='inner').dropna(subset=[f'e{k}' for k in range(1,7)])
ers=F[[f'e{k}' for k in range(1,7)]].values; ins=np.array([[str(k) in s for k in range(1,7)] for s in F.s])
F['worst']=np.where(ins,ers,0).max(1); F['sumr']=np.where(ins,ers,0).sum(1); F['outbest']=np.where(~ins,ers,9).min(1); F['e1r']=F.e1
R=lambda s: s.pay.sum()/(100*len(s)) if len(s) else np.nan
def rep(nm,s):
    ym=s.hd.str[:6]; mo=s.groupby(ym).pay.sum()/(100*s.groupby(ym).size())
    print(f'{nm} | 点 {len(s)} 的中 {(s.pay>0).mean():.3f} 平均払戻 {s[s.pay>0].pay.mean():.0f} 回収 {R(s):.3f} 前半 {R(s[s.hd<"20260201"]):.3f} 後半 {R(s[s.hd>="20260201"]):.3f} 勝ち月 {(mo>1).sum()}/{len(mo)}')
for lo,nm in ((.5,'確率50%以上'),(.4,'確率40%以上'),(.3,'確率30%以上')):
    s=F[F.q>=lo]; print('==',nm); rep('全部',s)
    rep('3艇とも展示3位以内',s[s.worst<=3]); rep('3艇とも展示4位以内',s[s.worst<=4]); rep('3艇のだれかが展示5〜6位',s[s.worst>=5]); rep('3艇のだれかが展示6位',s[s.worst==6])
    rep('外の3艇に展示1位がいる',s[s.outbest==1]); rep('外の3艇に展示1位がいない',s[s.outbest>1]); rep('外の3艇に展示1〜2位がいない',s[s.outbest>2])
    rep('①が展示1〜2位（①を含む組）',s[(s.s.str[0]=='1')&(s.e1r<=2)]); rep('①が展示5〜6位（①を含む組）',s[(s.s.str[0]=='1')&(s.e1r>=5)])
