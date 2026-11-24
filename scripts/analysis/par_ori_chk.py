import numpy as np, pandas as pd
exec(open('scripts/analysis/three_signals.py').read().split("S=pd.concat")[0])
exec(open('scripts/analysis/par_ori_deep.py').read().split("# ---- A")[0])
mine=R0[S1m&ok&has].assign(m=1); th=S1[key+['cb']].assign(t=1)
mm=mine.merge(th,on=key,how='outer'); print(len(mine),len(th),mm.m.isna().sum(),mm.t.isna().sum())
print(mm[mm.m.isna()|mm.t.isna()].merge(p,on=key,how='left').head(30))
print(len(th.merge(p,on=key)), len(X[key].drop_duplicates()), len(R0))
