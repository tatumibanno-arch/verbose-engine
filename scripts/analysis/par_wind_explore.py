import pandas as pd, numpy as np
exec(open('scripts/analysis/one_pick.py').read().split("run('公式展示だけ'")[0])
W=pd.read_pickle('/tmp/claude-0/-home-user-verbose-engine/968008a0-6ecf-5641-a3ff-006912b41722/scratchpad/wind.pkl')
Q=p.merge(W,on=key); Q['w1']=Q.res.str[0]=='1'; Q['n']=1
print(len(Q))
s=Q[Q.ws>=3]
print(s.groupby('wd').w1.agg(['mean','count']).T.round(3).to_string())
print(Q.groupby(pd.cut(Q.ws,[-1,0,1,2,3,4,5,7,20])).w1.agg(['mean','count']).round(3))
print(Q.groupby(pd.cut(Q.wave,[-1,0,1,2,3,5,30])).w1.agg(['mean','count']).round(3))
# per venue: code with extreme 1win
v=s.groupby(['jcd','wd']).w1.agg(['mean','count']); print(v[v['count']>150].sort_values('mean').iloc[[0,1,2,-3,-2,-1]])
