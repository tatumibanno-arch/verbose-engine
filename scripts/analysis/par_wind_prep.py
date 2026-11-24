"""風・波データのレース単位化（オッズ不使用）。展示時点の気象。"""
import sqlite3, json, pandas as pd
c=sqlite3.connect('data/boatrace.db')
rows=[]
for hd,jcd,rno,d in c.execute("select hd,jcd,rno,data from beforeinfo where lane=1"):
    j=json.loads(d); rows.append((hd,jcd,rno,j.get('wind_speed'),j.get('wind_direction'),j.get('wave_height'),j.get('air_temp')))
W=pd.DataFrame(rows,columns=['hd','jcd','rno','ws','wd','wave','temp'])
W.to_pickle('/tmp/claude-0/-home-user-verbose-engine/968008a0-6ecf-5641-a3ff-006912b41722/scratchpad/wind.pkl')
print(len(W)); print(W.describe()); print(W.wd.value_counts().sort_index())
