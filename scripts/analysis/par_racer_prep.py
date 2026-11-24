"""選手のスタイル（過去の成績から）。results の entrance_position から選手ごとの進入別成績を作る。"""
import sqlite3,pandas as pd,numpy as np
c=sqlite3.connect('data/boatrace.db')
R=pd.read_sql("select hd,jcd,rno,lane,position pos,json_extract(data,'$.racer_number') rn,json_extract(data,'$.entrance_position') cs from results where hd>='20240501'",c)
R.to_parquet('/tmp/claude-0/-home-user-verbose-engine/968008a0-6ecf-5641-a3ff-006912b41722/scratchpad/res.parquet')
print(len(R),R.isna().mean().to_dict())
