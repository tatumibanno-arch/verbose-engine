"""a074_1: 場×月の前年同月傾向。前年同月にその場で1号艇が1着になった割合が 0.6 超、
かつ今レースの1号艇の全国勝率が2〜4号艇の最大以上なら、単勝1を100円。
前年同月の結果は払戻ファイル（レース日の前年分だけ）から作る。前夜に分かる情報のみ。"""
import pandas as pd
from pathlib import Path
D = Path("/home/user/hunt")
PW_TH = 0.6

def _prev_year_winrate():
    rs, ps = [], []
    for per in ["A_disc", "A_valid"]:
        rs.append(pd.read_parquet(D / f"{per}_races.parquet", columns=["hd", "jcd", "rno"]))
        p = pd.read_parquet(D / f"{per}_payouts.parquet", filters=[("ticket_type", "==", "tansho")])
        ps.append(p[p.combination == "1"][["hd", "jcd", "rno"]].drop_duplicates())
    r = pd.concat(rs).drop_duplicates(); w = pd.concat(ps); w["win"] = 1
    r = r.merge(w, on=["hd", "jcd", "rno"], how="left"); r["win"] = r.win.fillna(0)
    r["ym"] = r.hd.str[:6]
    g = r.groupby(["jcd", "ym"]).win.mean().rename("pwin").reset_index()
    # 前年同月 → 翌年の同じ月に適用
    g["ym"] = (g.ym.str[:4].astype(int) + 1).astype(str) + g.ym.str[4:6]
    return g

def bets(df, odds=None):
    g = _prev_year_winrate()
    x = df[["hd", "jcd", "rno", "nw1", "nw2", "nw3", "nw4"]].copy(); x["ym"] = x.hd.str[:6]
    x = x.merge(g, on=["jcd", "ym"], how="left")
    m = (x.pwin > PW_TH) & (x.nw1 >= x[["nw2", "nw3", "nw4"]].max(axis=1))
    out = x.loc[m, ["hd", "jcd", "rno"]].copy()
    out["ticket_type"] = "tansho"; out["combination"] = "1"; out["amount"] = 100
    return out
