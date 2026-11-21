"""a075_1: 選手の過去3暦年の単勝回収率（枠の平均で割った補正値）が高い選手を、1号艇が弱いレースの2〜4号艇で単勝買い。
レース年 Y の判定には Y-3〜Y-1 年の結果だけを使う（前向き）。
条件: 2〜4号艇、補正回収率 >= ADJ、過去出走数 >= 300、1号艇の全国勝率 <= 5.0。買い方: その艇の単勝100円。"""
import pandas as pd
from pathlib import Path
D = Path("/home/user/hunt")
ADJ, NMIN, NW1 = 1.3, 300, 5.0

def _hist():
    Ls = []
    for per in ["A_disc", "A_valid"]:
        rp, pp = D / f"{per}_races.parquet", D / f"{per}_payouts.parquet"
        if not rp.exists(): continue
        r = pd.read_parquet(rp, columns=["hd", "jcd", "rno"] + [f"racer{k}" for k in range(1, 7)])
        p = pd.read_parquet(pp); p = p[p.ticket_type == "tansho"].groupby(["hd", "jcd", "rno", "combination"], as_index=False).payout.max()
        for k in range(1, 7):
            x = r[["hd", "jcd", "rno", f"racer{k}"]].rename(columns={f"racer{k}": "racer"}); x["lane"] = k; x["combination"] = str(k)
            Ls.append(x.merge(p, on=["hd", "jcd", "rno", "combination"], how="left"))
    L = pd.concat(Ls, ignore_index=True).drop_duplicates(["hd", "jcd", "rno", "lane"])
    L["payout"] = L.payout.fillna(0); L["yr"] = L.hd.str[:4].astype(int)
    return L

def bets(df, odds=None):
    L = _hist(); out = []
    df = df.copy(); df["yr"] = df.hd.str[:4].astype(int)
    for Y in sorted(df.yr.unique()):
        past = L[(L.yr >= Y - 3) & (L.yr < Y)]
        if len(past) == 0: continue
        lm = past.groupby("lane").payout.mean()
        past = past.assign(ex=past.lane.map(lm))
        s = past.groupby("racer").agg(n=("payout", "size"), tot=("payout", "sum"), ex=("ex", "sum"))
        s = s[(s.n >= NMIN) & (s.tot / s.ex >= ADJ)]
        good = set(s.index)
        d = df[(df.yr == Y) & (df.nw1 <= NW1)]
        for k in (2, 3, 4):
            m = d[f"racer{k}"].isin(good)
            x = d.loc[m, ["hd", "jcd", "rno"]].copy(); x["ticket_type"] = "tansho"; x["combination"] = str(k); x["amount"] = 100
            out.append(x)
    return pd.concat(out, ignore_index=True)
