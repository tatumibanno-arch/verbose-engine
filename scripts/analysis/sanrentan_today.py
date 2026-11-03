"""3連単「期待値最大の1点」（紙上）を当日のレースに当てる。rules.yaml strategies.sanrentan_ev1_v1。

候補（notes/analysis/multi_bet.md の 3T:EV上位k:g:p>=0.01:k=1:EV>=1.2）:
  各レースで、モデル確率 P ≥ min_prob の3連単のうち 期待値 = P × 予想配当 g / 100 が最大の1点を、
  期待値 ≥ min_ev のときだけ買う。

前夜の情報だけを使う:
  * 着順確率 P: nirensho.py today が作った pred/today_<date>.npz（予測月の直前24か月で学習）
  * 予想配当 g: 予測日より前に当たった3連単すべてで学習（trifecta.g_features、gamma 回帰）

使い方:
    python scripts/analysis/sanrentan_today.py --date 20261104
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import trifecta as T  # noqa: E402
from common import VENUES, load_rules  # noqa: E402

RULE = "sanrentan_ev1_v1"


def main():
    import lightgbm as lgb
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    ap.add_argument("--workdir", default=str(T.DEFAULT_WORK))
    a = ap.parse_args()
    work = Path(a.workdir)
    T.WORK = work
    D = a.date
    cfg = load_rules()["strategies"][RULE]

    # ---- 当日の材料 ----
    zt = np.load(work / "pred" / f"today_{D}.npz")
    te = pd.read_parquet(work / "feat" / f"today_{D}.parquet").sort_values(["hd", "jcd", "rno", "lane"])
    n6 = te.groupby(["hd", "jcd", "rno"])["lane"].transform("size")
    te = te[n6 == 6].reset_index(drop=True)
    tr = te.iloc[::6][["hd", "jcd", "rno"] + T.RACE_FLAGS].reset_index(drop=True)
    assert (tr.jcd.to_numpy().astype(str) == zt["jcd"]).all() and (tr.rno.to_numpy() == zt["rno"]).all()
    Pt, P1t = zt["P"].astype(float), zt["P1"].astype(float)
    nwr_t = te["national_win_rate"].to_numpy(np.float32).reshape(-1, 6)
    grd_t = te["grade"].to_numpy(np.float32).reshape(-1, 6)

    # ---- 予想配当 g: 当日より前の当たった3連単すべて ----
    races, R, _void3, nwr, grd = T.load_eval(work)
    hit = races["hit"].to_numpy()
    ix = np.where((races["hd"].to_numpy() < D) & (hit >= 0))[0]
    X = T.g_features(races.iloc[ix], R["P"][ix], R["P1"][ix], nwr[ix], grd[ix]).to_numpy().reshape(len(ix), 120, -1)
    Xh = X[np.arange(len(ix)), hit[ix]]
    del X
    gm = lgb.train(T.G_PARAMS, lgb.Dataset(Xh, races["payout"].to_numpy()[ix]), T.G_ROUNDS)
    T.log(f"g train wins={len(ix)}")
    G = gm.predict(T.g_features(tr, Pt, P1t, nwr_t, grd_t).to_numpy()).reshape(-1, 120)
    EV = Pt * G / 100

    E = np.where(Pt >= cfg["min_prob"], EV, -np.inf)
    best = E.argmax(1)
    names = te[["jcd", "rno", "lane", "racer_name"]] if "racer_name" in te.columns else None
    out = []
    for r in range(len(tr)):
        k = best[r]
        a_, b_, c_ = T.PERMS[k] + 1
        out.append(dict(jcd=str(tr.jcd[r]).zfill(2), rno=int(tr.rno[r]), comb=f"{a_}-{b_}-{c_}",
                        p=round(float(Pt[r, k]), 4), g=round(float(G[r, k])), ev=round(float(E[r, k]), 3),
                        buy=bool(E[r, k] >= cfg["min_ev"])))
    df = pd.DataFrame(out)
    df.to_csv(work / f"sanrentan_today_{D}.csv", index=False)
    b = df[df.buy].copy()
    b["venue"] = b.jcd.map(VENUES)
    print(b.to_string(index=False))
    print(f"buy {len(b)} / {len(df)} races")


if __name__ == "__main__":
    main()
