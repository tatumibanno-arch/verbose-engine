"""2連単「EV最大1点」候補を当日のレースに当てる（紙上の精査用。本番の選定ではない）。

候補（notes/analysis/multi_bet.md）: 2連単で、モデル確率 ≥ 5% の目のうち 期待値（確率×予想配当 g2+）が最大の1点を、
期待値 ≥ 1.2 のときだけ買う。

本番と同じく前夜の情報だけを使う:
  * 着順確率 P: nirensho.py today が作った pred/today_<date>.npz（予測月の直前24か月で学習）
  * 予想市場確率 q̂: 3連単の締切オッズがある全月（2026-03〜11-01）で学習した「前夜の情報 → 締切オッズ」モデル
  * 予想配当 g2+: 予測日より前の当たった2連単すべてで学習（素性に q̂ を含む）

使い方:
    python scripts/analysis/nirensho_today.py --date 20261103
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import multi_bet as M  # noqa: E402
import nirensho as N  # noqa: E402
import trifecta as T  # noqa: E402

PMIN, EVMIN = 0.05, 1.2


def main():
    import lightgbm as lgb
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    ap.add_argument("--workdir", default=str(T.DEFAULT_WORK))
    a = ap.parse_args()
    work = Path(a.workdir)
    T.WORK = work
    D = a.date

    # ---- 当日の材料 ----
    zt = np.load(work / "pred" / f"today_{D}.npz")
    te = pd.read_parquet(work / "feat" / f"today_{D}.parquet").sort_values(["hd", "jcd", "rno", "lane"])
    n6 = te.groupby(["hd", "jcd", "rno"])["lane"].transform("size")
    te = te[n6 == 6].reset_index(drop=True)
    tr_races = te.iloc[::6][["hd", "jcd", "rno"] + T.RACE_FLAGS].reset_index(drop=True)
    assert (tr_races.jcd.to_numpy().astype(str) == zt["jcd"]).all() and (tr_races.rno.to_numpy() == zt["rno"]).all()
    Pt = zt["P"].astype(float)
    P1t = zt["P1"].astype(float)
    nwr_t = te["national_win_rate"].to_numpy(np.float32).reshape(-1, 6)
    grd_t = te["grade"].to_numpy(np.float32).reshape(-1, 6)
    ex_t = np.stack([te[c].to_numpy(np.float32).reshape(-1, 6) for c in M.EXTRA], 2)
    zt_d = dict(P=Pt.astype(np.float32), P1=P1t.astype(np.float32), nwr=nwr_t, grd=grd_t, ex=ex_t)

    # ---- 予想オッズモデル（オッズのある全月） ----
    races, z = M.load(work)
    O = z["O"].astype(float)
    has = np.isfinite(O).all(1) & (O > 0).all(1)
    q, _ = M.market_q(np.where(has[:, None], O, 1.0))
    rows = np.where(has)[0]
    Xall = M.odds_features(races, z, rows)
    cols = list(Xall.columns)
    Xn = Xall.to_numpy(np.float32)
    del Xall
    y = np.log(q[rows]).ravel().astype(np.float32)
    T.log(f"odds model train races={len(rows)}")
    om = lgb.train(M.ODDS_PARAMS, lgb.Dataset(Xn, y, feature_name=cols, free_raw_data=True), M.ODDS_ROUNDS)
    del Xn, y
    LQt = om.predict(M.odds_features(tr_races, zt_d, np.arange(len(tr_races))).to_numpy(np.float32),
                     num_threads=M.NT).reshape(-1, 120)
    q2t = M.norm_q(LQt) @ M.AGG

    # ---- g2+（当日より前の当たった2連単すべて） ----
    qh = M.norm_q(np.load(M.mb(work) / "qhat.npz")["LQ"].astype(float))
    q2h = qh @ M.AGG
    P2 = z["P"].astype(float) @ M.AGG
    flags = races[T.RACE_FLAGS].to_numpy(np.float32)

    def X2(rc, P2_, P1_, nwr_, grd_, fl_, Q2):
        X = N.g2_features(P2_, P1_, nwr_, grd_, fl_)
        X["jcd"] = np.repeat(rc.jcd.astype(int).to_numpy(), 30)
        X["rno"] = np.repeat(rc.rno.to_numpy(), 30)
        X["lq2h"] = np.log(Q2).ravel()
        X["lq2h_rank"] = ((-Q2).argsort(1).argsort(1) + 1).ravel()
        X["dlog"] = X["logP"] - X["lq2h"]
        return X.to_numpy(np.float32)

    hit2, pay2 = races.hit2.to_numpy(), races.pay2.to_numpy()
    ix = np.where((races.hd.to_numpy() < D) & (hit2 >= 0))[0]
    Xh = X2(races.iloc[ix], P2[ix], z["P1"][ix].astype(float), z["nwr"][ix], z["grd"][ix], flags[ix], q2h[ix])
    Xh = Xh.reshape(len(ix), 30, -1)[np.arange(len(ix)), hit2[ix]]
    gm = lgb.train(M.GP_PARAMS, lgb.Dataset(Xh, pay2[ix]), T.G_ROUNDS)
    T.log(f"g2+ train wins={len(ix)}")
    P2t = Pt @ M.AGG
    G2t = gm.predict(X2(tr_races, P2t, P1t, nwr_t, grd_t, tr_races[T.RACE_FLAGS].to_numpy(np.float32), q2t),
                     num_threads=M.NT).reshape(-1, 30)
    EV = P2t * G2t / 100

    # ---- 候補: 確率≥5% の中で EV 最大の1点、EV≥1.2 ----
    E = np.where(P2t >= PMIN, EV, -np.inf)
    best = E.argmax(1)
    out = []
    for r in range(len(tr_races)):
        k = best[r]
        out.append(dict(jcd=str(tr_races.jcd[r]), rno=int(tr_races.rno[r]),
                        comb=f"{M.PAIRS[k][0]+1}-{M.PAIRS[k][1]+1}", p=P2t[r, k], g=G2t[r, k], ev=E[r, k],
                        mkt_odds=1 / q2t[r, k] * 0.75, buy=bool(E[r, k] >= EVMIN)))
    df = pd.DataFrame(out)
    df.to_csv(work / f"nirensho_today_{D}.csv", index=False)
    print(df[df.buy].to_string(index=False))
    print(f"buy {df.buy.sum()} / {len(df)} races")


if __name__ == "__main__":
    main()
