"""2連単「EV最大1点」候補を当日のレースに当てる（紙上の精査用。本番の選定ではない）。

候補（notes/analysis/multi_bet.md）: 2連単で、モデル確率 ≥ 5% の目のうち 期待値（確率×予想配当 g2+）が最大の1点を、
期待値 ≥ 1.2 のときだけ買う。

本番と同じく前夜の情報だけを使う:
  * 着順確率 P: nirensho.py today が作った pred/today_<date>.npz（予測月の直前24か月で学習）
  * 予想市場確率 q̂: 当日の月より前の「3連単オッズの揃った月」（multi_bet.py prep が判定）すべてで学習した
    「前夜の情報 → 締切オッズ」モデル。2連単オッズの揃った月が MIN_PRIOR_ODDS 以上あれば、2連単は直接学習した q̂2
  * 予想配当 g2+: 予測日より前の当たった2連単すべてで学習（素性に q̂2。学習行の q̂2 は oddswf のウォークフォワード版）
  * 前提: multi_bet.py prep → oddswf（direct なら odds2wf も）が済んでいること

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
    ap.add_argument("--q2", default="auto", choices=["auto", "agg", "direct"],
                    help="q̂2 の作り。auto: 2連単オッズの揃った月が MIN_PRIOR_ODDS 以上あれば direct")
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

    # ---- 予想オッズモデル: 当日の月より前の「揃ったオッズ月」すべて（v2 の oddswf と同じ作り） ----
    races, z = M.load(work)
    month = races.month.to_numpy()
    cur = f"{D[:4]}-{D[4:6]}"
    om3 = [m for m in M.odds_months(work, "3t") if m < cur]
    O = z["O"].astype(float)
    has = np.isfinite(O).all(1) & (O > 0).all(1) & np.isin(month, om3)
    om = M.fit_q_model(lambda ix: M.odds_features(races, z, ix), np.where(has)[0], O, 120)[0]
    T.log(f"q̂ (3連単) train months={om3[0]}..{om3[-1]} ({len(om3)})")
    LQt = M.predict_q(om, lambda ix: M.odds_features(tr_races, zt_d, ix), np.arange(len(tr_races)), 120)
    q2t = M.norm_q(LQt) @ M.AGG
    om2 = [m for m in M.odds_months(work, "2t") if m < cur]
    src = a.q2
    if src == "auto":
        src = "direct" if len(om2) >= M.MIN_PRIOR_ODDS else "agg"
    if src == "direct":
        LQ3, _, _ = M.load_wf(work, "q3", races, 120)
        O2 = z["O2"].astype(float)
        has2 = np.isfinite(O2).all(1) & (O2 > 0).all(1) & np.isin(month, om2) & np.isfinite(LQ3).all(1)
        m2 = M.fit_q_model(lambda ix: M.odds2_features(races, z, ix, LQ3), np.where(has2)[0], O2, 30)[0]
        q2t = M.norm_q(M.predict_q(m2, lambda ix: M.odds2_features(tr_races, zt_d, ix, LQt), np.arange(len(tr_races)), 30))
        T.log(f"q̂2 (2連単オッズで直接) train months={om2[0]}..{om2[-1]} ({len(om2)})")

    # ---- g2+（当日より前の当たった2連単すべて。学習行の q̂2 はウォークフォワード版 = 各月その月より前のオッズだけ） ----
    Q2, _ = M.wf_q2(work, races, src)
    hit2, pay2 = races.hit2.to_numpy(), races.pay2.to_numpy()
    ix = np.where((races.hd.to_numpy() < D) & (hit2 >= 0) & np.isfinite(Q2).all(1))[0]
    Xh = M.g2p_X(races, z, ix, Q2).reshape(len(ix), 30, -1)[np.arange(len(ix)), hit2[ix]]
    gm = lgb.train(M.GP_PARAMS, lgb.Dataset(Xh, pay2[ix]), T.G_ROUNDS)
    T.log(f"g2+ train wins={len(ix)} q2={src}")
    P2t = Pt @ M.AGG
    G2t = gm.predict(M.g2p_X(tr_races, zt_d, np.arange(len(tr_races)), q2t), num_threads=M.NT).reshape(-1, 30)
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
