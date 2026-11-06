"""2連単「EV最大1点」候補を当日のレースに当てる（紙上の精査用。本番の選定ではない）。

候補（notes/analysis/multi_bet.md）: 2連単で、モデル確率 ≥ 5% の目のうち 期待値（確率×予想配当 g2+）が最大の1点を、
期待値 ≥ 1.2 のときだけ買う（ev / buy 列。補正なし、比較用に残す）。
2026-11-07 から rules.yaml の strategies.nirentan_ev1cal_v1（紙上）: 同じ1点の期待値を calibration.md の (b) 直接補正
（当日の月より前の最大12か月の「選ばれた買い目」で当てた係数。nirentan_cal.py）で補正し、補正後 ev_cal ≥ min_ev_cal で買う（buy_cal 列）。
CSV には head_not1（頭が1号艇以外 ★、H2）・second（2着艇、H1 は 1-2）も出し、--picks-md で notes/picks/<日付>-nirentan.md を書く。

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
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import multi_bet as M  # noqa: E402
import nirensho as N  # noqa: E402
import nirentan_cal as C  # noqa: E402
import trifecta as T  # noqa: E402

PMIN, EVMIN = 0.05, 1.2      # 補正なしの旧候補（比較用）。補正あり（本番候補）の閾値は rules.yaml の min_ev_cal


def main():
    import lightgbm as lgb
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    ap.add_argument("--workdir", default=str(T.DEFAULT_WORK))
    ap.add_argument("--q2", default="auto", choices=["auto", "agg", "direct"],
                    help="q̂2 の作り。auto: 2連単オッズの揃った月が MIN_PRIOR_ODDS 以上あれば direct")
    ap.add_argument("--picks-md", action="store_true", help="notes/picks/<日付>-nirentan.md も書く（締切・選手名は DB の entries から）")
    a = ap.parse_args()
    R = C.rules()
    assert abs(float(R["min_prob"]) - PMIN) < 1e-12, "rules.yaml の min_prob と旧候補の下限が違う"
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

    # ---- 候補: 確率≥5% の中で生の EV 最大の1点（選ぶ目は補正なし・ありで同じ） ----
    E = np.where(P2t >= PMIN, EV, -np.inf)
    best = E.argmax(1)
    out = []
    for r in range(len(tr_races)):
        k = best[r]
        out.append(dict(jcd=str(tr_races.jcd[r]), rno=int(tr_races.rno[r]),
                        comb=f"{M.PAIRS[k][0]+1}-{M.PAIRS[k][1]+1}", head=int(M.PAIRS[k][0] + 1), second=int(M.PAIRS[k][1] + 1),
                        p=P2t[r, k], g=G2t[r, k], ev=E[r, k], mkt_odds=1 / q2t[r, k] * 0.75, buy=bool(E[r, k] >= EVMIN)))
    df = pd.DataFrame(out)
    df["head_not1"] = df["head"] != 1
    # ---- 補正あり期待値（nirentan_ev1cal_v1）: 当日の月より前の「選ばれた買い目」で係数を当てる（当日の情報は入らない） ----
    hist = C.history(work, use_strict=True)
    hist = hist[hist.hd < D]
    coef, pm = C.fit_for_month(hist, cur, R)
    if coef is None:
        T.log(f"補正の学習月が足りない（{pm}）。ev_cal は出さない")
        df["ev_cal"] = np.nan
        df["calib_src"] = ""
    else:
        fit_rows = hist[hist.month.isin(pm)]
        df["ev_cal"] = C.apply_calibration(coef, df.ev, df.p, df["head"], df.second, R["calib_features"])
        df["calib_src"] = "+".join(f"{k}:{v}" for k, v in fit_rows.src.value_counts().sort_index().items())
        names = C.design(df.ev.to_numpy(), df.p.to_numpy(), df["head"].to_numpy(), df.second.to_numpy(), R["calib_features"])[1]
        T.log(f"補正の係数 fit={pm[0]}..{pm[-1]} n={len(fit_rows)} src={df.calib_src[0]} "
              + " ".join(f"{n}={c:+.3f}" for n, c in zip(names, coef)))
    df["buy_cal"] = df.ev_cal >= float(R["min_ev_cal"])
    df["ev_min_cal"] = float(R["min_ev_cal"])
    df.to_csv(work / f"nirensho_today_{D}.csv", index=False)
    show = df[df.buy | df.buy_cal].sort_values("ev_cal", ascending=False)
    print(show[["jcd", "rno", "comb", "head_not1", "second", "p", "g", "ev", "ev_cal", "mkt_odds", "buy", "buy_cal"]].to_string(index=False))
    print(f"buy(raw EV>={EVMIN}) {df.buy.sum()} / buy_cal(EV_cal>={R['min_ev_cal']}) {df.buy_cal.sum()} / {len(df)} races")
    if a.picks_md:
        write_picks_md(D, df, R)


def write_picks_md(D, df, R):
    """notes/picks/<日付>-nirentan.md。表は補正あり期待値の降順。★ = 頭が1号艇以外（H2）。1-2 は H1。"""
    import sqlite3
    from common import VENUES  # scripts/common.py
    con = sqlite3.connect(f"file:{T.DB}?mode=ro", uri=True, timeout=600)
    en = pd.read_sql("select jcd, rno, lane, racer_name, close_time from entries where hd=?", con, params=(D,))
    nm = {(r.jcd, int(r.rno), int(r.lane)): str(r.racer_name).strip() for r in en.itertuples()}
    ct = {(r.jcd, int(r.rno)): r.close_time for r in en.drop_duplicates(["jcd", "rno"]).itertuples()}
    tau = float(R["min_ev_cal"])
    d = df.copy()
    d["close"] = [ct.get((j, r), "") for j, r in zip(d.jcd, d.rno)]
    lines = [f"# {D[:4]}-{D[4:6]}-{D[6:]} 2連単（前夜の情報で期待値最大1点、紙上）", "",
             f"`nirentan_ev1cal_v1`（rules v15、紙上）: モデル確率 ≥ {R['min_prob']:.0%} の中で生の期待値が最大の1点を、"
             f"**補正後の期待値 ≥ {tau}** のときだけ100円。補正は notes/analysis/nirentan_calibrated.md（(b) 直接補正、"
             f"当日の月より前の最大{R['calib_window_months']}か月で学習。学習行の q̂: {d.calib_src.iloc[0] if len(d) else ''}）。",
             "★ = 頭が1号艇以外（H2）。1-2 は H1。「生」は補正なしの期待値（旧候補 ≥ 1.2 は参考）。", ""]
    sel = d[d.buy_cal].sort_values(["close", "jcd", "rno"])
    lines += [f"## 買い目（補正後 ≥ {tau}）: {len(sel)} 点", "",
              "| 締切 | レース | 2連単 | ★ | 並び | 確率 | 予想配当 | 期待値(生) | 期待値(補正) |", "|---|---|---|---|---|---|---|---|---|"]
    for r in sel.itertuples():
        pair = f"{nm.get((r.jcd, r.rno, r.head), '?')}→{nm.get((r.jcd, r.rno, r.second), '?')}"
        lines.append(f"| {r.close} | {VENUES.get(r.jcd, r.jcd)}{r.rno}R | {r.comb} | {'★' if r.head_not1 else ''} | {pair} | "
                     f"{r.p:.1%} | {r.g/100:.0f}倍 | {r.ev:.2f} | {r.ev_cal:.2f} |")
    extra = d[d.buy & ~d.buy_cal].sort_values(["close", "jcd", "rno"])
    lines += ["", f"## 参考: 生の期待値 ≥ {EVMIN} だが補正後 < {tau}（買わない）: {len(extra)} 点", ""]
    if len(extra):
        lines += ["| 締切 | レース | 2連単 | ★ | 確率 | 予想配当 | 期待値(生) | 期待値(補正) |", "|---|---|---|---|---|---|---|---|"]
        for r in extra.itertuples():
            lines.append(f"| {r.close} | {VENUES.get(r.jcd, r.jcd)}{r.rno}R | {r.comb} | {'★' if r.head_not1 else ''} | "
                         f"{r.p:.1%} | {r.g/100:.0f}倍 | {r.ev:.2f} | {r.ev_cal:.2f} |")
    for t in R["report_thresholds"]:
        s = d[d.ev_cal >= float(t)]
        lines.append(f"- 補正後 ≥ {t}: {len(s)} 点（★ {int(s.head_not1.sum())}、1-2 {int(((s["head"] == 1) & (s.second == 2)).sum())}）")
    lines += ["", "## 実購入メモ（朝に人が書く）", "", "- "]
    out = T.ROOT / "notes" / "picks" / f"{D[:4]}-{D[4:6]}-{D[6:]}-nirentan.md"
    out.write_text("
".join(lines) + "
")
    T.log(f"wrote {out}")


if __name__ == "__main__":
    main()
