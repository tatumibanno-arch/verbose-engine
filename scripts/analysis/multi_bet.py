"""2連単・3連単の複数点買い: 予想オッズ（市場の見立て）のモデル、買い方の探索、モデルと市場のズレ。

使い方（作業ファイルは trifecta.py と同じ workdir の mb/ 以下）:
    python scripts/analysis/multi_bet.py prep        # 予測・払戻・返還・前夜の素性・締切オッズをレース単位にまとめる
    python scripts/analysis/multi_bet.py oddsmodel   # 前夜の情報 → 3連単締切オッズ のモデル（市場確率 q̂）と g の比較
    python scripts/analysis/multi_bet.py gplus       # 当たり配当モデル g に q̂ を足した g+（3連単・2連単）
    python scripts/analysis/multi_bet.py market      # モデルと市場のズレ（締切オッズのある 2026-03〜10）
    python scripts/analysis/multi_bet.py bets        # 事前に決めた買い方の候補を発見用・検証用で評価
結果は notes/analysis/multi_bet.md にまとめる。

リークの扱い:
  * 選定の入力は前夜の情報だけ（着順モデル P は OOS、g / g+ は予測月より前の当たりだけで学習）。
  * 予想オッズ q̂ のモデルは「締切オッズ」を目的変数にする。締切オッズは 2025-11〜2026-11-01 にしか無いので、
      - オッズの無い月（〜2025-10）: オッズのある全月で学習したモデル
      - オッズのある月 m: m を除いた月で学習したモデル（leave-one-month-out）
    とした。目的変数は市場のオッズで、レースの結果ではない。m の結果が他の月のオッズに入る経路は無い。
    （念のため、2026-01〜10 は「m より前のオッズだけ」で学習した厳密版でも数字を出す）
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import trifecta as T  # noqa: E402

PERMS = T.PERMS
PAIRS = np.array([(i, j) for i in range(6) for j in range(6) if i != j])
_PI = {tuple(p): n for n, p in enumerate(PAIRS.tolist())}
AGG = np.zeros((120, 30), np.float64)
for _n, (_a, _b, _c) in enumerate(PERMS.tolist()):
    AGG[_n, _PI[(_a, _b)]] = 1.0
DISC = [f"{y}-{m:02d}" for y, m in [(2024, 11), (2024, 12)] + [(2025, i) for i in range(1, 11)]]
VALID = [f"{y}-{m:02d}" for y, m in [(2025, 11), (2025, 12)] + [(2026, i) for i in range(1, 11)]]
ODDS_MONTHS = ["2025-11", "2025-12"] + [f"2026-{i:02d}" for i in range(1, 12)]   # 3連単の締切オッズがある月（2025-11〜2026-11-01）
EXTRA = ["local_win_rate", "national_2ren_rate", "motor_2ren_rate", "boat_2ren_rate", "age", "ser_n", "ser_avg_pos",
         "ser_win", "ser_F", "rs_avg_st", "r90_win", "rc_win_shrunk", "rs_ability"]
NT = 4


def mb(work):
    d = work / "mb"
    d.mkdir(exist_ok=True)
    return d


def log(m):
    T.log(m)


# ============================================================== prep
def cmd_prep(work: Path):
    races, R, void, nwr, grd = T.load_eval(work)
    con = T.connect()
    hd_lo, hd_hi = races.hd.min(), races.hd.max()
    # 2連単の払戻
    p2 = pd.read_sql("select hd, jcd, rno, combination c2, payout pay2 from payouts where ticket_type='nirensho' "
                       "and hd between ? and ?", con, params=(hd_lo, hd_hi)).drop_duplicates(["hd", "jcd", "rno"])
    tp = pd.read_sql("select hd, jcd, rno, cast(combination as int) tl, payout pay1 from payouts where ticket_type='tansho' "
                     "and hd between ? and ?", con, params=(hd_lo, hd_hi)).drop_duplicates(["hd", "jcd", "rno"])
    races = races.merge(p2, on=["hd", "jcd", "rno"], how="left").merge(tp, on=["hd", "jcd", "rno"], how="left")
    ok2 = races.c2.notna().to_numpy()
    hit2 = np.full(len(races), -1)
    yy = races.loc[ok2, "c2"].str.split("-", expand=True).astype(int).to_numpy() - 1
    hit2[ok2] = [_PI[tuple(r)] for r in yy.tolist()]
    races["hit2"] = hit2
    races["pay2"] = races.pay2.fillna(0).astype(float)
    races["win1"] = races.tl.fillna(0).astype(int) - 1
    races["pay1"] = races.pay1.fillna(0).astype(float)
    lane_void = np.zeros((len(races), 6), bool)
    for l in range(6):
        idx = np.where((PERMS == l).any(1))[0]
        lane_void[:, l] = void[:, idx].all(1)
    # 素性（前夜）
    df = T.load_features(work)
    df = df[df["hd"].between(hd_lo, hd_hi)]
    k6 = df[["hd", "jcd", "rno"]].iloc[::6].reset_index(drop=True)
    k6["fi"] = np.arange(len(k6))
    fi = races[["hd", "jcd", "rno"]].merge(k6, on=["hd", "jcd", "rno"], how="left")["fi"].to_numpy()
    ex = np.stack([df[c].to_numpy(np.float32).reshape(-1, 6)[fi] for c in EXTRA], 2)   # (nr, 6, E)
    # 締切オッズ（3連単）
    key = races.reset_index()[["index", "hd", "jcd", "rno"]]
    od = pd.read_sql("select hd, jcd, rno, combination comb, odds from odds where ticket_type='sanrensho'", con)
    od = od.merge(key, on=["hd", "jcd", "rno"])
    O = np.full(R["P"].shape, np.nan, np.float32)
    ci = T.combo_index(od["comb"].str.split("-", expand=True).astype(int).to_numpy() - 1)
    O[od["index"].to_numpy(), ci] = od["odds"].to_numpy()
    tq = pd.read_sql("select hd, jcd, rno, cast(combination as int) lane, odds from odds where ticket_type='tansho'", con)
    tq = tq.merge(key, on=["hd", "jcd", "rno"])
    Q1 = np.full((len(races), 6), np.nan, np.float32)
    Q1[tq["index"].to_numpy(), tq["lane"].to_numpy() - 1] = tq["odds"].to_numpy()
    G = np.load(work / "eval" / "g_pred.npy")
    out = mb(work)
    races.drop(columns=["comb", "c2", "tl"]).to_parquet(out / "races.parquet")
    np.savez(out / "arrays.npz", P=R["P"].astype(np.float32), P1=R["P1"].astype(np.float32), void=void,
             lane_void=lane_void, nwr=nwr.astype(np.float32), grd=grd.astype(np.float32), ex=ex, O=O, Q1=Q1, G=G)
    log(f"prep races={len(races)} odds races={np.isfinite(O).all(1).sum()} tansho-odds races={np.isfinite(Q1).all(1).sum()}")


def load(work: Path):
    d = mb(work)
    races = pd.read_parquet(d / "races.parquet")
    z = dict(np.load(d / "arrays.npz"))
    return races, z


# ============================================================== 予想オッズ（市場確率）モデル
def odds_features(races, z, idx, P2=False):
    """3連単の買い目ごとの前夜の素性（g_features + 追加の素性）。"""
    X = T.g_features(races.iloc[idx], z["P"][idx].astype(float), z["P1"][idx].astype(float), z["nwr"][idx], z["grd"][idx])
    ex = z["ex"][idx]
    a, b, c = PERMS[:, 0], PERMS[:, 1], PERMS[:, 2]
    for e, nm in enumerate(EXTRA):
        v = ex[:, :, e]
        X[f"{nm}_a"] = v[:, a].ravel()
        X[f"{nm}_b"] = v[:, b].ravel()
        X[f"{nm}_c"] = v[:, c].ravel()
    return X


ODDS_PARAMS = dict(objective="regression", learning_rate=0.1, num_leaves=127, min_data_in_leaf=500,
                   feature_fraction=0.8, bagging_fraction=0.7, bagging_freq=1, lambda_l2=10.0,
                   num_threads=NT, verbose=-1, seed=3)
ODDS_ROUNDS = 400


def market_q(O):
    """オッズの逆数を正規化した市場確率と、レースの払戻率。"""
    inv = 1.0 / O
    s = inv.sum(1, keepdims=True)
    return inv / s, 1.0 / s[:, 0]


def cmd_oddsmodel(work: Path):
    import lightgbm as lgb
    races, z = load(work)
    O = z["O"].astype(float)
    month = races.month.to_numpy()
    has = np.isfinite(O).all(1) & (O > 0).all(1)
    log(f"odds races {has.sum()} months {sorted(set(month[has]))}")
    q, rate = market_q(np.where(has[:, None], O, 1.0))
    rows = np.where(has)[0]
    # 学習データ: 1レースから全120通り（目的変数 log q）
    Xall = odds_features(races, z, rows)
    yall = np.log(q[rows]).ravel().astype(np.float32)
    mrow = np.repeat(month[rows], 120)
    log(f"X {Xall.shape}")
    Xn = Xall.to_numpy(np.float32)
    cols = list(Xall.columns)
    del Xall
    LQ = np.full(O.shape, np.nan, np.float32)    # 予測した log q̂（正規化前）
    LQs = np.full(O.shape, np.nan, np.float32)   # 厳密版（過去のオッズだけ）
    omonths = sorted(set(month[has]))
    fits = {}
    # (1) オッズのある月: leave-one-month-out。(2) 全月モデル → オッズの無い月
    for m in omonths + ["ALL"]:
        t0 = time.time()
        tr = mrow != m
        mdl = lgb.train(ODDS_PARAMS, lgb.Dataset(Xn[tr], yall[tr], feature_name=cols, free_raw_data=True), ODDS_ROUNDS)
        if m == "ALL":
            te = np.where(~np.isin(month, omonths))[0]
            imp = pd.Series(mdl.feature_importance("gain"), index=cols).sort_values(ascending=False)
            imp.to_csv(mb(work) / "odds_imp.csv")
        else:
            te = np.where(month == m)[0]
        for s in range(0, len(te), 20000):
            tt = te[s:s + 20000]
            LQ[tt] = mdl.predict(odds_features(races, z, tt).to_numpy(np.float32), num_threads=NT).reshape(len(tt), 120)
        log(f"odds model hold-out {m}: {time.time()-t0:.0f}s")
    # 厳密版（m より前のオッズだけ）: オッズのある最初の2か月より後
    for m in omonths:
        if m < ODDS_MONTHS[2]:
            continue
        t0 = time.time()
        tr = mrow < m
        mdl = lgb.train(ODDS_PARAMS, lgb.Dataset(Xn[tr], yall[tr], feature_name=cols), ODDS_ROUNDS)
        te = np.where(month == m)[0]
        LQs[te] = mdl.predict(odds_features(races, z, te).to_numpy(np.float32), num_threads=NT).reshape(len(te), 120)
        log(f"odds model strict {m}: {time.time()-t0:.0f}s")
    np.savez(mb(work) / "qhat.npz", LQ=LQ, LQs=LQs)
    report_odds(work)


def norm_q(LQ):
    e = np.exp(LQ - np.nanmax(LQ, 1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def report_odds(work: Path):
    races, z = load(work)
    O = z["O"].astype(float)
    month = races.month.to_numpy()
    has = np.isfinite(O).all(1) & (O > 0).all(1)
    qq = np.load(mb(work) / "qhat.npz")
    qh = norm_q(qq["LQ"].astype(float))
    qs = norm_q(qq["LQs"].astype(float))
    G = z["G"].astype(float)
    q, rate = market_q(np.where(has[:, None], O, 1.0))
    hit = races.hit.to_numpy()
    lines = ["## 予想オッズの精度（3連単、締切オッズとの log 誤差）", ""]
    rate_mean = rate[has].mean()
    lines.append(f"締切オッズのあるレース {has.sum()}（{sorted(set(month[has]))[0]}〜{sorted(set(month[has]))[-1]}）。"
                 f"払戻率（1/Σ(1/オッズ)）の平均 {rate_mean:.3f}")
    rows = []
    for nm, ms in [(f"{ODDS_MONTHS[0]}〜（LOMO）", ODDS_MONTHS), (f"{ODDS_MONTHS[2]}〜（厳密/LOMO 比較用）", ODDS_MONTHS[2:])]:
        r = has & np.isin(month, ms) & np.isfinite(G).all(1)
        lo = np.log(O[r])
        for est_nm, est in [("g/100（当たり配当で学習）", G[r] / 100), ("g/100 × 定数補正", None),
                            ("q̂（LOMO）→ 率/q̂", rate_mean / qh[r]), ("q̂（厳密: 過去のオッズだけ）", rate_mean / qs[r]),
                            ("モデル確率 P → 率/P", rate_mean / z["P"][r].astype(float))]:
            if est is None:
                lg = np.log(G[r] / 100)
                est = np.exp(lg + (lo - lg).mean())        # 後知恵の定数補正（g に有利）
            if not np.isfinite(est).all():
                continue
            le = np.log(est)
            e = le - lo
            # 当たった買い目の配当に対して
            hh = hit[r]
            okh = hh >= 0
            eh = le[okh, hh[okh]] - lo[okh, hh[okh]]
            rows.append({"期間": nm, "推定": est_nm, "RMSE(log, 全買い目)": np.sqrt((e ** 2).mean()),
                         "平均の偏り": e.mean(), "相関": np.corrcoef(le.ravel(), lo.ravel())[0, 1],
                         "1番人気一致": (le.argmin(1) == lo.argmin(1)).mean(),
                         "RMSE(log, 当たり目)": np.sqrt((eh ** 2).mean()), "偏り(当たり目)": eh.mean()})
    lines.append(pd.DataFrame(rows).to_markdown(index=False, floatfmt=".3f"))
    # 対数損失（結果に対する確率としての当たり具合）
    r = has & np.isin(month, ODDS_MONTHS) & (hit >= 0)
    ii = np.where(r)[0]
    ll = lambda Pm: -np.log(np.clip(Pm[ii, hit[ii]], 1e-12, None)).mean()
    lines += ["", f"3連単の対数損失（{len(ii)} レース）: モデル P {ll(z['P'].astype(float)):.4f} / 締切オッズの市場確率 {ll(q):.4f} / "
              f"予想市場確率 q̂ {ll(qh):.4f}"]
    for a in [0.3, 0.5, 0.7]:
        bl = np.exp(a * np.log(z["P"].astype(float)) + (1 - a) * np.log(np.clip(q, 1e-12, None)))
        bl /= bl.sum(1, keepdims=True)
        lines.append(f"- 混合 モデル{a:.1f}・締切市場{1-a:.1f}: {ll(bl):.4f}")
    imp = pd.read_csv(mb(work) / "odds_imp.csv", index_col=0).iloc[:, 0]
    lines += ["", "予想オッズモデルの重要度上位: " + ", ".join(f"{k}" for k in imp.index[:12])]
    txt = "\n".join(lines)
    (mb(work) / "odds_report.md").write_text(txt)
    print(txt)


# ============================================================== g+（当たり配当モデルに q̂ を足す）
GP_PARAMS = dict(objective="gamma", learning_rate=0.03, num_leaves=31, min_data_in_leaf=100,
                 feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0,
                 num_threads=NT, verbose=-1, seed=11)


def cmd_gplus(work: Path):
    """g+（3連単）と g2+（2連単）: 当たった買い目だけで E[配当 | 当たり] を学習。素性に予想市場確率 q̂ を足す。
    比較用に、q̂ 無しの 2連単 g2 も同じ手順で作る。月 m は m より前の当たりだけで学習。"""
    import lightgbm as lgb
    races, z = load(work)
    qq = np.load(mb(work) / "qhat.npz")
    qh = norm_q(qq["LQ"].astype(float))
    month = races.month.to_numpy()
    hit = races.hit.to_numpy()
    pay = races.payout.to_numpy()
    hit2 = races.hit2.to_numpy()
    pay2 = races.pay2.to_numpy()
    q2h = qh @ AGG
    P = z["P"].astype(float)
    P2 = P @ AGG
    allm = sorted(set(month))
    # ---- 3連単 g+ ----
    def X3(ix):
        X = T.g_features(races.iloc[ix], P[ix], z["P1"][ix].astype(float), z["nwr"][ix], z["grd"][ix])
        X["lqh"] = np.log(qh[ix]).ravel()
        X["lqh_rank"] = ((-qh[ix]).argsort(1).argsort(1) + 1).ravel()
        X["dlog"] = X["logP"] - X["lqh"]
        return X.to_numpy(np.float32)
    XH, YH = {}, {}
    for m in allm:
        ix = np.where((month == m) & (hit >= 0))[0]
        if len(ix) == 0:
            continue
        X = X3(ix).reshape(len(ix), 120, -1)
        XH[m] = X[np.arange(len(ix)), hit[ix]]
        YH[m] = pay[ix]
    Gp = np.full(P.shape, np.nan, np.float32)
    for m in allm:
        if m < T.EVAL_FIRST_MONTH:
            continue
        prior = [x for x in XH if x < m]
        mdl = lgb.train(GP_PARAMS, lgb.Dataset(np.concatenate([XH[x] for x in prior]), np.concatenate([YH[x] for x in prior])),
                        T.G_ROUNDS)
        te = np.where(month == m)[0]
        Gp[te] = mdl.predict(X3(te), num_threads=NT).reshape(len(te), 120)
        log(f"g+ {m}")
    # ---- 2連単 g2 / g2+ ----
    import nirensho as N
    flags = races[T.RACE_FLAGS].to_numpy(np.float32)
    def X2(ix, plus):
        X = N.g2_features(P2[ix], z["P1"][ix].astype(float), z["nwr"][ix], z["grd"][ix], flags[ix])
        X["jcd"] = np.repeat(races.jcd.astype(int).to_numpy()[ix], 30)
        X["rno"] = np.repeat(races.rno.to_numpy()[ix], 30)
        if plus:
            X["lq2h"] = np.log(q2h[ix]).ravel()
            X["lq2h_rank"] = ((-q2h[ix]).argsort(1).argsort(1) + 1).ravel()
            X["dlog"] = X["logP"] - X["lq2h"]
        return X.to_numpy(np.float32)
    out = {}
    for plus in [False, True]:
        XH, YH = {}, {}
        for m in allm:
            ix = np.where((month == m) & (hit2 >= 0))[0]
            X = X2(ix, plus).reshape(len(ix), 30, -1)
            XH[m] = X[np.arange(len(ix)), hit2[ix]]
            YH[m] = pay2[ix]
        G2 = np.full(P2.shape, np.nan, np.float32)
        for m in allm:
            if m < T.EVAL_FIRST_MONTH:
                continue
            prior = [x for x in XH if x < m]
            mdl = lgb.train(GP_PARAMS, lgb.Dataset(np.concatenate([XH[x] for x in prior]), np.concatenate([YH[x] for x in prior])),
                            T.G_ROUNDS)
            te = np.where(month == m)[0]
            G2[te] = mdl.predict(X2(te, plus), num_threads=NT).reshape(len(te), 30)
        out["G2p" if plus else "G2"] = G2
        log(f"g2 plus={plus} done")
    np.savez(mb(work) / "gplus.npz", Gp=Gp, **out)
    report_g(work)


def report_g(work: Path):
    """当たった買い目の log 誤差: g / g+、g2 / g2+、q̂ からの換算。検証期間・オッズ期間で。"""
    races, z = load(work)
    gp = np.load(mb(work) / "gplus.npz")
    qh = norm_q(np.load(mb(work) / "qhat.npz")["LQ"].astype(float))
    month = races.month.to_numpy()
    hit, pay = races.hit.to_numpy(), races.payout.to_numpy()
    hit2, pay2 = races.hit2.to_numpy(), races.pay2.to_numpy()
    rows = []
    for per, ms in [("発見用", DISC), ("検証用", VALID)]:
        r = np.isin(month, ms) & (hit >= 0)
        ii = np.where(r)[0]
        y = np.log(pay[ii])
        for nm, E in [("g（3連単・従来）", z["G"]), ("g+（q̂ を素性に）", gp["Gp"]), ("0.75/q̂（換算のみ）", 75 / qh)]:
            e = np.log(E[ii, hit[ii]].astype(float)) - y
            rows.append(dict(券種="3連単", 期間=per, 推定=nm, 件数=len(ii), RMSE=np.sqrt((e ** 2).mean()), 偏り=e.mean(),
                             相関=np.corrcoef(e + y, y)[0, 1]))
        r = np.isin(month, ms) & (hit2 >= 0)
        ii = np.where(r)[0]
        y = np.log(pay2[ii])
        for nm, E in [("g2（2連単・従来と同じ作り）", gp["G2"]), ("g2+（q̂ 集計を素性に）", gp["G2p"]),
                      ("0.75/q̂2（換算のみ）", 75 / (qh @ AGG))]:
            e = np.log(E[ii, hit2[ii]].astype(float)) - y
            rows.append(dict(券種="2連単", 期間=per, 推定=nm, 件数=len(ii), RMSE=np.sqrt((e ** 2).mean()), 偏り=e.mean(),
                             相関=np.corrcoef(e + y, y)[0, 1]))
    txt = "## 当たった買い目の配当の推定誤差（log）\n\n" + pd.DataFrame(rows).to_markdown(index=False, floatfmt=".3f")
    (mb(work) / "g_report.md").write_text(txt)
    print(txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "oddsmodel", "report_odds", "gplus", "report_g", "market", "bets", "report_bets", "tansho_combo", "check"])
    ap.add_argument("--workdir", default=str(T.DEFAULT_WORK))
    a = ap.parse_args()
    w = Path(a.workdir)
    T.WORK = w
    fn = {"prep": cmd_prep, "oddsmodel": cmd_oddsmodel, "report_odds": report_odds, "gplus": cmd_gplus,
          "report_g": report_g}
    if a.cmd in fn:
        fn[a.cmd](w)
    else:
        import multi_bet_eval as E
        getattr(E, f"cmd_{a.cmd}")(w)


if __name__ == "__main__":
    main()
