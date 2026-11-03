"""2連単・3連単の複数点買い: 予想オッズ（市場の見立て）のモデル、買い方の探索、モデルと市場のズレ。

使い方（作業ファイルは trifecta.py と同じ workdir の mb/ 以下）:
    python scripts/analysis/multi_bet.py prep        # 予測・払戻・返還・前夜の素性・締切オッズをレース単位にまとめる
    python scripts/analysis/multi_bet.py oddsmodel   # 前夜の情報 → 3連単締切オッズ のモデル（市場確率 q̂）と g の比較
    python scripts/analysis/multi_bet.py gplus       # 当たり配当モデル g に q̂ を足した g+（3連単・2連単）
    python scripts/analysis/multi_bet.py market      # モデルと市場のズレ（締切オッズのある 2026-03〜10）
    python scripts/analysis/multi_bet.py bets        # 事前に決めた買い方の候補を発見用・検証用で評価
結果は notes/analysis/multi_bet.md にまとめる。

v2（2026-11-03〜、notes/analysis/nirensho_v2_plan.md）:
    python scripts/analysis/multi_bet.py prep          # オッズの揃った月を DB から判定（mb/odds_months.json）して材料を作り直す
    python scripts/analysis/multi_bet.py coverage      # 月ごとのオッズの揃い具合だけ表示（重い処理なし）
    python scripts/analysis/multi_bet.py oddswf  [--months ...]   # 3連単 q̂ のウォークフォワード（各月 m: m より前のオッズ月すべてで学習）
    python scripts/analysis/multi_bet.py odds2wf [--months ...]   # 2連単の締切オッズから直接 q̂2 を学習（2連単オッズの揃った月が要る）
    python scripts/analysis/multi_bet.py g2wf --q2 agg|direct [--months ...]  # 厳密 q̂ だけを使う g2+
    python scripts/analysis/multi_bet.py g3wf [--months ...]      # 厳密 q̂ だけを使う 3連単 g+
    python scripts/analysis/multi_bet.py v2eval [--smoke]  # 事前に決めた候補（plan）を発見用・検証用で評価

リークの扱い:
  * 選定の入力は前夜の情報だけ（着順モデル P は OOS、g / g+ は予測月より前の当たりだけで学習）。
  * 予想オッズ q̂ のモデルは「締切オッズ」を目的変数にする。使うのは「結果のあるレースの95%以上に
    3連単120通りのオッズが揃った月」だけ（ODDS_COMPLETE。prep で DB から判定）。
  * 旧版（oddsmodel）: オッズのある月は leave-one-month-out、無い月は全月モデル。未来の市場も学習に入る。
  * v2（oddswf）: 各月 m の q̂ は「m より前のオッズ月すべて（拡大窓）」で学習。前に MIN_PRIOR_ODDS か月以上
    あれば「厳密」。足りない月（最初のオッズ月より前など）は、最初の MIN_PRIOR_ODDS か月で学習したモデル
    （その月自身は除く）で埋める。これは厳密な月の g2+ の学習行にだけ使い、そのときも使うオッズは全部 m より前。
    だから厳密な月 m の買い目は、m より前のオッズ・結果だけで決まる（本番と同じ条件）。
"""
from __future__ import annotations

import argparse
import json
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
EXTRA = ["local_win_rate", "national_2ren_rate", "motor_2ren_rate", "boat_2ren_rate", "age", "ser_n", "ser_avg_pos",
         "ser_win", "ser_F", "rs_avg_st", "r90_win", "rc_win_shrunk", "rs_ability"]
NT = 2                    # 共有マシン（4コア）なので2まで
ODDS_COMPLETE = 0.95      # 結果のあるレースのうち、全通りのオッズが揃ったレースがこの割合以上の月を「オッズのある月」とする
MIN_PRIOR_ODDS = 2        # 厳密 q̂ に要る「過去のオッズ月」の数
ODDS_MAX_TRAIN_RACES = 40000   # q̂ の学習に使うレース数の上限（拡大窓が大きくなってもメモリを抑える。超えたら一様に間引く）
ODDS_TICKETS = {"3t": ("sanrensho", 120), "2t": ("nirensho", 30)}   # scrape_web: odds3t → sanrensho、odds2tf → nirensho
WF_VERSION = 1


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
    # 締切オッズ: 揃った月だけ読む（揃っていない月の行は NaN のまま = 使わない）
    cov = {k: odds_coverage(con, tt, nc, hd_lo, hd_hi) for k, (tt, nc) in {**ODDS_TICKETS, "tansho": ("tansho", 6)}.items()}
    rmonths = set(races.month)
    om = {k: [m for m in complete_months(c) if m in rmonths] for k, c in cov.items()}
    key = races.reset_index()[["index", "hd", "jcd", "rno"]]
    O = np.full(R["P"].shape, np.nan, np.float32)
    O2 = np.full((len(races), 30), np.nan, np.float32)
    Q1 = np.full((len(races), 6), np.nan, np.float32)
    for kind, arr in [("3t", O), ("2t", O2), ("tansho", Q1)]:
        tt = {"3t": "sanrensho", "2t": "nirensho", "tansho": "tansho"}[kind]
        for m in om[kind]:
            lo, hi = m.replace("-", "") + "01", m.replace("-", "") + "31"
            od = pd.read_sql("select hd, jcd, rno, combination comb, odds from odds where ticket_type=? and hd between ? and ?",
                             con, params=(tt, lo, hi)).merge(key, on=["hd", "jcd", "rno"])
            if kind == "3t":
                ci = T.combo_index(od["comb"].str.split("-", expand=True).astype(int).to_numpy() - 1)
            elif kind == "2t":
                ci = np.array([_PI[tuple(r)] for r in (od["comb"].str.split("-", expand=True).astype(int).to_numpy() - 1).tolist()])
            else:
                ci = od["comb"].astype(int).to_numpy() - 1
            arr[od["index"].to_numpy(), ci] = od["odds"].to_numpy()
    out = mb(work)
    meta = {"computed": time.strftime("%Y-%m-%d %H:%M"), "threshold": ODDS_COMPLETE,
            "months": om, "coverage": {k: c.round(4).reset_index().to_dict("records") for k, c in cov.items()}}
    (out / "odds_months.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    log(f"odds months 3t={om['3t']} 2t={om['2t']}")
    G = np.load(work / "eval" / "g_pred.npy")
    races.drop(columns=["comb", "c2", "tl"]).to_parquet(out / "races.parquet")
    np.savez(out / "arrays.npz", P=R["P"].astype(np.float32), P1=R["P1"].astype(np.float32), void=void,
             lane_void=lane_void, nwr=nwr.astype(np.float32), grd=grd.astype(np.float32), ex=ex, O=O, O2=O2, Q1=Q1, G=G)
    log(f"prep races={len(races)} odds races={np.isfinite(O).all(1).sum()} 2t-odds races={np.isfinite(O2).all(1).sum()} "
        f"tansho-odds races={np.isfinite(Q1).all(1).sum()}")


# ============================================================== オッズの揃い具合
def odds_coverage(con, ticket_type, ncomb, hd_lo, hd_hi):
    """月ごと: 結果のあるレース数、全 ncomb 通りのオッズが揃ったレース数、その割合。"""
    r = pd.read_sql("select hd, jcd, rno from results where hd between ? and ? and position is not null "
                    "group by hd, jcd, rno", con, params=(hd_lo, hd_hi))
    o = pd.read_sql("select hd, jcd, rno, count(*) n from odds where ticket_type=? and hd between ? and ? "
                    "group by hd, jcd, rno", con, params=(ticket_type, hd_lo, hd_hi))
    m = r.merge(o, how="left", on=["hd", "jcd", "rno"])
    m["month"] = m.hd.str[:4] + "-" + m.hd.str[4:6]
    m["full"] = m.n.fillna(0) >= ncomb
    g = m.groupby("month").agg(races=("full", "size"), full=("full", "sum"))
    g["frac"] = g.full / g.races
    return g


def complete_months(cov, thr=None):
    thr = ODDS_COMPLETE if thr is None else thr
    return sorted(cov.index[cov.frac >= thr].tolist())


def odds_months(work: Path, kind="3t"):
    """prep が DB から判定した「オッズの揃った月」（予測 P のある月に限る）。"""
    f = mb(work) / "odds_months.json"
    if not f.exists():
        raise SystemExit("mb/odds_months.json がない。先に prep を実行する")
    return json.loads(f.read_text())["months"].get(kind, [])


def cmd_coverage(work: Path):
    con = T.connect()
    lo, hi = "20231101", "20991231"
    for k, (tt, nc) in ODDS_TICKETS.items():
        c = odds_coverage(con, tt, nc, lo, hi)
        c = c[c.full > 0]
        print(f"## {k} ({tt}, {nc}通り) 揃った月（≥{ODDS_COMPLETE:.0%}）: {complete_months(c)}")
        print(c.round(3).to_string())


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
        if m < omonths[MIN_PRIOR_ODDS]:
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
    OM = odds_months(work)
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
    for nm, ms in [(f"{OM[0]}〜（LOMO）", OM), (f"{OM[MIN_PRIOR_ODDS]}〜（厳密/LOMO 比較用）", OM[MIN_PRIOR_ODDS:])]:
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
    r = has & np.isin(month, OM) & (hit >= 0)
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


# ============================================================== v2: ウォークフォワード（拡大窓）の q̂ / q̂2 / g2+
def full_months(races):
    """暦月として揃っている月（開催日が25日以上）。評価はこの月だけ。"""
    d = races.groupby("month").hd.nunique()
    return sorted(d.index[d >= 25].tolist())


def wf_train_sets(months_all, omonths, k=MIN_PRIOR_ODDS):
    """月 m → (学習に使うオッズ月, 厳密か)。
    厳密: m より前のオッズ月が k 以上 → その全部（拡大窓）。
    それ以外: 最初の k か月（m 自身は除く）。厳密な月の g2+ の学習行を埋めるためだけに使う。"""
    out = {}
    if len(omonths) < k:
        return out
    base = list(omonths[:k])
    for m in months_all:
        prior = [x for x in omonths if x < m]
        if len(prior) >= k:
            out[m] = (tuple(prior), True)
        else:
            out[m] = (tuple(x for x in base if x != m), False)
    return out


def _build_X(featfn, idx, ncomb, chunk=4000):
    """featfn(idx)->DataFrame を chunk ごとに作って float32 の配列に詰める（ピークメモリを抑える）。"""
    first = featfn(idx[:1])
    cols = list(first.columns)
    X = np.empty((len(idx) * ncomb, len(cols)), np.float32)
    for s in range(0, len(idx), chunk):
        X[s * ncomb:(s + chunk) * ncomb] = featfn(idx[s:s + chunk]).to_numpy(np.float32)
    return X, cols


def fit_q_model(featfn, rows, O, ncomb, cap=None):
    """rows（全通りのオッズが揃ったレース）で「前夜の素性 → log 市場確率」を学習。cap を超えたら一様に間引く。"""
    import lightgbm as lgb
    cap = ODDS_MAX_TRAIN_RACES if cap is None else cap
    if len(rows) > cap:
        rows = np.sort(np.random.default_rng(0).choice(rows, cap, replace=False))
    q, _ = market_q(O[rows].astype(float))
    X, cols = _build_X(featfn, rows, ncomb)
    ds = lgb.Dataset(X, np.log(q).ravel().astype(np.float32), feature_name=cols, free_raw_data=True).construct()
    del X
    return lgb.train(ODDS_PARAMS, ds, ODDS_ROUNDS), rows


def predict_q(mdl, featfn, idx, ncomb, chunk=4000):
    return np.concatenate([mdl.predict(_build_X(featfn, idx[s:s + chunk], ncomb)[0], num_threads=NT).reshape(-1, ncomb)
                           for s in range(0, len(idx), chunk)])


def _wf_fit(work, tag, months_sel, month, has, O, featfn, ncomb, omonths, cap=None):
    """tag の q̂ を月ごとに作り mb/qwf/{tag}_{m}.npz に置く（学習月が同じなら1回の学習を共有。済みなら飛ばす）。"""
    cap = ODDS_MAX_TRAIN_RACES if cap is None else cap
    d = mb(work) / "qwf"
    d.mkdir(exist_ok=True)
    sets = wf_train_sets(sorted(set(month)), omonths)
    todo = {}
    for m in months_sel:
        if m not in sets:
            continue
        trm, strict = sets[m]
        f = d / f"{tag}_{m}.npz"
        if f.exists():
            old = np.load(f)
            if old["train"].tolist() == list(trm) and int(old["cap"]) == cap and int(old["ver"]) == WF_VERSION:
                continue
        todo.setdefault(trm, []).append((m, strict))
    for trm, ms in todo.items():
        t0 = time.time()
        mdl, rows = fit_q_model(featfn, np.where(has & np.isin(month, trm))[0], O, ncomb, cap)
        for m, strict in ms:
            LQ = predict_q(mdl, featfn, np.where(month == m)[0], ncomb)
            np.savez(d / f"{tag}_{m}.npz", LQ=LQ.astype(np.float32), train=np.array(trm), strict=strict, cap=cap,
                     ver=WF_VERSION, n_train=len(rows))
        log(f"{tag} train={trm[0]}..{trm[-1]} ({len(trm)}か月, {len(rows)} races) → {[m for m, _ in ms]}: {time.time()-t0:.0f}s")


def load_wf(work, tag, races, ncomb):
    """mb/qwf/{tag}_*.npz を全レースに並べる。戻り値: LQ (nr, ncomb; 無い月は NaN), strict (nr,) bool, 月→学習月数。"""
    month = races.month.to_numpy()
    LQ = np.full((len(races), ncomb), np.nan, np.float32)
    strict = np.zeros(len(races), bool)
    info = {}
    for f in sorted((mb(work) / "qwf").glob(f"{tag}_*.npz")):
        m = f.stem.split("_", 1)[1]
        zz = np.load(f)
        te = np.where(month == m)[0]
        if len(te) != len(zz["LQ"]):
            continue
        LQ[te] = zz["LQ"]
        strict[te] = bool(zz["strict"])
        info[m] = (len(zz["train"]), bool(zz["strict"]))
    return LQ, strict, info


def _months_arg(a, races):
    if a:
        return a.split(",")
    return sorted(set(races.month))


def cmd_oddswf(work: Path, months=None):
    """3連単の予想オッズ q̂: 各月 m を「m より前の揃ったオッズ月すべて」で学習したモデルで予測。"""
    races, z = load(work)
    om = odds_months(work, "3t")
    month = races.month.to_numpy()
    O = z["O"].astype(float)
    has = np.isfinite(O).all(1) & (O > 0).all(1) & np.isin(month, om)
    log(f"3t odds months {om}")
    _wf_fit(work, "q3", _months_arg(months, races), month, has, O, lambda ix: odds_features(races, z, ix), 120, om)


def odds2_features(races, z, idx, LQ3):
    """2連単の買い目ごとの前夜の素性: g2 の素性 + 追加の素性 + 3連単 q̂ を集計した q̂2（log）。"""
    import nirensho as N
    P2 = z["P"][idx].astype(float) @ AGG
    X = N.g2_features(P2, z["P1"][idx].astype(float), z["nwr"][idx], z["grd"][idx],
                      races.iloc[idx][T.RACE_FLAGS].to_numpy(np.float32))
    X["jcd"] = np.repeat(races.jcd.astype(int).to_numpy()[idx], 30)
    X["rno"] = np.repeat(races.rno.to_numpy()[idx], 30)
    ex = z["ex"][idx]
    a, b = PAIRS[:, 0], PAIRS[:, 1]
    for e, nm in enumerate(EXTRA):
        X[f"{nm}_a"] = ex[:, a, e].ravel()
        X[f"{nm}_b"] = ex[:, b, e].ravel()
    X["lq2agg"] = np.log(norm_q(LQ3[idx].astype(float)) @ AGG).ravel()
    return X


def cmd_odds2wf(work: Path, months=None):
    """2連単の締切オッズから直接 q̂2 を学習（3連単 q̂ の集計も素性に入れる）。2連単オッズの揃った月が MIN_PRIOR_ODDS 以上要る。"""
    races, z = load(work)
    if "O2" not in z:
        raise SystemExit("arrays.npz に O2 がない。prep をやり直す")
    om2 = odds_months(work, "2t")
    if len(om2) < MIN_PRIOR_ODDS:
        log(f"2連単オッズの揃った月が {len(om2)} か月しかない（{om2}）。直接学習はしない")
        return
    LQ3, st3, _ = load_wf(work, "q3", races, 120)
    month = races.month.to_numpy()
    O2 = z["O2"].astype(float)
    has = np.isfinite(O2).all(1) & (O2 > 0).all(1) & np.isin(month, om2) & np.isfinite(LQ3).all(1)
    log(f"2t odds months {om2}")
    _wf_fit(work, "q2", _months_arg(months, races), month, has, O2, lambda ix: odds2_features(races, z, ix, LQ3), 30, om2)


def wf_q2(work, races, src):
    """g2+ に入れる q̂2 と、その月が厳密か。src='agg': 3連単 q̂ の集計。'direct': 2連単オッズで学習した q̂2
    （無い月は agg で埋めるが、厳密なのは2連単・3連単とも厳密な月だけ）。"""
    LQ3, st3, _ = load_wf(work, "q3", races, 120)
    ok3 = np.isfinite(LQ3).all(1)
    Q2 = np.full((len(races), 30), np.nan)
    Q2[ok3] = norm_q(LQ3[ok3].astype(float)) @ AGG
    strict = st3.copy()
    if src == "direct":
        LQ2, st2, _ = load_wf(work, "q2", races, 30)
        ok2 = np.isfinite(LQ2).all(1)
        Q2[ok2] = norm_q(LQ2[ok2].astype(float))
        strict = st3 & st2
    return Q2, strict


def g2p_X(races, z, ix, Q2, flags=None):
    """g2+ の素性（cmd_gplus の X2(plus=True) と同じ形）。"""
    import nirensho as N
    flags = races[T.RACE_FLAGS].to_numpy(np.float32) if flags is None else flags
    P2 = z["P"][ix].astype(float) @ AGG
    X = N.g2_features(P2, z["P1"][ix].astype(float), z["nwr"][ix], z["grd"][ix], flags[ix])
    X["jcd"] = np.repeat(races.jcd.astype(int).to_numpy()[ix], 30)
    X["rno"] = np.repeat(races.rno.to_numpy()[ix], 30)
    q2 = Q2[ix]
    X["lq2h"] = np.log(q2).ravel()
    X["lq2h_rank"] = ((-q2).argsort(1).argsort(1) + 1).ravel()
    X["dlog"] = X["logP"] - X["lq2h"]
    return X.to_numpy(np.float32)


def cmd_g2wf(work: Path, q2src="agg", months=None):
    """厳密な月 m の g2+: m より前の当たった2連単（q̂2 は各月のウォークフォワード版）で学習し、m を予測。"""
    import lightgbm as lgb
    races, z = load(work)
    Q2, strict = wf_q2(work, races, q2src)
    month = races.month.to_numpy()
    hit2, pay2 = races.hit2.to_numpy(), races.pay2.to_numpy()
    okq = np.isfinite(Q2).all(1)
    flags = races[T.RACE_FLAGS].to_numpy(np.float32)
    f = mb(work) / f"g2wf_{q2src}.npz"
    G = np.load(f)["G2"] if f.exists() else np.full((len(races), 30), np.nan, np.float32)
    if G.shape != (len(races), 30):
        G = np.full((len(races), 30), np.nan, np.float32)
    fm = set(full_months(races))
    sel = [m for m in _months_arg(months, races) if m >= T.EVAL_FIRST_MONTH and m in fm]
    XH = {}
    for m in sorted(set(month)):
        ix = np.where((month == m) & (hit2 >= 0) & okq)[0]
        if len(ix):
            XH[m] = (g2p_X(races, z, ix, Q2, flags).reshape(len(ix), 30, -1)[np.arange(len(ix)), hit2[ix]], pay2[ix])
    done = []
    for m in sel:
        te = np.where(month == m)[0]
        if not strict[te].all():
            G[te] = np.nan
            continue
        prior = [x for x in XH if x < m]
        mdl = lgb.train(GP_PARAMS, lgb.Dataset(np.concatenate([XH[x][0] for x in prior]), np.concatenate([XH[x][1] for x in prior])),
                        T.G_ROUNDS)
        G[te] = mdl.predict(g2p_X(races, z, te, Q2, flags), num_threads=NT).reshape(len(te), 30)
        done.append(m)
    np.savez(f, G2=G, Q2=Q2.astype(np.float32), strict=strict)
    log(f"g2wf ({q2src}) 厳密な月: {done}")


def g3p_X(races, z, ix, Q3):
    """3連単 g+ の素性（cmd_gplus の X3 と同じ形。q̂ はウォークフォワード版）。"""
    P = z["P"][ix].astype(float)
    X = T.g_features(races.iloc[ix], P, z["P1"][ix].astype(float), z["nwr"][ix], z["grd"][ix])
    q = Q3[ix]
    X["lqh"] = np.log(q).ravel()
    X["lqh_rank"] = ((-q).argsort(1).argsort(1) + 1).ravel()
    X["dlog"] = X["logP"] - X["lqh"]
    return X.to_numpy(np.float32)


def cmd_g3wf(work: Path, months=None):
    """厳密な月 m の 3連単 g+: m より前の当たった3連単（q̂ はウォークフォワード版）で学習し、m を予測。"""
    import lightgbm as lgb
    races, z = load(work)
    LQ3, strict, _ = load_wf(work, "q3", races, 120)
    month = races.month.to_numpy()
    hit, pay = races.hit.to_numpy(), races.payout.to_numpy()
    okq = np.isfinite(LQ3).all(1)
    Q3 = np.full(LQ3.shape, np.nan)
    Q3[okq] = norm_q(LQ3[okq].astype(float))
    f = mb(work) / "g3wf.npz"
    G = np.load(f)["G3"] if f.exists() else np.full((len(races), 120), np.nan, np.float32)
    if G.shape != (len(races), 120):
        G = np.full((len(races), 120), np.nan, np.float32)
    fm = set(full_months(races))
    sel = [m for m in _months_arg(months, races) if m >= T.EVAL_FIRST_MONTH and m in fm]
    XH = {}
    for m in sorted(set(month)):
        ix = np.where((month == m) & (hit >= 0) & okq)[0]
        if len(ix):
            XH[m] = (g3p_X(races, z, ix, Q3).reshape(len(ix), 120, -1)[np.arange(len(ix)), hit[ix]], pay[ix])
    done = []
    for m in sel:
        te = np.where(month == m)[0]
        if not strict[te].all():
            G[te] = np.nan
            continue
        prior = [x for x in XH if x < m]
        mdl = lgb.train(GP_PARAMS, lgb.Dataset(np.concatenate([XH[x][0] for x in prior]), np.concatenate([XH[x][1] for x in prior])),
                        T.G_ROUNDS)
        G[te] = np.concatenate([mdl.predict(g3p_X(races, z, te[s:s + 4000], Q3), num_threads=NT).reshape(-1, 120)
                                for s in range(0, len(te), 4000)])
        done.append(m)
    np.savez(f, G3=G)
    log(f"g3wf 厳密な月: {done}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "coverage", "oddsmodel", "report_odds", "gplus", "report_g", "market", "bets",
                                    "report_bets", "tansho_combo", "check", "oddswf", "odds2wf", "g2wf", "g3wf", "v2eval"])
    ap.add_argument("--workdir", default=str(T.DEFAULT_WORK))
    ap.add_argument("--months", help="oddswf / odds2wf / g2wf: 対象の月（カンマ区切り）。省略で全月")
    ap.add_argument("--q2", default="agg", choices=["agg", "direct"], help="g2wf: q̂2 の作り")
    ap.add_argument("--smoke", action="store_true", help="v2eval: 月数が足りなくても動作確認として出す")
    a = ap.parse_args()
    w = Path(a.workdir)
    T.WORK = w
    fn = {"prep": cmd_prep, "coverage": cmd_coverage, "oddsmodel": cmd_oddsmodel, "report_odds": report_odds, "gplus": cmd_gplus,
          "report_g": report_g}
    if a.cmd in fn:
        fn[a.cmd](w)
    elif a.cmd in ("oddswf", "odds2wf"):
        {"oddswf": cmd_oddswf, "odds2wf": cmd_odds2wf}[a.cmd](w, a.months)
    elif a.cmd == "g2wf":
        cmd_g2wf(w, a.q2, a.months)
    elif a.cmd == "g3wf":
        cmd_g3wf(w, a.months)
    elif a.cmd == "v2eval":
        import multi_bet_eval as E
        E.cmd_v2eval(w, smoke=a.smoke)
    else:
        import multi_bet_eval as E
        getattr(E, f"cmd_{a.cmd}")(w)


if __name__ == "__main__":
    main()
