"""1着確率 p と予想配当 g のウォークフォワード補正、補正後の期待値で作る候補の評価。

使い方:
    python scripts/analysis/calibrate.py tansho     # 単勝: 補正の当てはめ・当たり具合・候補（6通り）
    python scripts/analysis/calibrate.py nirensho [--strict]  # 2連単: P2 と g2+ の補正・当たり具合・候補（3通り）
出力は WORK（scratchpad/calib）に *.md / *.parquet。まとめは notes/analysis/calibration.md。

リークの扱い:
  * 入力は前夜に分かるもの（OOS の p・g、番組表の艇番）だけ。
  * 補正の係数は、対象月 m より前の月の結果だけで当てる（p は最大12か月・2024-05〜、
    g は最大12か月・g がある 2024-11〜）。m の前月末の結果は m 初日の選定（前夜）までに出ている。
  * 2連単の g2+ は multi_bet.py の作業ファイル（mb/gplus.npz の写し）を読む。g2+ の素性 q̂ は締切オッズで
    学習したモデルで、gplus.npz の q̂ は LOMO（未来の月のオッズも学習に入る）。--strict では、厳密 q̂ がある月
    （2026-01〜10）だけ g2+ を作り直す（multi_bet_eval.g2plus_strict と同じ作り）。発見用の月の q̂ は
    どちらでも「後の月のオッズ」で学習したもの（2025-10 以前に締切オッズが無いため）。
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))
import model as Mo  # noqa: E402

TRI_WORK = Path("/tmp/claude-0/-home-user-verbose-engine/968008a0-6ecf-5641-a3ff-006912b41722/scratchpad/trifecta")
WORK = TRI_WORK.parent / "calib"

DISC = ["2024-11", "2024-12"] + [f"2025-{i:02d}" for i in range(1, 11)]
VALID = ["2025-11", "2025-12"] + [f"2026-{i:02d}" for i in range(1, 11)]
P_FROM, G_FROM, WIN_MONTHS, MIN_HITS = "2024-05", "2024-11", 12, 3000

# 事前に決めた候補（notes/analysis/calibration.md 0節）
TANSHO_GRID = [(p, e) for p in (0.20, 0.30) for e in (1.00, 1.05, 1.10)]
NIREN_GRID = [1.0, 1.1, 1.2]
NIREN_PMIN = 0.05


def log(m):
    print(time.strftime("%H:%M:%S"), m, flush=True)


def prior_months(allm, m, start):
    return [x for x in allm if start <= x < m][-WIN_MONTHS:]


# ============================================================== 当てはめの部品
def fit_softmax(F, win, theta0):
    """F: (n, K, d) 素性、win: (n,) 当たりの列。s = F·θ のレース内 softmax の対数損失最小。"""
    F = F.astype(np.float64)
    n = len(win)
    Fw = F[np.arange(n), win]

    def f(th):
        s = F @ th
        s -= s.max(1, keepdims=True)
        e = np.exp(s)
        Z = e.sum(1)
        q = e / Z[:, None]
        ll = -(s[np.arange(n), win] - np.log(Z)).mean()
        grad = -(Fw - np.einsum("nk,nkd->nd", q, F)).mean(0)
        return ll, grad

    r = minimize(f, theta0, jac=True, method="L-BFGS-B")
    return r.x


def apply_softmax(F, th):
    s = F.astype(np.float64) @ th
    s -= s.max(1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(1, keepdims=True)


def fit_poisson_offset(X, y, g):
    """E[y] = g·exp(Xβ)（平均を合わせる）。y, g は同じ単位。"""
    X = X.astype(np.float64)
    sc = g.mean()
    y, g = y / sc, g / sc

    def f(b):
        eta = X @ b
        mu = g * np.exp(eta)
        return (mu - y * eta).mean(), X.T @ (mu - y) / len(y)

    return minimize(f, np.zeros(X.shape[1]), jac=True, method="L-BFGS-B").x


def onehot(v, levels):
    return np.stack([(v == l).astype(np.float64) for l in levels], -1)


# ============================================================== 当たり具合の表
def reliability(p, w, bins=(0, .05, .1, .2, .3, .4, .5, .6, .7, .8, 1.0001)):
    b = pd.cut(p, bins, right=False)
    return pd.DataFrame({"p": p, "w": w, "b": b}).groupby("b", observed=True).agg(n=("w", "size"), 予想=("p", "mean"),
                                                                                  実際=("w", "mean"))


def ece(p, w, nb=20):
    q = np.clip((p * nb).astype(int), 0, nb - 1)
    d = pd.DataFrame({"p": p, "w": w, "q": q}).groupby("q").agg(n=("w", "size"), p=("p", "mean"), w=("w", "mean"))
    return float((d.n * (d.p - d.w).abs()).sum() / d.n.sum())


def payout_table(g, y, bins):
    b = pd.cut(g, bins)
    d = pd.DataFrame({"g": g, "y": y, "b": b, "r": y / g}).groupby("b", observed=True).agg(
        n=("y", "size"), 予想平均=("g", "mean"), 実際平均=("y", "mean"), 比の中央値=("r", "median"))
    d["平均の比"] = d.実際平均 / d.予想平均
    return d


def ev_table(ev, ret, bins=(0, .8, .9, 1.0, 1.05, 1.1, 1.2, 1.5, 100)):
    b = pd.cut(ev, bins, right=False)
    return pd.DataFrame({"ev": ev, "ret": ret, "b": b}).groupby("b", observed=True).agg(
        点=("ret", "size"), 予想EV=("ev", "mean"), 実際の回収率=("ret", lambda r: r.mean() / 100))


def month_summary(df, months):
    """df: 1点1行（month, ret）。暦月ごとの回収率。"""
    g = df.groupby("month").agg(bets=("ret", "size"), hits=("ret", lambda r: int((r > 0).sum())), ret=("ret", "sum"))
    g = g.reindex(months).fillna(0)
    roi_m = np.where(g.bets > 0, g.ret / 100 / g.bets.clip(lower=1), np.nan)
    tot = g.ret.sum() / 100 / max(g.bets.sum(), 1)
    return dict(roi=tot, win=int(np.nansum(roi_m > 1)), worst=float(np.nanmin(roi_m)) if np.isfinite(roi_m).any() else np.nan,
                bets=int(g.bets.sum()), hits=int(g.hits.sum())), pd.Series(roi_m, index=months)


def md(df, fmt=".3f"):
    return df.to_markdown(floatfmt=fmt)


# ============================================================== 単勝
def tansho_material():
    cfg = Mo.strategy_cfg("tansho_ev_v1")
    h = Mo.load_history(cfg)
    hm = Mo.month_of(h["hd"])
    h = h[(hm >= P_FROM) & (hm <= VALID[-1])].reset_index(drop=True)
    con = Mo.connect_ro(timeout=600)
    TP, win = Mo.tansho_winners(con, h)
    void = Mo.void_lanes(con, h.hd.min(), h.hd.max())
    k = h[["hd", "jcd", "rno"]].reset_index().merge(void, on=["hd", "jcd", "rno"])
    V = np.zeros((len(h), 6), bool)
    V[k["index"].to_numpy(), k["lane"].to_numpy().astype(int) - 1] = True
    P1r, _, _, _ = Mo.hist_arrays(h)
    P = Mo.apply_alpha(P1r, h["alpha"].to_numpy(float))
    G = h[[f"g_{i}" for i in range(1, 7)]].to_numpy(float)
    return h, Mo.month_of(h["hd"]), P, G, TP, win, V


def tansho_pfeat(P):
    lp = np.log(np.clip(P, 1e-6, 1))
    lane = np.broadcast_to(np.eye(6)[None, :, 1:], P.shape + (5,))
    return np.concatenate([lp[..., None], (lp ** 2)[..., None], lane], -1)


def tansho_gfeat(G, P):
    lg = np.log(G / 100)
    rk = (-P).argsort(1).argsort(1) + 1
    lane = np.broadcast_to(np.arange(1, 7)[None, :], G.shape)
    return np.concatenate([np.ones(G.shape + (1,)), lg[..., None], (lg ** 2)[..., None],
                           onehot(lane, range(2, 7)), onehot(rk, range(2, 7))], -1)


def cmd_tansho():
    from sklearn.isotonic import IsotonicRegression
    WORK.mkdir(parents=True, exist_ok=True)
    h, month, P, G, TP, win, V = tansho_material()
    log(f"races={len(h)}")
    allm = sorted(set(month))
    Pc = np.full(P.shape, np.nan)
    Pi = np.full(P.shape, np.nan)
    Gc = np.full(G.shape, np.nan)
    F = tansho_pfeat(P)
    GX = tansho_gfeat(np.where(np.isfinite(G), G, 100.0), P)
    th0 = np.r_[1.0, 0.0, np.zeros(5)]
    params = []
    for m in allm:
        te = np.where(month == m)[0]
        pm = prior_months(allm, m, P_FROM)
        tr = np.where(np.isin(month, pm) & (win >= 0))[0]
        if len(tr) >= MIN_HITS:
            th = fit_softmax(F[tr], win[tr], th0)
            Pc[te] = apply_softmax(F[te], th)
            iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1).fit(
                P[tr].ravel(), (np.arange(6)[None, :] == win[tr][:, None]).ravel())
            q = iso.predict(P[te].ravel()).reshape(-1, 6)
            Pi[te] = q / q.sum(1, keepdims=True)
        else:
            th = None
            Pc[te] = P[te]
            Pi[te] = P[te]
        gm = prior_months(allm, m, G_FROM)
        trg = np.where(np.isin(month, gm) & (win >= 0) & np.isfinite(G).all(1))[0]
        if len(trg) >= MIN_HITS:
            wi = win[trg]
            b = fit_poisson_offset(GX[trg, wi], TP[trg, wi], G[trg, wi])
            Gc[te] = G[te] * np.exp(GX[te] @ b)
        else:
            b = None
            Gc[te] = G[te]
        params.append(dict(month=m, p_n=len(tr), p_theta=None if th is None else np.round(th, 3).tolist(),
                           g_n=len(trg), g_beta=None if b is None else np.round(b, 3).tolist()))
        log(f"{m} p_n={len(tr)} g_n={len(trg)} th={None if th is None else np.round(th[:2], 3)}")
    pd.DataFrame(params).to_csv(WORK / "tansho_params.csv", index=False)
    np.savez(WORK / "tansho_cal.npz", Pc=Pc, Pi=Pi, Gc=Gc)

    out = ["# 単勝の補正", ""]
    # ---- 当たり具合（検証用 2025-11〜2026-10）
    v = np.isin(month, VALID) & (win >= 0)
    W = (np.arange(6)[None, :] == win[:, None])
    vv = np.where(v)[0]
    out.append(f"## 1着確率（検証用 {len(vv)} レース）")
    rows = []
    for nm, Q in [("補正なし", P), ("isotonic", Pi), ("softmax再校正（主）", Pc)]:
        q, w = Q[vv].ravel(), W[vv].ravel()
        rows.append(dict(方法=nm, 対数損失=-np.log(np.clip(Q[vv, win[vv]], 1e-9, 1)).mean(),
                         Brier=((Q[vv] - W[vv]) ** 2).sum(1).mean(), ECE=ece(q, w)))
    out += [pd.DataFrame(rows).set_index("方法").to_markdown(floatfmt=".4f"), ""]
    tab = pd.concat({nm: reliability(Q[vv].ravel(), W[vv].ravel()) for nm, Q in
                     [("補正なし", P), ("isotonic", Pi), ("softmax", Pc)]}, axis=1)
    out += ["### 予想の帯ごと（帯は補正なしの p で切ったものではなく、それぞれの予想で切る）", md(tab), ""]
    lane = pd.DataFrame({"実際": W[vv].mean(0), "補正なし": P[vv].mean(0), "isotonic": Pi[vv].mean(0), "softmax": Pc[vv].mean(0)},
                        index=pd.Index(range(1, 7), name="艇番"))
    out += ["### 艇番別の平均", md(lane), ""]
    # ---- 配当
    okg = v & np.isfinite(G).all(1)
    ii = np.where(okg)[0]
    y = TP[ii, win[ii]]
    gb = (0, 130, 160, 200, 300, 500, 1000, 1e6)
    out += [f"## 予想配当（検証用の当たった艇 {len(ii)} 本、帯はそれぞれの予想で切る）",
            "### 補正なし g", md(payout_table(G[ii, win[ii]], y, gb)), "",
            "### 補正後 g_cal", md(payout_table(Gc[ii, win[ii]], y, gb)), ""]
    lt = pd.DataFrame({"本数": pd.Series(win[ii] + 1).value_counts().sort_index(),
                       "補正なし 実際/予想（平均の比）": pd.Series(y).groupby(win[ii] + 1).sum() / pd.Series(G[ii, win[ii]]).groupby(win[ii] + 1).sum(),
                       "補正後": pd.Series(y).groupby(win[ii] + 1).sum() / pd.Series(Gc[ii, win[ii]]).groupby(win[ii] + 1).sum()})
    lt.index.name = "艇番"
    out += ["### 艇番別（平均の比）", md(lt), ""]
    # ---- 期待値の当たり具合（買える艇＝返還でない・p≥0.2）
    lanes = np.isfinite(G) & ~V & (win >= 0)[:, None]
    EVr = P * G / 100
    EVc = Pc * Gc / 100
    R = TP
    for nm, ms in [("発見用", DISC), ("検証用", VALID)]:
        msk = lanes & np.isin(month, ms)[:, None]
        rows = {}
        for en, E, Q in [("補正なし EV", EVr, P), ("補正後 EV", EVc, Pc)]:
            s = msk & (Q >= 0.2)
            rows[en] = ev_table(E[s], R[s])
        out += [f"## 期待値の帯と実際の回収率（{nm}、p≥0.20 の艇、帯はそれぞれの EV で切る）", md(pd.concat(rows, axis=1)), ""]
    # ---- 候補
    mon = np.broadcast_to(month[:, None], P.shape)
    cands = [("参考: tansho_ev_v1（補正なし p≥0.20 EV≥1.0）", (P >= 0.2) & (EVr >= 1.0))]
    for pm, em in TANSHO_GRID:
        cands.append((f"補正EV: p_cal≥{pm:.2f} EV_cal≥{em:.2f}", (Pc >= pm) & (EVc >= em)))
    res, mres = [], {}
    for nm, mk in cands:
        s = mk & lanes
        df = pd.DataFrame({"month": mon[s], "ret": R[s]})
        a, ma = month_summary(df, DISC)
        b, mb_ = month_summary(df, VALID)
        res.append(dict(候補=nm, **{f"{k}_発見": x for k, x in a.items()}, **{f"{k}_検証": x for k, x in b.items()}))
        mres[nm] = pd.concat([ma, mb_])
    res = pd.DataFrame(res).set_index("候補")
    res.to_csv(WORK / "tansho_cands.csv")
    out += [f"## 候補（新しい系統 {len(TANSHO_GRID)} 通り＋参考1）", md(res), ""]
    new = res.iloc[1:]
    el = new[new.hits_発見 >= 100]
    best = el.roi_発見.idxmax()
    out += [f"### 発見用で選んだ1本: {best}", md(res.loc[[best]]), ""]
    mt = pd.DataFrame(mres).T
    out += ["### 月別の回収率", mt.to_markdown(floatfmt=".2f"), ""]
    txt = "\n".join(out)
    (WORK / "tansho_report.md").write_text(txt)
    print(txt)


# ============================================================== 2連単
MB = WORK / "mb_snap"   # mb/ の races.parquet・arrays.npz・gplus.npz・qhat.npz の写し（2026-11-03 08:09、別の作業が上書きしても数字が変わらないように）
GP_PARAMS = dict(objective="gamma", learning_rate=0.03, num_leaves=31, min_data_in_leaf=100,
                 feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0,
                 num_threads=2, verbose=-1, seed=11)   # multi_bet.GP_PARAMS と同じ（スレッド数だけ 2）


def niren_material(strict=False):
    import trifecta as T
    PERMS = T.PERMS
    PAIRS = np.array([(i, j) for i in range(6) for j in range(6) if i != j])
    pi = {tuple(p): n for n, p in enumerate(PAIRS.tolist())}
    AGG = np.zeros((120, 30), np.float32)
    for n, (a, b, _c) in enumerate(PERMS.tolist()):
        AGG[n, pi[(a, b)]] = 1
    races = pd.read_parquet(MB / "races.parquet")
    z = np.load(MB / "arrays.npz")
    P2 = (z["P"] @ AGG).astype(np.float64)
    void2 = z["lane_void"][:, PAIRS].any(2)
    G2p = np.load(MB / "gplus.npz")["G2p"].astype(np.float64)
    smonths = []
    if strict:
        G2p, smonths = strict_g2p(races, z, P2.astype(np.float32), G2p, AGG, T)
    return races, PAIRS, P2, void2, G2p, smonths


def strict_g2p(races, z, P2, G2p, AGG, T):
    """multi_bet_eval.g2plus_strict と同じ作り（写し）: 厳密 q̂（各月 m の q̂ を m より前のオッズだけで学習、LQs）が
    そろう暦月 m について、g2+ を「m より前の当たり」で学習し直し、m の行は厳密 q̂ を素性にして予測する。"""
    import lightgbm as lgb
    import nirensho as N

    def normq(LQ):
        e = np.exp(LQ - np.nanmax(LQ, 1, keepdims=True))
        return e / e.sum(1, keepdims=True)
    qq = np.load(MB / "qhat.npz")
    q2h = normq(qq["LQ"].astype(float)) @ AGG
    LQs = qq["LQs"]
    okq = np.isfinite(LQs).all(1)
    q2s = normq(np.where(np.isnan(LQs), 0, LQs).astype(float)) @ AGG
    month = races.month.to_numpy()
    nday = races.groupby("month").hd.nunique()
    full = set(nday.index[nday >= 25])
    sm = sorted(m for m in set(month[okq]) if m in full and m <= VALID[-1])
    flags = races[T.RACE_FLAGS].to_numpy(np.float32)
    P1 = z["P1"].astype(float)
    nwr, grd = z["nwr"], z["grd"]
    hit2, pay2 = races.hit2.to_numpy(), races.pay2.to_numpy()

    def X2(ix, Q2):
        X = N.g2_features(P2[ix], P1[ix], nwr[ix], grd[ix], flags[ix])
        X["jcd"] = np.repeat(races.jcd.astype(int).to_numpy()[ix], 30)
        X["rno"] = np.repeat(races.rno.to_numpy()[ix], 30)
        X["lq2h"] = np.log(Q2[ix]).ravel()
        X["lq2h_rank"] = ((-Q2[ix]).argsort(1).argsort(1) + 1).ravel()
        X["dlog"] = X["logP"] - X["lq2h"]
        return X.to_numpy(np.float32)
    G = G2p.copy()
    for m in sm:
        tr = np.where((month < m) & (hit2 >= 0))[0]
        parts = []
        for ch in np.array_split(tr, max(1, len(tr) // 20000)):
            parts.append(X2(ch, q2h).reshape(len(ch), 30, -1)[np.arange(len(ch)), hit2[ch]])
        mdl = lgb.train(GP_PARAMS, lgb.Dataset(np.concatenate(parts), pay2[tr]), T.G_ROUNDS)
        te = np.where(month == m)[0]
        G[te] = mdl.predict(X2(te, q2s), num_threads=2).reshape(len(te), 30)
        log(f"strict g2+ {m}")
    return G, sm


def niren_pfeat(P2, PAIRS):
    lp = np.log(np.clip(P2, 1e-7, 1))
    a = np.broadcast_to(np.eye(6)[PAIRS[:, 0]][None, :, 1:], P2.shape + (5,))
    b = np.broadcast_to(np.eye(6)[PAIRS[:, 1]][None, :, 1:], P2.shape + (5,))
    return np.concatenate([lp[..., None], (lp ** 2)[..., None], a, b], -1).astype(np.float32)


def niren_gfeat_rows(g, P2row_rank, first, second):
    lg = np.log(g / 100)
    band = np.digitize(P2row_rank, [2, 3, 4, 7, 11])     # 1,2,3,4-6,7-10,11+ → 0..5
    return np.concatenate([np.ones((len(g), 1)), lg[:, None], (lg ** 2)[:, None],
                           onehot(first, range(1, 6)), onehot(second, range(1, 6)), onehot(band, range(1, 6))], 1)


def cmd_nirensho(strict=False):
    WORK.mkdir(parents=True, exist_ok=True)
    races, PAIRS, P2, void2, G2, smonths = niren_material(strict)
    tag = "_strict" if strict else ""
    races = races.reset_index(drop=True)
    month = races.month.to_numpy()
    sel = month <= VALID[-1]
    hit2, pay2 = races.hit2.to_numpy(), races.pay2.to_numpy()
    log(f"races={len(races)}")
    allm = sorted(set(month[sel]))
    rank = (-P2).argsort(1).argsort(1) + 1
    P2c = np.full(P2.shape, np.nan)
    G2c = np.full(G2.shape, np.nan)
    th0 = np.r_[1.0, 0.0, np.zeros(10)]
    first = np.broadcast_to(PAIRS[:, 0][None, :], P2.shape)
    second = np.broadcast_to(PAIRS[:, 1][None, :], P2.shape)
    params = []
    for m in allm:
        te = np.where(month == m)[0]
        pm = prior_months(allm, m, P_FROM)
        tr = np.where(np.isin(month, pm) & (hit2 >= 0))[0]
        if len(tr) >= MIN_HITS:
            th = fit_softmax(niren_pfeat(P2[tr], PAIRS), hit2[tr], th0)
            P2c[te] = apply_softmax(niren_pfeat(P2[te], PAIRS), th)
        else:
            th = None
            P2c[te] = P2[te]
        gm = prior_months(allm, m, G_FROM)
        trg = np.where(np.isin(month, gm) & (hit2 >= 0) & np.isfinite(G2).all(1))[0]
        if len(trg) >= MIN_HITS:
            wi = hit2[trg]
            X = niren_gfeat_rows(G2[trg, wi], rank[trg, wi], PAIRS[wi, 0], PAIRS[wi, 1])
            b = fit_poisson_offset(X, pay2[trg], G2[trg, wi])
            Xt = niren_gfeat_rows(G2[te].ravel(), rank[te].ravel(), first[te].ravel(), second[te].ravel())
            G2c[te] = G2[te] * np.exp(Xt @ b).reshape(len(te), 30)
        else:
            b = None
            G2c[te] = G2[te]
        params.append(dict(month=m, p_n=len(tr), p_theta=None if th is None else np.round(th, 3).tolist(),
                           g_n=len(trg), g_beta=None if b is None else np.round(b, 3).tolist()))
        log(f"{m} p_n={len(tr)} g_n={len(trg)} th={None if th is None else np.round(th[:2], 3)}")
    pd.DataFrame(params).to_csv(WORK / f"niren_params{tag}.csv", index=False)

    out = ["# 2連単の補正" + ("（g2+ は厳密 q̂ 版: " + ",".join(smonths) + "）" if strict else "（g2+ は mb/gplus.npz のまま＝LOMO q̂）"), ""]
    v = np.isin(month, VALID) & (hit2 >= 0)
    vv = np.where(v)[0]
    W = np.zeros(P2.shape, bool)
    W[vv, hit2[vv]] = True
    rows = []
    for nm, Q in [("補正なし", P2), ("softmax再校正", P2c)]:
        rows.append(dict(方法=nm, 対数損失=-np.log(np.clip(Q[vv, hit2[vv]], 1e-9, 1)).mean(), ECE=ece(Q[vv].ravel(), W[vv].ravel())))
    out += [f"## 2連単の確率（検証用 {len(vv)} レース）", pd.DataFrame(rows).set_index("方法").to_markdown(floatfmt=".4f"), ""]
    bins = (0, .01, .02, .05, .1, .15, .2, .3, .4, .5, 1.0001)
    tab = pd.concat({nm: reliability(Q[vv].ravel(), W[vv].ravel(), bins) for nm, Q in [("補正なし", P2), ("softmax", P2c)]}, axis=1)
    out += [md(tab), ""]
    ii = vv[np.isfinite(G2[vv]).all(1)]
    y = pay2[ii]
    gb = (0, 300, 600, 1000, 2000, 5000, 1e6)
    out += [f"## 2連単の予想配当 g2+（検証用の当たり {len(ii)} 本）", "### 補正なし", md(payout_table(G2[ii, hit2[ii]], y, gb)), "",
            "### 補正後", md(payout_table(G2c[ii, hit2[ii]], y, gb)), ""]
    ok = (hit2 >= 0) & np.isfinite(G2).all(1)
    EVr = P2 * np.nan_to_num(G2) / 100
    EVc = P2c * np.nan_to_num(G2c) / 100
    R = np.zeros(P2.shape)
    jj = np.where(hit2 >= 0)[0]
    R[jj, hit2[jj]] = pay2[jj]

    def top1(E, Q, tau):
        Em = np.where(Q >= NIREN_PMIN, E, -np.inf)
        j = Em.argmax(1)
        best = Em[np.arange(len(E)), j]
        buy = ok & (best >= tau) & sel
        buy &= ~void2[np.arange(len(E)), j]
        return buy, j

    for nm, ms in [("発見用", DISC), ("検証用", VALID)]:
        rows = {}
        for en, E, Q in [("補正なし EV", EVr, P2), ("補正後 EV", EVc, P2c)]:
            Em = np.where(Q >= NIREN_PMIN, E, -np.inf)
            j = Em.argmax(1)
            s = ok & np.isin(month, ms) & np.isfinite(Em.max(1)) & ~void2[np.arange(len(E)), j]
            rows[en] = ev_table(E[np.arange(len(E)), j][s], R[np.arange(len(E)), j][s])
        out += [f"## EV最大1点の期待値の帯と実際の回収率（{nm}、P2≥{NIREN_PMIN}）", md(pd.concat(rows, axis=1)), ""]
    cands = [("参考: g2+ p≥5% EV最大1点 EV≥1.2（補正なし、multi_bet.md）", EVr, P2, 1.2)]
    for t in NIREN_GRID:
        cands.append((f"補正EV最大1点: P2_cal≥{NIREN_PMIN} EV_cal≥{t:.1f}", EVc, P2c, t))
    res, mres = [], {}
    for nm, E, Q, t in cands:
        buy, j = top1(E, Q, t)
        ix = np.where(buy)[0]
        df = pd.DataFrame({"month": month[ix], "ret": R[ix, j[ix]]})
        a, ma = month_summary(df, DISC)
        b, mb_ = month_summary(df, VALID)
        extra = {}
        if strict:
            c_, _ = month_summary(df, smonths)
            extra = {f"{k}_厳密区間": x for k, x in c_.items()}
        res.append(dict(候補=nm, **{f"{k}_発見": x for k, x in a.items()}, **{f"{k}_検証": x for k, x in b.items()}, **extra))
        mres[nm] = pd.concat([ma, mb_])
    res = pd.DataFrame(res).set_index("候補")
    res.to_csv(WORK / f"niren_cands{tag}.csv")
    out += [f"## 候補（新しい系統 {len(NIREN_GRID)} 通り＋参考1）", md(res), ""]
    new = res.iloc[1:]
    el = new[new.hits_発見 >= 100]
    if len(el):
        best = el.roi_発見.idxmax()
        out += [f"### 発見用で選んだ1本: {best}", md(res.loc[[best]]), ""]
    out += ["### 月別の回収率", pd.DataFrame(mres).T.to_markdown(floatfmt=".2f"), ""]
    txt = "\n".join(out)
    (WORK / f"niren_report{tag}.md").write_text(txt)
    print(txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["tansho", "nirensho"])
    ap.add_argument("--strict", action="store_true", help="2連単: g2+ を厳密 q̂（過去のオッズだけ）で作り直した版で評価")
    a = ap.parse_args()
    if a.cmd == "tansho":
        cmd_tansho()
    else:
        cmd_nirensho(a.strict)


if __name__ == "__main__":
    main()
