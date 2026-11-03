"""選ばれた買い目の「見すぎ」（勝者の呪い）を直す補正の、ウォークフォワード評価。

使い方:
    python scripts/analysis/calibrate.py [--types niren,sanren,tansho]
出力は WORK（scratchpad/calib）に calib_*.md / *.csv。まとめは notes/analysis/calibration.md（0節に事前の宣言）。

補正（どちらも対象月 m より前の最大12か月だけで当てる。g が 2024-11〜なので 2025-01 から）:
  (a) 縮小: P_a = softmax(α log P + β log q̂ + γ log g)、g_a = g·exp(b·[1, log g, log q̂, log P])。EV_a で選び直す。
  (b) 直接: 補正なしの「選ぶ目」で E[払戻/100] = EV·exp(c·[1, log EV, log P])。選び方はそのまま。
リーク: 入力は前夜に分かるもの（OOS の P・g、q̂ は締切オッズを学習した予想。LQ は LOMO なので注意、LQs で感度を見る）。
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

TRI_WORK = Path("/tmp/claude-0/-home-user-verbose-engine/968008a0-6ecf-5641-a3ff-006912b41722/scratchpad/trifecta")
WORK = TRI_WORK.parent / "calib"
MB = WORK / "mb_snap"   # mb/ の races.parquet・arrays.npz・gplus.npz・qhat.npz の写し（2026-11-03 08:09）

DISC = [f"2025-{i:02d}" for i in range(1, 11)]
VALID = ["2025-11", "2025-12"] + [f"2026-{i:02d}" for i in range(1, 11)]
STRICT = [f"2026-{i:02d}" for i in range(1, 11)]
FIT_FROM, WIN_MONTHS, MIN_PRIOR = "2024-11", 12, 2
TAUS = [1.0, 1.05, 1.1, 1.2]
PMIN = {"niren": 0.05, "sanren": 0.01, "tansho": 0.20}
NAME = {"niren": "2連単 EV最大1点", "sanren": "3連単 EV最大1点", "tansho": "単勝 p≥0.20"}
MIN_DISC_HITS = 100
METHODS = ["raw", "a", "c", "b"]
MNAME = {"raw": "補正なし", "a": "(a)縮小", "c": "(a')混合", "b": "(b)直接"}


def log(m):
    print(time.strftime("%H:%M:%S"), m, flush=True)


# ============================================================== 材料
def norm_q(LQ):
    e = np.exp(LQ - np.nanmax(LQ, 1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def load_all(types):
    import trifecta as T
    races = pd.read_parquet(MB / "races.parquet").reset_index(drop=True)
    z = np.load(MB / "arrays.npz")
    gp = np.load(MB / "gplus.npz")
    qq = np.load(MB / "qhat.npz")
    month = races.month.to_numpy()
    Q3 = norm_q(qq["LQ"].astype(np.float64))
    LQs = qq["LQs"]
    oks = np.isfinite(LQs).all(1)
    Q3s = np.full(Q3.shape, np.nan)
    Q3s[oks] = norm_q(LQs[oks].astype(np.float64))
    PERMS = T.PERMS
    out = {}
    if "sanren" in types:
        out["sanren"] = dict(P=z["P"].astype(np.float64), G=gp["Gp"].astype(np.float64), Q=Q3, Qs=Q3s,
                             void=z["void"], hit=races.hit.to_numpy(), pay=races.payout.to_numpy())
    if "niren" in types:
        PAIRS = np.array([(i, j) for i in range(6) for j in range(6) if i != j])
        pi = {tuple(p): n for n, p in enumerate(PAIRS.tolist())}
        AGG = np.zeros((120, 30))
        for n, (a, b, _c) in enumerate(PERMS.tolist()):
            AGG[n, pi[(a, b)]] = 1
        out["niren"] = dict(P=z["P"].astype(np.float64) @ AGG, G=gp["G2p"].astype(np.float64), Q=Q3 @ AGG, Qs=Q3s @ AGG,
                            void=z["lane_void"][:, PAIRS].any(2), hit=races.hit2.to_numpy(), pay=races.pay2.to_numpy())
    if "tansho" in types:
        import model as Mo
        A1 = np.zeros((120, 6))
        A1[np.arange(120), PERMS[:, 0]] = 1
        h = pd.read_parquet(ROOT / "data" / "models" / "tansho_ev" / "history.parquet")
        h = h.drop_duplicates(["hd", "jcd", "rno"], keep="last")
        k = races[["hd", "jcd", "rno"]].merge(h, on=["hd", "jcd", "rno"], how="left")
        P1r = k[[f"p1r_{i}" for i in range(1, 7)]].to_numpy(float)
        okp = np.isfinite(P1r).all(1)
        P = np.full(P1r.shape, np.nan)
        P[okp] = Mo.apply_alpha(P1r[okp], k["alpha"].to_numpy(float)[okp])
        out["tansho"] = dict(P=P, G=k[[f"g_{i}" for i in range(1, 7)]].to_numpy(float), Q=Q3 @ A1, Qs=Q3s @ A1,
                             void=z["lane_void"], hit=races.win1.to_numpy(), pay=races.pay1.to_numpy())
    for d in out.values():
        d["month"] = month
        d["ok"] = (d["hit"] >= 0) & np.isfinite(d["P"]).all(1) & np.isfinite(d["G"]).all(1) & np.isfinite(d["Q"]).all(1) \
            & (d["G"] > 0).all(1)
        d["R"] = np.zeros(d["P"].shape)
        jj = np.where(d["hit"] >= 0)[0]
        d["R"][jj, d["hit"][jj]] = d["pay"][jj]
    return out


# ============================================================== 当てはめ
def lg(x):
    return np.log(np.clip(x, 1e-9, None))


def feats_a(P, Q, G):
    return np.stack([lg(P), lg(Q), lg(G / 100)], -1)


def fit_logit(F, valid, win, th0):
    """F: (n,K,d)、valid: (n,K) 選べる目、win: (n,)。s=F·θ のレース内 softmax の対数損失最小。"""
    n = len(win)
    Fw = F[np.arange(n), win]
    neg = np.where(valid, 0.0, -1e9)

    def f(th):
        s = F @ th + neg
        s -= s.max(1, keepdims=True)
        e = np.exp(s)
        Z = e.sum(1)
        q = e / Z[:, None]
        ll = -(s[np.arange(n), win] - np.log(Z)).mean()
        grad = -(Fw - np.einsum("nk,nkd->nd", q, F)).mean(0)
        return ll, grad

    return minimize(f, th0, jac=True, method="L-BFGS-B").x


def apply_logit(F, valid, th):
    s = F @ th + np.where(valid, 0.0, -1e9)
    s -= s.max(1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(1, keepdims=True)


def fit_poisson_offset(X, y, g):
    """E[y] = g·exp(Xβ)。"""
    sc = g.mean()
    y, g = y / sc, g / sc

    def f(b):
        eta = X @ b
        mu = g * np.exp(eta)
        return (mu - y * eta).mean(), X.T @ (mu - y) / len(y)

    return minimize(f, np.zeros(X.shape[1]), jac=True, method="L-BFGS-B").x


def fit_blend_p(LP, LQ, valid, win):
    """(a') log P' = (1-w) log P + w log q̂ のレース内正規化。w∈[0,1] を対数損失で。"""
    from scipy.optimize import minimize_scalar
    n = len(win)
    D = LQ - LP
    neg = np.where(valid, 0.0, -1e9)

    def f(w):
        s = LP + w * D + neg
        s -= s.max(1, keepdims=True)
        return -(s[np.arange(n), win] - np.log(np.exp(s).sum(1))).mean()

    return minimize_scalar(f, bounds=(0.0, 1.0), method="bounded").x


def apply_blend_p(LP, LQ, valid, w):
    s = LP + w * (LQ - LP) + np.where(valid, 0.0, -1e9)
    s -= s.max(1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(1, keepdims=True)


def fit_blend_g(G, Q, y):
    """(a') g' = exp(c)·g^(1-v)·(100/q̂)^v。v∈[0,1]。戻り値 (c, v)。"""
    L = lg(100 / Q) - lg(G)
    sc = G.mean()
    yy, gg = y / sc, G / sc

    def f(b):
        eta = b[0] + b[1] * L
        mu = gg * np.exp(eta)
        r = mu - yy
        return (mu - yy * eta).mean(), np.array([r.mean(), (r * L).mean()])

    return minimize(f, np.array([0.0, 0.5]), jac=True, method="L-BFGS-B", bounds=[(None, None), (0.0, 1.0)]).x


def apply_blend_g(G, Q, cv):
    return G * np.exp(cv[0] + cv[1] * (lg(100 / Q) - lg(G)))


def xg(G, Q, P):
    return np.stack([np.ones_like(G), lg(G / 100), lg(Q), lg(P)], -1)


def xb(EV, P):
    return np.stack([np.ones_like(EV), lg(EV), lg(P)], -1)


def eligible(kind, E, P, void, ok):
    """選ぶ目の (n,K) bool。単勝は p≥下限の艇すべて、2連単・3連単は p≥下限の中で E 最大の1点。"""
    can = (P >= PMIN[kind]) & ~void & ok[:, None]
    if kind == "tansho":
        return can
    Em = np.where(can, E, -np.inf)
    j = Em.argmax(1)
    el = np.zeros(P.shape, bool)
    r = np.where(np.isfinite(Em[np.arange(len(E)), j]))[0]
    el[r, j[r]] = True
    return el


def prior_months(allm, m):
    return [x for x in allm if FIT_FROM <= x < m][-WIN_MONTHS:]


# ============================================================== 1券種のウォークフォワード
def run_kind(kind, d):
    P, G, Q, void, ok, month, hit, R = d["P"], d["G"], d["Q"], d["void"], d["ok"], d["month"], d["hit"], d["R"]
    allm = sorted(set(month))
    target = [m for m in allm if m <= VALID[-1] and len(prior_months(allm, m)) >= MIN_PRIOR]
    EVr = P * G / 100
    rows, params = [], []
    for m in target:
        pm = prior_months(allm, m)
        tr = np.where(np.isin(month, pm) & ok)[0]
        te = np.where((month == m) & ok)[0]
        # ---- (a)
        valid_tr = ~void[tr]
        th = fit_logit(feats_a(P[tr], Q[tr], G[tr]), valid_tr, hit[tr], np.array([1.0, 0.0, 0.0]))
        wi = hit[tr]
        ar = np.arange(len(tr))
        bg = fit_poisson_offset(xg(G[tr, wi], Q[tr, wi], P[tr, wi]), R[tr, wi], G[tr, wi])
        # ---- (a')
        w = fit_blend_p(lg(P[tr]), lg(Q[tr]), valid_tr, hit[tr])
        cv = fit_blend_g(G[tr, wi], Q[tr, wi], R[tr, wi])
        # ---- (b)
        elr = eligible(kind, EVr[tr], P[tr], void[tr], np.ones(len(tr), bool))
        ri, ci = np.where(elr)
        bb = fit_poisson_offset(xb(EVr[tr][ri, ci], P[tr][ri, ci]), R[tr][ri, ci], EVr[tr][ri, ci] * 100)
        params.append(dict(month=m, n_fit=len(tr), alpha=th[0], beta=th[1], gamma=th[2], w_p=w, g_c=cv[0], v_g=cv[1],
                           **{f"g_b{i}": x for i, x in enumerate(bg)}, **{f"ev_c{i}": x for i, x in enumerate(bb)}))
        log(f"{kind} {m} n={len(tr)} th={np.round(th, 3)} bg={np.round(bg, 3)} bb={np.round(bb, 3)} w={w:.3f} cv={np.round(cv, 3)}")
        del ar
        # ---- 当てる月
        Pt, Gt, Qt, vt = P[te], G[te], Q[te], void[te]
        okt = np.ones(len(te), bool)
        sets = {}
        sets["raw"] = (Pt, EVr[te], eligible(kind, EVr[te], Pt, vt, okt))
        Pa = apply_logit(feats_a(Pt, Qt, Gt), ~vt, th)
        Ga = Gt * np.exp(xg(Gt, Qt, Pt) @ bg)
        EVa = Pa * Ga / 100
        sets["a"] = (Pa, EVa, eligible(kind, EVa, Pa, vt, okt))
        Pc = apply_blend_p(lg(Pt), lg(Qt), ~vt, w)
        EVc = Pc * apply_blend_g(Gt, Qt, cv) / 100
        sets["c"] = (Pc, EVc, eligible(kind, EVc, Pc, vt, okt))
        el = sets["raw"][2]
        EVb = np.where(el, EVr[te] * np.exp(xb(EVr[te], Pt) @ bb), np.nan)
        sets["b"] = (Pt, EVb, el)
        if m in STRICT and np.isfinite(d["Qs"][te]).all():
            Qs = d["Qs"][te]
            Pas = apply_logit(feats_a(Pt, Qs, Gt), ~vt, th)
            EVas = Pas * (Gt * np.exp(xg(Gt, Qs, Pt) @ bg)) / 100
            sets["a_strict"] = (Pas, EVas, eligible(kind, EVas, Pas, vt, okt))
            Pcs = apply_blend_p(lg(Pt), lg(Qs), ~vt, w)
            EVcs = Pcs * apply_blend_g(Gt, Qs, cv) / 100
            sets["c_strict"] = (Pcs, EVcs, eligible(kind, EVcs, Pcs, vt, okt))
        for meth, (PP, EE, EL) in sets.items():
            r_, c_ = np.where(EL)
            rows.append(pd.DataFrame(dict(month=m, method=meth, race=te[r_], comb=c_, p=PP[r_, c_], ev=EE[r_, c_],
                                          ev_raw=EVr[te][r_, c_], p_raw=Pt[r_, c_], ret=R[te][r_, c_])))
    return pd.concat(rows, ignore_index=True), pd.DataFrame(params)


# ============================================================== 集計
def month_summary(b, months):
    g = b.groupby("month").agg(bets=("ret", "size"), hits=("ret", lambda r: int((r > 0).sum())), ret=("ret", "sum"))
    g = g.reindex(months).fillna(0)
    roi_m = np.where(g.bets > 0, g.ret / 100 / g.bets.clip(lower=1), np.nan)
    tot = g.ret.sum() / 100 / max(g.bets.sum(), 1)
    return dict(roi=tot, win=int(np.nansum(roi_m > 1)), worst=float(np.nanmin(roi_m)) if np.isfinite(roi_m).any() else np.nan,
                bets_m=g.bets.sum() / len(months), hits=int(g.hits.sum())), pd.Series(roi_m, index=months)


def honesty(b):
    """選ばれた買い目の 予想 vs 実際。SE は払戻のばらつきから。"""
    n = len(b)
    if n == 0:
        return dict(n=0)
    r = b.ret.to_numpy() / 100
    se = r.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    return dict(n=n, 予想的中率=b.p.mean(), 実際の的中率=(r > 0).mean(), 予想EV=b.ev.mean(), 実際の回収率=r.mean(),
                SE=se, z=(r.mean() - b.ev.mean()) / se)


def report(kind, bets):
    out = [f"## {NAME[kind]}", ""]
    rows, mrows = [], {}
    for meth in METHODS:
        for t in TAUS:
            sel = bets[(bets.method == meth) & (bets.ev >= t)]
            a, ma = month_summary(sel[sel.month.isin(DISC)], DISC)
            v, mv = month_summary(sel[sel.month.isin(VALID)], VALID)
            hv = honesty(sel[sel.month.isin(VALID)])
            hd_ = honesty(sel[sel.month.isin(DISC)])
            nm = f"{MNAME[meth]} EV≥{t:.2f}"
            rows.append(dict(券種=NAME[kind], method=meth, tau=t, 候補=nm,
                             **{f"{k}_発見": x for k, x in a.items()}, **{f"{k}_検証": x for k, x in v.items()},
                             予想EV_発見=hd_.get("予想EV"), 実回収_発見=hd_.get("実際の回収率"),
                             予想EV_検証=hv.get("予想EV"), 実回収_検証=hv.get("実際の回収率"), SE_検証=hv.get("SE"), z_検証=hv.get("z"),
                             予想的中_検証=hv.get("予想的中率"), 実的中_検証=hv.get("実際の的中率")))
            mrows[nm] = pd.concat([ma, mv])
    res = pd.DataFrame(rows)
    show = res.drop(columns=["券種", "method", "tau"]).set_index("候補")
    out += ["### 候補（発見 2025-01〜10 の10か月 / 検証 2025-11〜2026-10 の12か月）",
            show[[c for c in show.columns if c.endswith("_発見") and not c.startswith(("予想", "実"))]
                 + [c for c in show.columns if c.endswith("_検証") and not c.startswith(("予想", "実", "SE", "z"))]]
            .to_markdown(floatfmt=".3f"), "",
            "### 選ばれた買い目の 予想 vs 実際（検証用。z = (実際の回収率−予想EV)/SE）",
            show[["予想EV_発見", "実回収_発見", "予想EV_検証", "実回収_検証", "SE_検証", "z_検証", "予想的中_検証", "実的中_検証"]]
            .to_markdown(floatfmt=".3f"), ""]
    # 発見用で1本
    picks = []
    for meth in ["a", "c", "b"]:
        r = res[(res.method == meth) & (res.hits_発見 >= MIN_DISC_HITS)]
        if len(r):
            picks.append(r.loc[r.roi_発見.idxmax(), "候補"])
    out += ["### 発見用で選んだ1本（券種×補正ごと）", show.loc[picks, ["roi_発見", "win_発見", "roi_検証", "win_検証", "worst_検証",
                                                       "bets_m_検証", "hits_検証"]].to_markdown(floatfmt=".3f"), ""]
    # 予想EVの帯（検証用、τ 最小の集合 = 選ぶ目すべて）
    bands = (0, 0.8, 0.9, 1.0, 1.05, 1.1, 1.2, 1.5, 1e9)
    tb = {}
    for meth in METHODS:
        s = bets[(bets.method == meth) & bets.month.isin(VALID)]
        bb = pd.cut(s.ev, bands, right=False)
        tb[MNAME[meth]] = s.groupby(bb, observed=True).agg(点=("ret", "size"), 予想EV=("ev", "mean"),
                                                          実際=("ret", lambda r: r.mean() / 100))
    out += ["### 選ぶ目すべての 予想EVの帯 → 実際の回収率（検証用）", pd.concat(tb, axis=1).to_markdown(floatfmt=".3f"), ""]
    # 厳密 q̂ の感度
    if (bets.method == "a_strict").any():
        sr = []
        for meth in ["a", "a_strict", "c", "c_strict"]:
            for t in TAUS:
                sel = bets[(bets.method == meth) & (bets.ev >= t) & bets.month.isin(STRICT)]
                s_, _ = month_summary(sel, STRICT)
                h_ = honesty(sel)
                sr.append(dict(補正=MNAME[meth[0]], q̂=("LQs（厳密）" if meth.endswith("strict") else "LQ（LOMO）"), tau=t, **s_,
                               予想EV=h_.get("予想EV"), 実際=h_.get("実際の回収率")))
        out += ["### (a)・(a') の q̂ を厳密版に替えた感度（2026-01〜10、係数は同じ）",
                pd.DataFrame(sr).to_markdown(index=False, floatfmt=".3f"), ""]
    out += ["### 月別の回収率", pd.DataFrame(mrows).T.to_markdown(floatfmt=".2f"), ""]
    return res, "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--types", default="niren,sanren,tansho")
    a = ap.parse_args()
    types = a.types.split(",")
    WORK.mkdir(parents=True, exist_ok=True)
    data = load_all(types)
    for kind in types:
        d = data.pop(kind)
        log(f"{kind}: races ok={int(d['ok'].sum())}")
        bets, params = run_kind(kind, d)
        del d
        bets.to_parquet(WORK / f"bets_{kind}.parquet")
        params.to_csv(WORK / f"params_{kind}.csv", index=False)
        res, txt = report(kind, bets)
        res.to_csv(WORK / f"cands_{kind}.csv", index=False)
        (WORK / f"calib_{kind}.md").write_text(txt)
        print(txt, flush=True)


if __name__ == "__main__":
    main()
