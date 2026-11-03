"""multi_bet.py の評価部分: 市場とのズレ（market）と、事前に決めた買い方の候補（bets）。

候補の一覧は CANDIDATES に**先に**全部書いてあり、評価の前に数を記録する。
発見用 2024-11〜2025-10 と 検証用 2025-11〜2026-10 の両方の月別成績を出す。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import multi_bet as M
import trifecta as T

PERMS, PAIRS, AGG = M.PERMS, M.PAIRS, M.AGG
DISC, VALID = M.DISC, M.VALID


# ============================================================== 共通の材料
def materials(work: Path):
    races, z = M.load(work)
    gp = np.load(M.mb(work) / "gplus.npz")
    qh = M.norm_q(np.load(M.mb(work) / "qhat.npz")["LQ"].astype(float))
    P = z["P"].astype(float)
    P1 = z["P1"].astype(float)
    P2 = P @ AGG
    lv = z["lane_void"]
    void3 = lv[:, PERMS].any(2)
    void2 = lv[:, PAIRS].any(2)
    hit, pay = races.hit.to_numpy(), races.payout.to_numpy()
    hit2, pay2 = races.hit2.to_numpy(), races.pay2.to_numpy()
    W3 = np.zeros(P.shape, bool)
    W3[np.where(hit >= 0)[0], hit[hit >= 0]] = True
    W2 = np.zeros(P2.shape, bool)
    W2[np.where(hit2 >= 0)[0], hit2[hit2 >= 0]] = True
    return dict(races=races, z=z, P=P, P1=P1, P2=P2, g=z["G"].astype(float), gp=gp["Gp"].astype(float),
                g2=gp["G2"].astype(float), g2p=gp["G2p"].astype(float), qh=qh, q2h=qh @ AGG,
                void3=void3, void2=void2, hit=hit, pay=pay, hit2=hit2, pay2=pay2, W3=W3, W2=W2,
                month=races.month.to_numpy())


# ============================================================== 買い方の部品（レース×買い目のマスクを返す）
def fav_first_mask(P1, P, t, k, pairs=False):
    """本命（1着確率最大）が P1≥t のレースで、本命1着の目を確率上位 k 点。"""
    fav = P1.argmax(1)
    first = (PAIRS[:, 0] if pairs else PERMS[:, 0])[None, :] == fav[:, None]
    Pm = np.where(first, P, -1)
    rk = (-Pm).argsort(1).argsort(1)
    return (rk < k) & first & (P1.max(1) >= t)[:, None]


def box_mask(P1, n, pairs=False):
    top = np.argsort(-P1, 1)[:, :n]
    inset = np.zeros(P1.shape, bool)
    np.put_along_axis(inset, top, True, 1)
    C = PAIRS if pairs else PERMS
    return inset[:, C].all(2)


def ev_topk_mask(EV, P, pmin, k, tau):
    E = np.where(P >= pmin, EV, -np.inf)
    rk = (-E).argsort(1).argsort(1)
    return (rk < k) & (E >= tau)


def set_ev(mask, EV):
    n = mask.sum(1)
    return np.where(n > 0, (EV * mask).sum(1) / np.maximum(n, 1), 0)


def build_candidates():
    """(名前, 券種, 関数(ctx)->mask)。ctx は P, P1, P2, EV3[est], EV2[est] を持つ。"""
    C = []
    # ---- 3連単 ----
    for est in ["g", "g+"]:
        for tau in [1.0, 1.1, 1.2, 1.3]:
            for pm in [0.005, 0.01, 0.02]:
                C.append((f"3T:EV全部:{est}:EV>={tau}:p>={pm}", 3, lambda c, e=est, t=tau, p=pm: (c["EV3"][e] >= t) & (c["P"] >= p)))
    for t in [0.5, 0.65]:
        for k in [4, 6, 10]:
            C.append((f"3T:1着固定流し:P1>={t}:{k}点", 3, lambda c, t=t, k=k: fav_first_mask(c["P1"], c["P"], t, k)))
            for est in ["g", "g+"]:
                for se in [1.0, 1.1]:
                    C.append((f"3T:1着固定流し:P1>={t}:{k}点:{est}:組EV>={se}", 3,
                              lambda c, t=t, k=k, e=est, s=se: (lambda m: m & (set_ev(m, c["EV3"][e]) >= s)[:, None])(
                                  fav_first_mask(c["P1"], c["P"], t, k))))
    for n in [3, 4]:
        C.append((f"3T:BOX上位{n}艇", 3, lambda c, n=n: box_mask(c["P1"], n)))
        for est in ["g", "g+"]:
            for se in [1.0, 1.1]:
                C.append((f"3T:BOX上位{n}艇:{est}:組EV>={se}", 3,
                          lambda c, n=n, e=est, s=se: (lambda m: m & (set_ev(m, c["EV3"][e]) >= s)[:, None])(box_mask(c["P1"], n))))
    for est in ["g", "g+"]:
        for pm in [0.005, 0.01]:
            for k in [1, 3, 5]:
                for tau in [1.0, 1.1, 1.2]:
                    C.append((f"3T:EV上位k:{est}:p>={pm}:k={k}:EV>={tau}", 3,
                              lambda c, e=est, p=pm, k=k, t=tau: ev_topk_mask(c["EV3"][e], c["P"], p, k, t)))
    # ---- 2連単 ----
    for est in ["g2", "g2+"]:
        for tau in [1.0, 1.1, 1.2]:
            for pm in [0.05, 0.10, 0.15]:
                C.append((f"2T:EV全部:{est}:EV>={tau}:p>={pm}", 2, lambda c, e=est, t=tau, p=pm: (c["EV2"][e] >= t) & (c["P2"] >= p)))
    for t in [0.5, 0.65]:
        for k in [2, 3]:
            C.append((f"2T:1着固定流し:P1>={t}:{k}点", 2, lambda c, t=t, k=k: fav_first_mask(c["P1"], c["P2"], t, k, True)))
            for est in ["g2", "g2+"]:
                for se in [1.0, 1.1]:
                    C.append((f"2T:1着固定流し:P1>={t}:{k}点:{est}:組EV>={se}", 2,
                              lambda c, t=t, k=k, e=est, s=se: (lambda m: m & (set_ev(m, c["EV2"][e]) >= s)[:, None])(
                                  fav_first_mask(c["P1"], c["P2"], t, k, True))))
    for n in [2, 3]:
        C.append((f"2T:BOX上位{n}艇", 2, lambda c, n=n: box_mask(c["P1"], n, True)))
        for est in ["g2", "g2+"]:
            for se in [1.0, 1.1]:
                C.append((f"2T:BOX上位{n}艇:{est}:組EV>={se}", 2,
                          lambda c, n=n, e=est, s=se: (lambda m: m & (set_ev(m, c["EV2"][e]) >= s)[:, None])(box_mask(c["P1"], n, True))))
    for est in ["g2", "g2+"]:
        for pm in [0.05, 0.10]:
            for k in [1, 2, 3]:
                for tau in [1.0, 1.1, 1.2]:
                    C.append((f"2T:EV上位k:{est}:p>={pm}:k={k}:EV>={tau}", 2,
                              lambda c, e=est, p=pm, k=k, t=tau: ev_topk_mask(c["EV2"][e], c["P2"], p, k, t)))
    return C


def month_table(mask, W, pay, void, ok, month):
    mk = mask & ~void & ok[:, None]
    n = mk.sum(1)
    ret = (mk & W).any(1) * pay
    df = pd.DataFrame(dict(month=month, bets=n, hits=(mk & W).sum(1), ret=ret, races=n > 0))
    g = df.groupby("month").agg(bets=("bets", "sum"), hits=("hits", "sum"), ret=("ret", "sum"), races=("races", "sum"))
    g["cost"] = g.bets * 100.0
    return g


def summarize(g, ms):
    x = g.reindex(ms).fillna(0)
    roi_m = np.where(x.cost > 0, x.ret / x.cost.clip(lower=1), np.nan)
    return dict(roi=x.ret.sum() / max(x.cost.sum(), 1), win=int(np.nansum(roi_m > 1)), worst=float(np.nanmin(roi_m)) if np.isfinite(roi_m).any() else np.nan,
                bets_m=x.bets.sum() / len(ms), hits=int(x.hits.sum()), races_m=x.races.sum() / len(ms))


def cmd_bets(work: Path):
    c = materials(work)
    month = c["month"]
    ok3 = (c["hit"] >= 0) & np.isfinite(c["g"]).all(1) & np.isfinite(c["gp"]).all(1)
    ok2 = (c["hit2"] >= 0) & np.isfinite(c["g2"]).all(1) & np.isfinite(c["g2p"]).all(1)
    ctx = dict(P=c["P"], P1=c["P1"], P2=c["P2"],
               EV3={"g": c["P"] * np.nan_to_num(c["g"]) / 100, "g+": c["P"] * np.nan_to_num(c["gp"]) / 100},
               EV2={"g2": c["P2"] * np.nan_to_num(c["g2"]) / 100, "g2+": c["P2"] * np.nan_to_num(c["g2p"]) / 100})
    cands = build_candidates()
    n3 = sum(1 for x in cands if x[1] == 3)
    T.log(f"candidates: {len(cands)} (3連単 {n3}, 2連単 {len(cands)-n3})")
    rows, mrows = [], []
    for name, kind, fn in cands:
        mask = fn(ctx)
        if kind == 3:
            g = month_table(mask, c["W3"], c["pay"], c["void3"], ok3, month)
        else:
            g = month_table(mask, c["W2"], c["pay2"], c["void2"], ok2, month)
        s1, s2 = summarize(g, DISC), summarize(g, VALID)
        rows.append(dict(cand=name, **{f"{k}_発見": v for k, v in s1.items()}, **{f"{k}_検証": v for k, v in s2.items()}))
        mrows.append(g.assign(cand=name).reset_index())
    res = pd.DataFrame(rows)
    cm = pd.concat(mrows)
    out = M.mb(work)
    res.to_csv(out / "bets_summary.csv", index=False)
    cm.to_parquet(out / "bets_cand_month.parquet")
    report_bets(work)


def report_bets(work: Path):
    out = M.mb(work)
    res = pd.read_csv(out / "bets_summary.csv")
    cm = pd.read_parquet(out / "bets_cand_month.parquet")
    L = [f"## 買い方の候補 {len(res)} 通り（3連単 {res.cand.str.startswith('3T').sum()}・2連単 {res.cand.str.startswith('2T').sum()}）", ""]
    cols = ["cand", "roi_発見", "win_発見", "worst_発見", "bets_m_発見", "hits_発見", "roi_検証", "win_検証", "worst_検証", "bets_m_検証", "hits_検証"]
    # 事前の選び方: 発見用で的中≥100本の中で通算回収率が最良の1本 → 検証へ（系統ごとにも）
    fam = res.cand.str.split(":").str[:2].str.join(":")
    res["fam"] = fam
    elig = res[res.hits_発見 >= 100]
    best = elig.sort_values("roi_発見", ascending=False).head(1)
    L.append("### 発見用で1本選ぶ（的中≥100本の中で発見用の回収率が最良）→ 検証用")
    L.append(best[cols].to_markdown(index=False, floatfmt=".3f"))
    L.append("")
    L.append("### 系統ごとに発見用で最良の1本（的中≥100本）→ 検証用")
    fb = elig.sort_values("roi_発見", ascending=False).groupby("fam").head(1)
    L.append(fb[cols].to_markdown(index=False, floatfmt=".3f"))
    L.append("")
    both = res[(res.roi_発見 > 1) & (res.roi_検証 > 1)]
    L.append(f"### 2期間とも 1.0 超え: {len(both)} 本（うち両期間の勝ち月≥8: {((both.win_発見 >= 8) & (both.win_検証 >= 8)).sum()} 本）")
    if len(both):
        L.append(both.sort_values("roi_検証", ascending=False)[cols].to_markdown(index=False, floatfmt=".3f"))
    L.append("")
    L.append("### 点数の多い（発見用で月≥300点）候補の上位10（発見用の回収率順）")
    L.append(res[res.bets_m_発見 >= 300].sort_values("roi_発見", ascending=False).head(10)[cols].to_markdown(index=False, floatfmt=".3f"))
    L.append("")
    L.append("### 検証用で合格ライン（>1.0・勝ち月≥8）を満たした候補（後知恵）")
    hs = res[(res.roi_検証 > 1) & (res.win_検証 >= 8)]
    L.append(hs[cols].to_markdown(index=False, floatfmt=".3f") if len(hs) else "なし")
    txt = "\n".join(L)
    (out / "bets_report.md").write_text(txt)
    print(txt)


# ============================================================== 市場とのズレ
def clogit(S_list, hitidx):
    """条件付きロジット: hit ~ Σ_k b_k S_k（各 S は nr×K）。係数と対数損失。"""
    from scipy.optimize import minimize
    S = np.stack(S_list, 2)
    n = len(hitidx)

    def f(b):
        u = S @ b
        u = u - u.max(1, keepdims=True)
        lse = np.log(np.exp(u).sum(1))
        ll = u[np.arange(n), hitidx] - lse
        p = np.exp(u - lse[:, None])
        grad = S[np.arange(n), hitidx] - (p[:, :, None] * S).sum(1)
        return -ll.mean(), -grad.mean(0)
    r = minimize(f, np.ones(S.shape[2]) * 0.5, jac=True, method="L-BFGS-B")
    return r.x, r.fun


def cmd_market(work: Path):
    c = materials(work)
    races, z = c["races"], c["z"]
    O = z["O"].astype(float)
    month = c["month"]
    om = [m for m in M.odds_months(work) if m in M.full_months(races)]
    has = np.isfinite(O).all(1) & (O > 0).all(1) & (c["hit"] >= 0) & np.isin(month, om)
    ii = np.where(has)[0]
    q, rate = M.market_q(O[ii])
    P = c["P"][ii]
    qh = c["qh"][ii]
    h = c["hit"][ii]
    W = c["W3"][ii]
    pay = c["pay"][ii]
    V = ~c["void3"][ii]
    L = [f"## モデルと市場のズレ（3連単の締切オッズがある 2026-03〜10、{len(ii)} レース）", ""]
    # (a) 条件付きロジット
    lP, lq, lqh = np.log(np.clip(P, 1e-12, None)), np.log(q), np.log(qh)
    for nm, S in [("市場だけ", [lq]), ("モデルだけ", [lP]), ("市場＋モデル", [lq, lP]), ("予想市場 q̂＋モデル", [lqh, lP]),
                  ("市場＋モデル＋q̂", [lq, lP, lqh])]:
        b, ll = clogit(S, h)
        L.append(f"- 条件付きロジット {nm}: 係数 {np.round(b, 3).tolist()}、対数損失 {ll:.4f}")
    L.append("")
    # (b) d = log P − log q の帯ごとの回収率（実際の締切オッズで買えたとした場合 = 後知恵）と、予想市場 q̂ での d̂
    def band_table(d, label, extra_mask=None):
        mk = V & (P >= 0.002)
        if extra_mask is not None:
            mk &= extra_mask
        bins = [-9, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1.0, 9]
        b = np.digitize(d, bins) - 1
        rows = []
        for k in range(len(bins) - 1):
            s = mk & (b == k)
            n = s.sum()
            if n < 100:
                continue
            ret = (s & W).sum(1) @ pay
            rows.append({label: f"{bins[k]}〜{bins[k+1]}", "買い目": int(n), "的中": int((s & W).sum()),
                         "モデル確率の平均": P[s].mean(), "市場確率の平均": q[s].mean(), "実際の的中率": (s & W).sum() / n,
                         "回収率": ret / (100 * n)})
        return pd.DataFrame(rows)
    L.append("### (b) d = log(モデル確率) − log(締切市場確率) の帯ごと（確率≥0.2%の目。締切オッズを見て選ぶ＝後知恵）")
    L.append(band_table(lP - lq, "d").to_markdown(index=False, floatfmt=".4f"))
    L.append("")
    L.append("### (c) d̂ = log(モデル確率) − log(予想市場確率 q̂) の帯ごと（前夜に分かる。同じ8か月）")
    L.append(band_table(lP - lqh, "d̂").to_markdown(index=False, floatfmt=".4f"))
    L.append("")
    # (d) 条件別: 「d≥0.5（モデルが市場より強気）」の目の回収率と、モデル/市場どちらが当たっているか
    a = PERMS[:, 0]
    grd = z["grd"][ii]
    qrank = (-q).argsort(1).argsort(1) + 1
    segs = {
        "1着の艇番": np.broadcast_to(a[None, :] + 1, P.shape),
        "1着艇の級別(4=A1)": grd[:, a],
        "市場の人気順位(帯)": np.digitize(qrank, [1, 4, 11, 31, 61]),
        "レース種別": np.broadcast_to(races.race_cat.to_numpy()[ii][:, None], P.shape),
        "場": np.broadcast_to(races.jcd.to_numpy()[ii][:, None], P.shape),
    }
    for dn, d in [("d（締切）", lP - lq), ("d̂（前夜）", lP - lqh)]:
        L.append(f"### (d) 条件別: {dn} ≥ 0.5 の目（モデルが市場より強気）と ≤ −0.5 の目")
        for sn, sv in segs.items():
            rows = []
            for v in np.unique(sv):
                base = V & (sv == v) & (P >= 0.002)
                r = {sn: v}
                for side, cond in [("強気", d >= 0.5), ("弱気", d <= -0.5)]:
                    s = base & cond
                    n = s.sum()
                    r[f"{side}:点"] = int(n)
                    r[f"{side}:実際/モデル"] = (s & W).sum() / max(P[s].sum(), 1e-9)
                    r[f"{side}:実際/市場"] = (s & W).sum() / max(q[s].sum(), 1e-9)
                    r[f"{side}:回収率"] = ((s & W).sum(1) @ pay) / max(100 * n, 1)
                rows.append(r)
            L.append(pd.DataFrame(rows).to_markdown(index=False, floatfmt=".3f"))
            L.append("")
    # (e) 市場の本命-大穴の偏り（締切オッズ帯ごとの回収率）
    rows = []
    for lo_, hi_ in [(0, 5), (5, 10), (10, 20), (20, 50), (50, 100), (100, 300), (300, 1e9)]:
        s = V & (O[ii] >= lo_) & (O[ii] < hi_)
        n = s.sum()
        rows.append({"締切オッズ": f"{lo_}〜{hi_}", "買い目": int(n), "回収率": ((s & W).sum(1) @ pay) / (100 * n),
                     "実際/市場確率": (s & W).sum() / q[s].sum()})
    L.append("### (e) 締切オッズ帯ごとの回収率（全部買った場合）")
    L.append(pd.DataFrame(rows).to_markdown(index=False, floatfmt=".3f"))
    L.append("")
    # (f) 2連単: 3連単オッズから集計した市場確率 q2 と実際の2連単払戻
    h2 = c["hit2"][ii]
    ok = h2 >= 0
    q2 = q @ AGG
    qq = q2[np.where(ok)[0], h2[ok]]
    p2 = c["pay2"][ii][ok]
    rr = rate[ok]
    est = 100 * 0.75 / qq
    L.append(f"### (f) 2連単: 3連単オッズの市場確率を集計した q2 で当たり目の配当を換算（{ok.sum()} レース）")
    L.append(f"- log(0.75/q2×100) と log(実際の2連単払戻) の相関 {np.corrcoef(np.log(est), np.log(p2))[0,1]:.3f}、"
             f"比(実際/換算) の中央値 {np.median(p2/est):.3f}、10%点 {np.quantile(p2/est,.1):.3f}、90%点 {np.quantile(p2/est,.9):.3f}")
    txt = "\n".join(L)
    (M.mb(work) / "market_report.md").write_text(txt)
    print(txt)


# ============================================================== 単勝ルールとの併用
def cmd_tansho_combo(work: Path):
    """単勝ルール tansho_ev_v1（P1≥0.2 かつ P1×G1/100≥1.0）で選んだ艇のレースに、2連単（その艇1着）を足す。
    G1 は trifecta.py cmd_tansho と同じ作り（月 m より前の単勝の当たりで学習）。候補は事前に 8 通り。"""
    import lightgbm as lgb
    c = materials(work)
    races, z = c["races"], c["z"]
    month = c["month"]
    win, TPv = races.win1.to_numpy(), races.pay1.to_numpy()
    has = win >= 0
    TP = np.zeros((len(races), 6))
    TP[np.where(has)[0], win[has]] = TPv[has]
    P1, _ = T.recalibrate(c["P1"], win, month, T.G_FIRST_MONTH)
    nwr, grd = z["nwr"], z["grd"]
    r1 = (-P1).argsort(1).argsort(1) + 1
    nrk = (-np.nan_to_num(nwr, nan=0)).argsort(1).argsort(1) + 1
    H1 = -(P1 * np.log(P1)).sum(1)

    def feats(ix):
        n = len(ix)
        rep = lambda x: np.repeat(np.asarray(x, float), 6)
        cols = {"lane": np.tile(np.arange(1, 7), n), "lp": np.log(P1[ix]).ravel(), "r1": r1[ix].ravel(),
                "nw": nwr[ix].ravel(), "g": grd[ix].ravel(), "nrk": nrk[ix].ravel(), "H1": rep(H1[ix]),
                "p1max": rep(P1[ix].max(1)), "nw1": rep(nwr[ix, 0]), "nwmax_o": rep(np.nanmax(nwr[ix, 1:], 1)),
                "jcd": rep(races["jcd"].to_numpy()[ix].astype(int)), "rno": rep(races["rno"].to_numpy()[ix])}
        for cc in T.RACE_FLAGS:
            cols[cc] = rep(races[cc].to_numpy()[ix])
        return pd.DataFrame(cols).to_numpy(np.float32)
    G1 = np.full((len(races), 6), np.nan)
    gpar = dict(T.G_PARAMS, num_threads=M.NT)
    for m in DISC + VALID:
        tr = np.where((month >= T.G_FIRST_MONTH) & (month < m) & has)[0]
        X = feats(tr).reshape(len(tr), 6, -1)[np.arange(len(tr)), win[tr]]
        mdl = lgb.train(gpar, lgb.Dataset(X, TP[tr, win[tr]]), T.G_ROUNDS)
        te = np.where(month == m)[0]
        G1[te] = mdl.predict(feats(te)).reshape(len(te), 6)
    lv = z["lane_void"]
    pick = (P1 >= 0.2) & (P1 * G1 / 100 >= 1.0) & ~lv & has[:, None]
    EV2 = c["P2"] * np.nan_to_num(c["g2"]) / 100
    ok2 = (c["hit2"] >= 0) & np.isfinite(c["g2"]).all(1)
    rows = []

    def summ(cost, ret, label):
        df = pd.DataFrame(dict(month=month, cost=cost, ret=ret)).groupby("month").sum()
        for per, ms in [("発見", DISC), ("検証", VALID)]:
            x = df.reindex(ms).fillna(0)
            r = x.ret / x.cost.clip(lower=1)
            rows.append(dict(買い方=label, 期間=per, 月の点数=x.cost.sum() / 100 / 12, 回収率=x.ret.sum() / x.cost.sum(),
                             勝ち月=int((r > 1).sum()), 最悪月=r.min()))
    cost1 = pick.sum(1) * 100.0
    ret1 = (pick * TP).sum(1)
    summ(cost1, ret1, "単勝ルールのみ（再現）")
    first_of = PAIRS[:, 0]
    for k in [1, 2, 3]:
        for evmin in [None, 1.0]:
            m2 = pick[:, first_of]                       # 単勝で選んだ艇が1着の2連単
            Pm = np.where(m2, c["P2"], -1)
            rk = (-Pm).argsort(1).argsort(1)
            m2 = m2 & (rk < k) & ~c["void2"] & ok2[:, None]
            if evmin is not None:
                m2 &= EV2 >= evmin
            cost2 = m2.sum(1) * 100.0
            ret2 = (m2 & c["W2"]).any(1) * c["pay2"]
            lab = f"2連単 選んだ艇→相手上位{k}" + (f"・EV(g2)≥{evmin}" if evmin else "")
            summ(cost2, ret2, lab + "（2連単部分のみ）")
            summ(cost1 + cost2, ret1 + ret2, lab + "（単勝と合算）")
    txt = "## 単勝ルールとの併用（2連単を足す、6候補）\n\n" + pd.DataFrame(rows).to_markdown(index=False, floatfmt=".3f")
    (M.mb(work) / "tansho_combo.md").write_text(txt)
    print(txt)


# ============================================================== 有望な候補の点検
CHECK = ["2T:EV上位k:g2+:p>=0.05:k=1:EV>=1.2", "2T:EV上位k:g2+:p>=0.05:k=1:EV>=1.1", "2T:EV全部:g2+:EV>=1.1:p>=0.1",
         "2T:EV全部:g2+:EV>=1.1:p>=0.05", "3T:EV全部:g+:EV>=1.2:p>=0.02", "3T:EV全部:g:EV>=1.1:p>=0.01"]


def strict_months_v1(work, c):
    """旧版 oddsmodel の厳密 q̂（LQs）がある、暦月の揃った月。"""
    LQs = np.load(M.mb(work) / "qhat.npz")["LQs"]
    ok = np.isfinite(LQs).all(1)
    fm = set(M.full_months(c["races"]))
    return sorted(m for m in set(c["month"][ok]) if m in fm)


def g2plus_strict(work, c):
    """2026-01〜10 の各月 m: テスト行の q̂ を「m より前のオッズだけ」で学習した厳密版に差し替えて g2+ を作り直す。"""
    import lightgbm as lgb
    import nirensho as N
    races, z = c["races"], c["z"]
    qq = np.load(M.mb(work) / "qhat.npz")
    q2h = c["q2h"]
    q2s = M.norm_q(np.where(np.isnan(qq["LQs"]), 0, qq["LQs"]).astype(float)) @ AGG
    month = c["month"]
    flags = races[T.RACE_FLAGS].to_numpy(np.float32)
    P2 = c["P2"]

    def X2(ix, Q2):
        X = N.g2_features(P2[ix], c["P1"][ix], z["nwr"][ix], z["grd"][ix], flags[ix])
        X["jcd"] = np.repeat(races.jcd.astype(int).to_numpy()[ix], 30)
        X["rno"] = np.repeat(races.rno.to_numpy()[ix], 30)
        X["lq2h"] = np.log(Q2[ix]).ravel()
        X["lq2h_rank"] = ((-Q2[ix]).argsort(1).argsort(1) + 1).ravel()
        X["dlog"] = X["logP"] - X["lq2h"]
        return X.to_numpy(np.float32)
    hit2, pay2 = c["hit2"], c["pay2"]
    G = c["g2p"].copy()
    for m in strict_months_v1(work, c):
        tr = np.where((month < m) & (hit2 >= 0))[0]
        X = X2(tr, q2h).reshape(len(tr), 30, -1)[np.arange(len(tr)), hit2[tr]]
        mdl = lgb.train(M.GP_PARAMS, lgb.Dataset(X, pay2[tr]), T.G_ROUNDS)
        te = np.where(month == m)[0]
        G[te] = mdl.predict(X2(te, q2s), num_threads=M.NT).reshape(len(te), 30)
        T.log(f"strict g2+ {m}")
    return G


def cmd_check(work: Path):
    c = materials(work)
    month = c["month"]
    cands = {n: (k, f) for n, k, f in build_candidates()}
    Gs = g2plus_strict(work, c)
    ok3 = (c["hit"] >= 0) & np.isfinite(c["g"]).all(1) & np.isfinite(c["gp"]).all(1)
    ok2 = (c["hit2"] >= 0) & np.isfinite(c["g2"]).all(1) & np.isfinite(c["g2p"]).all(1)
    rng = np.random.default_rng(0)
    L = ["## 有望な候補の点検", ""]
    races = c["races"]
    for variant in ["LOMO", "厳密（2026-01〜10 は過去のオッズだけ）"]:
        g2p = c["g2p"] if variant == "LOMO" else Gs
        ctx = dict(P=c["P"], P1=c["P1"], P2=c["P2"],
                   EV3={"g": c["P"] * np.nan_to_num(c["g"]) / 100, "g+": c["P"] * np.nan_to_num(c["gp"]) / 100},
                   EV2={"g2": c["P2"] * np.nan_to_num(c["g2"]) / 100, "g2+": c["P2"] * np.nan_to_num(g2p) / 100})
        L.append(f"### q̂ の作り: {variant}")
        rows = []
        for name in CHECK:
            kind, fn = cands[name]
            if kind == 3:
                W, pay, void, ok = c["W3"], c["pay"], c["void3"], ok3
            else:
                W, pay, void, ok = c["W2"], c["pay2"], c["void2"], ok2
            mk = fn(ctx) & ~void & ok[:, None]
            cost = mk.sum(1) * 100.0
            ret = (mk & W).any(1) * pay
            df = pd.DataFrame(dict(hd=races.hd, month=month, cost=cost, ret=ret, hit=(mk & W).any(1)))
            r = dict(cand=name)
            sm = strict_months_v1(work, c)
            for per, ms in [("発見", DISC), ("検証", VALID), (f"厳密区間 {sm[0]}〜{sm[-1]}", sm), ("24か月", DISC + VALID)]:
                x = df[df.month.isin(ms)]
                dd = x.groupby("hd")[["cost", "ret"]].sum()
                cc, rr = dd.cost.to_numpy(), dd.ret.to_numpy()
                bs = []
                for _ in range(1000):
                    i = rng.integers(0, len(dd), len(dd))
                    bs.append(rr[i].sum() / max(cc[i].sum(), 1))
                mm = x.groupby("month")[["cost", "ret"]].sum()
                r[per] = (f"{x.ret.sum()/max(x.cost.sum(),1):.3f} [{np.quantile(bs,.05):.2f}〜{np.quantile(bs,.95):.2f}] "
                          f"勝{int((mm.ret > mm.cost).sum())}/{len(mm)} 的中{int(x.hit.sum())}")
            rows.append(r)
            if variant == "LOMO" and name == CHECK[0]:
                mt = df.groupby("month").agg(点=("cost", lambda s: int(s.sum() / 100)), 的中=("hit", "sum"), 払戻=("ret", "sum"), 購入=("cost", "sum"))
                mt["回収率"] = mt.払戻 / mt.購入
                mt = mt.loc[DISC + VALID]
                monthly = mt[["点", "的中", "回収率"]]
                # 内訳: 1着の艇番・予想配当帯・レース種別
                sel = np.where(mk)
                g2v = np.nan_to_num(g2p)[sel] / 100
                det = pd.DataFrame(dict(per=np.where(month[sel[0]] < "2025-11", "発見", "検証"),
                                        first=PAIRS[sel[1], 0] + 1, second=PAIRS[sel[1], 1] + 1,
                                        oband=pd.cut(g2v, [0, 5, 10, 20, 50, 1e4]).astype(str),
                                        cat=races.race_cat.to_numpy()[sel[0]],
                                        hit=W[sel], ret=np.where(W[sel], pay[sel[0]], 0)))
                det = det[np.isin(month[sel[0]], DISC + VALID)]
                bd = []
                for col in ["first", "second", "oband", "cat"]:
                    t = det.groupby([col, "per"]).agg(n=("hit", "size"), hits=("hit", "sum"), ret=("ret", "sum"))
                    t["roi"] = t.ret / (100 * t.n)
                    bd.append(f"内訳 {col}\n" + t[["n", "hits", "roi"]].unstack("per").round(3).to_markdown())
        L.append(pd.DataFrame(rows).to_markdown(index=False))
        L.append("")
    L.append(f"### 月別: {CHECK[0]}（LOMO）")
    L.append(monthly.to_markdown(floatfmt=".3f"))
    L.append("")
    L += bd
    txt = "\n".join(L)
    (M.mb(work) / "check_report.md").write_text(txt)
    print(txt)


# ============================================================== v2: 事前に決めた候補（notes/analysis/nirensho_v2_plan.md）
# この一覧は評価の前に plan に書いたもの。中身を変えたら別の候補として plan に書き足す（数え直す）。
# (名前, q̂2 の作り, 買い方 'max1' | 'all', 確率の下限, EV の下限)
V2_CANDIDATES = [
    ("C1:EV最大1点:agg:p>=0.05:EV>=1.1", "agg", "max1", 0.05, 1.1),
    ("C2:EV最大1点:agg:p>=0.05:EV>=1.2", "agg", "max1", 0.05, 1.2),   # 主仮説（前回の候補そのもの）
    ("C3:EV最大1点:agg:p>=0.05:EV>=1.3", "agg", "max1", 0.05, 1.3),
    ("C4:EV最大1点:agg:p>=0.10:EV>=1.2", "agg", "max1", 0.10, 1.2),
    ("C5:EV全部:agg:p>=0.10:EV>=1.1", "agg", "all", 0.10, 1.1),
    ("C6:EV最大1点:direct:p>=0.05:EV>=1.2", "direct", "max1", 0.05, 1.2),  # 2連単オッズで q̂2 を直接学習できたときだけ
]
V2_PRIMARY = "C2:EV最大1点:agg:p>=0.05:EV>=1.2"
V2_MIN_DISC_MONTHS = 6
V2_MIN_DISC_HITS = 100


def _rules_eval():
    import yaml
    r = yaml.safe_load((T.ROOT / "config" / "rules.yaml").read_text())["evaluation"]
    return int(r["adopt_window_months"]), float(r["adopt_min_total_roi"]), int(r["adopt_min_win_months"])


def v2_windows(strict_months, n_valid):
    """検証用 = 厳密な月のうち新しい n_valid か月、発見用 = それより前の厳密な月すべて。"""
    sm = sorted(strict_months)
    return sm[:-n_valid] if len(sm) > n_valid else [], sm[-n_valid:]


def cmd_v2eval(work: Path, smoke=False):
    races, z = M.load(work)
    month = races.month.to_numpy()
    P2 = z["P"].astype(float) @ AGG
    void2 = z["lane_void"][:, PAIRS].any(2)
    hit2, pay2 = races.hit2.to_numpy(), races.pay2.to_numpy()
    W2 = np.zeros(P2.shape, bool)
    W2[np.where(hit2 >= 0)[0], hit2[hit2 >= 0]] = True
    nwin, roi_line, win_line = _rules_eval()
    fm = set(M.full_months(races))
    rng = np.random.default_rng(0)
    L = ["## 2連単 v2 の評価（事前に決めた候補、厳密 q̂ のウォークフォワード）", ""]
    if smoke:
        L.append("**動作確認（smoke）: 月数が足りないので数字は判断に使わない。**")
        L.append("")
    G2 = {}
    for src in ["agg", "direct"]:
        f = M.mb(work) / f"g2wf_{src}.npz"
        if f.exists():
            G2[src] = np.load(f)["G2"].astype(float)
    rows, monthly = [], {}
    windows = {}
    for src, G in G2.items():
        ok = np.isfinite(G).all(1)
        sm = sorted(m for m in set(month[ok]) if m in fm)
        windows[src] = v2_windows(sm, nwin)
    L.append("厳密な月（q̂ はその月より前のオッズだけ、g2+ はその月より前の当たりだけで学習）:")
    for src, (d, v) in windows.items():
        L.append(f"- q̂2={src}: 発見用 {d[0] if d else '-'}〜{d[-1] if d else '-'}（{len(d)}か月）、"
                 f"検証用 {v[0] if v else '-'}〜{v[-1] if v else '-'}（{len(v)}か月）")
    L.append("")
    for name, src, kind, pmin, tau in V2_CANDIDATES:
        if src not in G2:
            rows.append(dict(cand=name, 状態=f"q̂2={src} が無い（未評価）"))
            continue
        G = G2[src]
        ok = (hit2 >= 0) & np.isfinite(G).all(1)
        EV = P2 * np.nan_to_num(G) / 100
        mk = ev_topk_mask(EV, P2, pmin, 1, tau) if kind == "max1" else (EV >= tau) & (P2 >= pmin)
        mk = mk & ~void2 & ok[:, None]
        cost = mk.sum(1) * 100.0
        ret = (mk & W2).any(1) * pay2
        df = pd.DataFrame(dict(hd=races.hd, month=month, cost=cost, ret=ret, hit=(mk & W2).any(1), bets=mk.sum(1)))
        disc, valid = windows[src]
        r = dict(cand=name)
        for per, ms in [("発見", disc), ("検証", valid)]:
            x = df[df.month.isin(ms)]
            if len(ms) == 0 or x.cost.sum() == 0:
                r[f"{per}:回収率"] = np.nan
                continue
            dd = x.groupby("hd")[["cost", "ret"]].sum()
            cc, rr = dd.cost.to_numpy(), dd.ret.to_numpy()
            bs = [rr[i].sum() / max(cc[i].sum(), 1) for i in (rng.integers(0, len(dd), len(dd)) for _ in range(1000))]
            mm = x.groupby("month")[["cost", "ret"]].sum().reindex(ms).fillna(0)
            mr = np.where(mm.cost > 0, mm.ret / mm.cost.clip(lower=1), np.nan)
            r[f"{per}:回収率"] = x.ret.sum() / x.cost.sum()
            r[f"{per}:90%区間"] = f"{np.quantile(bs, .05):.2f}〜{np.quantile(bs, .95):.2f}"
            r[f"{per}:勝ち月"] = f"{int(np.nansum(mr > 1))}/{len(ms)}"
            r[f"{per}:最悪月"] = np.nanmin(mr) if np.isfinite(mr).any() else np.nan
            r[f"{per}:月の点数"] = x.bets.sum() / len(ms)
            r[f"{per}:的中"] = int(x.hit.sum())
            r[f"_{per}_win"] = int(np.nansum(mr > 1))
        monthly[name] = df.groupby("month").agg(点=("bets", "sum"), 的中=("hit", "sum"), 購入=("cost", "sum"), 払戻=("ret", "sum"))
        rows.append(r)
    res = pd.DataFrame(rows)
    show = [c for c in res.columns if not c.startswith("_")]
    L.append(f"### 候補 {len(V2_CANDIDATES)} 本（plan に事前登録。評価した数: {int(res['発見:回収率'].notna().sum() if '発見:回収率' in res else 0)}）")
    L.append(res[show].to_markdown(index=False, floatfmt=".3f"))
    L.append("")
    # 事前に決めた選び方: 発見用の的中≥V2_MIN_DISC_HITS の中で発見用の回収率が最良の1本 → 検証用で合格ラインを見る
    sel = None
    if "発見:的中" in res:
        el = res[(res["発見:的中"] >= V2_MIN_DISC_HITS) & res["発見:回収率"].notna()]
        if len(el):
            sel = el.sort_values("発見:回収率", ascending=False).iloc[0]
    ndisc = max((len(d) for d, _ in windows.values()), default=0)
    L.append(f"### 判定（合格ライン: 検証用 {nwin} か月で通算 > {roi_line}、勝ち月 ≥ {win_line}。rules.yaml evaluation）")
    if ndisc < V2_MIN_DISC_MONTHS and not smoke:
        L.append(f"発見用の厳密な月が {ndisc} か月（< {V2_MIN_DISC_MONTHS}）。判定しない。")
    for lab, rr in [("発見用で選んだ1本", sel), ("主仮説 " + V2_PRIMARY, res[res.cand == V2_PRIMARY].iloc[0] if (res.cand == V2_PRIMARY).any() else None)]:
        if rr is None or "検証:回収率" not in rr or not np.isfinite(rr.get("検証:回収率", np.nan)):
            L.append(f"- {lab}: なし")
            continue
        nv = len(windows[[s for n, s, *_ in V2_CANDIDATES if n == rr.cand][0]][1])
        passed = nv >= nwin and rr["検証:回収率"] > roi_line and rr["_検証_win"] >= win_line
        L.append(f"- {lab}: {rr.cand} → 検証用 {rr['検証:回収率']:.3f} 勝ち月 {rr['検証:勝ち月']}（{nv}か月）→ "
                 f"{'合格ラインを満たす（次は decisions.md に登録して前向きに確かめる）' if passed else '不合格'}")
    L.append("")
    for name in dict.fromkeys([n for n in [V2_PRIMARY, sel.cand if sel is not None else None] if n in monthly]):
        mt = monthly[name]
        mt["回収率"] = mt.払戻 / mt.購入.clip(lower=1)
        d, v = windows[[s for n, s, *_ in V2_CANDIDATES if n == name][0]]
        mt = mt.loc[[m for m in d + v if m in mt.index]]
        L.append(f"### 月別: {name}")
        L.append(mt[["点", "的中", "回収率"]].to_markdown(floatfmt=".3f"))
        L.append("")
    txt = "\n".join(L)
    (M.mb(work) / ("v2_report_smoke.md" if smoke else "v2_report.md")).write_text(txt)
    print(txt)
