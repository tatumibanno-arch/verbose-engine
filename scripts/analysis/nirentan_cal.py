"""2連単「EV最大1点」の補正あり期待値（calibration.md の (b) 直接補正）。本番の一覧（nirensho_today.py）と
厳密バックテストで同じ関数を使う。設定は config/rules.yaml の strategies.nirentan_ev1cal_v1（唯一の正）。

補正の形: 対象月 m より前の最大 W か月で「選ばれた買い目」（P2≥min_prob の中で生の EV 最大の1点）を集め、
  E[払戻/100] = EV·exp(c·X)、X = [1, log EV, log P, 1[頭≠1号艇], 1[2着艇=2..6]（1号艇が基準）]
のポアソン回帰。補正後 EV_cal = EV·exp(c·X)。選ぶ目は変えず、閾値を EV_cal に付ける。

使い方:
    python scripts/analysis/nirentan_cal.py strict-g2        # 厳密 q̂（qhat.npz の LQs がある月）で g2+ を作り直し mb/g2p_strict.npz に置く
    python scripts/analysis/nirentan_cal.py backtest          # 2024-11〜2026-10 のウォークフォワード（出力は scratchpad/nirentan_cal/）
リーク: 補正の係数は m より前の月の買い目・払戻だけ。q̂ は 2026-01〜10 が厳密（過去のオッズだけ）、それより前は LOMO。
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))
import multi_bet as M  # noqa: E402
import trifecta as T  # noqa: E402

PAIRS, AGG = M.PAIRS, M.AGG
STRATEGY = "nirentan_ev1cal_v1"
OUT = T.DEFAULT_WORK.parent / "nirentan_cal"
BT_FROM, BT_TO = "2024-11", "2026-10"


def rules():
    r = yaml.safe_load((ROOT / "config" / "rules.yaml").read_text())["strategies"][STRATEGY]
    return r


def log(m):
    T.log(m)


def lg(x):
    return np.log(np.clip(x, 1e-9, None))


# ============================================================== 補正
FEATURE_NAMES = ["const", "log_ev", "log_p", "head_not1"] + [f"second_{k}" for k in range(2, 7)]


def design(ev, p, head, second, features):
    """features: rules.yaml の calib_features。head/second は 1〜6 の艇番。"""
    cols = [np.ones_like(ev, dtype=float)]
    names = ["const"]
    if "log_ev" in features:
        cols.append(lg(ev)); names.append("log_ev")
    if "log_p" in features:
        cols.append(lg(p)); names.append("log_p")
    if "head_not1" in features:
        cols.append((np.asarray(head) != 1).astype(float)); names.append("head_not1")
    if "second_lane" in features:
        for k in range(2, 7):
            cols.append((np.asarray(second) == k).astype(float)); names.append(f"second_{k}")
    return np.stack(cols, -1), names


def fit_poisson_offset(X, y, g):
    """E[y] = g·exp(Xβ)（calibrate.py と同じ）。"""
    sc = g.mean()
    y, g = y / sc, g / sc

    def f(b):
        eta = X @ b
        mu = g * np.exp(eta)
        return (mu - y * eta).mean(), X.T @ (mu - y) / len(y)

    return minimize(f, np.zeros(X.shape[1]), jac=True, method="L-BFGS-B").x


def fit_calibration(bets: pd.DataFrame, features):
    """bets: 選ばれた買い目（ev, p, head, second, ret=払戻円）。戻り値 (係数, 名前)。"""
    X, names = design(bets.ev.to_numpy(float), bets.p.to_numpy(float), bets["head"].to_numpy(), bets.second.to_numpy(), features)
    c = fit_poisson_offset(X, bets.ret.to_numpy(float), bets.ev.to_numpy(float) * 100)
    return c, names


def apply_calibration(coef, ev, p, head, second, features):
    X, _ = design(np.asarray(ev, float), np.asarray(p, float), head, second, features)
    return np.asarray(ev, float) * np.exp(X @ coef)


# ============================================================== 材料（mb/）
def strict_g2_path(work):
    return M.mb(work) / "g2p_strict.npz"


def cmd_strict_g2(work: Path):
    """qhat.npz の LQs（過去のオッズだけで学習した q̂）がある月 m: テスト行の q̂2 を厳密版にして g2+ を作り直す
    （学習行は m より前の当たりで q̂2 は LOMO。multi_bet_eval.g2plus_strict と同じ）。"""
    import lightgbm as lgb
    races, z = M.load(work)
    qq = np.load(M.mb(work) / "qhat.npz")
    q2h = M.norm_q(qq["LQ"].astype(float)) @ AGG
    oks = np.isfinite(qq["LQs"]).all(1)
    q2s = np.full(q2h.shape, np.nan)
    q2s[oks] = M.norm_q(qq["LQs"][oks].astype(float)) @ AGG
    month = races.month.to_numpy()
    hit2, pay2 = races.hit2.to_numpy(), races.pay2.to_numpy()
    flags = races[T.RACE_FLAGS].to_numpy(np.float32)
    fm = set(M.full_months(races))
    sm = sorted(m for m in set(month[oks]) if m in fm)
    f = strict_g2_path(work)
    G = np.load(f)["G2"] if f.exists() else np.full((len(races), 30), np.nan, np.float32)
    done = list(np.load(f)["months"]) if f.exists() else []
    for m in sm:
        if m in done:
            continue
        tr = np.where((month < m) & (hit2 >= 0))[0]
        X = M.g2p_X(races, z, tr, q2h, flags).reshape(len(tr), 30, -1)[np.arange(len(tr)), hit2[tr]]
        mdl = lgb.train(M.GP_PARAMS, lgb.Dataset(X, pay2[tr]), T.G_ROUNDS)
        te = np.where(month == m)[0]
        G[te] = mdl.predict(M.g2p_X(races, z, te, q2s, flags), num_threads=M.NT).reshape(len(te), 30)
        done.append(m)
        np.savez(f, G2=G, months=np.array(sorted(done)))
        log(f"strict g2+ {m} (train wins={len(tr)})")
    log(f"strict g2+ months: {sorted(done)}")


def history(work: Path, use_strict=True):
    """全月の「選ばれた買い目」（P2≥min_prob の中で生の EV 最大の1点）。src 列: その月の g2+ が strict か lomo か。"""
    r = rules()
    pmin = float(r["min_prob"])
    races, z = M.load(work)
    gp = np.load(M.mb(work) / "gplus.npz")
    G = gp["G2p"].astype(float)
    src = np.array(["lomo"] * len(races), dtype=object)
    f = strict_g2_path(work)
    if use_strict and f.exists():
        zs = np.load(f)
        ok = np.isfinite(zs["G2"]).all(1)
        G[ok] = zs["G2"][ok]
        src[ok] = "strict"
    P2 = z["P"].astype(float) @ AGG
    void = z["lane_void"][:, PAIRS].any(2)
    hit2, pay2 = races.hit2.to_numpy(), races.pay2.to_numpy()
    ok = (hit2 >= 0) & np.isfinite(G).all(1) & (G > 0).all(1)
    EV = P2 * G / 100
    E = np.where((P2 >= pmin) & ~void, EV, -np.inf)
    j = E.argmax(1)
    sel = ok & np.isfinite(E[np.arange(len(E)), j])
    i = np.where(sel)[0]
    k = j[i]
    return pd.DataFrame(dict(race=i, hd=races.hd.to_numpy()[i], jcd=races.jcd.to_numpy()[i], rno=races.rno.to_numpy()[i],
                             month=races.month.to_numpy()[i], comb=k, head=PAIRS[k, 0] + 1, second=PAIRS[k, 1] + 1,
                             p=P2[i, k], g=G[i, k], ev=EV[i, k], ret=np.where(hit2[i] == k, pay2[i], 0.0), src=src[i]))


def prior_window(months_all, m, r):
    return [x for x in months_all if r["calib_fit_from"] <= x < m][-int(r["calib_window_months"]):]


def fit_for_month(hist: pd.DataFrame, m: str, r=None):
    """対象月 m（'YYYY-MM'）の係数。前の月が足りなければ None。"""
    r = r or rules()
    pm = prior_window(sorted(set(hist.month)), m, r)
    if len(pm) < int(r["calib_min_prior_months"]):
        return None, pm
    b = hist[hist.month.isin(pm)]
    c, _ = fit_calibration(b, r["calib_features"])
    return c, pm


# ============================================================== バックテスト
def month_summary(b, months):
    g = b.groupby("month").agg(bets=("ret", "size"), hits=("ret", lambda x: int((x > 0).sum())), ret=("ret", "sum"),
                               days=("hd", "nunique"))
    g = g.reindex(months).fillna(0)
    roi_m = np.where(g.bets > 0, g.ret / 100 / g.bets.clip(lower=1), np.nan)
    tot = g.ret.sum() / 100 / max(g.bets.sum(), 1)
    return dict(roi=tot, win=int(np.nansum(roi_m > 1)), nmon=int((g.bets > 0).sum()),
                worst=float(np.nanmin(roi_m)) if np.isfinite(roi_m).any() else np.nan,
                bets=int(g.bets.sum()), bets_day=g.bets.sum() / max(g.days.sum(), 1), hits=int(g.hits.sum())), pd.Series(roi_m, index=months)


def bootstrap_ci(ret, n=4000, seed=0):
    if len(ret) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    r = np.asarray(ret, float) / 100
    m = np.array([r[rng.integers(0, len(r), len(r))].mean() for _ in range(n)])
    return tuple(np.percentile(m, [2.5, 97.5]))


def cmd_backtest(work: Path):
    r = rules()
    taus = [float(t) for t in r["report_thresholds"]]
    feats = r["calib_features"]
    OUT.mkdir(parents=True, exist_ok=True)
    out_rows = []
    series = {}
    for label, use_strict in [("strict", True), ("lomo", False)]:
        hist = history(work, use_strict)
        months = [m for m in sorted(set(hist.month)) if BT_FROM <= m <= BT_TO]
        parts, coefs = [], []
        for m in months:
            c, pm = fit_for_month(hist, m, r)
            b = hist[hist.month == m].copy()
            if c is None:
                b["ev_cal"] = np.nan
            else:
                b["ev_cal"] = apply_calibration(c, b.ev, b.p, b["head"], b.second, feats)
                _, names = design(b.ev.to_numpy(), b.p.to_numpy(), b["head"].to_numpy(), b.second.to_numpy(), feats)
                coefs.append(dict(month=m, fit_months=f"{pm[0]}..{pm[-1]}", n_fit=int(hist.month.isin(pm).sum()),
                                  fit_src=",".join(sorted(set(hist[hist.month.isin(pm)].src))), **dict(zip(names, c))))
            parts.append(b)
            log(f"[{label}] {m} src={set(b.src)} n={len(b)} fit={pm[0] if pm else None}..{pm[-1] if pm else None}")
        bets = pd.concat(parts, ignore_index=True)
        bets.to_parquet(OUT / f"bets_{label}.parquet")
        pd.DataFrame(coefs).to_csv(OUT / f"coefs_{label}.csv", index=False)
        series[label] = bets
    # ---- 集計
    md = ["# 2連単 EV最大1点・補正あり期待値のバックテスト（nirentan_cal.py backtest）", ""]
    for label in ["strict", "lomo"]:
        bets = series[label]
        cal_months = sorted(set(bets.month[bets.ev_cal.notna()]))
        st_months = sorted(set(bets.month[bets.src == "strict"]))
        md += [f"## q̂/g2+ = {label}（strict の月: {st_months[0] if st_months else '-'}〜{st_months[-1] if st_months else '-'}、"
               f"それ以外は LOMO）", ""]
        windows = {"補正できる全月 " + f"{cal_months[0]}〜{cal_months[-1]}": cal_months,
                   "直近12か月 2025-11〜2026-10": [m for m in cal_months if m >= "2025-11"]}
        if label == "strict" and st_months:
            windows["厳密の月だけ " + f"{st_months[0]}〜{st_months[-1]}"] = [m for m in cal_months if m in st_months]
        for wname, ms in windows.items():
            rows, mrows = [], {}
            for kind, col in [("補正あり", "ev_cal"), ("補正なし", "ev")]:
                for t in taus:
                    s = bets[bets.month.isin(ms) & (bets[col] >= t)]
                    a, mser = month_summary(s, ms)
                    lo, hi = bootstrap_ci(s.ret)
                    nm = f"{kind} EV≥{t:.1f}"
                    rows.append(dict(候補=nm, 回収率=a["roi"], CI95=f"[{lo:.2f}, {hi:.2f}]", 勝ち月=f"{a['win']}/{len(ms)}",
                                     最悪月=a["worst"], 点数=a["bets"], 点_日=a["bets_day"], 的中=a["hits"],
                                     予想EV=s[col].mean() if len(s) else np.nan, 予想的中=s.p.mean() if len(s) else np.nan,
                                     実的中=(s.ret > 0).mean() if len(s) else np.nan))
                    mrows[nm] = mser
                    out_rows.append(dict(qhat=label, window=wname, kind=kind, tau=t, **rows[-1]))
            md += [f"### {wname}", pd.DataFrame(rows).to_markdown(index=False, floatfmt=".3f"), "",
                   "月別の回収率:", pd.DataFrame(mrows).T.to_markdown(floatfmt=".2f"), ""]
            # H1/H2 の切り口（補正あり、各 τ）
            for t in taus:
                s = bets[bets.month.isin(ms) & (bets.ev_cal >= t)].copy()
                if len(s) == 0:
                    continue
                s["head_not1"] = s["head"] != 1
                sl = []
                for name, mask in [("頭=1号艇", ~s.head_not1), ("頭≠1号艇 ★（H2）", s.head_not1)]:
                    a, _ = month_summary(s[mask], ms)
                    sl.append(dict(切り口=name, **{k: a[k] for k in ["roi", "win", "worst", "bets", "hits"]}, 予想EV=s[mask].ev_cal.mean()))
                for k2 in range(1, 7):
                    mask = s.second == k2
                    a, _ = month_summary(s[mask], ms)
                    sl.append(dict(切り口=f"2着={k2}号艇", **{k: a[k] for k in ["roi", "win", "worst", "bets", "hits"]},
                                   予想EV=s[mask].ev_cal.mean() if mask.any() else np.nan))
                mask = (s["head"] == 1) & (s.second == 2)
                a, _ = month_summary(s[mask], ms)
                sl.append(dict(切り口="1-2（H1）", **{k: a[k] for k in ["roi", "win", "worst", "bets", "hits"]},
                               予想EV=s[mask].ev_cal.mean() if mask.any() else np.nan))
                d = pd.DataFrame(sl).rename(columns=dict(roi="回収率", win="勝ち月", worst="最悪月", bets="点数", hits="的中"))
                md += [f"切り口別（補正あり EV≥{t:.1f}、{wname}。勝ち月は /{len(ms)}。全部載せる）:",
                       d.to_markdown(index=False, floatfmt=".3f"), ""]
                for _, row in d.iterrows():
                    out_rows.append(dict(qhat=label, window=wname, kind="補正あり/切り口", tau=t, 候補=row["切り口"],
                                         回収率=row["回収率"], 勝ち月=row["勝ち月"], 最悪月=row["最悪月"], 点数=row["点数"], 的中=row["的中"]))
        # 係数
        co = pd.read_csv(OUT / f"coefs_{label}.csv")
        md += ["### 補正の係数（月ごと。学習は fit_months の選ばれた買い目）", co.to_markdown(index=False, floatfmt=".3f"), ""]
    pd.DataFrame(out_rows).to_csv(OUT / "summary.csv", index=False)
    (OUT / "report.md").write_text("\n".join(md))
    print("\n".join(md))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["strict-g2", "backtest"])
    ap.add_argument("--workdir", default=str(T.DEFAULT_WORK))
    a = ap.parse_args()
    work = Path(a.workdir)
    T.WORK = work
    if a.cmd == "strict-g2":
        cmd_strict_g2(work)
    else:
        cmd_backtest(work)


if __name__ == "__main__":
    main()
