"""市場（3連単の締切オッズ）から出発して、前夜の素性・当日の展示・直前情報で少しだけ動かす3連単モデル。

研究用。本番の選定は変えない。DB は読み取り専用。説明と結果は notes/analysis/market_plus.md。

  python scripts/analysis/market_plus.py prep      # 材料をまとめる（mp/data.npz）
  python scripts/analysis/market_plus.py fit       # 構成ごと・月ごとに学習（拡大窓）→ 対数損失（mp/ll_*.csv, mp/padj_*.npy）
  python scripts/analysis/market_plus.py llreport  # 当てはまりの表（mp/ll_report.md）
  python scripts/analysis/market_plus.py bets      # 事前に決めた買い方 R1〜R9・対照 C1〜C3
  python scripts/analysis/market_plus.py phase2    # 直前情報（beforeinfo）を足した構成 NEB

モデル（条件付きロジット、1レース120通り）:
  score(a-b-c) = θ·[log q, (log q)², log P] + Σ_p β_p·x(p着の艇) + β1'·x(1着の艇)·[1着が1号艇]
"""
from __future__ import annotations

import glob
import json
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import trifecta as T  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
WORK = T.DEFAULT_WORK
MB = WORK / "mb"
MP = WORK / "mp"
PERMS = T.PERMS
M_LO, M_HI = "2025-11", "2026-10"
EVAL_MONTHS = [f"2026-{m:02d}" for m in range(1, 11)]
DISC = EVAL_MONTHS[:5]
VALID = EVAL_MONTHS[5:]
LAM = 1.0

N_FEATS = ["national_win_rate__dmean", "local_win_rate__dmean", "rc_win_shrunk__dmean", "grade__dmean",
           "rs_c_avg_st__dmean", "r90_avg_st__dmean", "sr_ex_res__dmean", "mm_ex_rank_shrunk__dmean", "ot_st_vs_inner"]
E_FEATS = ["ex_dmean", "ex_rank1", "ex_rank6", "ex_vs_meet"]
B_FEATS = ["bi_inward", "bi_st_dmean", "bi_st_F", "bi_tilt", "bi_parts", "bi_wdiff", "bi_adjw"]
CONFIGS = {
    "M0": dict(combo=["lq", "lq2"], boat=[]),
    "M0P": dict(combo=["lq", "lq2", "lp"], boat=[]),
    "N": dict(combo=["lq", "lq2", "lp"], boat=N_FEATS),
    "E": dict(combo=["lq", "lq2"], boat=E_FEATS),
    "NE": dict(combo=["lq", "lq2", "lp"], boat=N_FEATS + E_FEATS),
}
SAVE_P = {"M0", "N", "NE"}


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def connect():
    return sqlite3.connect(f"file:{ROOT / 'data' / 'boatrace.db'}?mode=ro", uri=True, timeout=600)


# ============================================================== 材料
def cmd_prep():
    MP.mkdir(exist_ok=True)
    races = pd.read_parquet(MB / "races.parquet")
    z = np.load(MB / "arrays.npz")
    sel = ((races.month >= M_LO) & (races.month <= M_HI)).to_numpy()
    O = z["O"][sel]
    void = z["void"][sel]
    P = z["P"][sel]
    races = races[sel].reset_index(drop=True)
    hit = races.hit.to_numpy()
    ok = (np.isfinite(O) | void).all(1) & (hit >= 0)
    ok[ok] &= ~void[np.where(ok)[0], hit[ok]]
    races, O, void, P = races[ok].reset_index(drop=True), O[ok], void[ok], P[ok]
    log(f"races {len(races)}")
    key = races[["hd", "jcd", "rno"]].copy()
    key["ri"] = np.arange(len(key))
    hd_lo, hd_hi = races.hd.min(), races.hd.max()

    def per_boat(df, cols):
        m = df.merge(key, on=["hd", "jcd", "rno"])
        out = np.full((len(key), 6, len(cols)), np.nan, np.float32)
        out[m.ri.to_numpy(), m.lane.to_numpy().astype(int) - 1] = m[cols].to_numpy(np.float32)
        return out

    # 前夜の素性
    fcols = [c for c in N_FEATS if not c.startswith(("sr_", "mm_", "ot_"))]
    xcols = [c for c in N_FEATS if c.startswith(("sr_", "mm_", "ot_"))] + ["sr_ex_res"]
    f = pd.concat([pd.read_parquet(p, columns=["hd", "jcd", "rno", "lane"] + fcols)
                   for p in sorted((WORK / "feat").glob("feat_2025110*_*.parquet")) + sorted((WORK / "feat").glob("feat_20260501_*.parquet"))])
    fx = pd.concat([pd.read_parquet(p, columns=["hd", "jcd", "rno", "lane"] + xcols)
                    for p in [WORK.parent / "fx" / "extra_20251101_20260430.parquet", WORK.parent / "fx" / "extra_20260501_20261031.parquet"]])
    for d in (f, fx):
        d["jcd"] = d.jcd.astype(str).str.zfill(2)
        d["rno"] = d.rno.astype(int)
        d["hd"] = d.hd.astype(str)
    XN = np.concatenate([per_boat(f, fcols), per_boat(fx, xcols)], 2)
    ncols = fcols + xcols
    log(f"night features {XN.shape} nan={np.isnan(XN).mean():.3f}")
    # 当日の展示タイム（results.data。直前情報の展示タイムと同じ値）
    con = connect()
    r = pd.read_sql("select hd, jcd, rno, lane, json_extract(data, '$.exhibition_time') ex from results "
                    "where hd between ? and ?", con, params=(hd_lo, hd_hi))
    r["rno"] = r.rno.astype(int)
    EX = per_boat(r, ["ex"])[:, :, 0]
    EX[(EX < 6.0) | (EX > 7.5)] = np.nan
    # 2連単の締切オッズ（DB ＋ data/web の Parquet）
    o2 = pd.read_sql("select hd, jcd, rno, combination comb, odds from odds where ticket_type='nirensho' and hd between ? and ?",
                     con, params=(hd_lo, hd_hi))
    webs = [pd.read_parquet(p) for p in sorted(glob.glob(str(ROOT / "data" / "web" / "odds2tf_*.parquet")))]
    if webs:
        w = pd.concat(webs)
        w = w[w.ticket_type == "nirensho"][["hd", "jcd", "rno", "combination", "odds"]].rename(columns={"combination": "comb"})
        o2 = pd.concat([o2, w])
    o2["rno"] = o2.rno.astype(int)
    o2 = o2.drop_duplicates(["hd", "jcd", "rno", "comb"]).merge(key, on=["hd", "jcd", "rno"])
    PI2 = {p: i for i, p in enumerate([(a, b) for a in range(6) for b in range(6) if a != b])}
    O2 = np.full((len(key), 30), np.nan, np.float32)
    ab = o2.comb.str.split("-", expand=True).astype(int).to_numpy() - 1
    O2[o2.ri.to_numpy(), [PI2[tuple(x)] for x in ab.tolist()]] = o2.odds.to_numpy(np.float32)
    log(f"2t odds races with all 30: {np.isfinite(O2).all(1).sum()}")
    np.savez(MP / "data.npz", O=O, void=void, P=P, XN=XN, EX=EX, O2=O2, ncols=np.array(ncols))
    races.to_parquet(MP / "races.parquet")


def load_beforeinfo(races):
    """data/web/beforeinfo_*.parquet（と DB の beforeinfo）を艇ごとの配列に。無い艇は NaN。"""
    parts = [pd.read_parquet(p) for p in sorted(glob.glob(str(ROOT / "data" / "web" / "beforeinfo_*.parquet")))]
    con = connect()
    db = pd.read_sql("select hd, jcd, rno, lane, exhibition_time, tilt, start_course, start_st, data from beforeinfo "
                     "where hd between ? and ?", con, params=(races.hd.min(), races.hd.max()))
    b = pd.concat(parts + [db])
    b["rno"] = b.rno.astype(int)
    b["lane"] = b.lane.astype(int)
    b = b.drop_duplicates(["hd", "jcd", "rno", "lane"])
    d = pd.json_normalize(b.data.map(lambda s: json.loads(s) if isinstance(s, str) else s).tolist())
    b = b.reset_index(drop=True)
    b["weight_bi"] = pd.to_numeric(d["weight"], errors="coerce")
    b["adjw"] = pd.to_numeric(d["adjust_weight"], errors="coerce")
    b["parts"] = d["parts_changed"].map(lambda x: float(len(x) > 0) if isinstance(x, list) else np.nan)
    b["stF"] = d["start_st_raw"].astype(str).str.startswith("F").astype(float)
    key = races[["hd", "jcd", "rno"]].copy()
    key["ri"] = np.arange(len(key))
    m = b.merge(key, on=["hd", "jcd", "rno"])
    cols = ["start_course", "start_st", "stF", "tilt", "parts", "weight_bi", "adjw"]
    out = np.full((len(key), 6, len(cols)), np.nan, np.float32)
    out[m.ri.to_numpy(), m.lane.to_numpy() - 1] = m[cols].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    # 番組表の体重
    ent = pd.read_sql("select hd, jcd, rno, lane, json_extract(data, '$.weight') w from entries where hd between ? and ?",
                      con, params=(races.hd.min(), races.hd.max()))
    ent["rno"] = ent.rno.astype(int)
    ent["lane"] = ent.lane.astype(int)
    m2 = ent.merge(key, on=["hd", "jcd", "rno"])
    W0 = np.full((len(key), 6), np.nan, np.float32)
    W0[m2.ri.to_numpy(), m2.lane.to_numpy() - 1] = pd.to_numeric(m2.w, errors="coerce").to_numpy(np.float32)
    return out, W0


# ============================================================== 素性の組み立て
def build(data, races, BI=None, W0=None):
    """combo 素性 C (N,120,3) と 艇素性 dict(name -> (N,6))。"""
    O, void, P = data["O"], data["void"], data["P"]
    inv = np.where(void, 0, 1.0 / np.where(np.isfinite(O) & (O > 0), O, np.inf))
    q = inv / inv.sum(1, keepdims=True)
    lq = np.log(np.clip(q, 1e-7, None))
    Pn = np.where(void, 0, P)
    Pn = Pn / Pn.sum(1, keepdims=True)
    lp = np.log(np.clip(Pn, 1e-7, None))
    combo = {"lq": lq, "lq2": lq ** 2 / 10, "lp": lp}
    boat = {}
    ncols = list(data["ncols"])
    for c in N_FEATS:
        boat[c] = data["XN"][:, :, ncols.index(c)]
    EX = data["EX"]
    mu = np.nanmean(EX, 1, keepdims=True)
    exd = np.clip(EX - mu, -0.3, 0.3)
    rk = np.argsort(np.argsort(np.where(np.isnan(EX), 99, EX), 1), 1)    # 0=最速
    nvalid = np.isfinite(EX).sum(1, keepdims=True)
    boat["ex_dmean"] = exd
    boat["ex_rank1"] = np.where(np.isfinite(EX), (rk == 0).astype(np.float32), np.nan)
    boat["ex_rank6"] = np.where(np.isfinite(EX), (rk == nvalid - 1).astype(np.float32), np.nan)
    srx = data["XN"][:, :, ncols.index("sr_ex_res")]
    boat["ex_vs_meet"] = exd - np.where(np.isnan(srx), 0, np.clip(srx, -0.3, 0.3))
    boat["ex_vs_meet"][np.isnan(srx)] = np.nan
    if BI is not None:
        sc, st, stF, tilt, parts, wbi, adjw = [BI[:, :, i] for i in range(7)]
        lane = np.arange(1, 7)[None, :]
        boat["bi_inward"] = np.clip(lane - sc, -3, 3)
        sts = np.where(stF == 1, -st, st)
        boat["bi_st_dmean"] = np.clip(sts - np.nanmean(sts, 1, keepdims=True), -0.3, 0.3)
        boat["bi_st_F"] = stF
        boat["bi_tilt"] = np.clip(tilt, -0.5, 3)
        boat["bi_parts"] = parts
        boat["bi_wdiff"] = np.clip(wbi - W0, -5, 5)
        boat["bi_adjw"] = np.clip(adjw, 0, 5)
    return combo, boat


ONEHOT = [np.eye(6)[PERMS[:, p]] for p in range(3)]       # 3 x (120, 6)
IS1 = (PERMS[:, 0] == 0)


class CLogit:
    def __init__(self, combo_names, boat_names, lam=LAM):
        self.cn, self.bn, self.lam = combo_names, boat_names, lam

    def _design(self, combo, boat, idx, fit=False):
        C = np.stack([combo[c][idx] for c in self.cn], 2).astype(np.float64)
        if not self.bn:
            return C, None
        X = np.stack([boat[b][idx] for b in self.bn], 2).astype(np.float64)
        if fit:
            self.mu = np.nanmean(X.reshape(-1, X.shape[2]), 0)
            self.sd = np.nanstd(X.reshape(-1, X.shape[2]), 0) + 1e-9
        X = (X - self.mu) / self.sd
        X = np.nan_to_num(X, nan=0.0)
        return C, X

    def _scores(self, w, C, X):
        kc = C.shape[2]
        s = C @ w[:kc]
        if X is not None:
            F = X.shape[2]
            B = w[kc:].reshape(4, F)
            for p in range(3):
                bs = X @ B[p]                       # (N,6)
                s += bs[:, PERMS[:, p]]
            s += np.where(IS1[None, :], (X[:, 0, :] @ B[3])[:, None], 0.0)
        return s

    def _nll(self, w, C, X, mask, y):
        s = self._scores(w, C, X)
        s = np.where(mask, s, -1e30)
        mx = s.max(1, keepdims=True)
        e = np.exp(s - mx) * mask
        Z = e.sum(1, keepdims=True)
        p = e / Z
        n = len(y)
        nll = -(s[np.arange(n), y] - mx[:, 0] - np.log(Z[:, 0])).sum()
        R = -p
        R[np.arange(n), y] += 1.0                  # y - p
        kc = C.shape[2]
        g = np.empty_like(w)
        g[:kc] = -np.einsum("nc,nck->k", R, C)
        reg = 0.0
        if X is not None:
            F = X.shape[2]
            B = w[kc:].reshape(4, F)
            gB = np.empty((4, F))
            for p in range(3):
                Wp = R @ ONEHOT[p]                 # (N,6)
                gB[p] = -np.einsum("nb,nbf->f", Wp, X)
            gB[3] = -(R[:, IS1].sum(1)[:, None] * X[:, 0, :]).sum(0)
            reg = 0.5 * self.lam * (B ** 2).sum()
            gB += self.lam * B
            g[kc:] = gB.ravel()
        return (nll + reg) / n, g / n

    def fit(self, combo, boat, idx, mask, y):
        C, X = self._design(combo, boat, idx, fit=True)
        k = C.shape[2] + (4 * X.shape[2] if X is not None else 0)
        w0 = np.zeros(k)
        w0[0] = 1.0
        r = minimize(self._nll, w0, args=(C, X, mask, y), jac=True, method="L-BFGS-B", options={"maxiter": 500})
        self.w = r.x
        return self

    def predict(self, combo, boat, idx, mask):
        C, X = self._design(combo, boat, idx)
        s = np.where(mask, self._scores(self.w, C, X), -1e30)
        s -= s.max(1, keepdims=True)
        e = np.exp(s) * mask
        return e / e.sum(1, keepdims=True)

    def coef_table(self):
        kc = len(self.cn)
        rows = [{"素性": c, "係数": self.w[i]} for i, c in enumerate(self.cn)]
        if self.bn:
            B = self.w[kc:].reshape(4, len(self.bn))
            for j, b in enumerate(self.bn):
                rows.append({"素性": b, "1着": B[0, j], "2着": B[1, j], "3着": B[2, j], "1着×1号艇": B[3, j]})
        return pd.DataFrame(rows)


# ============================================================== 評価
P2_IDX = np.array([[a, b] for a in range(6) for b in range(6) if a != b])
AGG2 = np.zeros((120, 30))
for c, (a, b, _) in enumerate(PERMS):
    AGG2[c, np.where((P2_IDX[:, 0] == a) & (P2_IDX[:, 1] == b))[0][0]] = 1
AGG1 = ONEHOT[0]


def lls(p, y):
    n = len(y)
    h2 = (AGG2[y] == 1).argmax(1)
    h1 = PERMS[y, 0]
    p2 = p @ AGG2
    p1 = p @ AGG1
    eps = 1e-12
    return (-np.log(p[np.arange(n), y] + eps), -np.log(p2[np.arange(n), h2] + eps), -np.log(p1[np.arange(n), h1] + eps))


def load_all(with_bi=False):
    data = dict(np.load(MP / "data.npz"))
    races = pd.read_parquet(MP / "races.parquet")
    BI = W0 = None
    if with_bi:
        BI, W0 = load_beforeinfo(races)
    combo, boat = build(data, races, BI, W0)
    if with_bi:
        pne = np.load(MP / "padj_NE.npy").astype(np.float64)
        combo["lne"] = np.log(np.clip(np.nan_to_num(pne, nan=1e-7), 1e-7, None))
    mask = ~data["void"]
    y = races.hit.to_numpy()
    return data, races, combo, boat, mask, y


def cmd_fit(only=None):
    data, races, combo, boat, mask, y = load_all()
    months = races.month.to_numpy()
    rows = []
    for name, cfg in CONFIGS.items():
        if only and name not in only:
            continue
        out = MP / f"ll_{name}.csv"
        if out.exists():
            log(f"skip {name}")
            continue
        Padj = np.full(mask.shape, np.nan, np.float32)
        rows = []
        coefs = []
        for m in ["2025-12"] + EVAL_MONTHS:
            tr = np.where(months < m)[0]
            te = np.where(months == m)[0]
            mdl = CLogit(cfg["combo"], cfg["boat"]).fit(combo, boat, tr, mask[tr], y[tr])
            p = mdl.predict(combo, boat, te, mask[te])
            Padj[te] = p
            l3, l2, l1 = lls(p, y[te])
            rows.append({"config": name, "month": m, "n": len(te), "ll3": l3.mean(), "ll2": l2.mean(), "ll1": l1.mean()})
            ct = mdl.coef_table()
            ct["month"] = m
            coefs.append(ct)
            log(f"{name} {m} train={len(tr)} ll3={l3.mean():.4f} ll2={l2.mean():.4f} ll1={l1.mean():.4f}")
        pd.DataFrame(rows).to_csv(out, index=False)
        pd.concat(coefs).to_csv(MP / f"coef_{name}.csv", index=False)
        if name in SAVE_P:
            np.save(MP / f"padj_{name}.npy", Padj)
    # 市場そのもの（素の q）
    out = MP / "ll_Q.csv"
    if not out.exists():
        q = np.exp(combo["lq"]) * mask
        q = q / q.sum(1, keepdims=True)
        rows = []
        for m in ["2025-12"] + EVAL_MONTHS:
            te = np.where(months == m)[0]
            l3, l2, l1 = lls(q[te], y[te])
            rows.append({"config": "Q", "month": m, "n": len(te), "ll3": l3.mean(), "ll2": l2.mean(), "ll1": l1.mean()})
        pd.DataFrame(rows).to_csv(out, index=False)
        # 前夜モデル P 単独
        p = np.where(mask, data["P"], 0)
        p = p / p.sum(1, keepdims=True)
        rows = []
        for m in ["2025-12"] + EVAL_MONTHS:
            te = np.where(months == m)[0]
            l3, l2, l1 = lls(p[te], y[te])
            rows.append({"config": "P", "month": m, "n": len(te), "ll3": l3.mean(), "ll2": l2.mean(), "ll1": l1.mean()})
        pd.DataFrame(rows).to_csv(MP / "ll_P.csv", index=False)


def boot_se(d, n=300, seed=0):
    rng = np.random.default_rng(seed)
    return np.std([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)])


def cmd_llreport():
    data, races, combo, boat, mask, y = load_all()
    months = races.month.to_numpy()
    ev = np.isin(months, EVAL_MONTHS)
    q = np.exp(combo["lq"]) * mask
    q /= q.sum(1, keepdims=True)
    base = lls(q[ev], y[ev])
    lines = []
    tab = []
    for name in ["Q", "P"] + list(CONFIGS):
        f = MP / f"ll_{name}.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f)
        d = d[d.month.isin(EVAL_MONTHS)]
        tot = {k: np.average(d[k], weights=d.n) for k in ["ll3", "ll2", "ll1"]}
        tab.append({"構成": name, **tot})
    T_ = pd.DataFrame(tab)
    lines.append(T_.to_markdown(index=False, floatfmt=".4f"))
    # 対にした差（M0 と NE 等）
    mm = {}
    for name in ["M0", "M0P", "N", "E", "NE"]:
        f = MP / f"padj_{name}.npy"
        if f.exists():
            mm[name] = np.load(f)
    rows = []
    ref = None
    for name, Pa in mm.items():
        l = lls(Pa[ev].astype(np.float64), y[ev])
        if name == "M0":
            ref = l
        rows.append((name, l))
    out = []
    for name, l in rows:
        r = {"構成": name}
        for k, (a, b) in enumerate(zip(l, base)):
            dd = a - b
            r[["Δ3連単", "Δ2連単", "Δ1着"][k] + "×1000(対 素の市場)"] = dd.mean() * 1000
            r[["se3", "se2", "se1"][k]] = boot_se(dd) * 1000
        if ref is not None:
            dd = l[0] - ref[0]
            r["Δ3連単×1000(対 M0)"] = dd.mean() * 1000
            mon = pd.Series(dd).groupby(months[ev]).mean()
            r["M0より良い月"] = f"{(mon < 0).sum()}/{len(mon)}"
        out.append(r)
    lines.append("")
    lines.append(pd.DataFrame(out).to_markdown(index=False, floatfmt=".2f"))
    txt = "\n".join(lines)
    (MP / "ll_report.md").write_text(txt)
    print(txt)


# ============================================================== 買い方（事前に決めた R1〜R9・対照 C1〜C3）
def _q1(combo, mask):
    q = np.exp(combo["lq"]) * mask
    q /= q.sum(1, keepdims=True)
    return q[:, IS1].sum(1)


def rule_top1(p, O, mask, rows, thr, pmin=0.0, sub=None):
    """各レース EV 最大の1点。返り値: list of (race, combos, stakes)."""
    ok = mask & (p >= pmin)
    if sub is not None:
        ok &= sub[None, :]
    ev = np.where(ok, p * O, -1)
    out = []
    for i in rows:
        c = int(ev[i].argmax())
        if ev[i, c] >= thr:
            out.append((i, [c], [100.0]))
    return out


def rule_all(p, O, mask, rows, thr, pmin):
    ev = np.where(mask & (p >= pmin), p * O, -1)
    out = []
    for i in rows:
        cs = np.where(ev[i] >= thr)[0]
        if len(cs):
            out.append((i, cs.tolist(), [100.0] * len(cs)))
    return out


def rule_set(p, O, mask, rows, thr=1.2, synth_min=None, kmax=6):
    """1-x-y を EV の高い順に足す。合成オッズで配分。"""
    cand = np.where(IS1)[0]
    out = []
    for i in rows:
        cs = cand[mask[i, cand]]
        if not len(cs):
            continue
        ev = p[i, cs] * O[i, cs]
        cs = cs[np.argsort(-ev)][:kmax]
        inv = np.cumsum(1.0 / O[i, cs])
        S = 1.0 / inv
        sev = np.cumsum(p[i, cs]) * S
        k = None
        if synth_min is None:
            good = np.where(sev >= thr)[0]
            k = good.max() + 1 if len(good) else None
        else:
            okk = np.where(S >= synth_min)[0]
            if len(okk):
                kk = okk.max() + 1
                if sev[kk - 1] >= thr:
                    k = kk
        if k:
            sel = cs[:k]
            w = (1.0 / O[i, sel]) / (1.0 / O[i, sel]).sum()
            out.append((i, sel.tolist(), (100.0 * k * w).tolist()))
    return out


def settle(bets, races, kind="3t"):
    hit = races.hit.to_numpy() if kind == "3t" else races.hit2.to_numpy()
    pay = races.payout.to_numpy() if kind == "3t" else races.pay2.to_numpy()
    rec = []
    for i, cs, st in bets:
        ret = 0.0
        h = 0
        for c, s in zip(cs, st):
            if c == hit[i]:
                ret += s * pay[i] / 100.0
                h = 1
        rec.append((i, sum(st), ret, h, len(cs)))
    d = pd.DataFrame(rec, columns=["ri", "stake", "ret", "hit", "pts"])
    d["month"] = races.month.to_numpy()[d.ri]
    d["hd"] = races.hd.to_numpy()[d.ri]
    return d


def summarize(d, months, ndays):
    g = d.groupby("month").agg(stake=("stake", "sum"), ret=("ret", "sum"), hits=("hit", "sum"), pts=("pts", "sum"),
                               races=("ri", "size")).reindex(months).fillna(0)
    g["roi"] = g.ret / g.stake.replace(0, np.nan)
    st, rt = g.stake.sum(), g.ret.sum()
    return dict(roi=rt / st if st else np.nan, win=int((g.roi > 1).sum()), worst=g.roi.min(),
                pts_day=g.pts.sum() / ndays, races=int(g.races.sum()), hits=int(g.hits.sum()), stake=st), g


def cmd_bets():
    data, races, combo, boat, mask, y = load_all()
    O = np.where(np.isfinite(data["O"]), data["O"], 0).astype(np.float64)
    months = races.month.to_numpy()
    q1 = _q1(combo, mask)
    a1 = races.race_n_a1.to_numpy() >= 5
    P = {k: np.load(MP / f"padj_{k}.npy").astype(np.float64) for k in ["NE", "N", "M0"]}
    ev_rows = np.where(np.isin(months, EVAL_MONTHS))[0]
    nd = races[np.isin(months, EVAL_MONTHS)].groupby("month").hd.nunique()
    fav = ev_rows[q1[ev_rows] >= 0.5]
    sg = fav[a1[fav]]
    # 2連単
    O2 = data["O2"].astype(np.float64)
    m2 = (mask.astype(float) @ AGG2) > 0
    has2 = np.isfinite(O2).all(1)
    O2z = np.where(np.isfinite(O2), O2, 0)
    rules = {
        "R1": lambda: (rule_top1(P["NE"], O, mask, ev_rows, 1.2, 0.01), "3t"),
        "R2": lambda: (rule_top1(P["NE"], O, mask, ev_rows, 1.5, 0.01), "3t"),
        "R3": lambda: (rule_all(P["NE"], O, mask, ev_rows, 1.3, 0.02), "3t"),
        "R4": lambda: (rule_set(P["NE"], O, mask, sg, 1.2), "3t"),
        "R5": lambda: (rule_set(P["NE"], O, mask, sg, 1.2, synth_min=7), "3t"),
        "R6": lambda: (rule_set(P["NE"], O, mask, fav, 1.2), "3t"),
        "R7": lambda: (rule_set(P["NE"], O, mask, fav, 1.2, synth_min=7), "3t"),
        "R8": lambda: (rule_top1(P["NE"], O, mask, fav, 1.2, 0.0, sub=IS1), "3t"),
        "R9": lambda: (rule_top1(P["NE"] @ AGG2, O2z, m2, ev_rows[has2[ev_rows]], 1.2, 0.0), "2t"),
        "C1": lambda: (rule_top1(P["M0"], O, mask, ev_rows, 1.2, 0.01), "3t"),
        "C2": lambda: (rule_top1(P["N"], O, mask, ev_rows, 1.2, 0.01), "3t"),
        "C3": lambda: (rule_set(P["M0"], O, mask, fav, 1.2), "3t"),
    }
    res, mon = [], []
    for name, f in rules.items():
        bets, kind = f()
        d = settle(bets, races, kind)
        d.to_parquet(MP / f"bets_{name}.parquet")
        for per, ms in [("発見", DISC), ("検証", VALID), ("10か月", EVAL_MONTHS)]:
            s, g = summarize(d[d.month.isin(ms)], ms, nd[ms].sum())
            res.append({"候補": name, "期間": per, **s})
            if per == "10か月":
                g["rule"] = name
                mon.append(g.reset_index())
        log(f"{name}: bets={len(d)}")
    R = pd.DataFrame(res)
    R.to_csv(MP / "bets_summary.csv", index=False)
    M = pd.concat(mon)
    M.to_csv(MP / "bets_month.csv", index=False)
    piv = R.pivot(index="候補", columns="期間", values=["roi", "win", "worst", "pts_day", "hits"])
    print(piv.round(3).to_string())
    print(M.pivot(index="month", columns="rule", values="roi").round(2).to_string())


# ============================================================== 第2段: 直前情報（beforeinfo）
def cmd_phase2():
    data, races, combo, boat, mask, y = load_all(with_bi=True)
    months = races.month.to_numpy()
    have = np.isfinite(boat["bi_inward"]).sum(1) >= 5
    bim = sorted(pd.Series(months[have]).value_counts().loc[lambda s: s >= 500].index)
    log(f"beforeinfo months {bim}: races {[int((have & (months == m)).sum()) for m in bim]}")
    NEB = N_FEATS + E_FEATS + B_FEATS
    rows, coefs = [], []
    q = np.exp(combo["lq"]) * mask
    q /= q.sum(1, keepdims=True)
    for mode in ["strict", "lomo"]:
        for m in bim:
            te = np.where(have & (months == m))[0]
            if mode == "strict":
                tr = np.where(have & (months < m))[0]
            else:
                tr = np.where(have & (months != m))[0]
            if len(tr) < 2000:
                continue
            res = {}
            for nm, bn in [("NE", N_FEATS + E_FEATS), ("NEB", NEB)]:
                mdl = CLogit(["lq", "lq2", "lp"], bn).fit(combo, boat, tr, mask[tr], y[tr])
                res[nm] = lls(mdl.predict(combo, boat, te, mask[te]), y[te])
                if nm == "NEB":
                    ct = mdl.coef_table()
                    ct["mode"], ct["month"] = mode, m
                    coefs.append(ct)
            # 素性の少ない版: NE（対象月より前の全月で学習）の log 確率を出発点にして、直前情報だけ足す
            for nm, bn in [("NEall+0", []), ("NEall+B", B_FEATS)]:
                mdl = CLogit(["lne"], bn).fit(combo, boat, tr, mask[tr], y[tr])
                res[nm] = lls(mdl.predict(combo, boat, te, mask[te]), y[te])
                if nm == "NEall+B":
                    ct = mdl.coef_table()
                    ct["mode"], ct["month"] = mode + "_small", m
                    coefs.append(ct)
            # 大きい学習（NE: 対象月より前の全月）との比較用
            big = np.load(MP / "padj_NE.npy")[te].astype(np.float64)
            res["NE_all"] = lls(big, y[te])
            res["Q"] = lls(q[te], y[te])
            r = {"mode": mode, "month": m, "train": len(tr), "n": len(te)}
            for k, v in res.items():
                r[f"ll3_{k}"] = v[0].mean()
                r[f"ll2_{k}"] = v[1].mean()
            d = res["NEB"][0] - res["NE"][0]
            r["d3_NEB_NE"] = d.mean() * 1000
            r["se"] = boot_se(d) * 1000
            r["d2_NEB_NE"] = (res["NEB"][1] - res["NE"][1]).mean() * 1000
            d = res["NEall+B"][0] - res["NEall+0"][0]
            r["d3_small"] = d.mean() * 1000
            r["se_small"] = boot_se(d) * 1000
            rows.append(r)
            log(str({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}))
    pd.DataFrame(rows).to_csv(MP / "phase2.csv", index=False)
    if coefs:
        pd.concat(coefs).to_csv(MP / "coef_NEB.csv", index=False)
        cc = pd.concat(coefs)
        print(cc[cc["mode"].str.endswith("_small")].groupby("素性")[["1着", "2着", "3着", "1着×1号艇"]].mean().loc[B_FEATS].round(3).to_string())
        print(cc[~cc["mode"].str.endswith("_small")].groupby("素性")[["1着", "2着", "3着", "1着×1号艇"]].mean().loc[B_FEATS].round(3).to_string())


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "prep":
        cmd_prep()
    elif cmd == "fit":
        cmd_fit(sys.argv[2].split(",") if len(sys.argv) > 2 else None)
    elif cmd == "llreport":
        cmd_llreport()
    elif cmd == "bets":
        cmd_bets()
    elif cmd == "phase2":
        cmd_phase2()
