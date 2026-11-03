"""追加特徴量（scripts/features_extra.py）の効き目を、1着モデル（trifecta.py の StageModel 第1段と同じ作り）で測る。

使い方（作業ファイルは --workdir。リポジトリには置かない）:
    python scripts/analysis/features_extra_eval.py run --configs base,S,ALL --months 2025-11,2025-12
    python scripts/analysis/features_extra_eval.py report

  * 基本の特徴量は trifecta.py の特徴量キャッシュ（<trifecta workdir>/feat、stats_lag_days=45）。
  * 追加特徴量は <workdir>/extra_*.parquet（features_extra.py で作る）を (hd,jcd,rno,lane) で横付け。
  * 予測月 m ごとに、直前24か月（最後の2か月は早期打ち切りと τ 用）で学習し、m を予測（ウォークフォワード）。
  * 指標: 1着が1艇に決まったレースでの 6艇 softmax の対数損失（レース単位）と、最大確率の艇が1着の割合。
  * 構成ごとの差はレース単位で対にして比べる（同じレース・同じ月）。乱数の種だけ変えた base も走らせて雑音の大きさを見る。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))
import trifecta as TF  # noqa: E402

SCRATCH = Path("/tmp/claude-0/-home-user-verbose-engine/968008a0-6ecf-5641-a3ff-006912b41722/scratchpad")
WORK = SCRATCH / "fx"
FEAT_DIR = SCRATCH / "trifecta" / "feat"
BASE_CHUNKS = [("20231101", "20240430"), ("20240501", "20241031"), ("20241101", "20250430"),
               ("20250501", "20251031"), ("20251101", "20260430"), ("20260501", "20261031")]
TEST_MONTHS = TF.TEST_MONTHS
GROUPS = {"S": "sr_", "M": "mm_", "C": "ce_", "V": "vc_", "O": "ot_"}
CONFIGS = {
    "base": [], "base_seed2": [],
    "S": ["S"], "M": ["M"], "C": ["C"], "V": ["V"], "O": ["O"],
    "ALL": ["S", "M", "C", "V", "O"], "ALL_seed2": ["S", "M", "C", "V", "O"],
    "SC": ["S", "C"], "SC_seed2": ["S", "C"],
}
PARAMS = dict(TF.LGB_PARAMS, num_threads=3)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", file=sys.stderr, flush=True)


def load_all() -> pd.DataFrame:
    parts = []
    for f, t in BASE_CHUNKS:
        b = pd.read_parquet(FEAT_DIR / f"feat_{f}_{t}.parquet")
        e = pd.read_parquet(WORK / f"extra_{f}_{t}.parquet")
        b = b.merge(e, on=["hd", "jcd", "rno", "lane"], how="left")
        for c in b.columns:
            if b[c].dtype == np.float64:
                b[c] = b[c].astype(np.float32)
        parts.append(b)
        log(f"loaded {f}-{t} rows={len(b)}")
    df = pd.concat(parts, ignore_index=True).sort_values(["hd", "jcd", "rno", "lane"]).reset_index(drop=True)
    n = df.groupby(["hd", "jcd", "rno"])["lane"].transform("size")
    df = df[n == 6].reset_index(drop=True)
    assert (df["lane"].to_numpy().reshape(-1, 6) == np.arange(1, 7)).all()
    df["jcd_i"] = df["jcd"].astype(int)
    df["month"] = TF.month_of(df["hd"])
    return df


def split_cols(df):
    extra_pref = tuple(GROUPS.values())
    allc = TF.feature_cols(df)
    base = [c for c in allc if not c.startswith(extra_pref)]
    grp = {g: [c for c in allc if c.startswith(p)] for g, p in GROUPS.items()}
    return base, grp


def fit_predict(df, feats, month, seed):
    import lightgbm as lgb
    months = df["month"].to_numpy()
    tr_lo, va_lo = TF.add_months(month, -TF.TRAIN_MONTHS), TF.add_months(month, -TF.VALID_MONTHS)
    out = {}
    sets = {"tr": (months >= tr_lo) & (months < va_lo), "va": (months >= va_lo) & (months < month),
            "te": months == month}
    for k, m in sets.items():
        X = df.loc[m, feats].to_numpy(np.float32)
        pos = df.loc[m, "y_pos"].to_numpy().reshape(-1, 6)
        ok = (pos == 1).sum(1) == 1
        out[k] = (X, pos, ok)
    X, pos, ok = out["tr"]
    Xv, posv, okv = out["va"]
    nf = X.shape[1]
    p = dict(PARAMS, seed=seed)
    dtr = lgb.Dataset(X.reshape(-1, 6, nf)[ok].reshape(-1, nf), (pos[ok] == 1).ravel(), free_raw_data=True)
    dva = lgb.Dataset(Xv.reshape(-1, 6, nf)[okv].reshape(-1, nf), (posv[okv] == 1).ravel(), reference=dtr)
    mdl = lgb.train(p, dtr, 3000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
    Sv = mdl.predict(Xv, raw_score=True).reshape(-1, 6)
    tau = TF.fit_tau(Sv[okv], np.argmax(posv[okv] == 1, 1))
    Xt, post, okt = out["te"]
    St = mdl.predict(Xt, raw_score=True).reshape(-1, 6)
    P = TF.softmax_groups(St * tau, None)
    gain = pd.Series(mdl.feature_importance("gain"), index=feats)
    return P, post, okt, tau, mdl.best_iteration, gain


def cmd_run(configs, months):
    df = load_all()
    base, grp = split_cols(df)
    log(f"base feats={len(base)} groups=" + ", ".join(f"{g}:{len(c)}" for g, c in grp.items()))
    rdir = WORK / "res"
    rdir.mkdir(exist_ok=True)
    for month in months:
        for cfg in configs:
            out = rdir / f"{cfg}_{month}.npz"
            if out.exists():
                continue
            feats = base + [c for g in CONFIGS[cfg] for c in grp[g]]
            t0 = time.time()
            seed = 11 if cfg.endswith("_seed2") else TF.LGB_PARAMS["seed"]
            P, pos, ok, tau, it, gain = fit_predict(df, feats, month, seed)
            y = np.argmax(pos == 1, 1)
            ll = -np.log(np.clip(P[np.arange(len(P)), y], 1e-12, None))
            acc = (P.argmax(1) == y)
            np.savez_compressed(out, P=P, y=y, ok=ok, tau=tau, it=it)
            gain.to_csv(rdir / f"gain_{cfg}_{month}.csv")
            log(f"{cfg} {month}: n={ok.sum()} ll={ll[ok].mean():.4f} acc={acc[ok].mean():.4f} "
                f"iters={it} tau={tau:.2f} ({time.time()-t0:.0f}s)")


def cmd_report():
    rdir = WORK / "res"
    rows = []
    per = {}
    for f in sorted(rdir.glob("*.npz")):
        cfg, month = f.stem.rsplit("_", 1)
        z = np.load(f)
        P, y, ok = z["P"], z["y"], z["ok"]
        ll = -np.log(np.clip(P[np.arange(len(P)), y], 1e-12, None))
        per[(cfg, month)] = (ll, P.argmax(1) == y, ok)
    cfgs = sorted({c for c, _ in per}, key=lambda c: list(CONFIGS).index(c))
    months = sorted({m for _, m in per})
    full = [m for m in months if all((c, m) in per for c in cfgs)]
    print(f"months with all configs: {len(full)} ({full[0]}..{full[-1]})")
    rng = np.random.default_rng(0)
    base_ll = np.concatenate([per[("base", m)][0][per[("base", m)][2]] for m in full])
    base_acc = np.concatenate([per[("base", m)][1][per[("base", m)][2]] for m in full])
    for c in cfgs:
        ll = np.concatenate([per[(c, m)][0][per[(c, m)][2]] for m in full])
        acc = np.concatenate([per[(c, m)][1][per[(c, m)][2]] for m in full])
        d = ll - base_ll
        # レース単位のブートストラップ（対）
        bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(300)]
        mon = [per[(c, m)][0][per[(c, m)][2]].mean() - per[("base", m)][0][per[("base", m)][2]].mean() for m in full]
        rows.append({"config": c, "races": len(ll), "logloss": ll.mean(), "d_ll_x1000": 1000 * d.mean(),
                     "se_x1000": 1000 * np.std(bs), "months_better": int(sum(x < 0 for x in mon)),
                     "top1_acc": acc.mean(), "d_acc_pt": 100 * (acc.mean() - base_acc.mean())})
    rep = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(rep.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    # 月別の Δ（×1000）
    tab = pd.DataFrame({c: [1000 * (per[(c, m)][0][per[(c, m)][2]].mean() - per[("base", m)][0][per[("base", m)][2]].mean())
                            for m in full] for c in cfgs}, index=full)
    print("\n月別 Δlogloss ×1000（負＝改善）")
    print(tab.round(2).to_string())
    # 追加特徴量の gain（ALL / ALL_U の平均順位）
    for c in ("ALL",):
        gs = [pd.read_csv(rdir / f"gain_{c}_{m}.csv", index_col=0).iloc[:, 0] for m in full
              if (rdir / f"gain_{c}_{m}.csv").exists()]
        if not gs:
            continue
        g = pd.concat(gs, axis=1).mean(1)
        g = (g / g.sum()).sort_values(ascending=False)
        rank = {k: i + 1 for i, k in enumerate(g.index)}
        ex = [k for k in g.index if k.startswith(tuple(GROUPS.values()))]
        print(f"\n{c}: 追加特徴量の gain シェア上位（全{len(g)}列中の順位）")
        for k in ex[:20]:
            print(f"  {rank[k]:4d}  {k:32s} {100*g[k]:.2f}%")
        print("  グループ別 gain シェア: " + ", ".join(
            f"{gname}={100*g[[k for k in g.index if k.startswith(p)]].sum():.1f}%" for gname, p in GROUPS.items()))
    return rep


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["run", "report"])
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--months", default=",".join(TEST_MONTHS))
    a = ap.parse_args()
    if a.cmd == "run":
        cmd_run(a.configs.split(","), a.months.split(","))
    else:
        cmd_report()


if __name__ == "__main__":
    main()
