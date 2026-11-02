"""3連単: 着順確率モデル × 買い方 のウォークフォワード検証（前夜に決める前提）。

使い方（作業ファイルは --workdir に置く。リポジトリには置かない）:
    python scripts/analysis/trifecta.py features   # 特徴量を半年ごとに作る（stats_lag_days=45、キャッシュあり）
    python scripts/analysis/trifecta.py predict    # 着順モデルの学習と月ごとの予測（ウォークフォワード）
    python scripts/analysis/trifecta.py evaluate   # 払戻モデル・買い方の探索・ウォークフォワード選択・月別成績
    python scripts/analysis/trifecta.py odds       # 締切オッズがある期間での参考値（前夜には使えない情報）
    python scripts/analysis/trifecta.py all

方針（詳細は notes/analysis/trifecta.md）:
  * 確率モデルは3段:  P(i,j,k) = P1(i) * P2(j | i) * P3(k | i, j)。
    各段 LightGBM の2値分類のスコアを、候補艇の中で softmax(τ・score) に通す（τ は検証用の直近2か月で決める）。
    比較用に Harville（P2 ∝ P1）と べき乗割引 Harville（P2 ∝ P1^λ2, P3 ∝ P1^λ3）も出す。
  * 学習は「予測する月の直前24か月」（うち最後の2か月は早期打ち切りと τ 用）。予測月より後のデータは使わない。
    前期間（2024-05〜2025-10）は四半期ごと、検証期間（2025-11〜2026-10）は毎月学習し直す。
  * 払戻モデル g: 「その買い目が当たったときの払戻」を、当たった買い目（各レース1つ）だけで学習する
    （E[払戻 | 当たり, 前夜の特徴] をそのまま推定できる）。期待値 EV = P(買い目) × g / 100。
    月 m の g は m より前の月の（その時点のウォークフォワード予測での）当たりだけで学習する。
  * 買い方は候補を全部月別に計算し、各検証月で「その月より前の12か月」の成績が最も良いものを選んで
    その月に当てる（パラメータ選びもその月より前だけ）。候補数を記録する。
  * 回収率 = 払戻合計 ÷ 購入額合計。1点100円。払戻表の3連単（確定配当）で当たりを判定。
  * 返還: results で position も st_timing も NULL の艇 = 欠場。その艇を含む買い目は返還（購入額0・払戻0）。
    position NULL で ST がある艇（F・転覆・失格など）は外れ扱い。3連単の払戻が無いレース（中止など）は買わなかった扱い。
  * DB は読み取り専用で開く。
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
DB = ROOT / "data" / "boatrace.db"
DEFAULT_WORK = Path("/tmp/claude-0/-home-user-verbose-engine/968008a0-6ecf-5641-a3ff-006912b41722/scratchpad/trifecta")

# ---- 期間・設定（分析の設定。選定ルールではないので rules.yaml には置かない） ----
FEAT_CHUNKS = [("20220501", "20221031"), ("20221101", "20230430"), ("20230501", "20231031"),
               ("20231101", "20240430"), ("20240501", "20241031"), ("20241101", "20250430"),
               ("20250501", "20251031"), ("20251101", "20260430"), ("20260501", "20261031"),
               ("20261101", "20261101")]
STATS_LAG_DAYS = 45
TRAIN_MONTHS = 24          # 学習窓（月）
VALID_MONTHS = 2           # 学習窓の末尾。早期打ち切りと τ・λ 用
PRE_FITS = ["2024-05", "2024-08", "2024-11", "2025-02", "2025-05", "2025-08"]   # 四半期ごと（各3か月を予測）
TEST_MONTHS = [f"{y}-{m:02d}" for y, m in [(2025, 11), (2025, 12)] + [(2026, i) for i in range(1, 11)]]
SELECT_WINDOW = 12         # 買い方の選択に使う直前の月数
G_FIRST_MONTH = "2024-05"  # 払戻モデルの学習に使う最初の月（OOS予測が始まる月）
EVAL_FIRST_MONTH = "2024-11"

PERMS = np.array(list(itertools.permutations(range(6), 3)), dtype=np.int64)   # 120 x 3（0始まりの艇）
ID_COLS = ["hd", "jcd", "rno", "lane", "racer_no", "motor_no", "boat_no"]
LGB_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=300,
                  feature_fraction=0.6, bagging_fraction=0.7, bagging_freq=1, lambda_l2=10.0,
                  max_bin=127, num_threads=4, verbose=-1, seed=7)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def connect():
    return sqlite3.connect(f"file:{DB}?mode=ro", uri=True, timeout=60)


def month_of(hd):
    hd = pd.Series(hd).astype(str)
    return (hd.str[:4] + "-" + hd.str[4:6]).to_numpy()


def add_months(m, k):
    y, mo = int(m[:4]), int(m[5:7])
    t = y * 12 + (mo - 1) + k
    return f"{t // 12}-{t % 12 + 1:02d}"


# ============================================================== features
def cmd_features(work: Path):
    d = work / "feat"
    d.mkdir(parents=True, exist_ok=True)
    for f, t in FEAT_CHUNKS:
        out = d / f"feat_{f}_{t}.parquet"
        if out.exists():
            continue
        log(f"features {f}-{t}")
        subprocess.run([sys.executable, str(ROOT / "scripts" / "features.py"), "--from", f, "--to", t,
                        "--stats-lag-days", str(STATS_LAG_DAYS), "--out", str(out)], check=True)


def load_features(work: Path) -> pd.DataFrame:
    parts = []
    for f, t in FEAT_CHUNKS:
        p = work / "feat" / f"feat_{f}_{t}.parquet"
        if p.exists():
            parts.append(pd.read_parquet(p))
    df = pd.concat(parts, ignore_index=True)
    df = df.sort_values(["hd", "jcd", "rno", "lane"]).reset_index(drop=True)
    n = df.groupby(["hd", "jcd", "rno"])["lane"].transform("size")
    df = df[(n == 6)].reset_index(drop=True)
    assert (df["lane"].to_numpy().reshape(-1, 6) == np.arange(1, 7)).all()
    df["jcd_i"] = df["jcd"].astype(int)
    df["month"] = month_of(df["hd"])
    for c in df.columns:
        if df[c].dtype == np.float64:
            df[c] = df[c].astype(np.float32)
    return df


def feature_cols(df):
    return [c for c in df.columns if c not in ID_COLS and not c.startswith("y_") and c not in ("month",)] + ["lane", "rno"]


def load_void(con, hd_from, hd_to):
    """欠場（返還）フラグ: (hd,jcd,rno,lane) の集合。position も st も NULL。"""
    res = pd.read_sql("select hd, jcd, rno, lane, position, json_extract(data,'$.st_timing') st from results "
                      "where position is null and hd between ? and ?", con, params=(hd_from, hd_to))
    return res[res["st"].isna()][["hd", "jcd", "rno", "lane"]].assign(void=True)


# ============================================================== 確率モデル
def softmax_groups(s, g):
    """s: (n, m) スコア, 無効は -inf。行ごとの softmax"""
    s = s - np.nanmax(np.where(np.isfinite(s), s, -1e30), axis=1, keepdims=True)
    e = np.where(np.isfinite(s), np.exp(s), 0.0)
    return e / e.sum(1, keepdims=True)


def fit_tau(S, y_idx, taus=np.linspace(0.3, 2.0, 35)):
    """S: (n, m) スコア行列(-inf=候補外), y_idx: 正解列。対数損失最小の τ"""
    best = None
    for t in taus:
        P = softmax_groups(S * t, None)
        ll = -np.log(np.clip(P[np.arange(len(S)), y_idx], 1e-12, None)).mean()
        if best is None or ll < best[1]:
            best = (t, ll)
    return best[0]


def lgb_train(X, y, Xv, yv, params=LGB_PARAMS, rounds=3000):
    import lightgbm as lgb
    dtr = lgb.Dataset(X, y, free_raw_data=True)
    dva = lgb.Dataset(Xv, yv, reference=dtr)
    m = lgb.train(params, dtr, rounds, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    return m


class StageModel:
    """3段の着順モデル。fit(train_df) → predict(df) で (nr, 120) の確率など。"""

    def __init__(self, feats):
        self.f1 = feats
        self.top = None

    # ---- 段ごとの行を作る ----
    @staticmethod
    def _key_feats():
        return ["national_win_rate", "rc_win_shrunk", "grade", "r90_avg_st", "m_res3_shrunk", "rs_c_win_shrunk"]

    def _stage2_rows(self, X6, first):
        """X6: (nr, 6, F) の特徴、first: (nr,) 1着艇。返り値: (nr*6, F2)（first 自身の行も作り、後で -inf）"""
        nr = X6.shape[0]
        kf = X6[np.arange(nr), first][:, self.kidx]                      # 1着艇の主要特徴
        own = X6[:, :, self.tidx]
        lane = np.broadcast_to(np.arange(1, 7), (nr, 6))
        fl = np.broadcast_to((first + 1)[:, None], (nr, 6))
        Z = np.concatenate([own, np.repeat(kf[:, None, :], 6, 1),
                            fl[..., None], (lane - fl)[..., None]], axis=2)
        return Z.reshape(nr * 6, -1).astype(np.float32)

    def _stage3_rows(self, X6, first, second):
        nr = X6.shape[0]
        k1 = X6[np.arange(nr), first][:, self.kidx]
        k2 = X6[np.arange(nr), second][:, self.kidx]
        own = X6[:, :, self.tidx]
        lane = np.broadcast_to(np.arange(1, 7), (nr, 6))
        fl = np.broadcast_to((first + 1)[:, None], (nr, 6))
        sl = np.broadcast_to((second + 1)[:, None], (nr, 6))
        Z = np.concatenate([own, np.repeat(k1[:, None, :], 6, 1), np.repeat(k2[:, None, :], 6, 1),
                            fl[..., None], sl[..., None], (lane - fl)[..., None], (lane - sl)[..., None]], axis=2)
        return Z.reshape(nr * 6, -1).astype(np.float32)

    def fit(self, tr: pd.DataFrame, va: pd.DataFrame):
        t0 = time.time()
        X = tr[self.f1].to_numpy(np.float32)
        Xv = va[self.f1].to_numpy(np.float32)
        pos = tr["y_pos"].to_numpy().reshape(-1, 6)
        posv = va["y_pos"].to_numpy().reshape(-1, 6)
        # stage 1: 1着が決まったレース
        ok = (pos == 1).sum(1) == 1
        okv = (posv == 1).sum(1) == 1
        m1 = lgb_train(X.reshape(-1, 6, X.shape[1])[ok].reshape(-1, X.shape[1]), (pos[ok] == 1).ravel(),
                       Xv.reshape(-1, 6, Xv.shape[1])[okv].reshape(-1, Xv.shape[1]), (posv[okv] == 1).ravel())
        self.m1 = m1
        gain = m1.feature_importance("gain")
        order = np.argsort(-gain)
        self.top = [self.f1[i] for i in order[:70]]
        self.tidx = np.array([self.f1.index(c) for c in self.top])
        self.kidx = np.array([self.f1.index(c) for c in self._key_feats() + ["lane"]])
        S1v = m1.predict(Xv, raw_score=True).reshape(-1, 6)
        self.tau1 = fit_tau(S1v[okv], np.argmax(posv[okv] == 1, 1))
        log(f"  stage1 iters={m1.best_iteration} tau={self.tau1:.2f} ({time.time()-t0:.0f}s)")

        # stage 2: 1着・2着が決まったレース。候補は1着以外・欠場以外
        def s2data(X_, pos_):
            X6 = X_.reshape(-1, 6, X_.shape[1])
            ok2 = ((pos_ == 1).sum(1) == 1) & ((pos_ == 2).sum(1) == 1)
            first = np.argmax(pos_ == 1, 1)
            Z = self._stage2_rows(X6[ok2], first[ok2])
            lab = (pos_[ok2] == 2)
            cand = (pos_[ok2] != 1)          # NaN（事故・欠場）も負例の候補に含める
            return Z, lab, cand, ok2
        Z, lab, cand, _ = s2data(X, pos)
        Zv, labv, candv, _ = s2data(Xv, posv)
        m2 = lgb_train(Z[cand.ravel()], lab.ravel()[cand.ravel()], Zv[candv.ravel()], labv.ravel()[candv.ravel()])
        S2v = np.where(candv, m2.predict(Zv, raw_score=True).reshape(-1, 6), -np.inf)
        self.m2 = m2
        self.tau2 = fit_tau(S2v, np.argmax(labv, 1))
        log(f"  stage2 iters={m2.best_iteration} tau={self.tau2:.2f} ({time.time()-t0:.0f}s)")

        def s3data(X_, pos_):
            X6 = X_.reshape(-1, 6, X_.shape[1])
            ok3 = ((pos_ == 1).sum(1) == 1) & ((pos_ == 2).sum(1) == 1) & ((pos_ == 3).sum(1) == 1)
            first = np.argmax(pos_ == 1, 1)
            second = np.argmax(pos_ == 2, 1)
            Z = self._stage3_rows(X6[ok3], first[ok3], second[ok3])
            p = pos_[ok3]
            lab = (p == 3)
            cand = ((p > 2) | np.isnan(p))
            return Z, lab, cand
        Z, lab, cand = s3data(X, pos)
        Zv, labv, candv = s3data(Xv, posv)
        m3 = lgb_train(Z[cand.ravel()], lab.ravel()[cand.ravel()], Zv[candv.ravel()], labv.ravel()[candv.ravel()])
        S3v = np.where(candv, m3.predict(Zv, raw_score=True).reshape(-1, 6), -np.inf)
        self.m3 = m3
        self.tau3 = fit_tau(S3v, np.argmax(labv, 1))
        log(f"  stage3 iters={m3.best_iteration} tau={self.tau3:.2f} ({time.time()-t0:.0f}s)")

        # べき乗割引 Harville の λ（検証2か月で）
        P1v = softmax_groups(S1v * self.tau1, None)
        ok3 = okv & ((posv == 2).sum(1) == 1) & ((posv == 3).sum(1) == 1)
        y3 = np.stack([np.argmax(posv == 1, 1), np.argmax(posv == 2, 1), np.argmax(posv == 3, 1)], 1)[ok3]
        best = None
        for l2 in np.arange(0.5, 1.21, 0.05):
            for l3 in np.arange(0.4, 1.21, 0.05):
                P = harville(P1v[ok3], l2, l3)
                ci = combo_index(y3)
                ll = -np.log(P[np.arange(len(P)), ci]).mean()
                if best is None or ll < best[2]:
                    best = (l2, l3, ll)
        self.lam2, self.lam3 = best[0], best[1]
        log(f"  harville-power lam2={self.lam2:.2f} lam3={self.lam3:.2f} ({time.time()-t0:.0f}s)")
        return self

    def predict(self, df: pd.DataFrame):
        X = df[self.f1].to_numpy(np.float32)
        nr = len(X) // 6
        X6 = X.reshape(nr, 6, -1)
        S1 = self.m1.predict(X, raw_score=True).reshape(nr, 6)
        P1 = softmax_groups(S1 * self.tau1, None)
        # stage 2: 1着候補ごと
        P2 = np.zeros((nr, 6, 6))
        for i in range(6):
            Z = self._stage2_rows(X6, np.full(nr, i))
            s = self.m2.predict(Z, raw_score=True).reshape(nr, 6)
            s[:, i] = -np.inf
            P2[:, i, :] = softmax_groups(s * self.tau2, None)
        # stage 3: (i,j) ごと（30通り）
        P3ij = np.zeros((nr, 6, 6, 6))
        for i in range(6):
            for j in range(6):
                if i == j:
                    continue
                Z = self._stage3_rows(X6, np.full(nr, i), np.full(nr, j))
                s = self.m3.predict(Z, raw_score=True).reshape(nr, 6)
                s[:, [i, j]] = -np.inf
                P3ij[:, i, j, :] = softmax_groups(s * self.tau3, None)
        a, b, c = PERMS[:, 0], PERMS[:, 1], PERMS[:, 2]
        P = P1[:, a] * P2[:, a, b] * P3ij[:, a, b, c]
        return dict(P=P.astype(np.float32), P1=P1.astype(np.float32),
                    Ph=harville(P1, 1.0, 1.0).astype(np.float32),
                    Pp=harville(P1, self.lam2, self.lam3).astype(np.float32))


def harville(P1, l2, l3):
    a, b, c = PERMS[:, 0], PERMS[:, 1], PERMS[:, 2]
    q2 = P1 ** l2
    q3 = P1 ** l3
    pa = P1[:, a]
    p2 = q2[:, b] / (q2.sum(1, keepdims=True) - q2[:, a])
    p3 = q3[:, c] / (q3.sum(1, keepdims=True) - q3[:, a] - q3[:, b])
    return pa * p2 * p3


_CI = {tuple(p): n for n, p in enumerate(PERMS.tolist())}


def combo_index(y3):
    return np.array([_CI[tuple(r)] for r in np.asarray(y3).tolist()])


def cmd_predict(work: Path, only=None):
    df = load_features(work)
    feats = feature_cols(df)
    log(f"features loaded rows={len(df)} races={len(df)//6} feats={len(feats)}")
    pdir = work / "pred"
    pdir.mkdir(exist_ok=True)
    fits = [(m, [m, add_months(m, 1), add_months(m, 2)]) for m in PRE_FITS] + [(m, [m]) for m in TEST_MONTHS]
    months = df["month"].to_numpy()
    for start, targets in fits:
        if only and start not in only:
            continue
        outs = [pdir / f"pred_{t}.npz" for t in targets]
        if all(o.exists() for o in outs):
            continue
        tr_lo, va_lo = add_months(start, -TRAIN_MONTHS), add_months(start, -VALID_MONTHS)
        tr = df[(months >= tr_lo) & (months < va_lo)]
        va = df[(months >= va_lo) & (months < start)]
        log(f"fit for {start}: train {tr_lo}..{add_months(va_lo,-1)} races={len(tr)//6}, valid races={len(va)//6}")
        # 1着が無いレース（中止・成績なし）は学習から外す（y_has_result）
        tr = tr[tr["y_has_result"].to_numpy()]
        va = va[va["y_has_result"].to_numpy()]
        sm = StageModel(feats).fit(tr, va)
        for t, o in zip(targets, outs):
            te = df[months == t]
            if te.empty:
                continue
            r = sm.predict(te)
            keys = te[["hd", "jcd", "rno"]].iloc[::6]
            np.savez_compressed(o, hd=keys["hd"].to_numpy().astype(str), jcd=keys["jcd"].to_numpy().astype(str),
                                rno=keys["rno"].to_numpy().astype(int), fit=start,
                                tau=np.array([sm.tau1, sm.tau2, sm.tau3, sm.lam2, sm.lam3]), **r)
            log(f"  predicted {t}: races={len(keys)}")
        imp = pd.Series(sm.m1.feature_importance("gain"), index=feats).sort_values(ascending=False)
        imp.head(40).to_csv(pdir / f"imp1_{start}.csv")


# ============================================================== main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["features", "predict", "evaluate", "odds", "all"])
    ap.add_argument("--workdir", default=str(DEFAULT_WORK))
    ap.add_argument("--only", help="predict: この学習開始月だけ（カンマ区切り）")
    a = ap.parse_args(argv)
    work = Path(a.workdir)
    work.mkdir(parents=True, exist_ok=True)
    if a.cmd in ("features", "all"):
        cmd_features(work)
    if a.cmd in ("predict", "all"):
        cmd_predict(work, a.only.split(",") if a.only else None)
    if a.cmd in ("evaluate", "all"):
        cmd_evaluate(work)
    if a.cmd in ("odds", "all"):
        cmd_odds(work)
    return 0


if __name__ == "__main__":
    sys.exit(main())
