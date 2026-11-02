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
WORK = DEFAULT_WORK
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
    fits = [(m, [m, add_months(m, 1), add_months(m, 2)]) for m in PRE_FITS] + [(m, [m]) for m in TEST_MONTHS[:-1]]
    fits.append((TEST_MONTHS[-1], [TEST_MONTHS[-1], add_months(TEST_MONTHS[-1], 1)]))  # 翌月分はオッズ参考用
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


# ============================================================== 評価用データ
RACE_FLAGS = ["r_final", "r_semifinal", "r_special", "r_qualify", "r_fixed_course", "race_n_a1", "close_hour"]


def load_eval(work: Path):
    """予測（OOS）と払戻・返還・前夜の素性をレース単位にそろえる。"""
    files = sorted((work / "pred").glob("pred_*.npz"))
    parts = []
    for f in files:
        z = np.load(f)
        parts.append(dict(hd=z["hd"], jcd=z["jcd"], rno=z["rno"], P=z["P"], P1=z["P1"], Ph=z["Ph"], Pp=z["Pp"]))
    R = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    races = pd.DataFrame({"hd": R["hd"], "jcd": R["jcd"], "rno": R["rno"]})
    races["month"] = month_of(races["hd"])
    hd_lo, hd_hi = races.hd.min(), races.hd.max()
    con = connect()
    pay = pd.read_sql("select hd, jcd, rno, combination comb, payout from payouts where ticket_type='sanrensho' "
                      "and hd between ? and ?", con, params=(hd_lo, hd_hi))
    pay = pay.drop_duplicates(["hd", "jcd", "rno"])     # 同着は無かった（確認済み）。あれば最初の1つ
    races = races.merge(pay, on=["hd", "jcd", "rno"], how="left")
    ok = races["comb"].notna()
    hit = np.full(len(races), -1)
    y3 = races.loc[ok, "comb"].str.split("-", expand=True).astype(int).to_numpy() - 1
    hit[ok.to_numpy()] = combo_index(y3)
    races["hit"] = hit
    races["payout"] = races["payout"].fillna(0).astype(float)
    vd = load_void(con, hd_lo, hd_hi)
    lane_void = np.zeros((len(races), 6), bool)
    key = races.reset_index()[["index", "hd", "jcd", "rno"]]
    vv = vd.merge(key, on=["hd", "jcd", "rno"])
    lane_void[vv["index"].to_numpy(), vv["lane"].to_numpy().astype(int) - 1] = True
    void = lane_void[:, PERMS].any(2)                                       # (nr, 120)
    # 前夜の素性（特徴量テーブルから）
    df = load_features(work)
    df = df[df["hd"].between(hd_lo, hd_hi)]
    k6 = df[["hd", "jcd", "rno"]].iloc[::6].reset_index(drop=True)
    k6["fi"] = np.arange(len(k6))
    races = races.merge(k6, on=["hd", "jcd", "rno"], how="left")
    assert races["fi"].notna().all()
    fi = races["fi"].to_numpy()
    nwr = df["national_win_rate"].to_numpy().reshape(-1, 6)[fi]
    grd = df["grade"].to_numpy().reshape(-1, 6)[fi]
    l1 = df.iloc[::6][RACE_FLAGS].to_numpy()[fi]
    for i, c in enumerate(RACE_FLAGS):
        races[c] = l1[:, i]
    races["race_cat"] = np.select([races.r_final == 1, races.r_semifinal == 1, races.r_special == 1, races.r_qualify == 1],
                                  ["優勝戦", "準優", "特選系", "予選"], "一般")
    log(f"eval races={len(races)} with payout={ok.sum()} void-combos={void.sum()}")
    return races, R, void, nwr, grd


def entropy(P):
    return -(P * np.log(np.clip(P, 1e-12, None))).sum(1)


def g_features(races, P, P1, nwr, grd):
    """払戻モデル g の入力（レース×120 を平たくする）。前夜に分かるものだけ。"""
    nr = len(races)
    a, b, c = PERMS[:, 0], PERMS[:, 1], PERMS[:, 2]
    rankP = (-P).argsort(1).argsort(1) + 1
    r1 = (-P1).argsort(1).argsort(1) + 1
    nr_rank = (-np.nan_to_num(nwr, nan=0)).argsort(1).argsort(1) + 1
    H = entropy(P)
    cols = {
        "cid": np.tile(np.arange(120), nr), "la": np.tile(a + 1, nr), "lb": np.tile(b + 1, nr), "lc": np.tile(c + 1, nr),
        "logP": np.log(np.clip(P, 1e-9, None)).ravel(), "rankP": rankP.ravel(),
        "lp1a": np.log(P1[:, a]).ravel(), "lp1b": np.log(P1[:, b]).ravel(), "lp1c": np.log(P1[:, c]).ravel(),
        "r1a": r1[:, a].ravel(), "r1b": r1[:, b].ravel(), "r1c": r1[:, c].ravel(),
        "nwa": nwr[:, a].ravel(), "nwb": nwr[:, b].ravel(), "nwc": nwr[:, c].ravel(),
        "nra": nr_rank[:, a].ravel(), "nrb": nr_rank[:, b].ravel(), "nrc": nr_rank[:, c].ravel(),
        "ga": grd[:, a].ravel(), "gb": grd[:, b].ravel(), "gc": grd[:, c].ravel(),
    }
    rep = lambda x: np.repeat(np.asarray(x, dtype=float), 120)
    cols.update({"H": rep(H), "lpmax": rep(np.log(P.max(1))), "p1max": rep(P1.max(1)),
                 "nw1": rep(nwr[:, 0]), "nwmax_o": rep(np.nanmax(nwr[:, 1:], 1)),
                 "jcd": rep(races["jcd"].astype(int)), "rno": rep(races["rno"])})
    for c_ in RACE_FLAGS:
        cols[c_] = rep(races[c_])
    return pd.DataFrame(cols).astype(np.float32)


G_PARAMS = dict(objective="gamma", learning_rate=0.03, num_leaves=31, min_data_in_leaf=100,
                feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0,
                num_threads=4, verbose=-1, seed=11)
G_ROUNDS = 600


def fit_g_all(work, races, R, nwr, grd, months_eval):
    """月ごとに g（当たったときの払戻の期待値）を、その月より前の当たりだけで学習して予測。"""
    import lightgbm as lgb
    cache = work / "eval" / "g_pred.npy"
    if cache.exists():
        G = np.load(cache)
        if G.shape == R["P"].shape:
            return G
    G = np.full(R["P"].shape, np.nan, dtype=np.float32)
    month = races["month"].to_numpy()
    hit = races["hit"].to_numpy()
    pay = races["payout"].to_numpy()
    # 当たった買い目の行だけ、月ごとに先に作っておく
    XH, YH = {}, {}
    for m in sorted(set(month)):
        if m < G_FIRST_MONTH or m > months_eval[-1]:
            continue
        ix = np.where((month == m) & (hit >= 0))[0]
        X = g_features(races.iloc[ix], R["P"][ix], R["P1"][ix], nwr[ix], grd[ix]).to_numpy().reshape(len(ix), 120, -1)
        XH[m] = X[np.arange(len(ix)), hit[ix]]; YH[m] = pay[ix]
    for m in months_eval:
        t0 = time.time()
        prior = [x for x in XH if x < m]
        Xtr = np.concatenate([XH[x] for x in prior]); ytr = np.concatenate([YH[x] for x in prior])
        mdl = lgb.train(G_PARAMS, lgb.Dataset(Xtr, ytr), G_ROUNDS)
        te = np.where(month == m)[0]
        Xte = g_features(races.iloc[te], R["P"][te], R["P1"][te], nwr[te], grd[te]).to_numpy()
        G[te] = mdl.predict(Xte).reshape(len(te), 120)
        log(f"  g for {m}: train hits={len(ytr)} ({time.time()-t0:.0f}s)")
    np.save(cache, G)
    return G


# ============================================================== 買い方
def topk_mask(P, k):
    idx = np.argsort(-P, 1)[:, :k]
    m = np.zeros(P.shape, bool)
    np.put_along_axis(m, idx, True, 1)
    return m


def build_candidates(H_cuts):
    """(名前, 系統, 関数(ctx)->mask)。ctx は月ごとの配列。"""
    C = []
    for k in [1, 2, 3, 4, 6, 8, 10, 15, 20, 30]:
        C.append((f"A:top{k}", "A 確率上位k点（全レース）", lambda x, k=k: topk_mask(x["P"], k)))
    for q, hc in H_cuts.items():
        for k in [1, 2, 3, 5, 8]:
            C.append((f"B:H<=q{q}:top{k}", "B 荒れにくいレース×上位k点",
                      lambda x, hc=hc, k=k: topk_mask(x["P"], k) & (x["H"] <= hc)[:, None]))
    for tau in [1.0, 1.1, 1.2, 1.3, 1.5, 1.75, 2.0, 2.5, 3.0]:
        for pmin in [0, 0.003, 0.01, 0.03]:
            C.append((f"C:EV>={tau}:p>={pmin}", "C 期待値（確率×推定配当）しきい値",
                      lambda x, tau=tau, pmin=pmin: (x["EV"] >= tau) & (x["P"] >= pmin)))
    for k in [3, 5, 10, 20]:
        for tau in [0.9, 1.0, 1.1, 1.2, 1.4]:
            C.append((f"D:top{k}&EV>={tau}", "D 上位k点のうち期待値しきい値以上",
                      lambda x, k=k, tau=tau: topk_mask(x["P"], k) & (x["EV"] >= tau)))
    for p1 in [0.5, 0.6, 0.7, 0.8]:
        for m in [2, 4, 6, 10, 20]:
            C.append((f"E:fav>={p1}:fix1:top{m}", "E 本命1着固定の流し（上位m点）",
                      lambda x, p1=p1, m=m: topk_mask(np.where(x["favfirst"], x["P"], -1), m) & (x["p1max"] >= p1)[:, None]))
    for p1 in [0.5, 0.6, 0.7]:
        for tau in [1.0, 1.2, 1.5]:
            C.append((f"F:fav>={p1}:fix1:EV>={tau}", "F 本命1着固定×期待値",
                      lambda x, p1=p1, tau=tau: x["favfirst"] & (x["EV"] >= tau) & (x["p1max"] >= p1)[:, None]))
    for r in [1.0, 1.1, 1.25, 1.5]:
        for h in [10, 30]:
            C.append((f"G:cell>={r}:hits>={h}", "G 型×確率帯のセルで過去の回収率が高いもの",
                      lambda x, r=r, h=h: x["cell_ok"](r, h)))
    # ---- 運用者の方向性（2026-11-02 追加）: 予想オッズ O = g/100（前夜推定の配当倍率）で配当帯・人気を推定 ----
    for lo, hi in [(4, 8), (3, 8), (4, 10), (5, 12)]:
        C.append((f"H:top1&O{lo}-{hi}", "H 本命・人気になりすぎない（1点）",
                  lambda x, lo=lo, hi=hi: topk_mask(x["P"], 1) & (x["O"] >= lo) & (x["O"] <= hi)))
        C.append((f"H:bestP_in_O{lo}-{hi}&top3", "H 本命・人気になりすぎない（1点）",
                  lambda x, lo=lo, hi=hi: topk_mask(np.where((x["O"] >= lo) & (x["O"] <= hi), x["P"], -1), 1)
                  & topk_mask(x["P"], 3) & (x["O"] >= lo) & (x["O"] <= hi)))
    for k in [2, 3, 4, 5, 6]:
        for so in [3, 4, 5, 6]:
            C.append((f"H:top{k}&synO>={so}", "H 本命・複数点で合成オッズ下限",
                      lambda x, k=k, so=so: topk_mask(x["P"], k) & (synth(x["O"], topk_mask(x["P"], k)) >= so)[:, None]))
    for tau in [0.8, 0.9, 1.0, 1.1, 1.2, 1.4]:
        C.append((f"I:O10-30&EV>={tau}", "I 中穴（予想配当10〜30倍）×期待値",
                  lambda x, tau=tau: (x["O"] >= 10) & (x["O"] <= 30) & (x["EV"] >= tau)))
    for m in [1, 2, 3]:
        C.append((f"I:O10-30:topP{m}", "I 中穴（予想配当10〜30倍）×期待値",
                  lambda x, m=m: topk_mask(np.where((x["O"] >= 10) & (x["O"] <= 30), x["P"], -1), m) & (x["O"] >= 10) & (x["O"] <= 30)))
    for k in [1, 2, 3, 5]:
        C.append((f"J:nofav:top{k}", "J 予想1番人気を外した上位",
                  lambda x, k=k: topk_mask(np.where(x["favO"], -1, x["P"]), k)))
        C.append((f"J:nofav:top{k}:modeltop!=fav", "J 予想1番人気を外した上位",
                  lambda x, k=k: topk_mask(np.where(x["favO"], -1, x["P"]), k) & ~(topk_mask(x["P"], 1) & x["favO"]).any(1)[:, None]))
    return C


def synth(O, mask):
    """合成オッズ = 1 / Σ(1/O)（買う点に均等100円のとき、当たったときの払戻 ÷ 購入額 の調和平均）"""
    inv = np.where(mask, 1.0 / np.maximum(O, 1e-9), 0).sum(1)
    return np.where(inv > 0, 1.0 / np.maximum(inv, 1e-12), 0)


def recalibrate(P, hit, month, first, grid=np.arange(0.6, 1.31, 0.05), window=12):
    """月ごとに P ∝ P^α（レース内で正規化）。α はその月より前の OOS 月（最大12か月）で対数損失最小。"""
    Pc = P.copy(); alphas = {}
    ms = sorted(set(month))
    for m in ms:
        prior = [x for x in ms if first <= x < m][-window:]
        tr = np.isin(month, prior) & (hit >= 0)
        a = 1.0
        if tr.sum() >= 3000:
            lp = np.log(np.clip(P[tr], 1e-12, None)); h = hit[tr]
            best = None
            for g_ in grid:
                ll = -(g_ * lp[np.arange(len(h)), h] - np.log(np.exp(g_ * lp).sum(1))).mean()
                if best is None or ll < best[1]:
                    best = (g_, ll)
            a = best[0]
        sel = month == m
        q = np.clip(P[sel], 1e-12, None) ** a
        Pc[sel] = q / q.sum(1, keepdims=True)
        alphas[m] = a
    return Pc, alphas


PBANDS = np.array([0.0, 0.002, 0.005, 0.01, 0.02, 0.04, 0.07, 0.12, 1.01])


def cmd_evaluate(work: Path):
    ev = work / "eval"
    ev.mkdir(exist_ok=True)
    races, R, void, nwr, grd = load_eval(work)
    month = races["month"].to_numpy()
    months_all = sorted(set(month))
    months_eval = [m for m in months_all if EVAL_FIRST_MONTH <= m <= TEST_MONTHS[-1]]
    hit = races["hit"].to_numpy()
    payout = races["payout"].to_numpy()
    has = hit >= 0
    P, P1 = R["P"].astype(np.float64), R["P1"].astype(np.float64)
    P_raw = P
    P, alph = recalibrate(P_raw, hit, month, G_FIRST_MONTH)
    P1, alph1 = recalibrate(P1, np.where(hit >= 0, hit // 20, -1), month, G_FIRST_MONTH)
    log(f"recalibration alpha (3連単): {alph}")
    log(f"recalibration alpha (1着): {alph1}")
    out = []

    # ---- 1. 当てはまり（検証期間） ----
    def ll_of(PP, sel):
        return -np.log(np.clip(PP[sel, hit[sel]], 1e-12, None)).mean()
    tst = has & np.isin(month, TEST_MONTHS)
    pre = has & (month >= G_FIRST_MONTH) & (month < TEST_MONTHS[0])
    fit_rows = []
    for nm, PP in [("3段LightGBM＋月次の再校正(べき乗)", P), ("3段LightGBM", P_raw), ("Harville", R["Ph"]), ("べき乗Harville", R["Pp"])]:
        fit_rows.append(dict(model=nm, ll_pre=ll_of(PP, pre), ll_test=ll_of(PP, tst),
                             top1_hit=(np.argmax(PP[tst], 1) == hit[tst]).mean(),
                             top10_hit=(topk_mask(PP[tst], 10)[np.arange(tst.sum()), hit[tst]]).mean()))
    fit_rows.append(dict(model="一様(1/120)", ll_pre=math.log(120), ll_test=math.log(120), top1_hit=1 / 120, top10_hit=10 / 120))
    win = hit // 20  # PERMS は1着艇ごとに20個ずつ並ぶ
    fit_rows.append(dict(model="(参考)1着モデル P1", ll_pre=-np.log(P1[pre, win[pre]]).mean(),
                         ll_test=-np.log(P1[tst, win[tst]]).mean(), top1_hit=(np.argmax(P1[tst], 1) == win[tst]).mean(),
                         top10_hit=np.nan))
    fit = pd.DataFrame(fit_rows)
    fit.to_csv(ev / "fit.csv", index=False)
    out.append("## 当てはまり\n" + fit.to_markdown(index=False, floatfmt=".4f"))
    # 校正（検証期間・全買い目）
    pt = P[tst].ravel()
    yt = np.zeros_like(P[tst], dtype=bool)
    yt[np.arange(tst.sum()), hit[tst]] = True
    b = np.digitize(pt, PBANDS) - 1
    cal = pd.DataFrame({"band": b, "p": pt, "y": yt.ravel()}).groupby("band").agg(n=("y", "size"), pred=("p", "mean"), actual=("y", "mean"))
    cal.index = [f"{PBANDS[i]:.3f}-{PBANDS[i+1]:.3f}" for i in cal.index]
    p1t = P1[tst].ravel(); y1 = np.zeros_like(P1[tst], dtype=bool); y1[np.arange(tst.sum()), win[tst]] = True
    b1 = np.digitize(p1t, [0, .05, .1, .2, .3, .4, .5, .6, .7, .8, 1.01]) - 1
    cal1 = pd.DataFrame({"band": b1, "p": p1t, "y": y1.ravel()}).groupby("band").agg(n=("y", "size"), pred=("p", "mean"), actual=("y", "mean"))
    out.append("## 校正: 3連単（検証期間）\n" + cal.to_markdown(floatfmt=".4f"))
    out.append("## 校正: 1着（検証期間）\n" + cal1.to_markdown(floatfmt=".4f"))

    # ---- 2. 払戻モデルと期待値 ----
    log("fit payout model g per month")
    G = fit_g_all(work, races, R, nwr, grd, months_eval)
    EV = P * G / 100.0
    # g の当てはまり（検証期間の当たり）
    gh = G[tst, hit[tst]]
    ah = payout[tst]
    out.append(f"## 払戻モデル g（検証期間の当たり {tst.sum()} 件）\n"
               f"- log(推定)とlog(実際)の相関 {np.corrcoef(np.log(gh), np.log(ah))[0,1]:.3f}、"
               f"推定平均 {gh.mean():.0f}円 / 実際平均 {ah.mean():.0f}円、中央値 推定 {np.median(gh):.0f} / 実際 {np.median(ah):.0f}")
    # 期待値帯ごとの実際の回収率（全買い目を買ったと仮定、返還は除く）
    rows = []
    evb = [0, .5, .7, .8, .9, 1.0, 1.1, 1.2, 1.5, 2.0, 3.0, 1e9]
    for per, sel in [("前期間", (month >= EVAL_FIRST_MONTH) & (month < TEST_MONTHS[0]) & has), ("検証期間", tst)]:
        E = EV[sel]; V = ~void[sel]
        hm = np.zeros(E.shape, bool); hm[np.arange(sel.sum()), hit[sel]] = True
        bb = np.digitize(E, evb) - 1
        for i in range(len(evb) - 1):
            m_ = (bb == i) & V
            n = m_.sum()
            if n == 0:
                continue
            ret = (hm & m_).sum(1) @ payout[sel]
            rows.append(dict(period=per, ev=f"{evb[i]}-{evb[i+1]}", combos=int(n), hits=int((hm & m_).sum()),
                             roi=ret / (100 * n), mean_ev=E[m_].mean()))
    evt = pd.DataFrame(rows)
    evt.to_csv(ev / "ev_bands.csv", index=False)
    out.append("## 推定期待値の帯ごとの実際の回収率（全買い目）\n" + evt.to_markdown(index=False, floatfmt=".3f"))

    # ---- 3. 市場の偏り: 確率帯 × 型 の実際の回収率（発見用=前期間の OOS） ----
    a = PERMS[:, 0]; bb2 = PERMS[:, 1]; cc = PERMS[:, 2]
    ctype = np.where(a == 0, np.where(np.isin(bb2, [1, 2]) & np.isin(cc, [1, 2, 3]), "1頭-内(2,3)-内", "1頭-その他"),
                     np.where(a == 1, "2頭", np.where(a == 2, "3頭", "4-6頭")))
    rows = []
    for per, sel in [("発見用 2024-05〜2025-10", has & (month >= G_FIRST_MONTH) & (month < TEST_MONTHS[0])), ("検証用 2025-11〜2026-10", tst)]:
        PPs = P[sel]; V = ~void[sel]
        hm = np.zeros(PPs.shape, bool); hm[np.arange(sel.sum()), hit[sel]] = True
        bands = np.digitize(PPs, PBANDS) - 1
        pay_m = hm * payout[sel][:, None]
        for t in np.unique(ctype):
            tm = (ctype == t)[None, :]
            for i in range(len(PBANDS) - 1):
                m_ = (bands == i) & tm & V
                n = m_.sum()
                if n < 200:
                    continue
                rows.append(dict(period=per, type=t, pband=f"{PBANDS[i]:.3f}-{PBANDS[i+1]:.3f}", combos=int(n),
                                 hits=int((hm & m_).sum()), roi=(pay_m * m_).sum() / (100 * n)))
    bias = pd.DataFrame(rows)
    bias.to_csv(ev / "bias.csv", index=False)
    piv = bias.pivot_table(index=["type", "pband"], columns="period", values="roi")
    out.append("## 確率帯×型の回収率（全買い目を買った場合）\n" + piv.to_markdown(floatfmt=".3f"))
    pb = bias.groupby(["period", "pband"]).apply(lambda d: pd.Series({"combos": d.combos.sum(),
                                                 "roi": (d.roi * d.combos).sum() / d.combos.sum()})).unstack(0)
    out.append("## 確率帯ごとの回収率（本命-大穴の偏り）\n" + pb.to_markdown(floatfmt=".3f"))

    # ---- 4. 買い方の候補を月ごとに計算 ----
    sel0 = (month >= G_FIRST_MONTH) & (month < EVAL_FIRST_MONTH) & has
    Hall = entropy(P)
    H_cuts = {q: float(np.quantile(Hall[sel0], q)) for q in [0.1, 0.2, 0.3, 0.5]}
    cands = build_candidates(H_cuts)
    log(f"candidates={len(cands)}")
    pbands_all = np.digitize(P, PBANDS) - 1
    cell_id = np.arange(120)[None, :] * 10 + pbands_all                     # (nr,120)
    # セルの累積成績（月ごと）
    cell_cost = {}; cell_ret = {}; cell_hit = {}
    for m in months_all:
        s_ = (month == m) & has
        ids = cell_id[s_][~void[s_]]
        cc_ = np.bincount(ids, minlength=1200)
        hm = cell_id[s_, hit[s_]]
        cell_cost[m] = cc_; cell_ret[m] = np.bincount(hm, weights=payout[s_], minlength=1200)
        cell_hit[m] = np.bincount(hm, minlength=1200)
    res = []   # (cand, month, races, bets, hits, cost, ret)
    for m in months_eval:
        s_ = np.where((month == m) & has)[0]
        prior = [mm for mm in months_all if G_FIRST_MONTH <= mm < m]
        cc_ = sum(cell_cost[mm] for mm in prior); cr = sum(cell_ret[mm] for mm in prior); ch = sum(cell_hit[mm] for mm in prior)
        roi_cell = cr / np.maximum(cc_ * 100, 1)
        Pm = P[s_]; P1m = P1[s_]
        fav = np.argmax(P1m, 1)
        Om = G[s_] / 100.0
        ctx = dict(O=Om, favO=topk_mask(-Om, 1), P=Pm, EV=EV[s_], H=Hall[s_], p1max=P1m.max(1), favfirst=(PERMS[:, 0][None, :] == fav[:, None]),
                   cell_ok=lambda r, h, cid=cell_id[s_]: ((roi_cell >= r) & (ch >= h))[cid])
        V = ~void[s_]
        hm = np.zeros(Pm.shape, bool); hm[np.arange(len(s_)), hit[s_]] = True
        for name, fam, fn in cands:
            mk = fn(ctx) & V
            nb = mk.sum()
            hits = (mk & hm).any(1)
            res.append((name, fam, m, int(mk.any(1).sum()), int(nb), int(hits.sum()), 100.0 * nb, float(payout[s_][hits].sum())))
        log(f"  strategies {m} done")
    res = pd.DataFrame(res, columns=["cand", "family", "month", "races", "bets", "hits", "cost", "ret"])
    res.to_parquet(ev / "cand_month.parquet")
    report = walkforward_report(res, cands, months_eval)
    out += report
    (ev / "report.md").write_text("\n\n".join(out))
    print("\n\n".join(out))


def summarize(df):
    """月別の行（cost, ret, bets, hits）→ 通算の要約"""
    roi_m = np.where(df.cost > 0, df.ret / df.cost.where(df.cost > 0, 1), np.nan)
    return dict(total_roi=df.ret.sum() / max(df.cost.sum(), 1), win_months=int((roi_m > 1.0).sum()),
                months=len(df), worst=np.nanmin(roi_m) if np.isfinite(roi_m).any() else np.nan,
                bets=int(df.bets.sum()), hits=int(df.hits.sum()), cost=df.cost.sum(), ret=df.ret.sum(),
                bets_per_month=df.bets.sum() / len(df), races_per_month=df.races.sum() / len(df))


def pick_best(res, months, min_hits=30, cands=None):
    """months の通算回収率が最大の候補（的中 min_hits 以上）"""
    d = res[res.month.isin(months)]
    if cands is not None:
        d = d[d.cand.isin(cands)]
    g = d.groupby("cand").agg(cost=("cost", "sum"), ret=("ret", "sum"), hits=("hits", "sum"))
    g = g[(g.hits >= min_hits) & (g.cost > 0)]
    if g.empty:
        return None, np.nan
    g["roi"] = g.ret / g.cost
    best = g.roi.idxmax()
    return best, g.roi.max()


def walkforward_report(res, cands, months_eval):
    out = []
    test = [m for m in months_eval if m in TEST_MONTHS]
    fams = sorted(set(f for _, f, _ in cands))
    # (1) 毎月選び直す: その月の前12か月で最良の候補を、その月に適用
    def rolling(cset, label):
        rows = []
        for m in test:
            i = months_eval.index(m)
            win = months_eval[max(0, i - SELECT_WINDOW):i]
            best, hroi = pick_best(res, win, cands=cset)
            if best is None:
                rows.append(dict(month=m, chosen="(なし)", hist_roi=np.nan, races=0, bets=0, hits=0, cost=0.0, ret=0.0))
                continue
            r = res[(res.cand == best) & (res.month == m)].iloc[0]
            rows.append(dict(month=m, chosen=best, hist_roi=hroi, races=r.races, bets=r.bets, hits=r.hits, cost=r.cost, ret=r.ret))
        d = pd.DataFrame(rows)
        d["roi"] = np.where(d.cost > 0, d.ret / d.cost.where(d.cost > 0, 1), np.nan)
        s = summarize(d)
        return d, s
    allc = [c for c, _, _ in cands]
    d, s = rolling(allc, "全候補")
    d.to_csv(WORK / "eval" / "wf_all.csv", index=False)
    out.append(f"## ウォークフォワード（毎月、直前12か月で全{len(allc)}候補から最良を選んで適用）\n"
               + d.to_markdown(index=False, floatfmt=".3f") + "\n\n" + fmt_summary(s))
    rows = []
    for fam in fams:
        cs = [c for c, f, _ in cands if f == fam]
        d2, s2 = rolling(cs, fam)
        rows.append(dict(family=fam, n_cands=len(cs), **s2))
    fam_t = pd.DataFrame(rows)
    out.append("## 系統ごとのウォークフォワード（系統内で毎月選び直し）\n" + fam_t.to_markdown(index=False, floatfmt=".3f"))
    # (2) 事前に1本固定: 発見用（前期間12か月 2024-11〜2025-10）で最良 → 検証期間に固定で適用
    disc = [m for m in months_eval if m < TEST_MONTHS[0]]
    rows = []
    for fam in fams + ["全候補"]:
        cs = allc if fam == "全候補" else [c for c, f, _ in cands if f == fam]
        best, hroi = pick_best(res, disc, cands=cs)
        if best is None:
            continue
        dd = res[(res.cand == best) & res.month.isin(test)]
        rows.append(dict(family=fam, chosen=best, disc_roi=hroi, **summarize(dd)))
    fx = pd.DataFrame(rows)
    out.append("## 事前に1本固定（発見用 2024-11〜2025-10 で最良 → 検証期間に固定で適用）\n" + fx.to_markdown(index=False, floatfmt=".3f"))
    # (3) 後知恵: 検証期間で最良（成績としては使わない。多重比較の目安）
    g = res[res.month.isin(test)].groupby(["cand", "family"]).apply(lambda x: pd.Series(summarize(x))).reset_index()
    g = g.sort_values("total_roi", ascending=False)
    g.to_csv(WORK / "eval" / "hindsight.csv", index=False)
    n_pass = int(((g.total_roi > 1) & (g.win_months >= 8)).sum())
    out.append(f"## 後知恵（検証期間で直接比べた上位15。{len(g)}候補中、合格ライン相当は {n_pass} 本。成績としては使わない）\n"
               + g.head(15)[["cand", "total_roi", "win_months", "worst", "bets_per_month", "races_per_month", "hits"]].to_markdown(index=False, floatfmt=".3f"))
    # 同じ候補の発見用の成績（後知恵の上位が発見用でも良かったか）
    dsum = res[res.month.isin(disc)].groupby("cand").apply(lambda x: x.ret.sum() / max(x.cost.sum(), 1))
    g["disc_roi"] = g["cand"].map(dsum)
    out.append("発見用ROIと検証用ROIの順位相関（全候補）: "
               f"{g[['disc_roi','total_roi']].corr(method='spearman').iloc[0,1]:.3f}")
    return out


def fmt_summary(s):
    return (f"通算回収率 {s['total_roi']:.3f}、勝ち月 {s['win_months']}/{s['months']}、最悪月 {s['worst']:.3f}、"
            f"購入 {s['bets']}点（月 {s['bets_per_month']:.0f}点 = {s['bets_per_month']*100:,.0f}円、{s['races_per_month']:.0f}レース）、的中 {s['hits']}")


def cmd_odds(work: Path):
    """締切オッズがある期間での参考（前夜・朝8時には使えない情報。本命の結果とは分ける）。"""
    import lightgbm as lgb
    ev = work / "eval"
    ev.mkdir(exist_ok=True)
    races, R, void, nwr, grd = load_eval(work)
    month = races["month"].to_numpy(); hit = races["hit"].to_numpy(); payout = races["payout"].to_numpy()
    con = connect()
    out = []
    key = races.reset_index()[["index", "hd", "jcd", "rno"]]
    # ---- (1) 3連単オッズ（ある日だけ） ----
    od = pd.read_sql("select hd, jcd, rno, combination comb, odds from odds where ticket_type='sanrensho'", con)
    od = od.merge(key, on=["hd", "jcd", "rno"])
    if len(od):
        idx = od["index"].to_numpy()
        ci = combo_index(od["comb"].str.split("-", expand=True).astype(int).to_numpy() - 1)
        O = np.full(R["P"].shape, np.nan); O[idx, ci] = od["odds"].to_numpy()
        rs = np.unique(idx)
        rs = rs[(hit[rs] >= 0) & np.isfinite(O[rs]).all(1)]
        # g をこの月より前の当たり全部で学習
        mo = month[rs[0]]
        Xs, ys = [], []
        for mm in sorted(set(month)):
            if not (G_FIRST_MONTH <= mm < mo):
                continue
            ix = np.where((month == mm) & (hit >= 0))[0]
            X = g_features(races.iloc[ix], R["P"][ix], R["P1"][ix], nwr[ix], grd[ix]).to_numpy().reshape(len(ix), 120, -1)
            Xs.append(X[np.arange(len(ix)), hit[ix]]); ys.append(payout[ix])
        mdl = lgb.train(G_PARAMS, lgb.Dataset(np.concatenate(Xs), np.concatenate(ys)), G_ROUNDS)
        Gr = mdl.predict(g_features(races.iloc[rs], R["P"][rs], R["P1"][rs], nwr[rs], grd[rs]).to_numpy()).reshape(len(rs), 120)
        Pc, _ = recalibrate(R["P"].astype(float), hit, month, G_FIRST_MONTH)
        Pm = Pc[rs]; Om = O[rs]; Oe = Gr / 100.0
        qm = (1 / Om) / (1 / Om).sum(1, keepdims=True)      # 市場の確率（オッズの逆数を正規化）
        hm = np.zeros(Pm.shape, bool); hm[np.arange(len(rs)), hit[rs]] = True
        from scipy.stats import spearmanr
        rho = np.nanmean([spearmanr(Oe[i], Om[i])[0] for i in range(len(rs))])
        fav_ok = (np.argmin(Oe, 1) == np.argmin(Om, 1)).mean()
        def band(o):
            return np.digitize(o, [4, 8, 10, 30])
        conf = pd.crosstab(pd.Series(band(Oe).ravel(), name="予想"), pd.Series(band(Om).ravel(), name="実際"))
        lab = {0: "<4", 1: "4-8", 2: "8-10", 3: "10-30", 4: ">30"}
        conf.index = [lab[i] for i in conf.index]; conf.columns = [lab[i] for i in conf.columns]
        lines = [f"3連単オッズがあるのは {len(rs)} レース（{sorted(set(races.hd.to_numpy()[rs]))}）。"
                 f"払戻率（Σ1/オッズ の逆数）の平均 {np.mean(1/(1/Om).sum(1)):.3f}",
                 f"- 予想オッズ（前夜の推定配当 g/100）と締切オッズの相関（log、全{Om.size}買い目）: {np.corrcoef(np.log(Oe).ravel(), np.log(Om).ravel())[0,1]:.3f}"
                 f"、レース内の順位相関の平均 {rho:.3f}、1番人気の一致 {fav_ok:.1%}",
                 f"- 予想オッズ/締切オッズ の比の中央値 {np.median(Oe/Om):.2f}（10%点 {np.quantile(Oe/Om,.1):.2f}、90%点 {np.quantile(Oe/Om,.9):.2f}）",
                 f"- モデル確率と市場確率の相関（log）: {np.corrcoef(np.log(Pm).ravel(), np.log(qm).ravel())[0,1]:.3f}",
                 f"- 対数損失: モデル {-np.log(Pm[hm]).mean():.3f} / 市場 {-np.log(qm[hm]).mean():.3f} / 一様 {math.log(120):.3f}",
                 "", "配当帯の当たり方（行=予想オッズ、列=締切オッズ、買い目数）", conf.to_markdown(), ""]
        H_ = entropy(Pm); fav = np.argmax(Pm.reshape(len(rs), 6, 20).sum(2), 1)
        rows = []
        for nm, fam, fn in build_candidates({}):
            if not nm.startswith(("C:", "H:", "I:", "J:")):
                continue
            r_ = {"cand": nm}
            for lbl, OO in [("予想オッズ", Oe), ("締切オッズ", Om)]:
                ctx = dict(O=OO, favO=topk_mask(-OO, 1), P=Pm, EV=Pm * OO, H=H_, p1max=None,
                           favfirst=(PERMS[:, 0][None, :] == fav[:, None]))
                mk = fn(ctx) & ~void[rs]
                n = mk.sum(); ret = (mk & hm).any(1) @ payout[rs]
                r_[f"{lbl}:点"] = int(n); r_[f"{lbl}:的中"] = int((mk & hm).sum()); r_[f"{lbl}:回収率"] = ret / max(100 * n, 1)
            rows.append(r_)
        lines.append("同じ買い方を、予想オッズで選んだ場合と締切オッズで選んだ場合（168レース・1日だけなので参考の参考）")
        lines.append(pd.DataFrame(rows).to_markdown(index=False, floatfmt=".3f"))
        out.append("## 3連単の締切オッズ（参考）\n" + "\n".join(lines))
    # ---- (2) 単勝オッズ（2026-09-08〜）: 1着確率の比較と単勝の参考 ----
    td = pd.read_sql("select hd, jcd, rno, cast(combination as int) lane, odds from odds where ticket_type='tansho'", con)
    td = td.merge(key, on=["hd", "jcd", "rno"])
    Q = np.full((len(races), 6), np.nan); Q[td["index"].to_numpy(), td["lane"].to_numpy() - 1] = td["odds"].to_numpy()
    tp = pd.read_sql("select hd, jcd, rno, cast(combination as int) lane, payout from payouts where ticket_type='tansho' and hd>='20260901'", con)
    tp = tp.merge(key, on=["hd", "jcd", "rno"]).drop_duplicates("index")
    win = np.full(len(races), -1); win[tp["index"].to_numpy()] = tp["lane"].to_numpy() - 1
    tpay = np.zeros(len(races)); tpay[tp["index"].to_numpy()] = tp["payout"].to_numpy()
    rs = np.where(np.isfinite(Q).all(1) & (win >= 0) & (month <= TEST_MONTHS[-1]))[0]
    if len(rs):
        P1 = R["P1"][rs].astype(float); Qm = Q[rs]
        q = (1 / Qm) / (1 / Qm).sum(1, keepdims=True)
        w = win[rs]
        lines = [f"単勝オッズがあるレース（検証期間内）: {len(rs)}（{races.hd.to_numpy()[rs].min()}〜{races.hd.to_numpy()[rs].max()}）",
                 f"- 1着の対数損失: モデル {-np.log(P1[np.arange(len(rs)), w]).mean():.4f} / 締切単勝オッズ {-np.log(q[np.arange(len(rs)), w]).mean():.4f}"]
        for al in [0.25, 0.5, 0.75]:
            bl = np.exp(al * np.log(P1) + (1 - al) * np.log(q)); bl /= bl.sum(1, keepdims=True)
            lines.append(f"  - 混合（モデル {al:.2f}・市場 {1-al:.2f}、対数で）: {-np.log(bl[np.arange(len(rs)), w]).mean():.4f}")
        for tau in [1.0, 1.1, 1.2, 1.5]:
            mk = P1 * Qm >= tau
            ret = (mk[np.arange(len(rs)), w] * tpay[rs]).sum()
            lines.append(f"- 単勝で モデル確率×締切オッズ ≥{tau}: {mk.sum()}点 回収率 {ret/max(mk.sum()*100,1):.3f}")
        # 3連単: 市場の1着確率を混ぜた確率で、前夜の推定配当 g を使った期待値（オッズの情報の上積みの目安）
        Gp = np.load(ev / "g_pred.npy")[rs]
        Pm = R["P"][rs].astype(float)
        bl = np.exp(0.5 * np.log(P1) + 0.5 * np.log(q)); bl /= bl.sum(1, keepdims=True)
        Padj = Pm * (bl / P1)[:, PERMS[:, 0]]           # 1着の確率だけ市場と混ぜた版に置き換え
        Padj /= Padj.sum(1, keepdims=True)
        hm = np.zeros(Pm.shape, bool); hs = hit[rs] >= 0
        hm[np.where(hs)[0], hit[rs][hs]] = True
        V = ~void[rs]
        for tau in [1.2, 1.5, 2.0]:
            for nm, PP in [("前夜のみ", Pm), ("1着に単勝オッズを混ぜる", Padj)]:
                mk = (PP * Gp / 100 >= tau) & V & hs[:, None]
                lines.append(f"- 3連単 期待値≥{tau}（{nm}）: {mk.sum()}点 回収率 {(mk & hm).sum(1) @ payout[rs] / max(mk.sum()*100,1):.3f}")
        out.append("## 単勝の締切オッズ（参考）\n" + "\n".join(lines))
    (ev / "odds.md").write_text("\n\n".join(out))
    print("\n\n".join(out))


# ============================================================== 単勝（同じ1着モデルで、前夜の推定配当と締切オッズの差を大きな標本で見る）
def cmd_tansho(work: Path):
    """3連単の締切オッズは1日分しかないので、単勝（締切オッズが 2026-07 末から数か月ある）で
    「前夜の推定配当で選ぶ」と「締切オッズで選ぶ（上限の参考）」の差を測る。1着確率は3連単と同じモデル。"""
    import lightgbm as lgb
    ev = work / "eval"; ev.mkdir(exist_ok=True)
    races, R, void, nwr, grd = load_eval(work)
    month = races["month"].to_numpy()
    con = connect()
    key = races.reset_index()[["index", "hd", "jcd", "rno"]]
    tp = pd.read_sql("select hd, jcd, rno, cast(combination as int) lane, payout from payouts where ticket_type='tansho' "
                     "and hd >= ?", con, params=(races.hd.min(),)).merge(key, on=["hd", "jcd", "rno"])
    TP = np.zeros((len(races), 6)); TP[tp["index"].to_numpy(), tp["lane"].to_numpy() - 1] = tp["payout"].to_numpy()
    has = TP.sum(1) > 0
    win = np.where(has, np.argmax(TP > 0, 1), -1)
    vd = load_void(con, races.hd.min(), races.hd.max()).merge(key, on=["hd", "jcd", "rno"])
    LV = np.zeros((len(races), 6), bool); LV[vd["index"].to_numpy(), vd["lane"].to_numpy().astype(int) - 1] = True
    P1, al = recalibrate(R["P1"].astype(float), win, month, G_FIRST_MONTH)
    r1 = (-P1).argsort(1).argsort(1) + 1
    nrk = (-np.nan_to_num(nwr, nan=0)).argsort(1).argsort(1) + 1
    H1 = -(P1 * np.log(P1)).sum(1)

    def feats(ix):
        n = len(ix); rep = lambda x: np.repeat(np.asarray(x, float), 6)
        cols = {"lane": np.tile(np.arange(1, 7), n), "lp": np.log(P1[ix]).ravel(), "r1": r1[ix].ravel(),
                "nw": nwr[ix].ravel(), "g": grd[ix].ravel(), "nrk": nrk[ix].ravel(), "H1": rep(H1[ix]),
                "p1max": rep(P1[ix].max(1)), "nw1": rep(nwr[ix, 0]), "nwmax_o": rep(np.nanmax(nwr[ix, 1:], 1)),
                "jcd": rep(races["jcd"].to_numpy()[ix].astype(int)), "rno": rep(races["rno"].to_numpy()[ix])}
        for c in RACE_FLAGS:
            cols[c] = rep(races[c].to_numpy()[ix])
        return pd.DataFrame(cols).to_numpy(np.float32)
    months_eval = [m for m in sorted(set(month)) if EVAL_FIRST_MONTH <= m <= TEST_MONTHS[-1]]
    G1 = np.full((len(races), 6), np.nan)
    for m in months_eval:
        tr = np.where((month >= G_FIRST_MONTH) & (month < m) & has)[0]
        X = feats(tr).reshape(len(tr), 6, -1)[np.arange(len(tr)), win[tr]]
        mdl = lgb.train(G_PARAMS, lgb.Dataset(X, TP[tr, win[tr]]), G_ROUNDS)
        te = np.where(month == m)[0]
        G1[te] = mdl.predict(feats(te)).reshape(len(te), 6)
    log("tansho g fitted")
    EV1 = P1 * G1 / 100
    rows = []
    cands = [(t, pm) for t in [1.0, 1.1, 1.2, 1.3, 1.5] for pm in [0.0, 0.1, 0.2, 0.3, 0.5]]
    for m in months_eval:
        s_ = np.where((month == m) & has)[0]
        for t, pm in cands:
            mk = (EV1[s_] >= t) & (P1[s_] >= pm) & ~LV[s_]
            rows.append(dict(cand=f"単勝:EV>={t}:p>={pm}", month=m, races=int(mk.any(1).sum()), bets=int(mk.sum()),
                             hits=int(mk[np.arange(len(s_)), win[s_]].sum()), cost=100.0 * mk.sum(),
                             ret=float((mk * TP[s_]).sum())))
    res = pd.DataFrame(rows)
    res.to_parquet(ev / "tansho_cand_month.parquet")
    out = []
    test = [m for m in months_eval if m in TEST_MONTHS]
    disc = [m for m in months_eval if m < TEST_MONTHS[0]]
    # 前夜の推定配当の当てはまり
    tst = np.isin(month, TEST_MONTHS) & has
    gh = G1[tst, win[tst]]; ah = TP[tst, win[tst]]
    out.append(f"推定配当（単勝）: log相関 {np.corrcoef(np.log(gh), np.log(ah))[0,1]:.3f}")
    # ウォークフォワード（毎月選び直し）と事前固定
    wf = []
    for m in test:
        i = months_eval.index(m)
        best, hroi = pick_best(res, months_eval[max(0, i - SELECT_WINDOW):i])
        r = res[(res.cand == best) & (res.month == m)].iloc[0]
        wf.append(dict(month=m, chosen=best, hist_roi=hroi, races=r.races, bets=r.bets, hits=r.hits, cost=r.cost, ret=r.ret, roi=r.ret / max(r.cost, 1)))
    wf = pd.DataFrame(wf)
    out.append(f"## 単勝: ウォークフォワード（{len(cands)}候補から毎月選び直し）\n" + wf.to_markdown(index=False, floatfmt=".3f")
               + "\n\n" + fmt_summary(summarize(wf)))
    best, hroi = pick_best(res, disc)
    out.append(f"## 単勝: 発見用で1本固定 → 検証期間\n{best}（発見用 {hroi:.3f}）: " + fmt_summary(summarize(res[(res.cand == best) & res.month.isin(test)])))
    g = res[res.month.isin(test)].groupby("cand").apply(lambda x: pd.Series(summarize(x)))
    gd = res[res.month.isin(disc)].groupby("cand").apply(lambda x: x.ret.sum() / max(x.cost.sum(), 1))
    g["disc_roi"] = gd
    out.append("## 単勝: 全候補（発見用と検証用）\n" + g[["disc_roi", "total_roi", "win_months", "worst", "bets_per_month", "hits"]].to_markdown(floatfmt=".3f"))
    # 締切オッズとの比較（オッズがある期間）
    od = pd.read_sql("select hd, jcd, rno, cast(combination as int) lane, odds from odds where ticket_type='tansho'", con).merge(key, on=["hd", "jcd", "rno"])
    Q = np.full((len(races), 6), np.nan); Q[od["index"].to_numpy(), od["lane"].to_numpy() - 1] = od["odds"].to_numpy()
    rs = np.where(np.isfinite(Q).all(1) & has & np.isfinite(G1).all(1))[0]
    lines = [f"締切オッズがあるレース: {len(rs)}（{races.hd.to_numpy()[rs].min()}〜{races.hd.to_numpy()[rs].max()}）",
             f"- 予想オッズ（推定配当/100）と締切オッズの相関（log、0倍を除く）: "
             f"{np.corrcoef(np.log(G1[rs][Q[rs] > 0]), np.log(Q[rs][Q[rs] > 0] * 100))[0,1]:.3f}、1番人気の一致 {(np.argmin(G1[rs],1)==np.argmin(np.where(Q[rs]>0,Q[rs],1e9),1)).mean():.1%}"]
    qv = np.where(Q[rs] > 0, 1 / np.where(Q[rs] > 0, Q[rs], 1), 0); qv /= qv.sum(1, keepdims=True)
    w_ = win[rs]
    lines.append(f"- 1着の対数損失: モデル {-np.log(P1[rs, w_]).mean():.4f} / 締切オッズ（1/オッズを正規化） {-np.log(np.clip(qv[np.arange(len(rs)), w_], 1e-9, None)).mean():.4f}")
    tbl = []
    for t, pm in [(1.0, 0.0), (1.2, 0.0), (1.5, 0.0), (1.2, 0.2), (1.2, 0.5)]:
        r_ = {"買い方": f"EV>={t}, p>={pm}"}
        for lbl, OO in [("予想オッズ", G1[rs] / 100), ("締切オッズ", Q[rs])]:
            mk = (P1[rs] * OO >= t) & (P1[rs] >= pm) & ~LV[rs] & (Q[rs] > 0)
            r_[f"{lbl}:点"] = int(mk.sum()); r_[f"{lbl}:回収率"] = (mk * TP[rs]).sum() / max(100 * mk.sum(), 1)
        tbl.append(r_)
    mo = races.month.to_numpy()[rs]
    for lbl, OO in [("予想オッズ", G1[rs] / 100), ("締切オッズ", Q[rs])]:
        mk = (P1[rs] * OO >= 1.2) & ~LV[rs] & (Q[rs] > 0)
        mm = {m: f"{(mk[mo==m]*TP[rs][mo==m]).sum()/max(100*mk[mo==m].sum(),1):.3f}" for m in sorted(set(mo))}
        lines.append(f"- EV>=1.2 の月別回収率（{lbl}で選ぶ）: {mm}")
    out.append("## 単勝: 予想オッズで選ぶ vs 締切オッズで選ぶ（上限の参考）\n" + "\n".join(lines) + "\n\n" + pd.DataFrame(tbl).to_markdown(index=False, floatfmt=".3f"))
    (ev / "tansho.md").write_text("\n\n".join(out))
    print("\n\n".join(out))


# ============================================================== main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["features", "predict", "evaluate", "odds", "tansho", "all"])
    ap.add_argument("--workdir", default=str(DEFAULT_WORK))
    ap.add_argument("--only", help="predict: この学習開始月だけ（カンマ区切り）")
    a = ap.parse_args(argv)
    global WORK
    work = WORK = Path(a.workdir)
    work.mkdir(parents=True, exist_ok=True)
    if a.cmd in ("features", "all"):
        cmd_features(work)
    if a.cmd in ("predict", "all"):
        cmd_predict(work, a.only.split(",") if a.only else None)
    if a.cmd in ("evaluate", "all"):
        cmd_evaluate(work)
    if a.cmd in ("odds", "all"):
        cmd_odds(work)
    if a.cmd in ("tansho", "all"):
        cmd_tansho(work)
    return 0


if __name__ == "__main__":
    sys.exit(main())
