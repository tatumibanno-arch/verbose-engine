"""1着確率モデルと予想単勝配当（運用用）。

scripts/analysis/trifecta.py の「単勝」（cmd_tansho）と同じ作り方を、毎晩の選定で使えるように切り出したもの。
  * 1着確率: 特徴量（features.build_features, stats_lag_days は rules.yaml）→ LightGBM 2値分類（trifecta.py の
    StageModel の1段目と同じ設定）→ レース内 softmax(τ・スコア)。τ は学習窓の末尾2か月で決める。
    さらに月ごとの再校正 P ∝ P^α（α はその月より前の OOS 月で決める。分析では α≈1.0）。
  * 予想単勝配当 g: 「その艇が1着になったときの単勝払戻」の期待値を、過去の OOS 予測と単勝払戻
    （当たった艇の行だけ）で LightGBM gamma 回帰。期待値 EV = 1着確率 × g / 100。
  * 学習は「月ごと・その月より前の24か月」（分析と同じ）。学習済みは data/models/<dir>/<YYYY-MM>/ に保存。
  * 予想配当の学習に使う「過去の OOS 予測」は data/models/<dir>/history.parquet（1レース1行）に貯める。
    毎晩の予測をそこに追記する。初期値は分析のウォークフォワード予測（seed-history）。

使い方:
    python scripts/model.py ensure --date 20261103      # その月の学習済みが無ければ学習（nightly 用）
    python scripts/model.py train --month 2026-11       # 学習し直す（既存を上書き）
    python scripts/model.py predict --date 20261103     # 1日分の1着確率・予想配当・EV を表示（確認用）
    python scripts/model.py seed-history --workdir <trifecta.py の作業ディレクトリ>   # 初回だけ
    python scripts/model.py backtest --from 202411 --to 202610   # history の OOS 予測にルールを当てる

設定（学習窓・stats_lag_days など）は config/rules.yaml の strategies.<名前>.model。
LightGBM のハイパーパラメータはモデルの定義なので、ここ（trifecta.py と同じ値）に置く。
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_DIR, DB_PATH, load_rules  # noqa: E402

MODELS_DIR = DATA_DIR / "models"
DEFAULT_STRATEGY = "tansho_ev_v1"

# ---- モデルの定義（trifecta.py と同じ値。変えると分析の数字と比べられなくなる） ----
# スレッド数だけは環境に合わせる（分析は4。コア数ちょうどで他のプロセスと取り合うと LightGBM は数十倍遅くなる
# ことがあったので、既定は「コア数-1（最大4）」。結果は変わらないことを確かめた）。環境変数 MODEL_THREADS で上書き可。
THREADS = int(os.environ.get("MODEL_THREADS") or min(4, max(1, (os.cpu_count() or 2) - 1)))
ID_COLS = ["hd", "jcd", "rno", "lane", "racer_no", "motor_no", "boat_no"]
RACE_FLAGS = ["r_final", "r_semifinal", "r_special", "r_qualify", "r_fixed_course", "race_n_a1", "close_hour"]
WIN_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=300,
                  feature_fraction=0.6, bagging_fraction=0.7, bagging_freq=1, lambda_l2=10.0,
                  max_bin=127, num_threads=THREADS, verbose=-1, seed=7)
WIN_MAX_ROUNDS = 3000
WIN_EARLY_STOP = 100
TAU_GRID = np.linspace(0.3, 2.0, 35)
PAYOUT_PARAMS = dict(objective="gamma", learning_rate=0.03, num_leaves=31, min_data_in_leaf=100,
                     feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0,
                     num_threads=THREADS, verbose=-1, seed=11)
PAYOUT_ROUNDS = 600
ALPHA_GRID = np.arange(0.6, 1.31, 0.05)
FEATURE_CHUNK_MONTHS = 6
LOCK_RETRIES = 10            # DB が別プロセスの書き込みで locked のときの取り直し回数
LOCK_WAIT_S = 30     # 学習用の特徴量を何か月ずつ作るか（メモリのため）

HIST_KEYS = ["hd", "jcd", "rno"]
HIST_COLS = (HIST_KEYS + ["model_month", "alpha"] + [f"p1r_{i}" for i in range(1, 7)]
             + [f"g_{i}" for i in range(1, 7)] + [f"nwr_{i}" for i in range(1, 7)]
             + [f"grd_{i}" for i in range(1, 7)] + RACE_FLAGS)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def connect_ro(timeout: int = 300) -> sqlite3.Connection:
    """読み取り専用。DB が無ければ db.connect() で Parquet から組み立ててから開く。"""
    if not DB_PATH.exists():
        from db import connect
        connect().close()
    return sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, timeout=timeout)


def strategy_cfg(name: str = DEFAULT_STRATEGY, rules: dict | None = None) -> dict:
    rules = rules or load_rules()
    s = (rules.get("strategies") or {}).get(name)
    if not s:
        raise KeyError(f"rules.yaml に strategies.{name} がありません")
    return s


def model_root(cfg: dict) -> Path:
    return MODELS_DIR / cfg["model"]["dir"]


# ============================================================== 日付
def month_of(hd) -> np.ndarray:
    hd = pd.Series(hd).astype(str)
    return (hd.str[:4] + "-" + hd.str[4:6]).to_numpy()


def add_months(m: str, k: int) -> str:
    y, mo = int(m[:4]), int(m[5:7])
    t = y * 12 + (mo - 1) + k
    return f"{t // 12}-{t % 12 + 1:02d}"


def month_first_hd(m: str) -> str:
    return m.replace("-", "") + "01"


def month_last_hd(m: str) -> str:
    return (pd.Timestamp(month_first_hd(add_months(m, 1))) - pd.Timedelta(days=1)).strftime("%Y%m%d")


# ============================================================== 特徴量
def build_frame(con, hd_from: str, hd_to: str, mcfg: dict, with_labels: bool) -> pd.DataFrame:
    """features.build_features → 6艇そろったレースだけ・並び順・float32（trifecta.load_features と同じ形）"""
    from features import build_features
    for attempt in range(LOCK_RETRIES + 1):
        try:
            df = build_features(con, hd_from, hd_to, mcfg["stats_lag_days"], with_labels=with_labels,
                                hist_days=mcfg.get("feature_history_days"))
            break
        except Exception as e:  # 別プロセスが書き込み中だと読み取りでも locked が返ることがある
            if "locked" not in str(e) or attempt == LOCK_RETRIES:
                raise
            log(f"  database is locked → {LOCK_WAIT_S}s 待って取り直し（{attempt + 1}/{LOCK_RETRIES}）")
            time.sleep(LOCK_WAIT_S)
    if df.empty:
        return df
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


def build_frame_months(con, m_from: str, m_to: str, mcfg: dict) -> pd.DataFrame:
    parts = []
    m = m_from
    while m <= m_to:
        m_end = min(add_months(m, FEATURE_CHUNK_MONTHS - 1), m_to)
        t0 = time.time()
        parts.append(build_frame(con, month_first_hd(m), month_last_hd(m_end), mcfg, with_labels=True))
        log(f"  features {m}..{m_end}: races={len(parts[-1]) // 6} ({time.time() - t0:.0f}s)")
        m = add_months(m_end, 1)
    df = pd.concat(parts, ignore_index=True)
    # チャンクの境目で dtype が揃わないことがあるので、そろえる
    for c in df.columns:
        if df[c].dtype == np.float64:
            df[c] = df[c].astype(np.float32)
    return df


def feature_cols(df: pd.DataFrame) -> list[str]:
    """trifecta.feature_cols と同じ並び。"""
    return [c for c in df.columns if c not in ID_COLS and not c.startswith("y_") and c not in ("month",)] + ["lane", "rno"]


# ============================================================== 1着確率
def softmax_rows(s: np.ndarray) -> np.ndarray:
    s = s - np.nanmax(np.where(np.isfinite(s), s, -1e30), axis=1, keepdims=True)
    e = np.where(np.isfinite(s), np.exp(s), 0.0)
    return e / e.sum(1, keepdims=True)


def fit_tau(S, y_idx, taus=TAU_GRID):
    best = None
    for t in taus:
        P = softmax_rows(S * t)
        ll = -np.log(np.clip(P[np.arange(len(S)), y_idx], 1e-12, None)).mean()
        if best is None or ll < best[1]:
            best = (t, ll)
    return float(best[0])


def fit_win_model(tr: pd.DataFrame, va: pd.DataFrame, feats: list[str]):
    """trifecta.StageModel.fit の1段目と同じ。返り値 (booster, tau)。"""
    import lightgbm as lgb
    X = tr[feats].to_numpy(np.float32)
    Xv = va[feats].to_numpy(np.float32)
    pos = tr["y_pos"].to_numpy().reshape(-1, 6)
    posv = va["y_pos"].to_numpy().reshape(-1, 6)
    ok = (pos == 1).sum(1) == 1
    okv = (posv == 1).sum(1) == 1
    F = X.shape[1]
    dtr = lgb.Dataset(X.reshape(-1, 6, F)[ok].reshape(-1, F), (pos[ok] == 1).ravel(), free_raw_data=True)
    dva = lgb.Dataset(Xv.reshape(-1, 6, F)[okv].reshape(-1, F), (posv[okv] == 1).ravel(), reference=dtr)
    m = lgb.train(WIN_PARAMS, dtr, WIN_MAX_ROUNDS, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(WIN_EARLY_STOP, verbose=False)])
    S1v = m.predict(Xv, raw_score=True).reshape(-1, 6)
    tau = fit_tau(S1v[okv], np.argmax(posv[okv] == 1, 1))
    return m, tau


def predict_p1_raw(booster, tau: float, feats: list[str], df: pd.DataFrame) -> np.ndarray:
    """(nr, 6) の1着確率（再校正前）。分析と同じく float32 で持つ。"""
    X = df[feats].to_numpy(np.float32)
    S = booster.predict(X, raw_score=True).reshape(-1, 6)
    return softmax_rows(S * tau).astype(np.float32)


def fit_alpha(P_raw: np.ndarray, win: np.ndarray, min_races: int) -> float:
    """P ∝ P^α の α（対数損失最小）。trifecta.recalibrate と同じ。標本が少なければ 1.0。"""
    tr = win >= 0
    if tr.sum() < min_races:
        return 1.0
    lp = np.log(np.clip(P_raw[tr].astype(float), 1e-12, None))
    h = win[tr]
    best = None
    for g_ in ALPHA_GRID:
        ll = -(g_ * lp[np.arange(len(h)), h] - np.log(np.exp(g_ * lp).sum(1))).mean()
        if best is None or ll < best[1]:
            best = (g_, ll)
    return float(best[0])


def apply_alpha(P_raw: np.ndarray, alpha) -> np.ndarray:
    a = np.asarray(alpha, dtype=float)
    if a.ndim == 1:
        a = a[:, None]
    q = np.clip(P_raw.astype(float), 1e-12, None) ** a
    return q / q.sum(1, keepdims=True)


# ============================================================== 予想単勝配当
def payout_features(P1: np.ndarray, nwr: np.ndarray, grd: np.ndarray, flags: np.ndarray,
                    jcd: np.ndarray, rno: np.ndarray) -> np.ndarray:
    """trifecta.cmd_tansho の feats() と同じ列・同じ順（1艇1行、(nr*6, 19)）。
    P1 は再校正後、nwr/grd は (nr,6)、flags は (nr, len(RACE_FLAGS))。"""
    n = len(P1)
    r1 = (-P1).argsort(1).argsort(1) + 1
    nrk = (-np.nan_to_num(nwr, nan=0)).argsort(1).argsort(1) + 1
    H1 = -(P1 * np.log(P1)).sum(1)
    rep = lambda x: np.repeat(np.asarray(x, float), 6)  # noqa: E731
    with np.errstate(all="ignore"):
        nwmax_o = np.nanmax(nwr[:, 1:], 1)
    cols = {"lane": np.tile(np.arange(1, 7), n), "lp": np.log(P1).ravel(), "r1": r1.ravel(),
            "nw": nwr.ravel(), "g": grd.ravel(), "nrk": nrk.ravel(), "H1": rep(H1),
            "p1max": rep(P1.max(1)), "nw1": rep(nwr[:, 0]), "nwmax_o": rep(nwmax_o),
            "jcd": rep(np.asarray(jcd).astype(int)), "rno": rep(rno)}
    for i, c in enumerate(RACE_FLAGS):
        cols[c] = rep(flags[:, i])
    return pd.DataFrame(cols).to_numpy(np.float32)


def tansho_winners(con, keys: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """keys(hd,jcd,rno) の順に (単勝払戻 (nr,6), 1着艇 0始まり or -1)。払戻表の単勝で判定（分析と同じ）。"""
    if keys.empty:
        return np.zeros((0, 6)), np.zeros(0, int)
    k = keys[HIST_KEYS].reset_index(drop=True).copy()
    k["i"] = np.arange(len(k))
    tp = pd.read_sql("select hd, jcd, rno, cast(combination as int) lane, payout from payouts "
                     "where ticket_type='tansho' and hd between ? and ?", con,
                     params=(k.hd.min(), k.hd.max())).merge(k, on=HIST_KEYS)
    tp = tp[tp["lane"].between(1, 6)]
    TP = np.zeros((len(k), 6))
    TP[tp["i"].to_numpy(), tp["lane"].to_numpy() - 1] = tp["payout"].to_numpy()
    has = TP.sum(1) > 0
    win = np.where(has, np.argmax(TP > 0, 1), -1)
    return TP, win


def hist_arrays(h: pd.DataFrame):
    g = lambda p: h[[f"{p}_{i}" for i in range(1, 7)]].to_numpy(float)  # noqa: E731
    return g("p1r"), g("nwr"), g("grd"), h[RACE_FLAGS].to_numpy(float)


def fit_payout_model(h: pd.DataFrame, win: np.ndarray, TP: np.ndarray):
    """h: history の行（再校正後の P1 を作るため alpha 列を使う）。当たった艇の行だけで gamma 回帰。"""
    import lightgbm as lgb
    P1r, nwr, grd, fl = hist_arrays(h)
    P1 = apply_alpha(P1r, h["alpha"].to_numpy(float))
    tr = np.where(win >= 0)[0]
    X = payout_features(P1[tr], nwr[tr], grd[tr], fl[tr], h["jcd"].to_numpy()[tr], h["rno"].to_numpy()[tr])
    X = X.reshape(len(tr), 6, -1)[np.arange(len(tr)), win[tr]]
    return lgb.train(PAYOUT_PARAMS, lgb.Dataset(X, TP[tr, win[tr]]), PAYOUT_ROUNDS), len(tr)


def predict_payout(booster, P1, nwr, grd, fl, jcd, rno) -> np.ndarray:
    return booster.predict(payout_features(P1, nwr, grd, fl, jcd, rno)).reshape(len(P1), 6)


# ============================================================== 保存
def _save_booster(b, path: Path):
    with gzip.open(path, "wt", encoding="utf-8", compresslevel=9) as f:
        f.write(b.model_to_string())


def _load_booster(path: Path):
    import lightgbm as lgb
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return lgb.Booster(model_str=f.read())


class MonthModel:
    """data/models/<dir>/<YYYY-MM>/ の学習済み一式。"""

    def __init__(self, path: Path):
        self.path = path
        self.meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        self.win = _load_booster(path / "win.txt.gz")
        self.payout = _load_booster(path / "payout.txt.gz")
        self.feats = self.meta["features"]
        self.tau = self.meta["tau"]
        self.alpha = self.meta["alpha"]

    @staticmethod
    def exists(cfg: dict, month: str, root: Path | None = None) -> bool:
        return ((root or model_root(cfg)) / month / "meta.json").exists()

    @classmethod
    def load(cls, cfg: dict, month: str, root: Path | None = None) -> "MonthModel":
        return cls((root or model_root(cfg)) / month)

    def predict_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """features の DataFrame（6艇そろい）→ 1レース1行の history 形式（p1r, g, nwr, grd, flags）。"""
        P1r = predict_p1_raw(self.win, self.tau, self.feats, df)
        P1 = apply_alpha(P1r, self.alpha)
        k = df[HIST_KEYS].iloc[::6].reset_index(drop=True)
        nwr = df["national_win_rate"].to_numpy(float).reshape(-1, 6)
        grd = df["grade"].to_numpy(float).reshape(-1, 6)
        fl = df.iloc[::6][RACE_FLAGS].to_numpy(float)
        G = predict_payout(self.payout, P1, nwr, grd, fl, k["jcd"].to_numpy(), k["rno"].to_numpy())
        return make_hist_rows(k, self.meta["month"], self.alpha, P1r, G, nwr, grd, fl)


def make_hist_rows(k, model_month, alpha, P1r, G, nwr, grd, fl) -> pd.DataFrame:
    out = k[HIST_KEYS].reset_index(drop=True).copy()
    out["rno"] = out["rno"].astype(int)
    out["model_month"] = model_month
    out["alpha"] = np.broadcast_to(np.asarray(alpha, float), (len(out),)).astype(np.float64)
    for p, A in [("p1r", P1r), ("g", G), ("nwr", nwr), ("grd", grd)]:
        for i in range(6):
            out[f"{p}_{i + 1}"] = np.asarray(A, float)[:, i].astype(np.float32)
    for i, c in enumerate(RACE_FLAGS):
        out[c] = np.asarray(fl, float)[:, i].astype(np.float32)
    return out[HIST_COLS]


# ============================================================== history（過去の OOS 予測）
def hist_path(cfg: dict, root: Path | None = None) -> Path:
    return (root or model_root(cfg)) / "history.parquet"


def load_history(cfg: dict, root: Path | None = None) -> pd.DataFrame:
    p = hist_path(cfg, root)
    if not p.exists():
        return pd.DataFrame(columns=HIST_COLS)
    return pd.read_parquet(p)


def save_history(cfg: dict, h: pd.DataFrame, root: Path | None = None) -> None:
    p = hist_path(cfg, root)
    p.parent.mkdir(parents=True, exist_ok=True)
    h = h.drop_duplicates(HIST_KEYS, keep="last").sort_values(HIST_KEYS).reset_index(drop=True)
    h.to_parquet(p, index=False, compression="zstd", compression_level=19)


def upsert_history(cfg: dict, rows: pd.DataFrame, root: Path | None = None) -> None:
    h = load_history(cfg, root)
    if len(h):
        h = h[~h["hd"].isin(set(rows["hd"]))]
    save_history(cfg, pd.concat([h, rows], ignore_index=True) if len(h) else rows, root)


# ============================================================== 学習
def train_month(cfg: dict, month: str, root: Path | None = None, con=None) -> dict:
    """month（YYYY-MM）の運用モデルを学習して保存。学習に使うのは month より前のデータだけ。"""
    mcfg = cfg["model"]
    root = root or model_root(cfg)
    con = con or connect_ro()
    t_all = time.time()
    timing = {}
    tr_lo = add_months(month, -mcfg["train_months"])
    va_lo = add_months(month, -mcfg["valid_months"])
    last = add_months(month, -1)
    log(f"train {month}: features {tr_lo}..{last}")
    t0 = time.time()
    df = build_frame_months(con, tr_lo, last, mcfg)
    timing["features_s"] = round(time.time() - t0, 1)
    feats = feature_cols(df)
    months = df["month"].to_numpy()
    tr = df[(months >= tr_lo) & (months < va_lo)]
    va = df[(months >= va_lo) & (months < month)]
    tr = tr[tr["y_has_result"].to_numpy().astype(bool)]
    va = va[va["y_has_result"].to_numpy().astype(bool)]
    log(f"  win model: train races={len(tr) // 6} valid races={len(va) // 6} feats={len(feats)}")
    t0 = time.time()
    win, tau = fit_win_model(tr, va, feats)
    timing["win_model_s"] = round(time.time() - t0, 1)
    log(f"  win model iters={win.best_iteration} tau={tau:.2f} ({timing['win_model_s']}s)")
    del df, tr, va

    # history の穴埋め（前月までで、nightly が予測を残さなかった日）。その月のモデルがあれば使う
    t0 = time.time()
    fill_history_gaps(cfg, con, add_months(month, -1), root)
    timing["fill_history_s"] = round(time.time() - t0, 1)

    # 再校正 α と 予想配当モデル（history の month より前の月で）
    h = load_history(cfg, root)
    hm = month_of(h["hd"]) if len(h) else np.array([])
    TP, winner = tansho_winners(con, h)
    prior = sorted(m for m in set(hm) if mcfg["recalib_from"] <= m < month)[-mcfg["recalib_window_months"]:]
    sel = np.isin(hm, prior)
    alpha = fit_alpha(h[[f"p1r_{i}" for i in range(1, 7)]].to_numpy(float)[sel], winner[sel],
                      mcfg["recalib_min_races"])
    t0 = time.time()
    selg = (hm >= mcfg["payout_train_from"]) & (hm < month)
    payout, n_hits = fit_payout_model(h[selg].reset_index(drop=True), winner[selg], TP[selg])
    timing["payout_model_s"] = round(time.time() - t0, 1)
    log(f"  alpha={alpha:.2f} payout model: hits={n_hits} ({timing['payout_model_s']}s)")

    out = root / month
    out.mkdir(parents=True, exist_ok=True)
    _save_booster(win, out / "win.txt.gz")
    _save_booster(payout, out / "payout.txt.gz")
    timing["total_s"] = round(time.time() - t_all, 1)
    meta = dict(month=month, features=feats, tau=tau, alpha=alpha, win_best_iteration=win.best_iteration,
                train=[tr_lo, add_months(va_lo, -1)], valid=[va_lo, last],
                payout_train=[mcfg["payout_train_from"], last], payout_hits=n_hits,
                model_cfg=mcfg, timing=timing, created=time.strftime("%Y-%m-%d %H:%M:%S"))
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"saved {out} ({sum(p.stat().st_size for p in out.iterdir()) / 1e6:.1f} MB, total {timing['total_s']}s)")
    return meta


def fill_history_gaps(cfg: dict, con, month: str, root: Path | None = None) -> int:
    """month の日のうち、番組表があるのに history に無い日を、month のモデルで予測して足す。"""
    root = root or model_root(cfg)
    if not MonthModel.exists(cfg, month, root):
        return 0
    h = load_history(cfg, root)
    have = set(h["hd"]) if len(h) else set()
    days = [r[0] for r in con.execute("select distinct hd from entries where hd between ? and ?",
                                      (month_first_hd(month), month_last_hd(month))).fetchall()]
    miss = sorted(d for d in days if d not in have)
    if not miss:
        return 0
    log(f"  history: {month} の {len(miss)} 日を {month} のモデルで補完")
    mm = MonthModel.load(cfg, month, root)
    df = build_frame(con, miss[0], miss[-1], cfg["model"], with_labels=False)
    df = df[df["hd"].isin(miss)].reset_index(drop=True)
    if df.empty:
        return 0
    rows = mm.predict_frame(df)
    upsert_history(cfg, rows, root)
    return len(miss)


def ensure_month(cfg: dict, month: str, root: Path | None = None) -> bool:
    """学習済みが無ければ学習。学習したら True。"""
    if MonthModel.exists(cfg, month, root):
        return False
    train_month(cfg, month, root)
    return True


# ============================================================== 1日分の予測（select_races から使う）
def predict_day(cfg: dict, hd: str, con=None, root: Path | None = None, record: bool = True) -> pd.DataFrame:
    """hd の全レース・全艇の 1着確率 p1、予想単勝配当 g、EV。record=True なら history に残す。
    返り値: 1艇1行 (hd, jcd, rno, lane, p1, g, ev, model_month)。学習済みが無ければ FileNotFoundError。"""
    month = f"{hd[:4]}-{hd[4:6]}"
    if not MonthModel.exists(cfg, month, root):
        raise FileNotFoundError(f"{month} の学習済みモデルがありません（python scripts/model.py ensure --date {hd}）")
    con = con or connect_ro()
    mm = MonthModel.load(cfg, month, root)
    df = build_frame(con, hd, hd, cfg["model"], with_labels=False)
    if df.empty:
        return pd.DataFrame(columns=["hd", "jcd", "rno", "lane", "p1", "g", "ev", "model_month"])
    rows = mm.predict_frame(df)
    if record:
        upsert_history(cfg, rows, root)
    return hist_to_lanes(rows)


def hist_to_lanes(rows: pd.DataFrame) -> pd.DataFrame:
    P1r, _, _, _ = hist_arrays(rows)
    P1 = apply_alpha(P1r, rows["alpha"].to_numpy(float))
    G = rows[[f"g_{i}" for i in range(1, 7)]].to_numpy(float)
    n = len(rows)
    out = pd.DataFrame({"hd": np.repeat(rows["hd"].to_numpy(), 6), "jcd": np.repeat(rows["jcd"].to_numpy(), 6),
                        "rno": np.repeat(rows["rno"].to_numpy(), 6), "lane": np.tile(np.arange(1, 7), n),
                        "p1": P1.ravel(), "g": G.ravel(),
                        "model_month": np.repeat(rows["model_month"].to_numpy(), 6)})
    out["ev"] = out["p1"] * out["g"] / 100
    return out


def select_lanes(lanes: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """ルール: p1 ≥ min_win_prob かつ p1×g/100 ≥ min_ev（しきい値は rules.yaml）。"""
    m = (lanes["p1"] >= cfg["min_win_prob"]) & (lanes["ev"] >= cfg["min_ev"]) & lanes["g"].notna()
    if cfg.get("venues"):
        m &= lanes["jcd"].isin([str(v).zfill(2) for v in cfg["venues"]])
    if cfg.get("race_numbers"):
        m &= lanes["rno"].isin(cfg["race_numbers"])
    return lanes[m].reset_index(drop=True)


# ============================================================== 初期 history（分析のウォークフォワード予測から）
def seed_history(cfg: dict, workdir: Path, root: Path | None = None) -> pd.DataFrame:
    """trifecta.py の pred_*.npz（月ごとの OOS 1着確率）と feat/*.parquet から history を作り、
    予想配当 g を cmd_tansho と同じウォークフォワード（月 m は m より前の当たりで学習）で付ける。"""
    mcfg = cfg["model"]
    files = sorted((workdir / "pred").glob("pred_*.npz"))
    parts = []
    for f in files:
        z = np.load(f)
        parts.append(pd.DataFrame({"hd": z["hd"].astype(str), "jcd": z["jcd"].astype(str), "rno": z["rno"].astype(int),
                                   "model_month": str(z["fit"])}).join(
            pd.DataFrame(z["P1"], columns=[f"p1r_{i}" for i in range(1, 7)])))
    k = pd.concat(parts, ignore_index=True)
    P1r = k[[f"p1r_{i}" for i in range(1, 7)]].to_numpy(np.float32)
    # 前夜の素性（分析の特徴量ファイルから）
    need = ["hd", "jcd", "rno", "lane", "national_win_rate", "grade"] + RACE_FLAGS
    fparts = []
    for p in sorted((workdir / "feat").glob("feat_*.parquet")):
        fparts.append(pd.read_parquet(p, columns=need))
    fe = pd.concat(fparts, ignore_index=True)
    fe = fe[fe["hd"].between(k.hd.min(), k.hd.max())].sort_values(["hd", "jcd", "rno", "lane"])
    n = fe.groupby(["hd", "jcd", "rno"])["lane"].transform("size")
    fe = fe[n == 6].reset_index(drop=True)
    f6 = fe.iloc[::6][["hd", "jcd", "rno"]].reset_index(drop=True)
    f6["fi"] = np.arange(len(f6))
    fi = k[HIST_KEYS].merge(f6, on=HIST_KEYS, how="left")["fi"]
    assert fi.notna().all()
    fi = fi.to_numpy().astype(int)
    nwr = fe["national_win_rate"].to_numpy(np.float32).astype(float).reshape(-1, 6)[fi]
    grd = fe["grade"].to_numpy(np.float32).astype(float).reshape(-1, 6)[fi]
    fl = fe.iloc[::6][RACE_FLAGS].to_numpy(np.float32).astype(float)[fi]

    con = connect_ro()
    TP, win = tansho_winners(con, k)
    month = month_of(k["hd"])
    # 再校正 α（trifecta.recalibrate と同じ: 各月、その月より前の OOS 月 最大12か月）
    alphas = {}
    for m in sorted(set(month)):
        prior = [x for x in sorted(set(month)) if mcfg["recalib_from"] <= x < m][-mcfg["recalib_window_months"]:]
        sel = np.isin(month, prior)
        alphas[m] = fit_alpha(P1r[sel], win[sel], mcfg["recalib_min_races"])
    alpha = np.array([alphas[m] for m in month])
    log(f"alphas: {alphas}")
    P1 = apply_alpha(P1r, alpha)
    import lightgbm as lgb
    G = np.full((len(k), 6), np.nan)
    for m in sorted(set(month)):
        if m < mcfg["payout_first_month"]:
            continue
        t0 = time.time()
        tr = np.where((month >= mcfg["payout_train_from"]) & (month < m) & (win >= 0))[0]
        X = payout_features(P1[tr], nwr[tr], grd[tr], fl[tr], k["jcd"].to_numpy()[tr], k["rno"].to_numpy()[tr])
        X = X.reshape(len(tr), 6, -1)[np.arange(len(tr)), win[tr]]
        mdl = lgb.train(PAYOUT_PARAMS, lgb.Dataset(X, TP[tr, win[tr]]), PAYOUT_ROUNDS)
        te = np.where(month == m)[0]
        G[te] = predict_payout(mdl, P1[te], nwr[te], grd[te], fl[te], k["jcd"].to_numpy()[te], k["rno"].to_numpy()[te])
        log(f"  g for {m}: hits={len(tr)} ({time.time() - t0:.0f}s)")
    rows = make_hist_rows(k, None, alpha, P1r, G, nwr, grd, fl)
    rows["model_month"] = k["model_month"].to_numpy()
    save_history(cfg, rows, root)
    return rows


# ============================================================== history でのバックテスト
def backtest_history(cfg: dict, m_from: str, m_to: str, root: Path | None = None, con=None) -> pd.DataFrame:
    """history（OOS 予測）に現行のしきい値を当てた月別成績。欠場艇は返還（購入0）。"""
    con = con or connect_ro()
    h = load_history(cfg, root)
    hm = month_of(h["hd"])
    h = h[(hm >= m_from) & (hm <= m_to)].reset_index(drop=True)
    TP, win = tansho_winners(con, h)
    lanes = hist_to_lanes(h)
    lanes["ret"] = TP.ravel()
    lanes["has"] = np.repeat(win >= 0, 6)
    void = void_lanes(con, h.hd.min(), h.hd.max())
    lanes = lanes.merge(void, on=["hd", "jcd", "rno", "lane"], how="left")
    lanes["void"] = lanes["void"].fillna(False).astype(bool)
    pk = select_lanes(lanes[lanes["has"] & ~lanes["void"]], cfg)
    pk["month"] = month_of(pk["hd"])
    unit = cfg["unit_yen"]
    g = pk.groupby("month").agg(races=("rno", "size"), bets=("lane", "size"), hits=("ret", lambda x: int((x > 0).sum())),
                                ret=("ret", "sum"))
    g["cost"] = g["bets"] * unit
    g["payout"] = g["ret"] * unit / 100
    g["roi"] = g["payout"] / g["cost"]
    return g.drop(columns=["ret", "races"]).reset_index()


def void_lanes(con, hd_from: str, hd_to: str) -> pd.DataFrame:
    """欠場（返還）: results で position も st_timing も NULL の艇。"""
    res = pd.read_sql("select hd, jcd, rno, lane, json_extract(data,'$.st_timing') st from results "
                      "where position is null and hd between ? and ?", con, params=(hd_from, hd_to))
    return res[res["st"].isna()][["hd", "jcd", "rno", "lane"]].assign(void=True)


# ============================================================== main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["ensure", "train", "predict", "seed-history", "backtest"])
    ap.add_argument("--strategy", default=DEFAULT_STRATEGY)
    ap.add_argument("--date")
    ap.add_argument("--month")
    ap.add_argument("--from", dest="m_from")
    ap.add_argument("--to", dest="m_to")
    ap.add_argument("--workdir")
    ap.add_argument("--root", help="学習済みの置き場（既定 data/models/<dir>）。動作確認用")
    a = ap.parse_args(argv)
    cfg = strategy_cfg(a.strategy)
    root = Path(a.root) if a.root else None
    if a.cmd == "ensure":
        if not a.date:
            ap.error("--date が必要")
        m = f"{a.date[:4]}-{a.date[4:6]}"
        t0 = time.time()
        did = ensure_month(cfg, m, root)
        print(f"{m}: {'学習した' if did else '学習済みあり'} ({time.time() - t0:.0f}s)")
    elif a.cmd == "train":
        meta = train_month(cfg, a.month, root)
        print(json.dumps({k: v for k, v in meta.items() if k != "features"}, ensure_ascii=False, indent=1))
    elif a.cmd == "predict":
        lanes = predict_day(cfg, a.date, root=root, record=False)
        pd.set_option("display.width", 200)
        print(select_lanes(lanes, cfg).to_string(index=False))
    elif a.cmd == "seed-history":
        rows = seed_history(cfg, Path(a.workdir), root)
        print(f"history rows={len(rows)} {rows.hd.min()}..{rows.hd.max()}")
    elif a.cmd == "backtest":
        f = lambda s: f"{s[:4]}-{s[4:6]}" if s and "-" not in s else s  # noqa: E731
        g = backtest_history(cfg, f(a.m_from), f(a.m_to), root)
        tot = g[["bets", "hits", "cost", "payout"]].sum()
        print(g.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        print(f"通算 {tot.payout / tot.cost:.3f}  勝ち月 {(g.roi > 1).sum()}/{len(g)}  点 {int(tot.bets)}  的中 {int(tot.hits)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
