"""2連単: 着順確率モデル（trifecta.py の3段モデル）× 予想配当 のウォークフォワード検証と、当日の分析。

使い方（作業ファイルは trifecta.py と同じ workdir）:
    python scripts/analysis/nirensho.py backtest              # 2024-11〜2026-10 の月別成績
    python scripts/analysis/nirensho.py today --date 20261103 # 当日の全レース分析（前夜の情報だけ）

方針:
  * 2連単の確率 P2(i,j) = Σ_k P(i,j,k)。P は trifecta.py の OOS 予測（pred_YYYY-MM.npz）。
  * 予想配当 g2: 「当たった2連単の払戻」を、当たった買い目（各レース1つ）だけで学習する LightGBM（gamma）。
    月 m の g2 は m より前の月の当たりだけで学習（ウォークフォワード）。
  * 期待値 EV = P2 × g2 / 100。買い方の候補（確率の下限 × EV の下限）を月別に全部出し、
    発見用（2024-11〜2025-10）と検証用（2025-11〜2026-10）に分けて見る。候補数を記録する。
  * 返還: 欠場艇を含む買い目は購入0・払戻0。2連単の払戻が無いレースは買わなかった扱い。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import trifecta as T  # noqa: E402

PAIRS = np.array([(i, j) for i in range(6) for j in range(6) if i != j])   # 30 x 2
_PI = {tuple(p): n for n, p in enumerate(PAIRS.tolist())}
# 120通り → 30通りの集計行列
AGG = np.zeros((120, 30), np.float32)
for n, (a, b, c) in enumerate(T.PERMS.tolist()):
    AGG[n, _PI[(a, b)]] = 1.0

G_PARAMS = dict(objective="gamma", learning_rate=0.05, num_leaves=31, min_data_in_leaf=200,
                feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0,
                num_threads=4, verbose=-1, seed=7)
PROB_MIN = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]
EV_MIN = [0.9, 1.0, 1.1, 1.2, 1.3, 1.5]
DISC = [f"{y}-{m:02d}" for y, m in [(2024, 11), (2024, 12)] + [(2025, i) for i in range(1, 11)]]
VALID = [f"{y}-{m:02d}" for y, m in [(2025, 11), (2025, 12)] + [(2026, i) for i in range(1, 11)]]


def g2_features(P2, P1, nwr, grd, flags):
    nr = len(P2)
    a, b = PAIRS[:, 0], PAIRS[:, 1]
    rankP = (-P2).argsort(1).argsort(1) + 1
    r1 = (-P1).argsort(1).argsort(1) + 1
    nrk = (-np.nan_to_num(nwr, nan=0)).argsort(1).argsort(1) + 1
    H = T.entropy(P2)
    cols = {
        "cid": np.tile(np.arange(30), nr), "la": np.tile(a + 1, nr), "lb": np.tile(b + 1, nr),
        "logP": np.log(np.clip(P2, 1e-9, None)).ravel(), "rankP": rankP.ravel(),
        "lp1a": np.log(P1[:, a]).ravel(), "lp1b": np.log(P1[:, b]).ravel(),
        "r1a": r1[:, a].ravel(), "r1b": r1[:, b].ravel(),
        "nwa": nwr[:, a].ravel(), "nwb": nwr[:, b].ravel(),
        "nra": nrk[:, a].ravel(), "nrb": nrk[:, b].ravel(),
        "ga": grd[:, a].ravel(), "gb": grd[:, b].ravel(),
        "H": np.repeat(H, 30), "p1max": np.repeat(P1.max(1), 30), "p1_1": np.repeat(P1[:, 0], 30),
    }
    for i in range(flags.shape[1]):
        cols[f"f{i}"] = np.repeat(flags[:, i], 30)
    return pd.DataFrame(cols).astype(np.float32)


def load_nirensho(con, hd_lo, hd_hi):
    pay = pd.read_sql("select hd, jcd, rno, combination comb, payout from payouts where ticket_type='nirensho' "
                      "and hd between ? and ?", con, params=(hd_lo, hd_hi))
    return pay.drop_duplicates(["hd", "jcd", "rno"])


def cmd_backtest(work: Path):
    import lightgbm as lgb
    races, R, void3, nwr, grd = T.load_eval(work)
    P2 = R["P"].astype(np.float64) @ AGG.astype(np.float64)
    P1 = R["P1"].astype(np.float64)
    con = T.connect()
    pay = load_nirensho(con, races.hd.min(), races.hd.max()).rename(columns={"comb": "c2", "payout": "pay2"})
    races = races.drop(columns=["comb", "payout", "hit"]).merge(pay, on=["hd", "jcd", "rno"], how="left")
    ok = races["c2"].notna().to_numpy()
    hit = np.full(len(races), -1)
    yy = races.loc[ok, "c2"].str.split("-", expand=True).astype(int).to_numpy() - 1
    hit[ok] = [_PI[tuple(r)] for r in yy.tolist()]
    pay2 = races["pay2"].fillna(0).to_numpy(float)
    # 欠場艇を含む2連単 = 返還
    lane_void = np.zeros((len(races), 6), bool)
    for n, (a, b, c) in enumerate(T.PERMS.tolist()):
        pass
    # void3 (nr,120) から艇ごとの欠場を復元: その艇を含む3連単がすべて返還なら欠場
    for l in range(6):
        idx = np.where((T.PERMS == l).any(1))[0]
        lane_void[:, l] = void3[:, idx].all(1)
    void2 = lane_void[:, PAIRS].any(2)
    flags = races[T.RACE_FLAGS].to_numpy(np.float32)
    X = g2_features(P2, P1, nwr, grd, flags)
    months = races["month"].to_numpy()
    allm = sorted(set(months))
    g = np.full(P2.shape, np.nan)
    for m in allm:
        if m < T.EVAL_FIRST_MONTH:
            continue
        trm = (months < m) & ok
        rows = np.where(trm)[0] * 30 + hit[trm]
        mdl = lgb.train(G_PARAMS, lgb.Dataset(X.iloc[rows], pay2[trm]), num_boost_round=300)
        te = np.where(months == m)[0]
        xr = (te[:, None] * 30 + np.arange(30)).ravel()
        g[te] = mdl.predict(X.iloc[xr]).reshape(-1, 30)
        T.log(f"g2 {m}: train wins={trm.sum()}")
    EV = P2 * g / 100
    out = []
    for pm in PROB_MIN:
        for em in EV_MIN:
            sel = (P2 >= pm) & (EV >= em) & ~void2 & ok[:, None] & ~np.isnan(g)
            win = np.zeros_like(sel)
            win[np.arange(len(sel))[ok], hit[ok]] = True
            for m in allm:
                mm = months == m
                s = sel[mm]
                n = s.sum()
                if n == 0:
                    continue
                ret = (s & win[mm]).sum(1) @ pay2[mm]
                out.append(dict(pmin=pm, evmin=em, month=m, bets=int(n), hits=int((s & win[mm]).sum()),
                                bet=n * 100, ret=float(ret)))
    cm = pd.DataFrame(out)
    cm.to_parquet(work / "eval" / "nirensho_cand_month.parquet")
    # 基準: モデル1番手の2連単、1-2 固定
    base = []
    top = P2.argmax(1)
    for name, pick in [("model_top1", top), ("1-2", np.full(len(P2), _PI[(0, 1)]))]:
        valid = ok & ~void2[np.arange(len(P2)), pick]
        for per, ms in [("発見", DISC), ("検証", VALID)]:
            mm = valid & np.isin(months, ms)
            base.append(dict(rule=name, period=per, bets=int(mm.sum()),
                             hit=float((hit[mm] == pick[mm]).mean()),
                             roi=float(((hit[mm] == pick[mm]) * pay2[mm]).sum() / (mm.sum() * 100))))
    print(pd.DataFrame(base).to_string(index=False))

    def summ(ms):
        d = cm[cm.month.isin(ms)]
        a = d.groupby(["pmin", "evmin"]).agg(bets=("bets", "sum"), hits=("hits", "sum"), bet=("bet", "sum"), ret=("ret", "sum"))
        a["roi"] = a.ret / a.bet
        d = d.assign(r=d.ret / d.bet)
        a["win_m"] = d.groupby(["pmin", "evmin"]).apply(lambda x: int((x.r > 1).sum()))
        a["worst"] = d.groupby(["pmin", "evmin"]).r.min()
        return a
    s1, s2 = summ(DISC), summ(VALID)
    tab = s1[["bets", "roi", "win_m", "worst"]].join(s2[["bets", "roi", "win_m", "worst"]], lsuffix="_発見", rsuffix="_検証")
    print(f"候補 {len(tab)} 通り（確率下限{len(PROB_MIN)} × EV下限{len(EV_MIN)}）")
    print(tab.round(3).to_string())
    tab.to_csv(work / "eval" / "nirensho_summary.csv")


def cmd_today(work: Path, date: str):
    import lightgbm as lgb
    # 着順モデル: 予測月の直前24か月で学習（trifecta.py と同じ窓）
    start = f"{date[:4]}-{date[4:6]}"
    df = T.load_features(work)
    feats = T.feature_cols(df)
    months = df["month"].to_numpy()
    tr_lo, va_lo = T.add_months(start, -T.TRAIN_MONTHS), T.add_months(start, -T.VALID_MONTHS)
    tr = df[(months >= tr_lo) & (months < va_lo)]
    va = df[(months >= va_lo) & (months < start)]
    tr = tr[tr["y_has_result"].to_numpy()]
    va = va[va["y_has_result"].to_numpy()]
    del df
    T.log(f"fit {start}: train {tr_lo}.. races={len(tr)//6} valid={len(va)//6}")
    sm = T.StageModel(feats).fit(tr, va)
    del tr, va
    te = pd.read_parquet(work / "feat" / f"today_{date}.parquet")
    te = te.sort_values(["hd", "jcd", "rno", "lane"]).reset_index(drop=True)
    n = te.groupby(["hd", "jcd", "rno"])["lane"].transform("size")
    te = te[n == 6].reset_index(drop=True)
    te["jcd_i"] = te["jcd"].astype(int)
    te["month"] = start
    for c in feats:
        if c not in te.columns:
            te[c] = np.nan
    r = sm.predict(te)
    keys = te[["hd", "jcd", "rno"]].iloc[::6].reset_index(drop=True)
    np.savez_compressed(work / "pred" / f"today_{date}.npz", hd=keys.hd.to_numpy().astype(str),
                        jcd=keys.jcd.to_numpy().astype(str), rno=keys.rno.to_numpy().astype(int), **r)
    T.log(f"predicted today races={len(keys)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["backtest", "today"])
    ap.add_argument("--workdir", default=str(T.DEFAULT_WORK))
    ap.add_argument("--date")
    a = ap.parse_args()
    w = Path(a.workdir)
    T.WORK = w
    if a.cmd == "backtest":
        cmd_backtest(w)
    else:
        cmd_today(w, a.date)


if __name__ == "__main__":
    main()
