"""3連単市場の「帯」と「並び」の偏り（H5・H6）。モデル不要。確定オッズ（odds.sanrensho）と結果だけで出す。

使い方:
    python scripts/analysis/market_shape.py [--out DIR]   # 全表を markdown で標準出力、CSV を --out に

事前登録は notes/analysis/market_shape.md の 0 節。ここに書いた以外の集計はしない。
  * 期間: 発見用 2025-11〜2026-04 / 検証用 2026-05〜2026-10。
  * q = (1/オッズ) をレース内（オッズがあり欠場艇を含まない目）で正規化。
  * 1点100円。回収率 = 払戻合計 ÷ 購入合計。欠場（position も st_timing も NULL）を含む目・オッズ NULL の目は買わない。
  * DB は読み取り専用で開く。
"""
from __future__ import annotations

import argparse
import itertools
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DB_URI = f"file:{ROOT / 'data' / 'boatrace.db'}?mode=ro"
DISC_MONTHS = ["202511", "202512", "202601", "202602", "202603", "202604"]
VAL_MONTHS = ["202605", "202606", "202607", "202608", "202609", "202610"]
MONTHS = DISC_MONTHS + VAL_MONTHS
QUIRKY = {"02", "03", "04", "11", "14", "22"}   # 戸田・江戸川・平和島・びわこ・鳴門・福岡

Q_BANDS = [(0, .005, "0〜0.5%"), (.005, .01, "0.5〜1%"), (.01, .02, "1〜2%"), (.02, .05, "2〜5%"),
           (.05, .10, "5〜10%"), (.10, .20, "10〜20%"), (.20, 1.01, "20%以上")]
ODDS_BANDS = [(0, 5, "〜5倍"), (5, 10, "5〜10倍"), (10, 20, "10〜20倍"), (20, 50, "20〜50倍"), (50, 100, "50〜100倍"),
              (100, 300, "100〜300倍"), (300, 1000, "300〜1000倍"), (1000, 1e9, "1000倍以上")]
HEAD1_BANDS = [(0, .3, "〜30%"), (.3, .4, "30〜40%"), (.4, .5, "40〜50%"), (.5, .6, "50〜60%"), (.6, .7, "60〜70%"),
               (.7, .8, "70〜80%"), (.8, 1.01, "80%以上")]
HEADX_BANDS = [(0, .05, "〜5%"), (.05, .10, "5〜10%"), (.10, .20, "10〜20%"), (.20, .30, "20〜30%"), (.30, 1.01, "30%以上")]
RATIO_BANDS = [(1, 2, "1〜2倍"), (2, 3, "2〜3倍"), (3, 1e9, "3倍以上")]

COMBOS = [f"{a}-{b}-{c}" for a, b, c in itertools.permutations(range(1, 7), 3)]
CIDX = {s: i for i, s in enumerate(COMBOS)}
ABC = np.array([[int(x) for x in s.split("-")] for s in COMBOS])  # (120,3)


def log(*a):
    print(*a, file=sys.stderr, flush=True)


# ---------------------------------------------------------------- データ
def load_month(con, m):
    """1か月分を (races, 120) の配列にする。返り値: dict(q, odds, hit, valid, jcd, hd)"""
    lo, hi = f"{m}01", f"{m}31"
    od = pd.read_sql("select hd, jcd, rno, combination comb, odds from odds "
                     "where ticket_type='sanrensho' and hd between ? and ?", con, params=(lo, hi))
    pay = pd.read_sql("select hd, jcd, rno, combination comb, payout from payouts "
                      "where ticket_type='sanrensho' and hd between ? and ?", con, params=(lo, hi))
    res = pd.read_sql("select hd, jcd, rno, lane from results where position is null "
                      "and json_extract(data,'$.st_timing') is null and hd between ? and ?", con, params=(lo, hi))
    races = od[["hd", "jcd", "rno"]].drop_duplicates().merge(pay[["hd", "jcd", "rno"]].drop_duplicates())
    races = races.sort_values(["hd", "jcd", "rno"]).reset_index(drop=True)
    races["rid"] = np.arange(len(races))
    key = races.set_index(["hd", "jcd", "rno"])["rid"]
    n = len(races)
    O = np.full((n, 120), np.nan, np.float32)
    H = np.zeros((n, 120), np.float32)       # 払戻（円、100円あたり）
    V = np.ones((n, 120), bool)
    od = od.join(key, on=["hd", "jcd", "rno"], how="inner")
    od = od[od.comb.isin(CIDX)]
    O[od.rid.to_numpy(), od.comb.map(CIDX).to_numpy()] = od.odds.to_numpy(np.float32)
    pay = pay.join(key, on=["hd", "jcd", "rno"], how="inner")
    pay = pay[pay.comb.isin(CIDX)]
    H[pay.rid.to_numpy(), pay.comb.map(CIDX).to_numpy()] = pay.payout.to_numpy(np.float32)
    res = res.join(key, on=["hd", "jcd", "rno"], how="inner")
    for rid, lane in zip(res.rid.to_numpy(), res.lane.to_numpy()):
        V[rid, (ABC == lane).any(1)] = False
    V &= np.isfinite(O) & (O > 0)
    inv = np.where(V, 1.0 / np.where(V, O, 1), 0.0)
    s = inv.sum(1, keepdims=True)
    Q = np.where(s > 0, inv / np.where(s > 0, s, 1), 0).astype(np.float32)
    return dict(q=Q, odds=O, pay=H, valid=V, jcd=races.jcd.to_numpy(), n=n, month=m)


def load_all(con):
    parts = [load_month(con, m) for m in MONTHS]
    for p in parts:
        log(f"[load] {p['month']}: {p['n']} レース")
    D = {k: np.concatenate([p[k] for p in parts]) for k in ("q", "odds", "pay", "valid", "jcd")}
    D["month"] = np.concatenate([np.full(p["n"], i) for i, p in enumerate(parts)])
    D["is_val"] = D["month"] >= len(DISC_MONTHS)
    D["hit"] = (D["pay"] > 0).astype(np.float32)
    D["quirky"] = np.isin(D["jcd"], list(QUIRKY))
    return D


# ---------------------------------------------------------------- 集計
def cell(q, hit, pay, month, mask):
    """mask (bool, 同じ形) の目を全部100円で買ったときの成績。"""
    n = int(mask.sum())
    if n == 0:
        return dict(n=0, q=np.nan, hit=np.nan, ratio=np.nan, roi=np.nan, win_m="-", worst=np.nan)
    qm = float(q[mask].mean()); hm = float(hit[mask].mean())
    ret = pay[mask]; roi = float(ret.sum() / (100.0 * n))
    mo = np.broadcast_to(month[:, None], q.shape)[mask]
    df = pd.DataFrame({"m": mo, "r": ret}).groupby("m")["r"].agg(["sum", "size"])
    mroi = df["sum"] / (100 * df["size"])
    return dict(n=n, q=qm, hit=hm, ratio=hm / qm if qm > 0 else np.nan, roi=roi,
                win_m=f"{int((mroi > 1).sum())}/{len(mroi)}", worst=float(mroi.min()))


def fmt(c):
    if c["n"] == 0:
        return "0 | - | - | - | - | -"
    return f"{c['n']:,} | {c['q']*100:.2f}% | {c['hit']*100:.2f}% | {c['ratio']:.3f} | **{c['roi']:.3f}** | {c['win_m']}"


def band_table(D, value, bands, mask_extra=None, title="", rows=None):
    """value: (races,120) の値で帯分け。両期間の表を markdown で返し、rows に CSV 行を積む。"""
    out = [f"\n#### {title}", "",
           "| 帯 | 発見 点数 | q平均 | 的中率 | 実際÷q | 回収率 | 勝ち月 | 検証 点数 | q平均 | 的中率 | 実際÷q | 回収率 | 勝ち月 | 両期間>1 |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    base = D["valid"] if mask_extra is None else (D["valid"] & mask_extra)
    isv = np.broadcast_to(D["is_val"][:, None], D["q"].shape)
    for lo, hi, name in bands:
        m = base & (value >= lo) & (value < hi)
        cd = cell(D["q"], D["hit"], D["pay"], D["month"], m & ~isv)
        cv = cell(D["q"], D["hit"], D["pay"], D["month"], m & isv)
        both = "○" if (cd["n"] and cv["n"] and cd["roi"] > 1 and cv["roi"] > 1) else ""
        out.append(f"| {name} | {fmt(cd)} | {fmt(cv)} | {both} |")
        if rows is not None:
            rows.append(dict(table=title, band=name, **{f"d_{k}": v for k, v in cd.items()}, **{f"v_{k}": v for k, v in cv.items()}))
    return "\n".join(out)


def head_prob(D):
    """Q_L: 艇 L が1着の20点の q 合計。(races,6)"""
    QL = np.zeros((D["q"].shape[0], 6), np.float32)
    for L in range(1, 7):
        QL[:, L - 1] = D["q"][:, ABC[:, 0] == L].sum(1)
    return QL


def h6(D, rows):
    out = ["\n### H6 市場確率の帯（T6-1〜T6-4）"]
    out.append(band_table(D, D["q"], Q_BANDS, title="T6-1 市場確率 q の帯（全場）", rows=rows))
    out.append(band_table(D, np.nan_to_num(D["odds"], nan=-1), ODDS_BANDS, title="T6-2 オッズ帯（全場）", rows=rows))
    QL = head_prob(D)
    head = ABC[:, 0]
    # 1号艇の1着20点: 値は Q_1 をレースごとに全目へ広げ、頭=1 の目だけ対象
    v1 = np.broadcast_to(QL[:, [0]], D["q"].shape)
    out.append(band_table(D, v1, HEAD1_BANDS, mask_extra=np.broadcast_to(head == 1, D["q"].shape),
                          title="T6-3a 1号艇1着の市場確率 Q_1 の帯（1号艇頭の20点を全部買う）", rows=rows))
    vx = np.zeros_like(D["q"])
    for L in range(2, 7):
        vx[:, head == L] = QL[:, [L - 1]]
    out.append(band_table(D, vx, HEADX_BANDS, mask_extra=np.broadcast_to(head != 1, D["q"].shape),
                          title="T6-3b 1号艇以外の艇 L の1着市場確率 Q_L の帯（L 頭の20点を全部買う）", rows=rows))
    # 1号艇の実際の1着率 vs Q_1（レース単位）
    out.append("\n#### T6-3a 補足: 1号艇の実際の1着率（レース単位）と市場 Q_1")
    out.append("| Q_1 の帯 | 発見 レース | Q_1平均 | 実際1着率 | 検証 レース | Q_1平均 | 実際1着率 |\n|---|---|---|---|---|---|---|")
    win1 = (D["hit"][:, head == 1].sum(1) > 0)
    ok = D["valid"].sum(1) > 0
    for lo, hi, name in HEAD1_BANDS:
        m = ok & (QL[:, 0] >= lo) & (QL[:, 0] < hi)
        s = []
        for per in (~D["is_val"], D["is_val"]):
            mm = m & per
            s.append(f"{mm.sum():,} | {QL[mm, 0].mean()*100:.1f}% | {win1[mm].mean()*100:.1f}%" if mm.sum() else "0 | - | -")
        out.append(f"| {name} | {s[0]} | {s[1]} |")
    for tag, mk in (("癖のある6場", D["quirky"]), ("その他18場", ~D["quirky"])):
        mk2 = np.broadcast_to(mk[:, None], D["q"].shape)
        out.append(band_table(D, D["q"], Q_BANDS, mask_extra=mk2, title=f"T6-4 市場確率 q の帯（{tag}）", rows=rows))
        out.append(band_table(D, v1, HEAD1_BANDS, mask_extra=mk2 & np.broadcast_to(head == 1, D["q"].shape),
                              title=f"T6-4 1号艇1着 Q_1 の帯（{tag}、1号艇頭の20点）", rows=rows))
    return "\n".join(out)


def pair_table(D, idx_a, idx_b, title, rows):
    """idx_a/idx_b: 対になる目の列番号の配列（同じ長さ）。市場が人気にしている方を人気側、他方を裏。"""
    q, hit, pay, V = D["q"], D["hit"], D["pay"], D["valid"]
    qa, qb = q[:, idx_a], q[:, idx_b]
    va = V[:, idx_a] & V[:, idx_b] & (qa > 0) & (qb > 0)
    a_pop = qa >= qb
    q_hi = np.where(a_pop, qa, qb); q_lo = np.where(a_pop, qb, qa)
    h_hi = np.where(a_pop, hit[:, idx_a], hit[:, idx_b]); h_lo = np.where(a_pop, hit[:, idx_b], hit[:, idx_a])
    p_hi = np.where(a_pop, pay[:, idx_a], pay[:, idx_b]); p_lo = np.where(a_pop, pay[:, idx_b], pay[:, idx_a])
    ratio = q_hi / np.where(q_lo > 0, q_lo, np.nan)
    head1 = np.broadcast_to(ABC[idx_a, 0] == 1, qa.shape)
    isv = np.broadcast_to(D["is_val"][:, None], qa.shape)
    out = [f"\n#### {title}", "",
           "| 頭 | 比 r | 側 | 発見 点数 | q平均 | 的中率 | 実際÷q | 回収率 | 勝ち月 | 検証 点数 | q平均 | 的中率 | 実際÷q | 回収率 | 勝ち月 |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for hname, hm in (("全体", np.ones_like(va)), ("1号艇", head1), ("1号艇以外", ~head1)):
        for lo, hi, rname in RATIO_BANDS:
            m = va & hm & (ratio >= lo) & (ratio < hi)
            for sname, qq, hh, pp in (("人気側", q_hi, h_hi, p_hi), ("裏", q_lo, h_lo, p_lo)):
                cd = cell(qq, hh, pp, D["month"], m & ~isv)
                cv = cell(qq, hh, pp, D["month"], m & isv)
                out.append(f"| {hname} | {rname} | {sname} | {fmt(cd)} | {fmt(cv)} |")
                rows.append(dict(table=title, band=f"{hname}/{rname}/{sname}", **{f"d_{k}": v for k, v in cd.items()},
                                 **{f"v_{k}": v for k, v in cv.items()}))
    return "\n".join(out)


def h5(D, rows):
    out = ["\n### H5 並びの偏り（T5-1〜T5-3）"]
    ia, ib = [], []
    for a, b, c in itertools.permutations(range(1, 7), 3):
        if b < c:
            ia.append(CIDX[f"{a}-{b}-{c}"]); ib.append(CIDX[f"{a}-{c}-{b}"])
    out.append(pair_table(D, np.array(ia), np.array(ib), "T5-1 2・3着の入れ替え a-b-c と a-c-b（1レース60組）", rows))
    ia, ib = [], []
    for a, b, c in itertools.permutations(range(1, 7), 3):
        if a < b:
            ia.append(CIDX[f"{a}-{b}-{c}"]); ib.append(CIDX[f"{b}-{a}-{c}"])
    out.append(pair_table(D, np.array(ia), np.array(ib), "T5-2 1・2着の入れ替え a-b-c と b-a-c（1レース60組）", rows))
    # T5-3 昇順
    asc = (ABC[:, 0] < ABC[:, 1]) & (ABC[:, 1] < ABC[:, 2])
    ascm = np.broadcast_to(asc, D["q"].shape)
    out.append("\n#### T5-3 艇番が昇順の目（20点）と それ以外（100点）、q の帯ごと")
    out.append("| q の帯 | 形 | 発見 点数 | q平均 | 的中率 | 実際÷q | 回収率 | 勝ち月 | 検証 点数 | q平均 | 的中率 | 実際÷q | 回収率 | 勝ち月 |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    isv = np.broadcast_to(D["is_val"][:, None], D["q"].shape)
    for lo, hi, name in [(0, 1.01, "全帯")] + Q_BANDS:
        for sname, sm in (("昇順", ascm), ("非昇順", ~ascm)):
            m = D["valid"] & sm & (D["q"] >= lo) & (D["q"] < hi)
            cd = cell(D["q"], D["hit"], D["pay"], D["month"], m & ~isv)
            cv = cell(D["q"], D["hit"], D["pay"], D["month"], m & isv)
            out.append(f"| {name} | {sname} | {fmt(cd)} | {fmt(cv)} |")
            rows.append(dict(table="T5-3", band=f"{name}/{sname}", **{f"d_{k}": v for k, v in cd.items()}, **{f"v_{k}": v for k, v in cv.items()}))
    out.append("\n#### T5-3 補足: 1-2-3 単独、昇順で頭1号艇（10点）／頭1号艇以外（10点）")
    out.append("| 目 | 発見 点数 | q平均 | 的中率 | 実際÷q | 回収率 | 勝ち月 | 検証 点数 | q平均 | 的中率 | 実際÷q | 回収率 | 勝ち月 |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    only123 = np.zeros(120, bool); only123[CIDX["1-2-3"]] = True
    for sname, sm in (("1-2-3", only123), ("昇順・頭1号艇", asc & (ABC[:, 0] == 1)), ("昇順・頭1号艇以外", asc & (ABC[:, 0] != 1))):
        m = D["valid"] & np.broadcast_to(sm, D["q"].shape)
        cd = cell(D["q"], D["hit"], D["pay"], D["month"], m & ~isv)
        cv = cell(D["q"], D["hit"], D["pay"], D["month"], m & isv)
        out.append(f"| {sname} | {fmt(cd)} | {fmt(cv)} |")
        rows.append(dict(table="T5-3補足", band=sname, **{f"d_{k}": v for k, v in cd.items()}, **{f"v_{k}": v for k, v in cv.items()}))
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="CSV の出力先ディレクトリ")
    args = ap.parse_args()
    con = sqlite3.connect(DB_URI, uri=True, timeout=600)
    D = load_all(con)
    con.close()
    nd, nv = int((~D["is_val"]).sum()), int(D["is_val"].sum())
    void_r = int((~D["valid"] & np.isfinite(D["odds"])).any(1).sum())
    allbuy = cell(D["q"], D["hit"], D["pay"], D["month"], D["valid"])
    print(f"- 対象レース: 発見用 {nd:,} / 検証用 {nv:,}。欠場で返還の目があるレース {void_r:,}。")
    prop = float((D["pay"][D["valid"]] * D["q"][D["valid"]]).sum() / (100.0 * D["q"][D["valid"]].sum()))
    print(f"- 全120点を全レースで均等に買う: 点数 {allbuy['n']:,}、回収率 {allbuy['roi']:.3f}"
          f"（＝平均配当÷12,000円。q に比例して買うと {prop:.3f}、これが控除率の確認で約0.75のはず）")
    rows = []
    print(h6(D, rows))
    print(h5(D, rows))
    if args.out:
        Path(args.out).mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(Path(args.out) / "market_shape_cells.csv", index=False)
    log(f"[done] セル数 {len(rows)}")


if __name__ == "__main__":
    main()
