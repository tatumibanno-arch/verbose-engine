"""穴狙いの紙上検証（ana-s5-v1・ana-s6-v1）の採点。notes/decisions.md 2026-11-14 の固定条件どおり。お金は使わない。

    python scripts/analysis/ana_paper.py                # 2026-11-14 以降を全部採点して notes/bets/ana-paper.md を作り直す
    python scripts/analysis/ana_paper.py --from 20261001 --to 20261031 --out /tmp/x.md   # 動作確認用

毎晩、成績・締切時オッズ・直前情報を取り込んで Parquet に書き出した後に呼ぶ（Parquet を直接読む。pyarrow と duckdb が要る）。
何度呼んでも同じ結果になる（毎回、期間の全レースを計算し直してファイルを作り直す）。
判定と配分は締切時オッズ＝実際より甘い。1レース1,000円を 1/オッズ に比例して100円単位で配分（各点最低100円）。
"""
import argparse, subprocess, sys, tempfile, os
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from common import VENUES  # noqa: E402

START = "20261114"   # 決めた日。これより前は採点に入れない
STAKE_UNITS = 10     # 1,000円 = 100円×10
RULES = {
    "ana-s5-v1": dict(tickets=["512", "513", "514"], desc="①人気＋①モーター25%以下＋⑤良 → 5-1-2・5-1-3・5-1-4"),
    "ana-s6-v1": dict(tickets=["321", "324", "325", "326"], desc="①B級＋⑥A1＋③展示3位以内＋合成15〜60倍 → 3-2-全"),
}


def match(d):
    """固定条件。戻り値: ルール名 → (真偽の配列, 理由の文字列の配列)"""
    f = lambda c: np.asarray(d[c].values, dtype=float)
    s5 = (f("p_H1") >= .5) & (f("mo1") <= 25) & ((f("er5") <= 2) | (f("mo5") >= 40))
    r5 = [f"①頭シェア{a:.0%}・①モーター{b:.0f}%・⑤展示{c:.0f}位・⑤モーター{e:.0f}%" for a, b, c, e in zip(f("p_H1"), f("mo1"), f("er5"), f("mo5"))]
    co6 = 1 / f("q_P32")
    s6 = (f("gr1") <= 2) & (f("gr6") == 4) & (f("er3") <= 3) & (co6 >= 15) & (co6 <= 60)
    r6 = [f"①B級・⑥A1・③展示{a:.0f}位・3-2-全の合成{b:.1f}倍" for a, b in zip(f("er3"), co6)]
    nz = lambda m: np.nan_to_num(m.astype(float)).astype(bool)
    return {"ana-s5-v1": (nz(s5), r5), "ana-s6-v1": (nz(s6), r6)}


def allocate(q):
    """1/オッズ に比例して100円単位で10口。各点最低1口。"""
    q = np.asarray(q, dtype=float); n = len(q)
    u = np.ones(n, dtype=int); rest = STAKE_UNITS - n
    ideal = q / q.sum() * STAKE_UNITS
    for _ in range(rest):
        i = int(np.argmax(ideal - u)); u[i] += 1
    return u


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="f", default=START); ap.add_argument("--to", dest="t", default="20991231")
    ap.add_argument("--out", default=str(ROOT / "notes/bets/ana-paper.md"))
    a = ap.parse_args()
    tmp = os.path.join(tempfile.mkdtemp(), "ana_paper.parquet")
    r = subprocess.run([sys.executable, str(ROOT / "scripts/analysis/ana_prep.py"), tmp, a.f, a.t], cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit("ana_prep.py が失敗: " + r.stderr[-500:])
    rows = []
    if os.path.exists(tmp):
        d = pd.read_parquet(tmp).sort_values(["hd", "jcd", "rno"]).reset_index(drop=True)
        win = d.cb.str.replace("-", "")
        for rule, (m, why) in match(d).items():
            tk = RULES[rule]["tickets"]
            for i in np.where(m)[0]:
                q = d.loc[i, ["t_" + t for t in tk]].values.astype(float)
                if np.isnan(q).any():
                    continue
                u = allocate(q)
                pay = int(u[tk.index(win[i])] * d.payout[i]) if win[i] in tk else 0
                rows.append(dict(rule=rule, hd=d.hd[i], jcd=int(d.jcd[i]), rno=int(d.rno[i]), why=why[i],
                                 bets=" ".join(f"{t[0]}-{t[1]}-{t[2]}×{k}00円" for t, k in zip(tk, u)),
                                 comp=1 / q.sum(), res=d.cb[i], pay=pay, stake=STAKE_UNITS * 100))
    t = pd.DataFrame(rows)
    L = ["# 穴狙いの紙上検証（ana-s5-v1・ana-s6-v1）", "",
         "本番ではない。お金は使わない。条件は notes/decisions.md の 2026-11-14 の項で固定。このファイルは scripts/analysis/ana_paper.py が毎晩作り直す（手で書き換えない）。",
         "判定と配分は締切時オッズ＝実際より甘い。合格ライン: 12か月の通算回収1.0超かつ勝ち月8以上。", "",
         f"採点した期間: {a.f}〜{t.hd.max() if len(t) else '（該当なし）'}", ""]
    for rule, info in RULES.items():
        x = t[t.rule == rule] if len(t) else t
        L += [f"## {rule}: {info['desc']}", ""]
        if not len(x):
            L += ["まだ該当レースなし。", ""]; continue
        g = x.assign(m=x.hd.str[:6]).groupby("m").agg(レース=("pay", "size"), 的中=("pay", lambda s: int((s > 0).sum())), 投入=("stake", "sum"), 払戻=("pay", "sum"))
        L += ["| 月 | レース | 的中 | 投入 | 払戻 | 回収 |", "|---|---|---|---|---|---|"]
        L += [f"| {m} | {r.レース} | {r.的中} | {r.投入:,} | {r.払戻:,} | {r.払戻/r.投入:.2f} |" for m, r in g.iterrows()]
        L += [f"| 通算 | {len(x)} | {int((x.pay>0).sum())} | {x.stake.sum():,} | {x.pay.sum():,} | {x.pay.sum()/x.stake.sum():.2f} |",
              f"\n勝ち月 {int((g.払戻 > g.投入).sum())}/{len(g)}", "",
              "| 日付 | レース | 条件に合った理由 | 買い目 | 合成 | 結果 | 払戻 |", "|---|---|---|---|---|---|---|"]
        L += [f"| {r.hd} | {VENUES.get(f'{r.jcd:02d}', r.jcd)}{r.rno}R | {r.why} | {r.bets} | {r.comp:.1f}倍 | {r.res} | {r.pay:,} |" for r in x.itertuples()]
        L += [""]
    Path(a.out).write_text("\n".join(L), encoding="utf-8")
    print(f"{a.out} を書いた。" + "、".join(f"{k}: {int((t.rule==k).sum()) if len(t) else 0}レース" for k in RULES))


if __name__ == "__main__":
    main()
