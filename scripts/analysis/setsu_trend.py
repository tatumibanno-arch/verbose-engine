"""節間の流れ（着順・展示・部品交換・チルト・ST）、モーターの前の節、丸亀の予選得点率と準優ボーダーの目安を1本にまとめる。
運用者の観戦・趣味用。舟券は買わない。少ない本数で良し悪しを断定しない。

使い方:
  python scripts/analysis/setsu_trend.py --day 20261118 --out notes/reads/extra-2026-11-18-c.md

使うのは「その日の朝までに分かる情報」（前日までの成績・直前情報、当日の番組表）だけ。
得点率は G1 の一般的な点（1着10・2着8・3着6・4着4・5着2・6着1、事故・欠場0）で計算した目安。
公式の減点（F・L・待機行動違反など）や同点時の並べ方（着順点の上位など）は反映できない。
"""
import argparse, json, sqlite3, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from common import VENUES

ap = argparse.ArgumentParser()
ap.add_argument("--day", default="20261118")
ap.add_argument("--out", default="notes/reads/extra-2026-11-18-c.md")
ap.add_argument("--sims", type=int, default=40000)
a = ap.parse_args()
con = sqlite3.connect("data/boatrace.db", timeout=600)

# 今日の候補（plan-2026-11-18.md）と各場の節の初日・現モーターの使用開始日（g1_prep の資料と同じ）
MEETS = {
    "11": dict(start="20261114", motor_since="20260408", races=[4, 5, 9, 11]),
    "15": dict(start="20261115", motor_since="20260917", races=[6, 8, 9, 11]),
}
PTS = {1: 10, 2: 8, 3: 6, 4: 4, 5: 2, 6: 1}
DAY = a.day
BORDER_RANK = 18


def md(hd):
    return f"{int(hd[4:6])}/{int(hd[6:])}"


def f2(x, n=2):
    return "-" if x is None or pd.isna(x) else f"{x:.{n}f}"


# ---------- 節間の成績（番組・成績・直前情報） ----------
def load_meet(jcd, d0, d1):
    df = pd.read_sql("""
        select e.hd, e.rno, e.lane, e.racer_no, e.racer_name, e.motor_no, e.race_name,
               json_extract(e.data,'$.national_win_rate') nwr,
               r.position, json_extract(r.data,'$.st_timing') st, json_extract(r.data,'$.entrance_position') course,
               b.exhibition_time ex, b.tilt, json_extract(b.data,'$.parts_changed') parts, b.start_st exst
        from entries e
        left join results r on r.hd=e.hd and r.jcd=e.jcd and r.rno=e.rno and r.lane=e.lane
        left join beforeinfo b on b.hd=e.hd and b.jcd=e.jcd and b.rno=e.rno and b.lane=e.lane
        where e.jcd=? and e.hd>=? and e.hd<=?""", con, params=(jcd, d0, d1))
    for k in ["position", "st", "course", "ex", "tilt", "exst", "nwr"]:
        df[k] = pd.to_numeric(df[k], errors="coerce")
    df["racer_no"] = df.racer_no.astype(str)
    df["exr"] = df.groupby(["hd", "rno"]).ex.rank(method="min")
    df["parts"] = df.parts.apply(lambda s: json.loads(s) if isinstance(s, str) else [])
    return df.sort_values(["hd", "rno"])


def trend(g):
    """節間の上向き／下向き（着順と展示順位の前半→直近2走の変化）。目安。"""
    g = g[g.position.notna() | g.exr.notna()]
    if len(g) < 3:
        return "判断材料少"
    late, early = g.tail(2), g.iloc[:-2]
    dpos = early.position.mean() - late.position.mean()
    dex = early.exr.mean() - late.exr.mean()
    s = 0.6 * (dpos if pd.notna(dpos) else 0) + 0.4 * (dex if pd.notna(dex) else 0)
    lab = "上向き" if s >= 0.8 else "下向き" if s <= -0.8 else "横ばい"
    return f"{lab}（着 {f2(early.position.mean(),1)}→{f2(late.position.mean(),1)}、展示順位 {f2(early.exr.mean(),1)}→{f2(late.exr.mean(),1)}）"


def racer_trend_lines(m, racer_no):
    """日ごとに「・」で区切る。日の中の複数走は「/」。"""
    g = m[(m.racer_no == racer_no) & (m.hd < DAY)]
    pos, exs, sts = [], [], []
    for hd, gd in g.groupby("hd"):
        pos.append("".join(str(int(p)) if pd.notna(p) else "欠" for p in gd.position))
        exs.append("/".join(f"{x:.2f}({int(r)})" if pd.notna(x) else "-" for x, r in zip(gd.ex, gd.exr.fillna(0))))
        sts.append("/".join(("F" if s < 0 else "") + f"{abs(s):.2f}".lstrip("0") if pd.notna(s) else "-" for s in gd.st))
    parts = [f"{md(r.hd)}{r.rno}R {'・'.join(r.parts)}" for r in g.itertuples() if r.parts]
    tilts = g.tilt.dropna().tolist()
    tv = [t for i, t in enumerate(tilts) if i == 0 or t != tilts[i - 1]]  # 続けて同じ値は1つに
    tl = "→".join(f"{t:+.1f}".replace("+0.0", "0") for t in tv) if tv else "-"
    return dict(pos="・".join(pos), ex="→".join(exs), st="→".join(sts), st_mean=g.st.mean(),
                parts="、".join(parts) if parts else "なし", tilt=tl, trend=trend(g))


# ---------- モーターの前の節 ----------
def motor_hist(jcd, since, meet_start):
    h = pd.read_sql("""select e.hd, e.rno, e.lane, e.racer_name, e.grade, e.motor_no,
                              json_extract(e.data,'$.national_win_rate') nwr, r.position,
                              json_extract(r.data,'$.exhibition_time') ex
                       from entries e left join results r on r.hd=e.hd and r.jcd=e.jcd and r.rno=e.rno and r.lane=e.lane
                       where e.jcd=? and e.hd>=? and e.hd<?""", con, params=(jcd, since, meet_start))
    for k in ["nwr", "position", "ex"]:
        h[k] = pd.to_numeric(h[k], errors="coerce")
    days = sorted(h.hd.unique())
    mid, prev, mp = 0, None, {}
    for d in days:
        if prev is not None and (pd.Timestamp(d) - pd.Timestamp(prev)).days > 1:
            mid += 1
        mp[d] = mid; prev = d
    h["meet"] = h.hd.map(mp)
    h["exr"] = h[h.ex > 0].groupby(["hd", "rno"]).ex.rank(method="min")
    hr = h[h.position.notna()]
    # 着順 ~ 枠 + 全国勝率 の当てはめからの残り（マイナスほど乗り手と枠のわりに着が良い）。g1_prep と同じ考え方
    X = np.column_stack([(hr.lane == k).astype(float) for k in range(1, 7)] + [hr.nwr.fillna(hr.nwr.mean())])
    beta, *_ = np.linalg.lstsq(X, hr.position.values, rcond=None)
    Xa = np.column_stack([(h.lane == k).astype(float) for k in range(1, 7)] + [h.nwr.fillna(hr.nwr.mean())])
    h["resid"] = h.position - Xa @ beta
    return h


def motor_prev(h, mno, n=2):
    g = h[h.motor_no == mno]
    out = []
    for m in sorted(g.meet.unique())[-n:][::-1]:
        gm = g[g.meet == m].sort_values(["hd", "rno"])
        pos = "".join(str(int(p)) if pd.notna(p) else "・" for p in gm.position)
        exr = "".join(str(int(x)) if pd.notna(x) else "-" for x in gm.exr)
        res, exm = gm.resid.mean(), gm.exr.mean()
        lab = "良かった" if res <= -0.4 else "悪かった" if res >= 0.4 else "普通"
        if pd.notna(exm):
            lab += "・展示も出ていた" if exm <= 2.9 else "・展示は出ていない" if exm >= 4.1 else ""
        who = gm.racer_name.iloc[0].replace(" ", "").replace("　", "")
        out.append(f"{md(gm.hd.min())}〜 {who}({gm.grade.iloc[0]}・勝率{f2(gm.nwr.iloc[0])}) 着{pos} 展示順位{exr}(平均{f2(exm, 1)}) 残り{res:+.2f}→{lab}")
    return out


# ---------- 丸亀の得点率と準優ボーダー ----------
def marugame_points(m):
    past = m[(m.hd < DAY) & (m.race_name.str.contains("予選"))].copy()
    past["pt"] = past.position.map(PTS).fillna(0)
    past["acc"] = past.position.isna()
    allr = m[m.hd < DAY][["racer_no", "racer_name"]].drop_duplicates("racer_no")
    t = past.groupby("racer_no").agg(pts=("pt", "sum"), n=("pt", "size"), acc=("acc", "sum")).reset_index()
    t = allr.merge(t, on="racer_no", how="left").fillna({"pts": 0, "n": 0, "acc": 0})
    seq = m[m.hd < DAY].groupby("racer_no").apply(
        lambda g: " ".join("".join(("D" if "ドリーム" in rn else "") + (str(int(p)) if pd.notna(p) else "・")
                                   for p, rn in zip(gd.position, gd.race_name)) for _, gd in g.groupby("hd")))
    t["seq"] = t.racer_no.map(seq)
    today = m[m.hd == DAY]
    t["today_all"] = t.racer_no.map(today.groupby("racer_no").rno.apply(lambda s: ",".join(f"{r}R" for r in s)))
    tq = today[today.race_name.str.contains("予選")]
    t["today_q"] = t.racer_no.map(tq.groupby("racer_no").rno.apply(list)).apply(lambda x: x if isinstance(x, list) else [])
    t["active"] = t.racer_no.isin(today.racer_no)
    t["rate"] = t.pts / t.n.replace(0, np.nan)
    return t, tq


def lane_weights(jcd):
    r = pd.read_sql("select lane, avg(position=1) w from results where jcd=? and hd>=? and hd<? and position is not null group by lane",
                    con, params=(jcd, "20231101", DAY))
    return r.set_index("lane").w.reindex(range(1, 7)).values


def simulate(t, tq, w, sims, seed=0):
    """今日の予選の着順を枠だけの強さ（当地3年の枠別1着率）で並べ替えて回す（Plackett-Luce）。選手の力は入れていない目安。"""
    rng = np.random.default_rng(seed)
    races = sorted(tq.rno.unique())
    act = t[t.active].reset_index(drop=True)
    idx = {r: i for i, r in enumerate(act.racer_no)}
    base_pts, base_n = act.pts.values.astype(float), act.n.values.astype(float)
    tot = np.tile(base_pts, (sims, 1)); cnt = np.tile(base_n, (sims, 1))
    finish = {}  # (rno, racer_no) -> 着順の配列
    logw = np.log(w)
    for rno in races:
        g = tq[tq.rno == rno].sort_values("lane")
        key = logw[g.lane.values - 1] + rng.gumbel(size=(sims, len(g)))
        order = np.argsort(-key, axis=1)
        pos = np.empty_like(order); pos[np.arange(sims)[:, None], order] = np.arange(1, len(g) + 1)
        for j, rn in enumerate(g.racer_no.values):
            i = idx[rn]
            tot[:, i] += np.vectorize(PTS.get)(pos[:, j]); cnt[:, i] += 1
            finish[(rno, rn)] = pos[:, j]
    rate = tot / np.where(cnt == 0, np.nan, cnt)
    rr = np.where(np.isnan(rate), -1, rate)
    border = -np.sort(-rr, axis=1)[:, BORDER_RANK - 1]
    # 18位の値と同じ得点率は「乗る」側に数える（公式の同点の並べ方は反映できない）
    inside = rr >= border[:, None] - 1e-9
    return act, idx, rate, border, inside, finish


def main():
    out = ["# 候補8レースの節間の流れ・モーターの前の節・丸亀の勝負駆け（2026-11-18）", "",
           "`scripts/analysis/setsu_trend.py` で作成。観戦・検討用の目安で、舟券は買わない。少ない本数で良し悪しを断定しない。",
           "使った情報は 11/17 までの成績・直前情報と 11/18 の番組表だけ（今日の展示・オッズは入っていない）。", ""]
    m15 = load_meet("15", MEETS["15"]["start"], DAY)
    t, tq = marugame_points(m15)
    w = lane_weights("15")
    act, idx, rate, border, inside, finish = simulate(t, tq, w, a.sims)
    q = np.percentile(border, [10, 50, 90])

    # ---- 1. 得点率の順位表 ----
    t = t.sort_values(["rate", "pts"], ascending=False, na_position="last").reset_index(drop=True)
    t["rank"] = t.rate.rank(ascending=False, method="min")
    pin = dict(zip(act.racer_no, inside.mean(axis=0)))
    out += ["## 1. 丸亀の予選得点率（11/15〜11/17、ドリーム戦を除く）", "",
            "点は G1 の一般的な点（1着10・2着8・3着6・4着4・5着2・6着1、事故・欠場0）。公式の減点（F・L 等）と同点の並べ方は反映していない目安。",
            "11/15〜17 の丸亀で事故（着順なし）の記録は無かった。山田祐也は 11/15 の1走のあと番組に無い（途中帰郷とみられる）ので順位から外した。",
            "11/18 の 12R は番組上「一般」（予選でない）なので得点に数えていない。もし予選扱いなら、その6人（石丸・原田・佐藤隆・船岡・渡邉和・前田）の数字は変わる。",
            f"今日の予選（1〜11R）を枠だけの強さ（当地3年の枠別1着率 {'/'.join(f'{x:.0%}' for x in w)}）で {a.sims:,} 回並べ替えた試算で、"
            f"最終の18位の得点率は 10%点 {q[0]:.2f}・中央 {q[1]:.2f}・90%点 {q[2]:.2f}。今の18位は {t[t['rank']<=18].rate.min():.2f}。",
            "「準優確率」はその試算で18位以内（同点は乗る側に数えた）に入った割合。選手の力・モーターは入れていないので粗い目安。", "",
            "| 順 | 選手 | 節間（D=ドリーム） | 得点 | 走 | 得点率 | 今日の出走 | 今日の予選 | 準優確率（目安） |", "|---|---|---|---|---|---|---|---|---|"]
    for r in t.itertuples():
        rk = "-" if not r.active else int(r.rank)
        pq = "-" if not r.active else f"{pin.get(r.racer_no, 0):.0%}"
        name = r.racer_name + ("（帰郷？）" if not r.active else "")
        out.append(f"| {rk} | {name} | {r.seq} | {int(r.pts)} | {int(r.n)} | {f2(r.rate)} | {r.today_all if isinstance(r.today_all, str) else '-'} | {len(r.today_q)}走 | {pq} |")
    out.append("")

    # ---- 2・3. レースごと ----
    hm = {j: motor_hist(j, MEETS[j]["motor_since"], MEETS[j]["start"]) for j in MEETS}
    mts = {"11": load_meet("11", MEETS["11"]["start"], DAY), "15": m15}
    tt = t.set_index("racer_no")
    out += ["## 2. レースごと（節間の流れ・モーターの前の節・丸亀は勝負駆け）", "",
            "- 着順は日ごと（11/14〜、丸亀は 11/15〜）。展示は「タイム(レース内順位)」、ST は本番のスタート（F は F）。",
            "- 上向き／下向きは、前半と直近2走の平均着・平均展示順位の差で機械的に付けた目安。",
            "- モーターの前の節は、この節より前の直近2節（新しい順）。残り＝着順を枠と乗り手の全国勝率で説明した残り（マイナスほど良い）。",
            "  良かった＝残り -0.40 以下、悪かった＝+0.40 以上。展示順位はその節の6艇中の順位（成績ファイルの展示タイムから）。", ""]
    for jcd in ["11", "15"]:
        m = mts[jcd]
        for rno in MEETS[jcd]["races"]:
            g = m[(m.hd == DAY) & (m.rno == rno)].sort_values("lane")
            out += [f"### {VENUES[jcd]} {rno}R（{g.race_name.iloc[0]}）", ""]
            for r in g.itertuples():
                x = racer_trend_lines(m, r.racer_no)
                out.append(f"- **{r.lane} {r.racer_name}**（{int(r.motor_no)}号機）**{x['trend']}**")
                out.append(f"  - 着 {x['pos']}／展示 {x['ex']}")
                out.append(f"  - ST {x['st']}（平均{f2(x['st_mean'])}）／チルト {x['tilt']}／部品 {x['parts']}")
                mp = motor_prev(hm[jcd], int(r.motor_no))
                out.append("  - 機の前節: " + (" ／ ".join(mp) if mp else "記録なし"))
                if jcd == "15":
                    out.append("  - 勝負駆け: " + kakegake(tt, r.racer_no, rno, act, idx, inside, finish, q))
            out.append("")
    Path(a.out).write_text("\n".join(out) + "\n", encoding="utf-8")
    print("wrote", a.out)


def min_combos(nd):
    """2走合計で nd 点以上になる「一番悪い」着の組み合わせ（どちらかを1つ悪くすると届かないもの）。"""
    ok = lambda x, y: PTS[x] + PTS[y] >= nd
    res = []
    for x in range(1, 7):
        for y in range(x, 7):
            if ok(x, y) and not (y < 6 and ok(x, y + 1)) and not (x < y and ok(x + 1, y)):
                res.append(f"{x}着+{y}着")
    return "・".join(res)


def kakegake(tt, rn, rno, act, idx, inside, finish, q):
    r = tt.loc[rn]
    base = f"{f2(r.rate)}（{int(r['rank'])}位、{int(r.pts)}点/{int(r.n)}走）・今日の予選{len(r.today_q)}走"
    if rn not in idx or (rno, rn) not in finish:
        return base + "。このレースは得点に数えない番組。"
    i = idx[rn]; f = finish[(rno, rn)]
    probs = [inside[f == p, i].mean() if (f == p).sum() else np.nan for p in range(1, 7)]
    ps = " ".join(f"{p}着{pr:.0%}" for p, pr in zip(range(1, 7), probs))
    ok50 = [p for p, pr in zip(range(1, 7), probs) if pr >= 0.5]
    ok90 = [p for p, pr in zip(range(1, 7), probs) if pr >= 0.9]
    if probs[0] < 0.05:
        need = "1着でもほぼ届かない（圏外）"
    elif probs[5] >= 0.9:
        need = "6着でもほぼ乗る（当確に近い）"
    else:
        need = (f"五分は{max(ok50)}着以内" if ok50 else "1着でも五分に届かない") + "、" + (f"安全は{max(ok90)}着以内" if ok90 else "1着でも安全とは言えない")
    if len(r.today_q) == 1:
        rates = " ".join(f"{p}着{(r.pts + PTS[p]) / (r.n + 1):.2f}" for p in range(1, 7))
        detail = f"最終 {rates}"
    else:
        other = ",".join(f"{x}R" for x in r.today_q if x != rno)
        nd = [int(np.ceil(b * (r.n + 2) - r.pts - 1e-9)) for b in q]
        def c(n):
            return "6着2本でも可" if n <= 2 else "1着2本でも不足" if n > 20 else f"{n}点（{min_combos(n)}）"
        detail = f"もう1走は{other}。2走合計で 低め{q[0]:.2f}なら{c(nd[0])}、高め{q[2]:.2f}なら{c(nd[2])}"
    return f"{base}。**{need}**。{detail}。準優確率 {ps}"


if __name__ == "__main__":
    main()
