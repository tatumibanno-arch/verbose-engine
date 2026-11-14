"""穴狙い5回目（2026-11-14）: 決まり手から出した「選手の癖」の筋書き V1〜V12（notes/patterns.md に先に固定）。

    python scripts/analysis/ana_kimarite.py IN.parquet [シャッフル回数=200]     （IN は ana_prep.py の表）

癖は「レース日より前の365日」の成績だけで作る（前夜に分かる）。払戻は締切時オッズの合成買い＝実際より甘い。
"""
import sys, itertools
import numpy as np, pandas as pd, duckdb

SPLIT = "20260501"; BAND = (15, 60); KSHR = 20
df = pd.read_parquet(sys.argv[1]).reset_index(drop=True)
NPERM = int(sys.argv[2]) if len(sys.argv) > 2 else 200
N = len(df); disc = (df.hd < SPLIT).values
c = duckdb.connect(); P = "data/parquet"

# ---- 選手×コース×日 の集計（決まり手は勝者のもの）----
c.sql(f"""create table daily as
select cast(json_extract(r.data,'$.racer_number') as int) racer, cast(json_extract(r.data,'$.entrance_position') as int) course,
  strptime(r.hd,'%Y%m%d')::date d, count(*) n,
  sum(case when r.position=1 and k.kimarite='まくり' then 1 else 0 end) w_mk,
  sum(case when r.position=1 and k.kimarite='まくり差し' then 1 else 0 end) w_mz,
  sum(case when r.position=1 and k.kimarite='差し' then 1 else 0 end) w_ss,
  sum(case when r.position=1 and k.kimarite='逃げ' then 1 else 0 end) w_ng,
  sum(case when coalesce(r.position,9)<>1 and k.kimarite='まくり' then 1 else 0 end) l_mk,
  sum(case when coalesce(r.position,9)<>1 and k.kimarite in ('差し','まくり差し') then 1 else 0 end) l_ss
from '{P}/results_*.parquet' r join '{P}/kimarite_*.parquet' k on k.hd=r.hd and k.jcd=cast(r.jcd as int) and k.rno=r.rno
where r.hd>='20240901' and json_extract(r.data,'$.entrance_position') is not null group by 1,2,3""")
base = c.sql("select course, sum(w_mk)/sum(n) mk, sum(w_mz)/sum(n) mz, sum(w_ss)/sum(n) ss, sum(w_ng)/sum(n) ng, sum(l_mk)/sum(n) lmk, sum(l_ss)/sum(n) lss from daily where d<'2025-11-01' group by 1 order by 1").df().set_index("course")
print("コースごとの平均（2024-09〜2025-10）:\n", (base * 100).round(1).to_string())
c.register("races", df[["hd", "jcd", "rno"]])
f = c.sql(f"""with e as (select e.hd, cast(e.jcd as int) jcd, e.rno, e.lane, cast(e.racer_no as int) racer, strptime(e.hd,'%Y%m%d')::date d
    from '{P}/entries_*.parquet' e join races x on x.hd=e.hd and x.jcd=cast(e.jcd as int) and x.rno=e.rno)
select e.hd, e.jcd, e.rno, e.lane, coalesce(sum(y.n),0) n, coalesce(sum(y.w_mk),0) w_mk, coalesce(sum(y.w_mz),0) w_mz, coalesce(sum(y.w_ss),0) w_ss,
  coalesce(sum(y.w_ng),0) w_ng, coalesce(sum(y.l_mk),0) l_mk, coalesce(sum(y.l_ss),0) l_ss
from e left join daily y on y.racer=e.racer and y.course=e.lane and y.d < e.d and y.d >= e.d - 365 group by 1,2,3,4""").df()
for a, b in (("mk", "w_mk"), ("mz", "w_mz"), ("ss", "w_ss"), ("ng", "w_ng"), ("lmk", "l_mk"), ("lss", "l_ss")):
    f[a] = (f[b] + KSHR * f.lane.map(base[a])) / (f.n + KSHR)
w = f.pivot_table(index=["hd", "jcd", "rno"], columns="lane", values=["mk", "mz", "ss", "ng", "lmk", "lss", "n"]); w.columns = [f"{a}{int(l)}" for a, l in w.columns]
df = df.merge(w.reset_index(), on=["hd", "jcd", "rno"], how="left")
assert len(df) == N
top = lambda col, q: (df[col] >= df[col].quantile(1 - q)).values
print("しきい値（上位1割）: " + "、".join(f"{n}={df[cn].quantile(.9)*100:.1f}%" for n, cn in (("④まくり", "mk4"), ("③まくり", "mk3"), ("①まくられ", "lmk1"), ("①差され", "lss1"), ("②差し", "ss2"), ("⑤まくり差し", "mz5"), ("①逃げ", "ng1"), ("③まくり差し", "mz3"))))

ALL = ["".join(map(str, x)) for x in itertools.permutations(range(1, 7), 3)]; CI = {x: i for i, x in enumerate(ALL)}
Q = df[["t_" + x for x in ALL]].fillna(0).values; wcb = df.cb.str.replace("-", "").map(CI).values
qw, pay, S = df.qw.values, df.payout.values / 100, df.S.values; p1 = df.p_H1.values; fav = p1 >= .5


def tk(*pats):
    m = np.zeros(120, bool)
    for p in pats:
        for x in ALL:
            if all(a == "x" or a == b for a, b in zip(p, x)): m[CI[x]] = True
    return m


FEAT = ["mk4", "mk3", "lmk1", "lss1", "ss2", "mz5", "ng1", "mz3"]
TOP = {(k, q): top(k, q) for k in FEAT for q in (.1, .3)}


def stories(T, perm=None):
    g = (lambda k, q: T[(k, q)]) if perm is None else (lambda k, q: T[(k, q)][perm])
    return {
        "V1 ④まくり屋→頭④": (g("mk4", .1), tk("4xx")),
        "V2 ④まくり屋→5-4,5-1": (g("mk4", .1), tk("54x", "51x")),
        "V3 ④まくり屋→2-1,2-4": (g("mk4", .1), tk("21x", "24x")),
        "V4 ③まくり屋→頭③": (g("mk3", .1), tk("3xx")),
        "V5 ③まくり屋→4-3,4-1": (g("mk3", .1), tk("43x", "41x")),
        "V6 ①まくられやすい＋①人気→3-1,3-4,4-1,4-5": (fav & g("lmk1", .1), tk("31x", "34x", "41x", "45x")),
        "V7 ①差されやすい＋①人気→2-1": (fav & g("lss1", .1), tk("21x")),
        "V8 ②差し屋＋①差されやすい→2-1,2-3": (g("ss2", .1) & g("lss1", .3), tk("21x", "23x")),
        "V9 ⑤まくり差し屋＋④まくり屋→5-1,5-4": (g("mz5", .1) & g("mk4", .3), tk("51x", "54x")),
        "V10 ①よく逃げるのに①頭6割以下→頭①": (g("ng1", .1) & (p1 <= .6), tk("1xx")),
        "V11 ③まくり差し屋→3-1": (g("mz3", .1), tk("31x")),
        "V12 ①人気＋④まくり屋＋①まくられやすい→4-1,4-5": (fav & g("mk4", .1) & g("lmk1", .3), tk("41x", "45x")),
    }


RET = {}
def evaluate(st):
    rows = []
    for name, (cond, m) in st.items():
        if name not in RET:
            sq = Q[:, m].sum(axis=1); hit = m[wcb]
            with np.errstate(all="ignore"):
                RET[name] = (np.where(hit, qw / sq * pay, 0.0), 1 / sq, sq / S)
        ret, co, pm = RET[name]
        for bn, bm in (("全部", cond), ("合成15〜60倍", cond & (co >= BAND[0]) & (co <= BAND[1]))):
            r = []
            for per in (disc, ~disc):
                x = bm & per
                r += [int(x.sum()), int((ret[x] > 0).sum())] + ([pm[x].mean(), (ret[x] > 0).mean(), ret[x].mean()] if x.any() else [np.nan] * 3)
            rows.append([name, bn] + r + [np.median(co[bm]) if bm.any() else np.nan, bm])
    return rows


cols = ["筋書き", "帯", "n発", "的中発", "見立て発", "実際発", "回収発", "n検", "的中検", "見立て検", "実際検", "回収検", "合成中央"]
rows = evaluate(stories(TOP)); t = pd.DataFrame([r[:-1] for r in rows], columns=cols); t["月あたり"] = ((t.n発 + t.n検) / 12).round(0)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 50)
print(t.round(3).to_string(index=False))
ok = lambda a, th: int(((a.回収発 >= th) & (a.回収検 >= th) & (a.的中発 >= 15) & (a.的中検 >= 10)).sum())
print(f"\n{len(t)}区分のうち 両期間とも回収1.0以上（的中 発見15・検証10以上）: {ok(t,1)}、両期間0.9以上: {ok(t,.9)}")
rng = np.random.default_rng(0); a1 = []; a9 = []
for it in range(NPERM):
    perm = np.arange(N)
    for msk in (disc, ~disc):
        idx = np.where(msk)[0]; perm[idx] = rng.permutation(idx)
    t2 = pd.DataFrame([r[:-1] for r in evaluate(stories(TOP, perm))], columns=cols); a1.append(ok(t2, 1)); a9.append(ok(t2, .9))
print(f"シャッフル{NPERM}回: 両方1.0以上 平均{np.mean(a1):.2f}（最大{max(a1)}）、両方0.9以上 平均{np.mean(a9):.2f}（最大{max(a9)}）")
print("\n== 月ごとの回収（帯=全部）=="); mon = df.hd.str[:6]
for r in rows:
    if r[1] != "全部" or not r[-1].any(): continue
    g = pd.DataFrame({"m": mon[r[-1]], "r": RET[r[0]][0][r[-1]]}).groupby("m").r.agg(["size", "mean"])
    print(r[0], " ".join(f"{m[2:]}:{v:.2f}({int(n)})" for m, n, v in zip(g.index, g["size"], g["mean"])), f" 勝ち月{int((g['mean']>1).sum())}/{len(g)}")

# ---- 癖そのものは効いているか（市場の見立てと比べる。帯ごと）----
print("\n== 癖の強さごとの、頭の実際の率と市場の見立て（全期間）==")
for lane, col, lab in ((4, "mk4", "④のまくり率"), (3, "mk3", "③のまくり率"), (2, "ss2", "②の差し率"), (1, "ng1", "①の逃げ率"), (1, "lmk1", "①のまくられ率")):
    b = pd.qcut(df[col], [0, .3, .7, .9, 1], labels=["下位3割", "中", "上位3割〜1割", "上位1割"], duplicates="drop")
    hit = (df.wh == lane).values; ret = RET_H = np.where(hit, df.qw.values / df[f"q_H{lane}"].values * pay, 0)
    g = pd.DataFrame({"b": b, "hit": hit, "pm": df[f"p_H{lane}"].values, "ret": ret}).groupby("b", observed=True).agg(n=("hit", "size"), 実際=("hit", "mean"), 見立て=("pm", "mean"), 回収=("ret", "mean"))
    print(lab, f"→ 頭{lane}"); print(g.round(3).to_string())
