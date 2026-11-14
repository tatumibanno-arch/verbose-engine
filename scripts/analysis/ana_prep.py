"""穴狙いパターン探しの下ごしらえ（2026-11-14）。Parquet から直接、1レース1行の表を作る（SQLite は使わない）。

    python scripts/analysis/ana_prep.py OUT.parquet

列:
  艇ごと（k=1..6）: et{k} 展示タイム, er{k} 展示順位, sc{k} スタ展の進入, sx{k} スタ展ST（Fは負）,
                    ms{k} 今節平均ST（前日までの本番、2走以上）, mo{k} モーター2連率, gr{k} 級(4=A1..1=B2),
                    nw{k} 全国勝率, cw{k} その枠（コース）の期別1着率, cs{k} その枠の期別平均ST
  レース: wind 風速, wdir 風向コード(1-16, 17=無風), wave 波高, waku 枠なりか
  市場（締切時の3連単オッズ。実際に買う時点より後の情報＝甘い）:
    形ごと（H{i}=頭i、P{i}{j}=i-j-全）: q_ 形の 1/オッズ 合計（合成オッズ=1/q）, p_ 市場の見立て確率,
    r_ 合成買い（払戻が同じになる配分）の払戻倍率（外れは0）, f_ 100円均等買いの払戻倍率
返還（F・欠場）は考えず外れ扱い（少し辛め）。3連単の払戻が無いレースは除く。
"""
import sys, glob, itertools
import duckdb, numpy as np, pandas as pd

FROM, TO = "20251101", "20261031"
out = sys.argv[1]
c = duckdb.connect()
P = "data/parquet"
K = ["hd", "jcd", "rno"]

# ---- 直前情報 ----
bi = c.sql(f"""select hd, cast(jcd as int) jcd, rno, lane, exhibition_time et, start_course sc, start_st sx,
  cast(json_extract(data,'$.wind_speed') as double) wind, cast(json_extract(data,'$.wind_direction') as int) wdir,
  cast(json_extract(data,'$.wave_height') as double) wave
  from '{P}/beforeinfo_*.parquet' where hd between '{FROM}' and '{TO}'""").df()
bi["er"] = bi.groupby(K).et.rank(method="min")

# ---- 番組表 ----
en = c.sql(f"""select hd, cast(jcd as int) jcd, rno, lane, cast(racer_no as int) racer, grade,
  cast(json_extract(data,'$.motor_2ren_rate') as double) mo, cast(json_extract(data,'$.national_win_rate') as double) nw
  from '{P}/entries_*.parquet' where hd between '{FROM}' and '{TO}'""").df()
en["gr"] = en.grade.map({"A1": 4, "A2": 3, "B1": 2, "B2": 1})

# ---- 今節平均ST: 同じ場・同じ選手の、前日まで6日以内の本番ST ----
rs = c.sql(f"""select hd, cast(jcd as int) jcd, cast(json_extract(data,'$.racer_number') as int) racer,
  cast(json_extract(data,'$.st_timing') as double) st
  from '{P}/results_*.parquet' where hd between '20251020' and '{TO}' and json_extract(data,'$.st_timing') is not null""").df()
rs["d"] = pd.to_datetime(rs.hd, format="%Y%m%d")
day = rs.groupby(["jcd", "racer", "d"]).st.agg(["sum", "count"]).reset_index()
q = en[["hd", "jcd", "racer"]].drop_duplicates(); q["d0"] = pd.to_datetime(q.hd, format="%Y%m%d")
m = q.merge(day, on=["jcd", "racer"]); m = m[(m.d < m.d0) & (m.d >= m.d0 - pd.Timedelta(days=6))]
ms = m.groupby(["hd", "jcd", "racer"])[["sum", "count"]].sum().reset_index()
ms["ms"] = np.where(ms["count"] >= 2, ms["sum"] / ms["count"], np.nan)
en = en.merge(ms[["hd", "jcd", "racer", "ms"]], on=["hd", "jcd", "racer"], how="left")

# ---- 期別のコース別成績（レース日より前に算出期間が終わった最新の期）----
st = c.sql(f"select hd shd, cast(racer_no as int) racer, data from '{P}/racer_stats_*.parquet' where hd>='20250430'").df()
import json
rows = []
for shd, racer, d in st.itertuples(index=False):
    j = json.loads(d)
    for k in range(1, 7):
        rows.append((shd, racer, k, j.get(f"c{k}_win_rate"), j.get(f"c{k}_avg_st"), j.get(f"c{k}_starts")))
cs = pd.DataFrame(rows, columns=["shd", "racer", "lane", "cw", "cs", "cn"])
cs.loc[cs.cn.fillna(0) < 8, ["cw", "cs"]] = np.nan
shds = sorted(cs.shd.unique())
en["shd"] = [max([s for s in shds if s < h], default=None) for h in en.hd]
en = en.merge(cs, on=["shd", "racer", "lane"], how="left")

b = bi.merge(en[K + ["lane", "mo", "nw", "gr", "ms", "cw", "cs"]], on=K + ["lane"], how="inner")
wide = b.pivot_table(index=K, columns="lane", values=["et", "er", "sc", "sx", "ms", "mo", "gr", "nw", "cw", "cs"], dropna=False)
wide.columns = [f"{a}{int(l)}" for a, l in wide.columns]
race = b.groupby(K).agg(wind=("wind", "first"), wdir=("wdir", "first"), wave=("wave", "first"), n=("lane", "size"),
                        waku=("sc", lambda s: False)).reset_index()
wk = b.assign(ok=b.sc == b.lane).groupby(K).ok.all().rename("waku").reset_index()
race = race.drop(columns="waku").merge(wk, on=K)
race = race[race.n == 6].merge(wide.reset_index(), on=K)

# ---- 市場（3連単）と払戻 ----
o = c.sql(f"""select hd, cast(jcd as int) jcd, rno, combination cb, odds from '{P}/odds_*.parquet'
  where ticket_type='sanrensho' and odds>0 and hd between '{FROM}' and '{TO}'""").df()
o["q"] = 1 / o.odds
o["h"] = o.cb.str[0].astype(int); o["s"] = o.cb.str[2].astype(int)
py = c.sql(f"""select hd, cast(jcd as int) jcd, rno, combination cb, payout from '{P}/payouts_*.parquet'
  where ticket_type='sanrensho' and hd between '{FROM}' and '{TO}'""").df()
py = py.sort_values("payout").drop_duplicates(K, keep=False)  # 同着で2通りのレースは除く
tot = o.groupby(K).agg(S=("q", "sum"), nc=("q", "size")).reset_index()
tot = tot[tot.nc >= 100]
H = o.pivot_table(index=K, columns="h", values="q", aggfunc="sum"); H.columns = [f"q_H{i}" for i in H.columns]
Pq = o.pivot_table(index=K, columns=["h", "s"], values="q", aggfunc="sum"); Pq.columns = [f"q_P{i}{j}" for i, j in Pq.columns]
T = o.pivot_table(index=K, columns="cb", values="q", aggfunc="sum"); T.columns = ["t_" + x.replace("-", "") for x in T.columns]  # 1点ごとの 1/オッズ
mk = tot.merge(H.reset_index(), on=K).merge(Pq.reset_index(), on=K).merge(T.reset_index(), on=K)
w = py.merge(o[K + ["cb", "q"]], on=K + ["cb"], how="inner").rename(columns={"q": "qw"})
w["wh"] = w.cb.str[0].astype(int); w["ws"] = w.cb.str[2].astype(int); w["wt"] = w.cb.str[4].astype(int)
mk = mk.merge(w[K + ["cb", "payout", "qw", "wh", "ws", "wt"]], on=K, how="inner")
shapes = [f"H{i}" for i in range(1, 7)] + [f"P{i}{j}" for i, j in itertools.permutations(range(1, 7), 2)]
new = {}
for s in shapes:
    hit = (mk.wh == int(s[1])) if s[0] == "H" else ((mk.wh == int(s[1])) & (mk.ws == int(s[2])))
    ncomb = 20 if s[0] == "H" else 4
    new["p_" + s] = mk["q_" + s] / mk.S
    new["r_" + s] = np.where(hit, mk.qw / mk["q_" + s] * mk.payout / 100, 0.0)
    new["f_" + s] = np.where(hit, mk.payout / 100 / ncomb, 0.0)
mk = pd.concat([mk, pd.DataFrame(new)], axis=1)
df = race.merge(mk, on=K, how="inner")
df.to_parquet(out)
print("レース数", len(df), "期間", df.hd.min(), df.hd.max(), "列", df.shape[1])
print("今節STあり(1号艇)", df.ms1.notna().mean().round(3), "コース別あり(1号艇)", df.cw1.notna().mean().round(3))
print("全体: 頭ごとの 市場見立て / 実際 / 合成買い回収")
for i in range(1, 7):
    print(i, round(df[f"p_H{i}"].mean(), 4), round((df.wh == i).mean(), 4), round(df[f"r_H{i}"].mean(), 3))
