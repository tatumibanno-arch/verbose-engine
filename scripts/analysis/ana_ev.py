"""「期待値が高いのに売れていない艇」を今日のレースから探す（2026-11-21）。判断材料のみ。

    python scripts/analysis/ana_ev.py IN.parquet OUT.txt     （IN は ana_prep.py の表＝学習と過去の成績確認に使う）

1. 前夜に分かる材料（各艇のその枠の期別1着率・全国勝率・モーター2連率・級）だけで「どの艇が頭か」の確率を学習（2025-11〜2026-04）。
2. 過去（2026-05〜10）で「見立て÷市場の頭シェア」の帯ごとに、その艇の頭20点を合成買いした回収を出す（締切時オッズ＝実際より甘い）。
3. 今日の締切前のレースで、今の3連単オッズの頭シェアと見立てを比べ、見立てのほうが高い順に並べる。
"""
import sys, json, subprocess, datetime as dt
from pathlib import Path
import numpy as np, pandas as pd, duckdb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from common import VENUES  # noqa: E402
from ingest import parse_schedule_text  # noqa: E402
import scrape_web as sw  # noqa: E402

df = pd.read_parquet(sys.argv[1]).reset_index(drop=True); OUT = open(sys.argv[2], "w")
def P(*a):
    print(*a); print(*a, file=OUT); OUT.flush()
FE = [f"{n}{k}" for n in ("cw", "nw", "mo", "gr") for k in range(1, 7)]
med = df[FE].median()
X = df[FE].fillna(med).values; y = df.wh.values; disc = (df.hd < "20260501").values
m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0)).fit(X[disc], y[disc])
pr = m.predict_proba(X)
ll = lambda p, yy: -np.mean(np.log(np.clip(p[np.arange(len(yy)), yy - 1], 1e-9, 1)))
mk = np.column_stack([df[f"p_H{k}"].values for k in range(1, 7)])
P(f"検証期間(2026-05〜10)の当てやすさ（小さいほど良い）: 見立て {ll(pr[~disc], y[~disc]):.4f} / 市場 {ll(mk[~disc], y[~disc]):.4f}")
rows = []
for k in range(1, 7):
    rows.append(pd.DataFrame({"lane": k, "per": np.where(disc, "発見", "検証"), "ratio": pr[:, k - 1] / mk[:, k - 1], "mp": mk[:, k - 1], "pp": pr[:, k - 1],
                              "hit": (y == k), "ret": df[f"r_H{k}"].values, "co": 1 / df[f"q_H{k}"].values}))
b = pd.concat(rows); b = b[b.per == "検証"]
b["帯"] = pd.cut(b.ratio, [0, .7, .9, 1.1, 1.3, 1.6, 2.0, 99], labels=["〜0.7", "0.7〜0.9", "0.9〜1.1", "1.1〜1.3", "1.3〜1.6", "1.6〜2.0", "2.0〜"])
P("\n== 過去の成績（検証期間）: 見立て÷市場 の帯ごと。その艇の頭20点を合成買い ==")
P(b.groupby("帯", observed=True).agg(艇数=("hit", "size"), 見立て=("pp", "mean"), 市場=("mp", "mean"), 実際=("hit", "mean"), 回収=("ret", "mean")).round(3).to_string())
P("\n== 同じく、市場の頭シェア5%以上の艇だけ ==")
P(b[b.mp >= .05].groupby("帯", observed=True).agg(艇数=("hit", "size"), 見立て=("pp", "mean"), 市場=("mp", "mean"), 実際=("hit", "mean"), 回収=("ret", "mean")).round(3).to_string())

# ---- 今日 ----
JST = dt.timezone(dt.timedelta(hours=9)); now = dt.datetime.now(JST); hd = now.strftime("%Y%m%d")
f = ROOT / f"data/lzh_cache/B/b{hd[2:]}.lzh"
text = subprocess.run(["7z", "x", "-so", str(f)], capture_output=True).stdout.decode("cp932", errors="replace")
e = pd.DataFrame(parse_schedule_text(text))
c = duckdb.connect(); PQ = str(ROOT / "data/parquet")
st = c.sql(f"select cast(racer_no as int) racer_no, data from '{PQ}/racer_stats_*.parquet' where hd=(select max(hd) from '{PQ}/racer_stats_*.parquet')").df()
J = {r: json.loads(d) for r, d in st.itertuples(index=False)}
def cw(r, l):
    j = J.get(r, {}); return j.get(f"c{l}_win_rate") if (j.get(f"c{l}_starts") or 0) >= 8 else np.nan
e["cw"] = [cw(r, l) for r, l in zip(e.racer_no, e.lane)]
e["nw"] = e.national_win_rate; e["mo"] = e.motor_2ren_rate; e["gr"] = e.grade.map({"A1": 4, "A2": 3, "B1": 2, "B2": 1})
w = e.pivot_table(index=["jcd", "rno", "close_time"], columns="lane", values=["cw", "nw", "mo", "gr"], dropna=False); w.columns = [f"{a}{int(l)}" for a, l in w.columns]
w = w.reset_index(); w = w[w.close_time > (now + dt.timedelta(minutes=12)).strftime("%H:%M")].sort_values("close_time").reset_index(drop=True)
pt = m.predict_proba(w[FE].astype(float).fillna(med).values)
names = e.set_index(["jcd", "rno", "lane"]).racer_name
res = []
for i, r in w.iterrows():
    url = f"{sw.BASE}odds3t?jcd={r.jcd}&hd={hd}&rno={r.rno}"
    h = subprocess.run(["curl", "-sS", "-f", "-m", "60", "--retry", "1", url], capture_output=True).stdout.decode("utf-8", errors="replace")
    q = {cb.replace("-", ""): 1 / o for _, cb, o, _ in sw.parse_odds3t(h) if o}
    if len(q) < 100: continue
    S = sum(q.values())
    for k in range(1, 7):
        qh = sum(v for kk, v in q.items() if kk[0] == str(k))
        res.append(dict(レース=f"{VENUES[r.jcd]}{r.rno}R", 締切=r.close_time, 艇=k, 選手=names.get((r.jcd, r.rno, k), ""), 見立て=pt[i, k - 1], 市場=qh / S, 比=pt[i, k - 1] / (qh / S), 頭の合成=1 / qh,
                        材料=f"枠1着率{r[f'cw{k}']}% 全国{r[f'nw{k}']} モーター{r[f'mo{k}']}% 級{r[f'gr{k}']}"))
t = pd.DataFrame(res); t.to_csv(sys.argv[2].replace(".txt", ".csv"), index=False)
P(f"\n== 今日（{now:%H:%M}時点、{t.レース.nunique()}レース）: 見立てのほうが市場より高い艇（市場の頭5%以上、比の高い順）==")
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 60)
P(t[(t.市場 >= .05)].sort_values("比", ascending=False).head(25).round(3).to_string(index=False))
P("\n== 逆に、市場のほうが見立てよりずっと高い艇（売れすぎ候補、市場10%以上）==")
P(t[(t.市場 >= .10)].sort_values("比").head(12).round(3).to_string(index=False))
