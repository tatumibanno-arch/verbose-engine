"""総当たり検証の共通採点。ルールは Python ファイルで、bets(df, odds) を定義する。
    python scripts/analysis/hunt_eval.py RULE.py --period A_disc --agent t001
  bets(df, odds) -> DataFrame[hd,jcd,rno,ticket_type,combination,amount]
    df   : 1レース1行（結果と LEAK_ 列は渡さない）。odds: B期間だけ 3連単の締切時オッズ（hd,jcd,rno,cb,odds）、A期間は None
    ticket_type: tansho fukusho nirensho nirenfuku wide sanrensho sanrenfuku
    combination: 着順どおり '1-2-3'。順不同の券種（nirenfuku, wide, sanrenfuku）は小さい順 '1-3-4'。amount は100円単位
返還は考えず外れ扱い（少し辛め）。実行のたびに ledger.jsonl に1行残す（何通り試したかの記録）。
"""
import argparse, importlib.util, json, hashlib, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd
D = Path("/home/user/hunt"); H = Path("/home/user/hunt_hold")
ap = argparse.ArgumentParser(); ap.add_argument("rule"); ap.add_argument("--period", required=True); ap.add_argument("--agent", default="x"); ap.add_argument("--json", action="store_true")
a = ap.parse_args()
PERIODS = ["A_disc", "A_valid", "B_disc", "B_valid", "A_hold", "B_hold"]
if a.period not in PERIODS: sys.exit("period は " + " ".join(PERIODS[:4]))
hold = a.period.endswith("hold")
if hold and os.environ.get("HUNT_KEY") != "orchestrator-only": sys.exit("この期間は見られません（最後の確認用）")
led = D / "ledger.jsonl"
if a.period.endswith("valid") and led.exists():
    n = sum(1 for l in open(led) if (j := json.loads(l))["agent"] == a.agent and j["period"].endswith("valid"))
    if n >= 3: sys.exit("確かめる期間の採点は1体3回まで。これ以上は回せません")
base = H if hold else D
df = pd.read_parquet(base / f"{a.period}_races.parquet"); py = pd.read_parquet(base / f"{a.period}_payouts.parquet")
odds = pd.read_parquet(base / f"{a.period}_odds3t.parquet") if a.period[0] == "B" else None
safe = df.drop(columns=[c for c in df.columns if c.startswith("LEAK_") or c in ("w1", "w2", "w3")])
if a.period[0] == "A": safe = safe.drop(columns=[c for c in safe.columns if c.startswith("ph") or c == "nodds"])
spec = importlib.util.spec_from_file_location("rule", a.rule); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
b = m.bets(safe, odds)
b = b[["hd", "jcd", "rno", "ticket_type", "combination", "amount"]].copy()
b["jcd"] = b.jcd.astype(int); b["rno"] = b.rno.astype(int); b["amount"] = b.amount.astype(float)
assert (b.amount > 0).all() and (b.amount % 100 == 0).all(), "amount は100円単位"
assert not b.duplicated(["hd", "jcd", "rno", "ticket_type", "combination"]).any(), "同じ買い目が重複"
py = py.groupby(["hd", "jcd", "rno", "ticket_type", "combination"], as_index=False).payout.max()
b = b.merge(py, on=["hd", "jcd", "rno", "ticket_type", "combination"], how="left")
b["ret"] = b.amount * b.payout.fillna(0) / 100; b["ym"] = b.hd.str[:6]
allm = sorted(df.ym.unique()); mo = b.groupby("ym").agg(s=("amount", "sum"), r=("ret", "sum")).reindex(allm).fillna(0)
mroi = (mo.r / mo.s.where(mo.s > 0)); rc = b.groupby(["hd", "jcd", "rno"]).agg(s=("amount", "sum"), r=("ret", "sum"))
stake, ret = float(b.amount.sum()), float(b.ret.sum()); top = float(b.ret.max()) if len(b) else 0.0
half = len(allm) // 2; h1 = mo.iloc[:half]; h2 = mo.iloc[half:]
out = dict(agent=a.agent, rule=Path(a.rule).name, sha=hashlib.md5(open(a.rule, "rb").read()).hexdigest()[:8], period=a.period,
           races=int(len(rc)), all_races=int(len(df)), bets=int(len(b)), stake=stake, ret=ret, roi=round(ret / stake, 4) if stake else None,
           hit_race_rate=round(float((rc.r > 0).mean()), 4) if len(rc) else None, months=len(allm), months_with_bets=int((mo.s > 0).sum()),
           win_months=int((mroi > 1).sum()), worst_month=round(float(mroi.min()), 3) if mroi.notna().any() else None,
           roi_wo_top_hit=round((ret - top) / stake, 4) if stake else None,
           roi_first_half=round(float(h1.r.sum() / h1.s.sum()), 3) if h1.s.sum() else None, roi_second_half=round(float(h2.r.sum() / h2.s.sum()), 3) if h2.s.sum() else None,
           avg_stake_per_race=round(stake / len(rc)) if len(rc) else None)
if not hold:
    with open(led, "a") as f: f.write(json.dumps(out, ensure_ascii=False) + "\n")
if a.json: print(json.dumps(out, ensure_ascii=False))
else:
    print(f"[{a.period}] {out['rule']} 対象{out['races']}/{out['all_races']}レース 買い目{out['bets']}点 購入{stake:,.0f}円 払戻{ret:,.0f}円")
    print(f"  回収率 {out['roi']}  的中レース率 {out['hit_race_rate']}  勝ち月 {out['win_months']}/{out['months_with_bets']}（全{out['months']}か月） 最悪月 {out['worst_month']}")
    print(f"  最大の1本を除いた回収率 {out['roi_wo_top_hit']}  前半 {out['roi_first_half']} / 後半 {out['roi_second_half']}  1レース平均 {out['avg_stake_per_race']}円")
