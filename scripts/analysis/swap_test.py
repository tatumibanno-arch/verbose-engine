"""notes/analysis/swap_plan.md の6通り。"""
import numpy as np, pandas as pd, sys
sys.path.insert(0, "scripts/analysis"); import trifecta as T
W = "/tmp/claude-0/-home-user-verbose-engine/968008a0-6ecf-5641-a3ff-006912b41722/scratchpad/trifecta/mb"
z = np.load(f"{W}/arrays.npz"); r = pd.read_parquet(f"{W}/races.parquet")
P = z["P"].astype(float); O = z["O"].astype(float); hit = r.hit.to_numpy(); pay = r.payout.to_numpy(); m = r.month.to_numpy()
ok = np.isfinite(O).all(1) & (O > 0).all(1) & (hit >= 0) & (m >= "2025-11") & (m <= "2026-10")
P, O, hit, pay, m = P[ok], O[ok], hit[ok], pay[ok], m[ok]
q = (1 / O) / (1 / O).sum(1, keepdims=True)
idx = {tuple(p): n for n, p in enumerate(T.PERMS.tolist())}
first1 = T.PERMS[:, 0] == 0
P1q = q[:, first1].sum(1); race = P1q >= 0.5
DISC = sorted({x for x in m if x <= "2026-04"}); VAL = sorted({x for x in m if x >= "2026-05"})
def report(name, stake, ret):   # stake, ret: per race arrays (円)
    out = []
    for lab, ms in [("発見", DISC), ("検証", VAL), ("12か月", DISC + VAL)]:
        sel = np.isin(m, ms) & (stake > 0)
        mm = pd.DataFrame({"m": m[sel], "s": stake[sel], "r": ret[sel]}).groupby("m").sum(); mr = mm.r / mm.s
        days = len(set(m[np.isin(m, ms)])) * 30.4
        out.append(f"{lab}: 1日{sel.sum()/days:.1f}件 回収率{ret[sel].sum()/stake[sel].sum():.3f} 勝ち月{(mr>1).sum()}/{len(mr)} 最悪月{mr.min():.2f}")
    print(name, " | ".join(out))
# A
pairs = [(x, y) for x in range(1, 6) for y in range(1, 6) if x != y]
for name, Rm, Rp in [("A1", 2, 1.2), ("A2", 2, 1.0), ("A3", 3, 1.2), ("A4", 3, 1.0)]:
    stake = np.zeros(len(P)); ret = np.zeros(len(P))
    for x, y in pairs:
        a, b = idx[(0, x, y)], idx[(0, y, x)]
        cond = race & (q[:, a] / q[:, b] >= Rm) & (P[:, a] / P[:, b] <= Rp)
        stake += cond * 100; ret += cond * (hit == b) * pay
    report(name, stake, ret)
# B
ones = np.where(first1)[0]
for name, k in [("B1", 1.5), ("B2", 2.0)]:
    stake = np.zeros(len(P)); ret = np.zeros(len(P))
    for i in np.where(race)[0]:
        o = ones[np.argsort(O[i, ones])]          # 1着1号艇を人気順
        c, S = o[0], o[1:3]
        syn = 1 / (1 / O[i, S]).sum()
        if P[i, S].sum() >= P[i, c] and syn >= k * O[i, c]:
            stake[i] = 100
            if hit[i] in S:
                w = (1 / O[i, hit[i]]) / (1 / O[i, S]).sum()   # 合成オッズ配分の比率
                ret[i] = pay[i] * w
    report(name, stake, ret)
