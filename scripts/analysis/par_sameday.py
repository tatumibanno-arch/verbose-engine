"""同日・同開催の「前のレースの成績」を使った3連単の条件探し（探索のみ）。
前走成績はそのレースの締切前に分かる情報。払戻は確定払戻（締切前オッズとは違う）。
"""
import sqlite3, itertools, sys
import numpy as np, pandas as pd

DB = '/home/user/verbose-engine/data/boatrace.db'
SPLIT = '20251101'
c = sqlite3.connect(DB)
r = pd.read_sql("select r.hd,r.jcd,r.rno,r.lane,r.position,e.racer_no from results r join entries e "
                "on e.hd=r.hd and e.jcd=r.jcd and e.rno=r.rno and e.lane=r.lane where r.hd>='20231001'", c)
r['d'] = pd.to_datetime(r.hd)
pts = {1: 10, 2: 8, 3: 6, 4: 4, 5: 2, 6: 1}
r['pt'] = r.position.map(pts).fillna(0)
r['dnf'] = r.position.isna().astype(int)
r['w'] = (r.position == 1).astype(int)
r = r.sort_values(['racer_no', 'jcd', 'd', 'rno']).reset_index(drop=True)
g = r.groupby(['racer_no', 'jcd'])
gap = g.d.diff().dt.days
r['meet'] = ((gap.isna()) | (gap > 2)).groupby([r.racer_no, r.jcd]).cumsum()
gm = r.groupby(['racer_no', 'jcd', 'meet'])
r['n_prev'] = gm.cumcount()
r['pt_prev'] = gm.pt.cumsum() - r.pt
r['dnf_prev'] = gm.dnf.cumsum() - r.dnf
r['last_pt'] = gm.pt.shift(1)
gt = r.groupby(['racer_no', 'jcd', 'hd'])
r['nt'] = gt.cumcount()
r['pt_t'] = gt.pt.cumsum() - r.pt
r['w_t'] = gt.w.cumsum() - r.w
r['last_pt_t'] = np.where(r.nt > 0, r.last_pt, np.nan)
r['m_avg'] = np.where(r.n_prev >= 2, r.pt_prev / r.n_prev.replace(0, np.nan), np.nan)
r['t_avg'] = np.where(r.nt >= 1, r.pt_t / r.nt.replace(0, np.nan), np.nan)
r = r[r.hd >= '20231101']
key = ['hd', 'jcd', 'rno']
def mat(col):
    return r.drop_duplicates(key+['lane']).set_index(key+['lane'])[col].unstack('lane')
M = {k: mat(k) for k in ['m_avg', 't_avg', 'w_t', 'last_pt_t', 'dnf_prev', 'n_prev', 'nt']}
M['nt'] = M['nt']
idx = M['m_avg'].index
ok = (M['nt'].notna().sum(axis=1) == 6)
for k in M: M[k] = M[k][ok.to_numpy()]
idx = M['m_avg'].index
A = {k: M[k].reindex(columns=range(1, 7)).to_numpy(float) for k in M}
pay = pd.read_sql("select hd,jcd,rno,combination,payout from payouts where ticket_type='sanrensho' and hd>='20231101'", c)
pay = pay.set_index(key)
pay = pay[~pay.index.duplicated()]
pj = pay.reindex(idx)
okp = pj.payout.notna().to_numpy()
win = np.array([[int(x) for x in s.split('-')] if isinstance(s, str) else [0, 0, 0] for s in pj.combination])
payout = pj.payout.fillna(0).to_numpy(float)
hd = np.array([i[0] for i in idx])
disc = hd < SPLIT; val = ~disc
month = np.array([h[:6] for h in hd])
R = len(idx); print('races', R, file=sys.stderr)
rows = np.arange(R)

def sel_lanes(score, head, k, excl_head=True):
    """score (R,6) higher=better; return bool mask (R,6) of top-k lanes excluding head"""
    s = np.where(np.isnan(score), 5.0, score).copy() + np.random.RandomState(0).rand(*score.shape) * 1e-6 * 0
    s[rows, head - 1] = -99
    order = np.argsort(-s, axis=1, kind='stable')[:, :k]
    m = np.zeros((R, 6), bool)
    for j in range(k): m[rows, order[:, j]] = True
    return m
def all_others(head):
    m = np.ones((R, 6), bool); m[rows, head - 1] = False; return m

def evaluate(gate, head, S2, S3):
    """tickets head-x-y, x in S2, y in S3, x!=y; return per-race bet and return"""
    n = S2.sum(1) * S3.sum(1) - (S2 & S3).sum(1)
    h = win[:, 0] == head
    a = S2[rows, np.clip(win[:, 1] - 1, 0, 5)]
    b = S3[rows, np.clip(win[:, 2] - 1, 0, 5)]
    hit = h & a & b & okp & (win[:, 0] > 0)
    use = gate & okp & (n > 0)
    bet = np.where(use, n * 100, 0)
    ret = np.where(use & hit, payout, 0)
    return use, bet, ret, hit & use

def summarize(use, bet, ret, hit):
    out = {}
    for nm, msk in (('all', np.ones(R, bool)), ('disc', disc), ('val', val)):
        b = bet[msk].sum(); rr = ret[msk].sum()
        out[nm] = (b / 100, rr / b if b else np.nan)
    def ex(k, msk):
        rt = np.sort(ret[msk])[::-1].copy(); rt[:k] = 0
        b = bet[msk].sum(); return (ret[msk].sum() - np.sort(ret[msk])[::-1][:k].sum()) / b if b else np.nan
    out['val_ex'] = [ex(k, val) for k in (1, 3, 5)]
    out['all_ex'] = [ex(k, np.ones(R, bool)) for k in (1, 3, 5)]
    m = pd.DataFrame({'m': month, 'b': bet, 'r': ret}).groupby('m').sum()
    m = m[m.b > 0]
    mr = m.r / m.b
    out['months'] = (int((mr > 1).sum()), len(mr))
    out['nraces'] = int(use.sum()); out['hit'] = hit.sum() / max(use.sum(), 1)
    out['nhit'] = int(hit.sum())
    return out

# ---- gates (relative to head boat) ----
def feat(name, head):
    return A[name][rows, head - 1]
def gates_for(head):
    G = {}
    mav = feat('m_avg', head); w_t = feat('w_t', head); lp = feat('last_pt_t', head)
    dn = feat('dnf_prev', head); npv = feat('n_prev', head); nt = feat('nt', head); tav = feat('t_avg', head)
    G['none'] = np.ones(R, bool)
    G['won_today'] = w_t >= 1
    G['last_today_bad(4-6/DNF)'] = lp <= 4
    G['last_today_good(1-2)'] = lp >= 8
    G['meet_avg>=7'] = mav >= 7
    G['meet_avg>=6'] = mav >= 6
    G['meet_avg<=4.5'] = mav <= 4.5
    G['meet_avg<=3.5'] = mav <= 3.5
    G['dnf_in_meet'] = dn >= 1
    G['first_race_of_meet'] = npv == 0
    G['has_today_races'] = nt >= 1
    G['today_avg>=8'] = tav >= 8
    G['today_avg<=4'] = tav <= 4
    # competitors relative: head's meet avg minus best of others
    oth = A['m_avg'].copy(); oth[rows, head - 1] = np.nan
    with np.errstate(all='ignore'):
        omax = np.nanmax(np.where(np.isnan(oth), -1, oth), axis=1)
        omean = np.nanmean(oth, axis=1)
    G['head_meet_edge>=1.5(vs others mean)'] = (mav - omean) >= 1.5
    G['head_meet_edge<=-1.5'] = (mav - omean) <= -1.5
    G['head_best_in_meet_by>=0.5'] = (mav - omax) >= 0.5
    # others weak today: nobody else won today
    ow = A['w_t'].copy(); ow[rows, head - 1] = 0
    G['others_won_today_0'] = (np.nansum(ow, 1) == 0) & (nt >= 1)
    G['others_won_today_>=2'] = np.nansum(ow, 1) >= 2
    return G

S2opts = {}; S3opts = {}
def selectors(head):
    S = {}
    S['all'] = all_others(head)
    for k in (2, 3, 4):
        S[f'top{k}_meet'] = sel_lanes(A['m_avg'], head, k)
        S[f'top{k}_today'] = sel_lanes(np.where(np.isnan(A['t_avg']), A['m_avg'], A['t_avg']), head, k)
    S['bot2_meet'] = sel_lanes(-A['m_avg'], head, 2)
    return S

res = []
tried = 0
for head in (1, 2, 3):
    G = gates_for(head); S = selectors(head)
    names = list(S)
    for gn, gate in G.items():
        gate = gate & ~np.isnan(feat('m_avg', head)) if gn not in ('none', 'won_today', 'last_today_bad(4-6/DNF)', 'last_today_good(1-2)', 'dnf_in_meet', 'first_race_of_meet', 'has_today_races', 'today_avg>=8', 'today_avg<=4', 'others_won_today_0', 'others_won_today_>=2') else gate
        if gate.sum() < 200: 
            continue
        for s2, s3 in itertools.product(names, names):
            if s2 == 'all' and s3 == 'all' and False: pass
            use, bet, ret, hit = evaluate(gate, head, S[s2], S[s3])
            tried += 1
            o = summarize(use, bet, ret, hit)
            res.append((f'head={head} gate={gn} 2nd={s2} 3rd={s3}', o))
print('TRIED', tried)
def fmt(o):
    return (f"bets={o['all'][0]:.0f} races={o['nraces']} ROI all={o['all'][1]:.3f} disc={o['disc'][1]:.3f} val={o['val'][1]:.3f}(bets {o['val'][0]:.0f}) "
            f"valEx1/3/5={'/'.join(f'{x:.2f}' for x in o['val_ex'])} allEx1/3/5={'/'.join(f'{x:.2f}' for x in o['all_ex'])} "
            f"winmonths={o['months'][0]}/{o['months'][1]} hit={o['hit']:.3f}")
cand = [(n, o) for n, o in res if o['val'][0] >= 150 and o['val'][1] > 1.0 and o['val_ex'][1] > 1.0]
print('PROMISING (val>1, valEx3>1, valbets>=150):', len(cand))
for n, o in sorted(cand, key=lambda x: -x[1]['val'][1])[:10]:
    print(n, fmt(o))
print('--- top 10 by val ROI among bets>=150 val, disc ROI>0.9')
c2 = [(n, o) for n, o in res if o['val'][0] >= 150]
for n, o in sorted(c2, key=lambda x: -x[1]['val'][1])[:10]:
    print(n, fmt(o))
print('--- top 10 by discovery ROI (val bets>=150)')
for n, o in sorted(c2, key=lambda x: -(x[1]['disc'][1] if not np.isnan(x[1]['disc'][1]) else 0))[:10]:
    print(n, fmt(o))
print('--- top 5 by all ROI excl top3 (bets>=300)')
c3 = [(n, o) for n, o in res if o['all'][0] >= 300]
for n, o in sorted(c3, key=lambda x: -x[1]['all_ex'][1])[:5]:
    print(n, fmt(o))
