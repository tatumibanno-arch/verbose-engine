"""ライブ判定（lap_runner/lap_one/score_one）とバックテスト（three_signals）の比較。読み取りのみ。"""
import sys, glob, os, re, json, sqlite3, warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
sys.path.insert(0, 'scripts'); import lap_one, score_one
LOG = 'data/live/log'
dates = sorted(re.search(r'(\d{8})_s0_lapone', f).group(1) for f in glob.glob(f'{LOG}/*_lapone.csv'))
SIG = {'lap': '知らせ1', 'score': '知らせ2', 'score4': '知らせ3'}
def rule_of(r):
    return 'score4' if 'score4' in r else ('score' if '（score）' in r else 'lap')
# 1) ライブ
live = {}; judged = {}
for d in dates:
    l = pd.read_csv(f'{LOG}/{d}_s0_lapone.csv', dtype=str); l['rule'] = l.result.map(rule_of)
    for rule, g in l.groupby('rule'):
        judged[(d, rule)] = set(zip(g.jcd, g.rno.astype(int)))
        live[(d, rule)] = set(zip(g[g.result.str.startswith('該当')].jcd, g[g.result.str.startswith('該当')].rno.astype(int)))
# 2) バックテスト（three_signals のまま）
src = open('scripts/analysis/three_signals.py').read(); head = src.split("R=lambda s,col")[0]
ns = {}; exec(head, ns); S = ns['S']; X = ns['X']; key = ns['key']
print('ob_ent に入っている最新日', pd.read_parquet('/tmp/claude-0/-home-user-verbose-engine/968008a0-6ecf-5641-a3ff-006912b41722/scratchpad/n2/ob_ent.parquet', columns=['hd']).hd.max())
bt = lambda d, sg: set(zip(S[(S.hd == d) & (S.sig == sg)].jcd, S[(S.hd == d) & (S.sig == sg)].rno.astype(int)))
# 3) 統計列を補って（ライブと同じ load_static）バックテストを作り直す
for d in dates:
    st = score_one.load_static(d)
    m = X.hd == d
    for q, i in (('nw', 0), ('mo', 1), ('c2r', 2)):
        v = np.array([pd.to_numeric((st.get((j, int(r)), {}).get(int(l)) or (None,)*3)[i], errors='coerce') for j, r, l in zip(X[m].jcd, X[m].rno, X[m].lane)], dtype=float)
        X[q] = pd.to_numeric(X[q], errors='coerce').astype(float); X.loc[m, q] = v
rk = lambda col, asc=True: X.groupby(key)[col].rank(ascending=asc, method='average')
X['r_nw'] = rk('nw', False); X['r_mo'] = rk('mo', False); X['r_c2'] = rk('c2r', False)
ns['X'] = X
exec("def picks" + src.split("def picks")[1].split("R=lambda s,col")[0], ns); S2 = ns['S']
bt2 = lambda d, sg: set(zip(S2[(S2.hd == d) & (S2.sig == sg)].jcd, S2[(S2.hd == d) & (S2.sig == sg)].rno.astype(int)))
print('\n== 表1: 日付 / 種別 | ライブ該当 | ライブが判定したレース数 | BT(そのまま) | BT(統計補正) | 重なり(ライブ∧BT補正) | 重なり(ライブ判定済みレース内のBT補正)')
for d in dates:
    for rule, sg in SIG.items():
        L = live.get((d, rule), set()); J = judged.get((d, rule), set()); B = bt(d, sg); B2 = bt2(d, sg)
        print(d, rule, len(L), len(J), len(B), len(B2), len(L & B2), len(J & B2), '| ライブのみ', sorted(L - B2), '| BTのみ(判定済内)', sorted((J & B2) - L), '| BTのみ(未判定)', sorted(B2 - J))
# 4) 入力の比較（11/24 のみ DB に確定値あり）
print('\n== 表2: ログ時点のオリジナル展示 vs DB(夜の取り込み) の差')
con = sqlite3.connect('data/boatrace.db')
for d in dates:
    o = pd.read_csv(f'{LOG}/{d}_s0_oriten.csv', dtype={'hd': str, 'jcd': str})
    o = o[o.lane.notna()]; o['lane'] = o.lane.astype(int); o['rno'] = o.rno.astype(int)
    db_o = pd.read_sql(f"select jcd,rno,lane,lap,mawari,chokusen from oriten where hd='{d}'", con)
    if db_o.empty: print(d, 'DB に oriten 無し（夜の取り込み前）'); continue
    m = o.merge(db_o, on=['jcd', 'rno', 'lane'], suffixes=('_log', '_db'))
    for c in ('lap', 'mawari', 'chokusen'):
        a, b = pd.to_numeric(m[c + '_log'], errors='coerce'), pd.to_numeric(m[c + '_db'], errors='coerce')
        print(d, c, '行', len(m), '一致', int(((a - b).abs() < 1e-9).sum()), 'ログ欠×DBあり', int((a.isna() & b.notna()).sum()), 'ログあり×DB欠', int((a.notna() & b.isna()).sum()), '値違い', int(((a - b).abs() > 1e-9).sum()))
    # ライブ判定をDBの確定値で再現
    bi = pd.read_sql(f"select jcd,rno,lane,exhibition_time from beforeinfo where hd='{d}'", con); bi['exhibition_time'] = pd.to_numeric(bi.exhibition_time, errors='coerce')
    st = score_one.load_static(d); rep = {r: set() for r in SIG}; rep_log = {r: set() for r in SIG}
    for (j, r), g in db_o.groupby(['jcd', 'rno']):
        rows = g.to_dict('records'); b = bi[(bi.jcd == j) & (bi.rno == r)].to_dict('records'); s = st.get((j, int(r)), {})
        for rule, fn in (('lap', lambda: lap_one.judge(rows, b)), ('score', lambda: score_one.judge(rows, b, s)), ('score4', lambda: score_one.judge4(rows, b, s))):
            x = fn()
            if x and x.get('combo'): rep[rule].add((j, int(r)))
    for rule, sg in SIG.items():
        print(d, rule, 'ライブ判定を確定データで再現', len(rep[rule]), '／BT補正', len(bt2(d, sg)), '／一致', len(rep[rule] & bt2(d, sg)))
    # 判定時点のスナップショットでの再現（ログのoriten、展示はDB確定）
    snap = o.sort_values('checked_at').groupby(['jcd', 'rno', 'lane']).tail(1)
    for (j, r), g in snap.groupby(['jcd', 'rno']):
        rows = g.assign(lap=pd.to_numeric(g.lap, errors='coerce'), mawari=pd.to_numeric(g.mawari, errors='coerce'), chokusen=pd.to_numeric(g.chokusen, errors='coerce')).to_dict('records')
        b = bi[(bi.jcd == j) & (bi.rno == r)].to_dict('records'); s = st.get((j, int(r)), {})
        for rule, fn in (('lap', lambda: lap_one.judge(rows, b)), ('score', lambda: score_one.judge(rows, b, s)), ('score4', lambda: score_one.judge4(rows, b, s))):
            x = fn()
            if x and x.get('combo'): rep_log[rule].add((j, int(r)))
    for rule in SIG: print(d, rule, 'ログのスナップショット+確定展示で再現', len(rep_log[rule]), '／ライブ実際', len(live[(d, rule)]), '／一致', len(rep_log[rule] & live[(d, rule)]))
# 5) 判定した時刻の範囲
for d in dates:
    l = pd.read_csv(f'{LOG}/{d}_s0_lapone.csv', dtype=str); l['rule'] = l.result.map(rule_of)
    print(d, '各ルールの最初のチェック時刻', l.groupby('rule').checked_at.min().to_dict(), 'ルール別レース数', l.groupby('rule').apply(lambda g: g[['jcd', 'rno']].drop_duplicates().shape[0]).to_dict())
    tot = pd.read_sql(f"select count(distinct jcd||'-'||rno) n from entries where hd='{d}'", con).n[0]; print(d, '開催レース総数', tot)

# 6) BT に載らないライブ該当の原因（払戻/ W の途中）
print('\n== 表3: ライブ該当レースが BT(S) に無い理由')
for d in dates:
    for rule, sg in SIG.items():
        for (j, r) in sorted(live[(d, rule)] - bt2(d, sg)):
            has_pay = pd.read_sql(f"select count(*) n from payouts where hd='{d}' and jcd='{j}' and rno={r} and ticket_type='sanrensho'", con).n[0]
            bi_ = pd.read_sql(f"select count(*) n from beforeinfo where hd='{d}' and jcd='{j}' and rno={r}", con).n[0]
            print(d, rule, j, r, '3連単払戻の行', has_pay, 'beforeinfo行', bi_)

print('\n== 表4: 11/24 BT(そのまま)知らせ3 の5レース詳細')
d = '20261124'; l = pd.read_csv(f'{LOG}/{d}_s0_lapone.csv', dtype=str)
for (j, r) in sorted(bt(d, '知らせ3')):
    x = l[(l.jcd == j) & (l.rno.astype(int) == r) & l.result.str.contains('score4')]
    cl = S2[(S2.hd == d) & (S2.jcd == j) & (S2.rno == r) & (S2.sig == '知らせ3')]
    g = X[(X.hd == d) & (X.jcd == j) & (X.rno == r)]
    print(j, r, 'ライブscore4ログ:', x[['checked_at', 'result']].values.tolist(), '| 補正後BTに残る' , len(cl) > 0, '| mo列がNaNの艇数(補正前)', int(X.loc[g.index, 'mo'].isna().sum()))
print('\n07-9 lap 一周(DB):')
w = pd.read_sql(f"select lane,lap from oriten where hd='{d}' and jcd='07' and rno=9 order by lap", con); print(w.values.tolist(), 'lap4-lap3 =', repr(sorted(w.lap)[3] - sorted(w.lap)[2]))

print('\n== 表5: 統計の同一性 ob_ent(BTの源) vs score_one.load_static(ライブ) 20261123')
E_ = pd.read_parquet('/tmp/claude-0/-home-user-verbose-engine/968008a0-6ecf-5641-a3ff-006912b41722/scratchpad/n2/ob_ent.parquet'); E_ = E_[E_.hd == '20261123']
st = score_one.load_static('20261123'); n = bad = 0
for r in E_.itertuples():
    v = st.get((r.jcd, int(r.rno)), {}).get(int(r.lane))
    if not v: continue
    c2 = (r.c2 + 2) / (r.cn + 5) if pd.notna(r.cn) else None
    for a, b in ((pd.to_numeric(r.nw, errors='coerce'), v[0]), (pd.to_numeric(r.mo, errors='coerce'), v[1]), (c2, v[2])):
        n += 1; bad += not ((pd.isna(a) and (b is None or pd.isna(b))) or (pd.notna(a) and b is not None and abs(float(a) - float(b)) < 1e-9))
print('比較', n, '不一致', bad)
