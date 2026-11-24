"""par_aff: 事前に決めた15仮説を検証期間(2023-11〜2026-11)で評価。オッズ不使用。"""
import numpy as np, pandas as pd, sys
S = sys.argv[1]
d = pd.read_parquet(S+'/feat.parquet'); p = pd.read_parquet(S+'/p.parquet')
d = d.sort_values(['hd','jcd','rno','lane'])
cnt = d.groupby(['hd','jcd','rno']).lane.transform('size'); d = d[cnt==6]
keys = d[['hd','jcd','rno']].values[::6]
R = d[['hd','jcd','rno']].iloc[::6].reset_index(drop=True)
def A(c): return np.asarray(d[c].to_numpy(dtype=object if c=='grade' else float)).reshape(-1,6)
nw, grade, aff, mom, day, rest, dist, loc, nv = A('nw'), A('grade'), A('aff'), A('mom'), A('dayno'), A('rest'), A('dist'), A('local'), A('n_v')
nw = np.nan_to_num(nw, nan=0.0)
p = p.drop_duplicates(['hd','jcd','rno'], keep='first')
R = R.merge(p, on=['hd','jcd','rno'], how='left'); ok = R.comb.notna().values
res = np.array([[int(x) for x in c.split('-')] if isinstance(c,str) else [0,0,0] for c in R.comb]); pay = R.payout.fillna(0).values
N = len(R); idx = np.arange(N)
print('races', N, 'with payout', ok.sum())
mon = R.hd.str[:6].values; hdv = R.hd.values; mid = np.median(R.hd.astype(int).values.astype(float)); half = R.hd.astype(int).values < mid
# --- ticket generators: return (N,T,3) lane numbers
def top_others(excl_lanes, m):
    """レースごとに excl に含まれない枠を nw降順(同点は小さい枠先)で並べた (N,5-ish) 枠番"""
    sc = nw.copy().astype(float); sc = sc - np.arange(6)*1e-6
    for L in excl_lanes: sc[idx if np.isscalar(L) else idx, (np.asarray(L)-1) if not np.isscalar(L) else L-1] = -9
    return np.argsort(-sc, 1)+1
def G_head1():
    o = top_others([1], None); a,b,c = o[:,0],o[:,1],o[:,2]; one = np.ones(N,int)
    return np.stack([np.stack([one,a,b],1),np.stack([one,b,a],1),np.stack([one,a,c],1),np.stack([one,b,c],1)],1)
def G_anti1():
    o = top_others([1], None); a,b,c = o[:,0],o[:,1],o[:,2]; one = np.ones(N,int)
    return np.stack([np.stack([a,one,b],1),np.stack([a,one,c],1)],1)
def G_k1(k):  # k は枠番(スカラ or 配列)
    k = np.broadcast_to(np.asarray(k), (N,)); o = top_others([1,k], None); a,b = o[:,0],o[:,1]; one = np.ones(N,int)
    return np.stack([np.stack([one,k,a],1),np.stack([one,k,b],1),np.stack([k,one,a],1),np.stack([k,one,b],1)],1)
def G_ab(): pass
def evaluate(name, m, T):
    m = m & ok; n = m.sum()
    if n < 30: return dict(name=name, n=int(n))
    t = T[m]; r = res[m]; hit = (t == r[:,None,:]).all(2).any(1)
    ret = np.where(hit, pay[m], 0).astype(float); stake = 100.0*T.shape[1]
    roi = ret.sum()/(stake*n); h = half[m]
    o = np.sort(ret)[::-1]
    ex1 = o[1:].sum()/(stake*n); ex3 = o[3:].sum()/(stake*n)
    ms = pd.Series(ret).groupby(mon[m]).agg(['sum','size']); mr = ms['sum']/(stake*ms['size'])
    return dict(name=name, n=int(n), tk=T.shape[1], hit=hit.mean(), roi=roi, se=ret.std()/stake/np.sqrt(n), h1=ret[h].sum()/(stake*h.sum()) if h.sum() else np.nan, h2=ret[~h].sum()/(stake*(~h).sum()) if (~h).sum() else np.nan, ex1=ex1, ex3=ex3, wm=f'{(mr>1).sum()}/{len(mr)}')
Tall = {'head1':G_head1(), 'anti1':G_anti1()}
k_best = lambda lo,hi: None
rows = []
def add(name, m, T): rows.append(evaluate(name, m, T))
# 基準: 全レース
add('BASE head1 全レース', np.ones(N,bool), Tall['head1']); add('BASE anti1 全レース', np.ones(N,bool), Tall['anti1'])
add('BASE 1-2系(②×①) 全レース', np.ones(N,bool), G_k1(2)); add('BASE ③×① 全レース', np.ones(N,bool), G_k1(3))
a1 = (grade[:,0]=='A1'); AFF=0.04; MOM=0.25
good = lambda c: (nv[:,c]>=30)
# 1 ①A1で当地相性が悪い -> ①飛び
add('H1 ①A1 相性悪(<=-.04) anti1', a1 & (aff[:,0]<=-AFF) & good(0), Tall['anti1'])
# 2 ①(A1以外) 相性良(>=.04) -> ①頭
add('H2 ①非A1 相性良 head1', ~a1 & (aff[:,0]>=AFF) & good(0), Tall['head1'])
# 3 ② 相性良 / 4 ③ 相性良
add('H3 ②相性良 ①②流し', (aff[:,1]>=AFF) & good(1), G_k1(2))
add('H4 ③相性良 ①③流し', (aff[:,2]>=AFF) & good(2), G_k1(3))
# 5 ①A1 絶不調 / 6 ①絶好調
add('H5 ①A1 直近5走コールド anti1', a1 & (mom[:,0]<=-MOM), Tall['anti1'])
add('H6 ①ホット head1', (mom[:,0]>=MOM), Tall['head1'])
add('H7 ②ホット ①②流し', (mom[:,1]>=MOM), G_k1(2))
# 8 疲労: ①が5日目以降 / 9 連続開催(休み<=1日) / 10 遠征700km以上
add('H8 ①開催5日目以降 anti1', (day[:,0]>=5), Tall['anti1'])
add('H9 ①連続開催(休み<=1日) anti1', (rest_ := np.nan_to_num(A('rest'),nan=99))[:,0]<=1, Tall['anti1'])
add('H10 ①遠征700km以上 anti1', dist[:,0]>=700, Tall['anti1'])
# 11 地元①(A1以外) head1 / 12 ①他所者で2-4に地元
add('H11 ①地元 非A1 head1', (loc[:,0]==1)&~a1, Tall['head1'])
lm = (loc[:,1:4]==1)&(loc[:,[0]]==0); has = lm.any(1)
sc = np.where(lm, nw[:,1:4], -9); kk = sc.argmax(1)+2
add('H12 ①他所者&2-4に地元 地元×①流し', has, G_k1(kk))
# 13/14/15 組合せ
add('H13 ①A1 相性悪&コールド(<=-.15) anti1', a1 & (aff[:,0]<=-AFF) & good(0) & (mom[:,0]<=-0.15), Tall['anti1'])
add('H14 ①A1 相性悪&4日目以降 anti1', a1 & (aff[:,0]<=-AFF) & good(0) & (day[:,0]>=4), Tall['anti1'])
add('H15 ①相性良&ホット(>=.15) head1', (aff[:,0]>=AFF) & good(0) & (mom[:,0]>=0.15), Tall['head1'])
df = pd.DataFrame(rows)
pd.set_option('display.width',250); pd.set_option('display.max_columns',30)
print(df.round(3).to_string(index=False))
print('aff quantiles ①', np.nanquantile(aff[:,0],[.05,.1,.5,.9,.95]).round(3), 'mom', np.nanquantile(mom[:,0],[.05,.1,.5,.9,.95]).round(3))
