"""par_aff: 特徴量を作ってレース単位の配列(N,6)で保存。使う情報は全部そのレースより前に分かるもの。"""
import numpy as np, pandas as pd, sys
S = sys.argv[1]
e = pd.read_parquet(S+'/e.parquet'); r = pd.read_parquet(S+'/r.parquet')
d = e.merge(r, on=['hd','jcd','rno','lane'], how='left')
d = d.dropna(subset=['racer']); d['racer']=d.racer.astype(int)
d['o'] = pd.to_datetime(d.hd, format='%Y%m%d').map(pd.Timestamp.toordinal)
d = d.sort_values(['racer','o','rno']).reset_index(drop=True)   # 同じ日は場が1つなので rno 順=時間順
d['t2'] = np.where(d.pos.isna(), np.nan, (d.pos<=2).astype(float))
base = d[d.hd<'20160101'].groupby('lane').t2.mean()
d['adj'] = d.t2 - d.lane.map(base)          # 枠の有利を引いた2連対(＋なら期待以上)
a0 = d.adj.fillna(0).values; v = (~d.adj.isna()).values.astype(float)
g = d.racer.values
first = np.r_[True, g[1:]!=g[:-1]]
gid = np.cumsum(first)-1
def excl_cumsum(x, keys):
    """同じkeysの中で、自分より前の行だけの累積和"""
    s = pd.Series(x).groupby(keys).cumsum().values
    return s - x
S_all = excl_cumsum(a0, gid); N_all = excl_cumsum(v, gid)
vk = d.racer.astype(np.int64).values*100 + d.jcd.astype(int).values
S_v = excl_cumsum(a0, vk); N_v = excl_cumsum(v, vk)
mu = np.where(N_all>=60, S_all/np.maximum(N_all,1), np.nan)
K=20
d['aff'] = np.where((N_all>=60)&(N_v>=1), (S_v - N_v*mu)/(N_v+K), np.nan)   # 当地相性（収縮つき）
d['n_v'] = N_v
# 直近5走
cs = pd.Series(a0).groupby(gid).cumsum().values; cv = pd.Series(v).groupby(gid).cumsum().values
idx = np.arange(len(d)); pos_in = idx - np.maximum.reduce([np.where(first, idx, 0)]) 
start = pd.Series(np.where(first, idx, np.nan)).ffill().values.astype(int)
k = idx - start    # グループ内の通し番号
def prev(arr, back):  # arr[i-back] (グループ内なら), 無ければ0
    j = idx-back; ok = k>=back
    return np.where(ok, arr[np.maximum(j,0)], 0.0)
last5 = (cs - a0) - prev(cs,5) ; n5 = (cv - v) - prev(cv,5)
# last5 = 直前5走分 (cs[i-1]-cs[i-6])
last5 = np.where(k>=1, cs[np.maximum(idx-1,0)], 0) - np.where(k>=6, cs[np.maximum(idx-6,0)], 0)
n5 = np.where(k>=1, cv[np.maximum(idx-1,0)], 0) - np.where(k>=6, cv[np.maximum(idx-6,0)], 0)
d['mom'] = np.where((n5>=5)&(k>=6)&(N_all>=60), last5/np.maximum(n5,1) - mu, np.nan)
# 連続開催・疲労
po = np.r_[0, d.o.values[:-1]]; pj = np.r_[0, d.jcd.values[:-1]]
newmeet = first | ((d.o.values-po)>1) | (d.jcd.values!=pj)
mid = np.cumsum(newmeet)
d['mstart'] = pd.Series(d.o.values).groupby(mid).transform('min').values
d['dayno'] = d.o - d.mstart + 1                      # 開催何日目
# 開催前の休み日数（直前に走った日から）
prev_last = np.where(newmeet & ~first, po, np.nan)
rest = d.mstart.values - pd.Series(prev_last).groupby(mid).transform('max').values - 1
d['rest'] = rest
# 距離
vc = {'01':(36.4,139.3),'02':(35.8,139.7),'03':(35.7,139.9),'04':(35.6,139.7),'05':(35.65,139.5),'06':(34.7,137.6),'07':(34.8,137.2),'08':(34.9,136.8),'09':(34.7,136.5),'10':(36.2,136.2),'11':(35.0,135.9),'12':(34.6,135.5),'13':(34.7,135.4),'14':(34.2,134.6),'15':(34.3,133.8),'16':(34.5,133.8),'17':(34.3,132.3),'18':(34.0,131.8),'19':(33.95,130.9),'20':(33.9,130.7),'21':(33.9,130.7),'22':(33.6,130.4),'23':(33.4,130.0),'24':(32.9,129.9)}
bc = {'群馬':(36.4,139.1),'埼玉':(35.9,139.6),'東京':(35.7,139.7),'静岡':(35.0,138.4),'愛知':(35.2,137.0),'三重':(34.7,136.5),'福井':(36.1,136.2),'滋賀':(35.0,135.9),'大阪':(34.7,135.5),'兵庫':(34.7,135.2),'徳島':(34.1,134.6),'香川':(34.3,134.0),'岡山':(34.7,133.9),'広島':(34.4,132.5),'山口':(34.0,131.5),'福岡':(33.6,130.4),'佐賀':(33.3,130.3),'長崎':(32.8,129.9)}
vb = {'01':'群馬','02':'埼玉','03':'東京','04':'東京','05':'東京','06':'静岡','07':'愛知','08':'愛知','09':'三重','10':'福井','11':'滋賀','12':'大阪','13':'兵庫','14':'徳島','15':'香川','16':'岡山','17':'広島','18':'山口','19':'山口','20':'福岡','21':'福岡','22':'福岡','23':'佐賀','24':'長崎'}
def hav(a,b):
    la1,lo1,la2,lo2 = map(np.radians,[a[0],a[1],b[0],b[1]])
    h = np.sin((la2-la1)/2)**2+np.cos(la1)*np.cos(la2)*np.sin((lo2-lo1)/2)**2
    return 6371*2*np.arcsin(np.sqrt(h))
tab = {}
for br,bxy in bc.items():
    for j,vxy in vc.items(): tab[(br,j)] = hav(bxy,vxy)
d['dist'] = [tab.get((b,j),np.nan) for b,j in zip(d.branch.values, d.jcd.values)]
d['local'] = (d.branch.values == d.jcd.map(vb).values).astype(float)
d['local'] = np.where(d.branch.isna(), np.nan, d.local)
CUT = sys.argv[2] if len(sys.argv)>2 else '20231101'; END = sys.argv[3] if len(sys.argv)>3 else '99999999'
d = d[(d.hd>=CUT)&(d.hd<END)].copy()     # 検証に使う期間だけ残す(特徴量は全履歴から作成済み)
d.to_parquet(S+'/feat.parquet')
print(len(d)); print(d[['aff','mom','dayno','rest','dist','local','n_v']].describe().T)
