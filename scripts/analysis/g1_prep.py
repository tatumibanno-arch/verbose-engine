"""G1・SG の節の資料（モーター・選手・場の傾向・節間の気配）を1本で作る。運用者の観戦・趣味用。
使い方:
  python scripts/analysis/g1_prep.py --jcd 11 --from 20261114 --to 20261119 --title "びわこ G1 開設74周年記念 びわこ大賞" --out notes/reads/biwako-taisho-2026-11.md
毎晩その日の成績を取り込んだあとに同じコマンドで作り直す（節間の章が増える）。
前検（公式「モーター抽選結果・前検タイム」）は初回に取り、notes/reads/sources/ に保存して以後はそれを読む。
使うのは「その時点までに分かっている情報」だけ。舟券は買わない。少ない本数で良し悪しを断定しない。
"""
import argparse, json, re, sqlite3, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from common import VENUES

COMPASS = {1: "北", 3: "北東", 5: "東", 7: "南東", 9: "南", 11: "南西", 13: "西", 15: "北西"}
ap = argparse.ArgumentParser()
ap.add_argument("--jcd", required=True); ap.add_argument("--from", dest="d0", required=True); ap.add_argument("--to", dest="d1", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--title", default="")
ap.add_argument("--motor-since", default="", help="現モーターの使用開始日 YYYYMMDD（省略時は公式2連率が0に戻った日を自動で探す）")
ap.add_argument("--no-web", action="store_true")
a = ap.parse_args()
con = sqlite3.connect("data/boatrace.db", timeout=600)
VN = VENUES.get(a.jcd, a.jcd)


def pct(x):
    return "-" if x is None or pd.isna(x) else f"{x:.0%}"


def f2(x, n=2):
    return "-" if x is None or pd.isna(x) else f"{x:.{n}f}"


def prev_day(hd, n=1):
    return (pd.Timestamp(hd) - pd.Timedelta(days=n)).strftime("%Y%m%d")


# ---------- 前検（公式） ----------
def zenken():
    p = Path(f"notes/reads/sources/zenken_{a.jcd}_{a.d0}.json")
    if p.exists():
        return json.loads(p.read_text())
    if a.no_web:
        return None
    import requests
    import warnings
    from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
    warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
    url = f"https://www.boatrace.jp/owpc/pc/race/rankingmotor?jcd={a.jcd}&hd={a.d0}"
    try:
        html = requests.get(url, timeout=30).text
    except Exception as ex:
        print("前検ページ取得失敗", ex); return None
    s = BeautifulSoup(html, "lxml")
    rows = []
    for tr in s.select("table tbody tr"):
        c = [re.sub(r"\s+", " ", td.get_text(" ", strip=True)) for td in tr.find_all("td")]
        if len(c) >= 9 and c[1].isdigit():
            rows.append(dict(racer_no=int(c[1]), name=c[2].replace(" ", "").replace("　", ""), grade=c[3].strip(), motor_no=int(c[4]),
                             motor_2ren=float(c[5].strip(" %")), boat_no=int(c[6]), boat_2ren=float(c[7].strip(" %")),
                             pre_time=float(c[8]) if re.match(r"^\d", c[8]) else None))
    if not rows:
        return None
    txt = re.sub(r"\s+", " ", s.get_text(" ", strip=True))
    m1 = re.search(r"モーター抽選結果 (?!・)(.*?) 前検タイム (.*?)(?: スマートフォン版| お好み| PAGE TOP|$)", txt)
    out = dict(url=url, rows=rows, comment_motor=m1.group(1) if m1 else "", comment_time=m1.group(2) if m1 else "")
    p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    return out


def assen():
    """前検の前でも分かる「出場予定レーサー一覧」（公式）。モーターは未定。"""
    p = Path(f"notes/reads/sources/assen_{a.jcd}_{a.d0}.json")
    if p.exists():
        return json.loads(p.read_text())
    if a.no_web:
        return None
    import requests
    url = f"https://www.boatrace.jp/owpc/pc/race/assen?jcd={a.jcd}&hd={a.d0}"
    try:
        html = requests.get(url, timeout=30).text
    except Exception as ex:
        print("あっせんページ取得失敗", ex); return None
    txt = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "|", html))
    rows = [dict(racer_no=int(n), name=nm.replace(" ", "").replace("\u3000", ""), grade=g)
            for n, nm, g in re.findall(r"\|(\d{4})\|[ |]*([^|]+?)[ |]*\|[ |]*級別：[ |]*(A1|A2|B1|B2)", txt)]
    if not rows:
        return None
    out = dict(url=url, rows=rows)
    p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    return out


Z = zenken()
zdf = pd.DataFrame(Z["rows"]) if Z else pd.DataFrame()

# ---------- 番組表（この節） ----------
e = pd.read_sql("select hd,rno,lane,racer_no,racer_name,grade,motor_no,boat_no,close_time,race_name,data from entries where jcd=? and hd between ? and ?",
                con, params=(a.jcd, a.d0, a.d1))
if not e.empty:
    d = e.data.map(json.loads)
    for k in ["branch", "age", "national_win_rate", "local_win_rate", "motor_2ren_rate"]:
        e[k] = d.map(lambda x: x.get(k))
AS = None
if zdf.empty and e.empty:
    AS = assen()
    if not AS:
        sys.exit("前検も番組表も出場予定一覧もまだありません（出場選手が分からない）")
    field = pd.DataFrame(AS["rows"]).assign(motor_no=-1, motor_2ren=np.nan, pre_time=np.nan)
elif not zdf.empty:
    field = zdf[["racer_no", "name", "grade", "motor_no", "motor_2ren", "pre_time"]].copy()
else:
    g = e.sort_values("hd").groupby("racer_no").first().reset_index()
    field = pd.DataFrame(dict(racer_no=g.racer_no, name=g.racer_name, grade=g.grade, motor_no=g.motor_no, motor_2ren=g.motor_2ren_rate, pre_time=np.nan))
field["racer_no"] = field.racer_no.astype(int); field["motor_no"] = field.motor_no.astype(int)
if not e.empty:
    g = e.sort_values("hd").groupby("racer_no").first()
    for k in ["branch", "age", "national_win_rate", "local_win_rate"]:
        field[k] = field.racer_no.map(g[k].to_dict())
else:
    for k in ["branch", "age", "national_win_rate", "local_win_rate"]:
        field[k] = None
RN = set(field.racer_no)

# ---------- モーター（この場・現モーターの使用開始から前節まで） ----------
since = a.motor_since
h = pd.read_sql("""select e.hd, e.rno, e.lane, e.racer_no, e.racer_name, e.grade, e.motor_no,
                          json_extract(e.data,'$.motor_2ren_rate') m2, json_extract(e.data,'$.national_win_rate') nwr,
                          r.position, json_extract(r.data,'$.exhibition_time') ex, json_extract(r.data,'$.st_timing') st,
                          json_extract(r.data,'$.entrance_position') course
                   from entries e left join results r on r.hd=e.hd and r.jcd=e.jcd and r.rno=e.rno and r.lane=e.lane
                   where e.jcd=? and e.hd>=? and e.hd<?""", con, params=(a.jcd, prev_day(a.d0, 500), a.d0))
for k in ["m2", "nwr", "position", "ex", "st", "course"]:
    h[k] = pd.to_numeric(h[k], errors="coerce")
if not since and not h.empty:
    z = h.groupby("hd").m2.apply(lambda s: (s.fillna(0) == 0).mean())
    zz = z[z > 0.8]
    since = zz.index.max() if len(zz) else h.hd.min()
    # 0 が続く最初の日（使用開始日）まで戻る
    days = sorted(h.hd.unique())
    i = days.index(since)
    while i > 0 and z[days[i - 1]] > 0.8:
        i -= 1
    since = days[i]
h = h[h.hd >= since].copy()
days = sorted(h.hd.unique())
meet_id, mid, prev = {}, 0, None
for dd in days:
    if prev is not None and (pd.Timestamp(dd) - pd.Timestamp(prev)).days > 1:
        mid += 1
    meet_id[dd] = mid; prev = dd
h["meet"] = h.hd.map(meet_id)
hr = h[h.position.notna()].copy()
hr["exr"] = hr.groupby(["hd", "rno"]).ex.rank(method="min")
# 乗り手と枠の分を引いた「着順の残り」: 着順 ~ 枠 + 全国勝率 の当てはめからのずれ（マイナスが良い）
X = np.column_stack([(hr.lane == k).astype(float) for k in range(1, 7)] + [hr.nwr.fillna(hr.nwr.mean())])
beta, *_ = np.linalg.lstsq(X, hr.position.values, rcond=None)
hr["resid"] = hr.position.values - X @ beta
last_day = hr.hd.max() if len(hr) else None
recent_from = prev_day(a.d0, 60)


def motor_row(mno):
    g = hr[hr.motor_no == mno]
    if g.empty:
        return None
    r60 = g[g.hd >= recent_from]
    ms = sorted(g.meet.unique())[-3:]
    hist = []
    for m in reversed(ms):
        gm = h[(h.meet == m) & (h.motor_no == mno)].sort_values(["hd", "rno"])
        who = gm.racer_name.iloc[0]; gr = gm.grade.iloc[0]
        pos = "".join(str(int(p)) if pd.notna(p) else "・" for p in gm.position)
        hist.append(f"{gm.hd.min()[4:6]}/{gm.hd.min()[6:]} {who}({gr}) {pos}")
    return dict(motor_no=mno, n=len(g), win=(g.position == 1).mean(), top2=(g.position <= 2).mean(), top3=(g.position <= 3).mean(),
                avg=g.position.mean(), resid=g.resid.mean(), exr=g.exr.mean(), nwr=g.nwr.mean(),
                n60=len(r60), top3_60=(r60.position <= 3).mean() if len(r60) else np.nan, resid60=r60.resid.mean() if len(r60) else np.nan,
                exr60=r60.exr.mean() if len(r60) else np.nan, hist=" ／ ".join(hist))


if AS:  # モーター未抽選: 直近45日に使われた全モーターを並べる
    mlist = sorted(h[h.hd >= prev_day(a.d0, 45)].motor_no.dropna().astype(int).unique())
    m2last = h.sort_values("hd").groupby("motor_no").m2.last()
    mfield = pd.DataFrame(dict(motor_no=mlist, name="未定", motor_2ren=[m2last.get(m) for m in mlist], pre_time=np.nan))
else:
    mlist = sorted(field.motor_no.unique()); mfield = field[["motor_no", "name", "motor_2ren", "pre_time"]]
mot = pd.DataFrame([x for x in (motor_row(m) for m in mlist) if x])
if not mot.empty:
    mot = mot.merge(mfield, on="motor_no", how="left")
    # 総合の目安: 着順の残り（全期間と直近60日）と展示順位の順位を足す。小さいほど上位
    mot["score"] = mot.resid.rank() + mot.resid60.rank(na_option="bottom") + mot.exr.rank() - mot.motor_2ren.rank()
    mot = mot.sort_values("score").reset_index(drop=True)

# ---------- 選手（直近365日の全場成績、当地は3年） ----------
y0 = prev_day(a.d0, 365)
rr = pd.read_sql("""select hd,jcd,rno,lane,position, json_extract(data,'$.racer_number') racer_no, json_extract(data,'$.entrance_position') course,
                           json_extract(data,'$.st_timing') st, json_extract(data,'$.exhibition_time') ex
                    from results where hd>=? and hd<?""", con, params=(prev_day(a.d0, 365 * 3), a.d0))
for k in ["position", "racer_no", "course", "st", "ex"]:
    rr[k] = pd.to_numeric(rr[k], errors="coerce")
rr = rr[rr.racer_no.isin(RN)].copy()
rr["exr"] = np.nan
ry = rr[rr.hd >= y0]
r90 = rr[rr.hd >= prev_day(a.d0, 90)]
rl = rr[rr.jcd == a.jcd]
# 期別成績（レース日より前の最新）
rs = pd.read_sql("select hd,racer_no,win_rate,top2_rate,avg_st,data from racer_stats where hd<? order by hd", con, params=(a.d0,))
rs["racer_no"] = pd.to_numeric(rs.racer_no, errors="coerce")
rs = rs[rs.racer_no.isin(RN)].groupby("racer_no").last()
rs_hd = rs.hd.max() if len(rs) else None


def racer_row(x):
    g = ry[ry.racer_no == x.racer_no]; g9 = r90[r90.racer_no == x.racer_no]; gl = rl[rl.racer_no == x.racer_no]
    out = dict(racer_no=x.racer_no, name=x["name"], n=len(g), win=(g.position == 1).mean() if len(g) else np.nan,
               top3=(g.position <= 3).mean() if len(g) else np.nan, st=g.st.mean(), avg=g.position.mean(),
               n90=len(g9), win90=(g9.position == 1).mean() if len(g9) else np.nan, top3_90=(g9.position <= 3).mean() if len(g9) else np.nan,
               avg90=g9.position.mean(), st90=g9.st.mean(),
               nl=len(gl), winl=(gl.position == 1).mean() if len(gl) else np.nan, top3l=(gl.position <= 3).mean() if len(gl) else np.nan, avgl=gl.position.mean(),
               mae=((g.course < g.lane).mean() if len(g) else np.nan))
    for c in range(1, 7):
        gc = g[g.course == c]
        out[f"c{c}"] = f"{len(gc)}走 {pct((gc.position == 1).mean())}/{pct((gc.position <= 3).mean())} ST{f2(gc.st.mean())}" if len(gc) else "-"
        out[f"c{c}n"] = len(gc); out[f"c{c}w"] = (gc.position == 1).mean() if len(gc) else np.nan
    if x.racer_no in rs.index:
        out["term_wr"] = rs.loc[x.racer_no, "win_rate"]; out["term_st"] = rs.loc[x.racer_no, "avg_st"]
    return out


rc = pd.DataFrame([racer_row(x) for _, x in field.iterrows()]).merge(
    field[["racer_no", "grade", "branch", "age", "national_win_rate", "local_win_rate", "motor_no"]], on="racer_no")

# ---------- 場の傾向 ----------
v = pd.read_sql("""select r.hd,r.rno,r.lane,r.position, json_extract(r.data,'$.entrance_position') course, e.grade
                   from results r left join entries e on e.hd=r.hd and e.jcd=r.jcd and e.rno=r.rno and e.lane=r.lane
                   where r.jcd=? and r.hd>=? and r.hd<?""", con, params=(a.jcd, "20150101", a.d0))
v["course"] = pd.to_numeric(v.course, errors="coerce"); v["position"] = pd.to_numeric(v.position, errors="coerce")
v["key"] = v.hd + "-" + v.rno.astype(str)
na1 = v.groupby("key").grade.apply(lambda s: (s == "A1").sum())
v["a1"] = v.key.map(na1)
rc_ = pd.read_sql("select hd,rno,weather,wind_direction,wind_speed,wave_height from races where jcd=? and hd>=? and hd<?", con, params=(a.jcd, "20150101", a.d0))
rc_["key"] = rc_.hd + "-" + rc_.rno.astype(str)
po = pd.read_sql("select hd,rno,combination,payout from payouts where jcd=? and hd>=? and hd<? and ticket_type='sanrensho'", con, params=(a.jcd, "20150101", a.d0))
po["key"] = po.hd + "-" + po.rno.astype(str)
po = po.groupby("key").first()
race = rc_.set_index("key").join(po[["combination", "payout"]])
race["a1"] = na1
w1 = v[(v.position == 1)].groupby("key").course.first()
race["win_course"] = w1
race["month"] = race.hd.str[4:6]


def course_table(keys, label):
    g = v[v.key.isin(keys) & v.course.between(1, 6)]
    if g.empty:
        return None
    t = g.groupby("course").position.agg(n="size", win=lambda s: (s == 1).mean(), top2=lambda s: (s <= 2).mean(), top3=lambda s: (s <= 3).mean())
    return f"| {label} | {g.key.nunique()} | " + " | ".join(f"{pct(t.win.get(c))}（{pct(t.top3.get(c))}）" for c in range(1, 7)) + " |"


def pay_line(rs_, label):
    p = rs_.payout.dropna()
    if p.empty:
        return None
    return (f"| {label} | {len(p)} | {pct((rs_.win_course == 1).mean())} | {p.median():,.0f}円 | {p.mean():,.0f}円 | {pct((p >= 10000).mean())} | "
            f"{pct((p < 1500).mean())} | {pct(rs_.combination.dropna().str.startswith('1-').mean())} |")


y1 = race[race.hd >= y0]; y3 = race[race.hd >= prev_day(a.d0, 365 * 3)]
g1 = y3[y3.a1 >= 5]; g1all = race[race.a1 >= 5]
mm = a.d0[4:6]
# 過去の記念級の節（A1 が5人以上のレースが1日8本以上ある日が4日以上続く節）
dd = race.groupby("hd").a1.apply(lambda s: (s >= 5).sum())
gd = sorted(dd[dd >= 8].index)
meets, cur = [], []
for x in gd:
    if cur and (pd.Timestamp(x) - pd.Timestamp(cur[-1])).days > 1:
        meets.append(cur); cur = []
    cur.append(x)
if cur:
    meets.append(cur)
meets = [m for m in meets if len(m) >= 4]

# ---------- 節間（この節の成績） ----------
t = pd.read_sql("""select e.hd,e.rno,e.lane,e.racer_no,e.racer_name,e.motor_no,e.race_name, json_extract(e.data,'$.national_win_rate') nwr,
                          r.position, json_extract(r.data,'$.exhibition_time') ex, json_extract(r.data,'$.st_timing') st, json_extract(r.data,'$.entrance_position') course
                   from entries e join results r on r.hd=e.hd and r.jcd=e.jcd and r.rno=e.rno and r.lane=e.lane
                   where e.jcd=? and e.hd between ? and ?""", con, params=(a.jcd, a.d0, a.d1))
for k in ["nwr", "position", "ex", "st", "course"]:
    t[k] = pd.to_numeric(t[k], errors="coerce")
bi = pd.read_sql("select hd,rno,lane,exhibition_time,tilt,start_course,start_st,data from beforeinfo where jcd=? and hd between ? and ?", con, params=(a.jcd, a.d0, a.d1))

# ---------- 書き出し ----------
L = [f"# {a.title or VN + ' ' + a.d0 + '〜'}", "",
     f"運用者の観戦・趣味用の資料（{VN}、{a.d0[4:6]}/{a.d0[6:]}〜{a.d1[4:6]}/{a.d1[6:]}）。`scripts/analysis/g1_prep.py` で作成、毎晩作り直す。",
     "数字は公式ダウンロード（番組表・競走成績）と公式サイト（前検）から。舟券は買わない。少ない本数で良し悪しを断定しない。", ""]
if Z:
    L += ["## 前検（公式「モーター抽選結果・前検タイム」）", "", f"出典: {Z['url']}", "",
          f"- 記者コメント（モーター）: {Z['comment_motor']}", f"- 記者コメント（前検タイム）: {Z['comment_time']}", ""]
    zt = zdf.dropna(subset=["pre_time"]).sort_values("pre_time")
    if len(zt):
        L += ["- 前検タイム上位10: " + "、".join(f"{x['name']}({x.motor_no}) {x.pre_time:.2f}" for _, x in zt.head(10).iterrows()),
              "- 前検タイム下位10: " + "、".join(f"{x['name']}({x.motor_no}) {x.pre_time:.2f}" for _, x in zt.tail(10).iloc[::-1].iterrows()),
              f"- 前検タイムの中央値 {zt.pre_time.median():.2f}（{len(zt)}人）。前検タイムは1回の計測で、伸びの目安にしかならない。", ""]
else:
    L += ["## 前検", "", "公式「モーター抽選結果・前検タイム」はまだ出ていない（確認できず）。出たらこのスクリプトを回し直すと入る。"
          + (f" 出場選手は公式「出場予定レーサー一覧」（{AS['url']}、{len(AS['rows'])}人）による。モーターは未抽選なので、モーターの表は直近45日に使われた全機。" if AS else ""), ""]

L += ["## 1. モーター", "",
      f"集計: {VN}の現モーター（{since[:4]}/{since[4:6]}/{since[6:]} 〜 {last_day[4:6] + '/' + last_day[6:] if last_day else '-'}、{hr.meet.nunique() if len(hr) else 0}節・{len(hr)}走）。"
      "使用開始日は公式2連率が0に戻った日から自動で判定。",
      "「乗り手込みの残り」＝着順を枠と乗り手の全国勝率で説明した残り。マイナスほど『乗り手と枠のわりに着が良い』。展示順位は6艇中の順位の平均（小さいほど速い）。",
      "並びは総合の目安（残り・直近60日の残り・展示順位・公式2連率の順位を足したもの）。目安であって、当たりを約束するものではない。", "",
      "| 順 | 機 | 今節の乗り手 | 公式2連率 | 走 | 1着率 | 2連率 | 3連率 | 平均着 | 残り | 展示順位 | 直近60日（走・3連率・残り） | 乗り手の平均勝率 | 前検 | 直近3節（新しい順: 初日 乗り手 着順） |",
      "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for i, x in mot.iterrows():
    L.append(f"| {i + 1} | {x.motor_no} | {x['name']} | {f2(x.motor_2ren, 1)} | {x.n} | {pct(x.win)} | {pct(x.top2)} | {pct(x.top3)} | {f2(x.avg)} | {x.resid:+.2f} | {f2(x.exr)} | "
             f"{x.n60}走・{pct(x.top3_60)}・{'-' if pd.isna(x.resid60) else f'{x.resid60:+.2f}'} | {f2(x.nwr)} | {f2(x.pre_time)} | {x['hist']} |")
if len(mot) >= 10:
    L += ["", "- 上位の目安: " + "、".join(f"{x.motor_no}号機（{x['name']}）" for _, x in mot.head(8).iterrows()),
          "- 下位の目安: " + "、".join(f"{x.motor_no}号機（{x['name']}）" for _, x in mot.tail(8).iloc[::-1].iterrows()),
          "- 直近60日で上向き（残りが全期間より 0.3 以上良い、20走以上）: " + ("、".join(
              f"{x.motor_no}号機（{x['name']}、{x.resid:+.2f}→{x.resid60:+.2f}）" for _, x in mot[(mot.n60 >= 20) & (mot.resid60 < mot.resid - 0.3)].sort_values("resid60").iterrows()) or "なし"),
          "- 直近60日で下向き（0.3 以上悪い、20走以上）: " + ("、".join(
              f"{x.motor_no}号機（{x['name']}、{x.resid:+.2f}→{x.resid60:+.2f}）" for _, x in mot[(mot.n60 >= 20) & (mot.resid60 > mot.resid + 0.3)].sort_values("resid60", ascending=False).iterrows()) or "なし")]

L += ["", "## 2. 選手", "",
      f"集計: 直近365日の全場の成績（{y0[4:6]}/{y0[6:]}〜前日）。当地は直近3年。コース別は「走数 1着率/3連率 平均ST」。前づけ率＝枠より内のコースに入った割合。"
      + (f" 期別成績は {rs_hd} までの期（それより新しい期はまだ取り込んでいない）。" if rs_hd else ""), "",
      "| 選手 | 級 | 支部 | 機 | 全国勝率 | 当地勝率 | 1年: 走・1着率・3連率・ST | 直近90日: 走・1着率・3連率・ST | 当地3年: 走・1着率・3連率 | 前づけ率 | 1コース | 2コース | 3コース | 4コース | 5コース | 6コース |",
      "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for _, x in rc.sort_values("win", ascending=False).iterrows():
    L.append(f"| {x['name']} | {x.grade} | {x.branch or ''} | {x.motor_no if x.motor_no > 0 else '未定'} | {x.national_win_rate if x.national_win_rate is not None else f2(x.get('term_wr'))} | {x.local_win_rate if x.local_win_rate is not None else '-'} | "
             f"{x.n}・{pct(x.win)}・{pct(x.top3)}・{f2(x.st)} | {x.n90}・{pct(x.win90)}・{pct(x.top3_90)}・{f2(x.st90)} | {x.nl}・{pct(x.winl)}・{pct(x.top3l)} | {pct(x.mae)} | "
             + " | ".join(x[f"c{c}"] for c in range(1, 7)) + " |")


def names(df, fmt):
    return "、".join(fmt(x) for _, x in df.iterrows()) or "なし"


q = rc[rc.n90 >= 20]
L += ["", "### 注目（数字の上位）",
      "- 直近90日の3連率: " + names(q.sort_values("top3_90", ascending=False).head(6), lambda x: f"{x['name']} {pct(x.top3_90)}（1着{pct(x.win90)}）"),
      "- 直近90日が1年平均より上向き（平均着の差）: " + names(q.assign(d=q.avg90 - q.avg).sort_values("d").head(6), lambda x: f"{x['name']} {x.avg:.2f}→{x.avg90:.2f}"),
      "- 当地3年の3連率（30走以上）: " + names(rc[rc.nl >= 30].sort_values("top3l", ascending=False).head(6), lambda x: f"{x['name']} {pct(x.top3l)}（{x.nl}走、1着{pct(x.winl)}）"),
      "- 平均STが早い（1年）: " + names(rc.sort_values("st").head(6), lambda x: f"{x['name']} {x.st:.3f}"),
      "- 1コースの1着率が高い（20走以上）: " + names(rc[rc.c1n >= 20].sort_values("c1w", ascending=False).head(6), lambda x: f"{x['name']} {pct(x.c1w)}（{x.c1n}走）"),
      "- センター・外で勝てる（3〜6コースの1着率、合計40走以上）: " + names(
          rc.assign(on=rc[[f"c{c}n" for c in (3, 4, 5, 6)]].sum(axis=1),
                    ow=sum(rc[f"c{c}n"] * rc[f"c{c}w"].fillna(0) for c in (3, 4, 5, 6)) / rc[[f"c{c}n" for c in (3, 4, 5, 6)]].sum(axis=1).replace(0, np.nan))
          .query("on>=40").sort_values("ow", ascending=False).head(6), lambda x: f"{x['name']} {pct(x.ow)}（{int(x.on)}走）"),
      "- 前づけに動く（前づけ率が高い）: " + names(rc.sort_values("mae", ascending=False).head(5), lambda x: f"{x['name']} {pct(x.mae)}"),
      "", "### 不安（数字の下位）",
      "- 直近90日の3連率が低い: " + names(q.sort_values("top3_90").head(6), lambda x: f"{x['name']} {pct(x.top3_90)}（1着{pct(x.win90)}）"),
      "- 直近90日が1年平均より下向き: " + names(q.assign(d=q.avg90 - q.avg).sort_values("d", ascending=False).head(6), lambda x: f"{x['name']} {x.avg:.2f}→{x.avg90:.2f}"),
      "- 当地3年の3連率が低い（30走以上）: " + names(rc[rc.nl >= 30].sort_values("top3l").head(6), lambda x: f"{x['name']} {pct(x.top3l)}（{x.nl}走）"),
      "- 当地の経験が少ない（3年で30走未満）: " + names(rc[rc.nl < 30].sort_values("nl"), lambda x: f"{x['name']} {x.nl}走"),
      "- 平均STが遅い（1年）: " + names(rc.sort_values("st", ascending=False).head(6), lambda x: f"{x['name']} {x.st:.3f}"),
      "- 1コースの1着率が低い（20走以上）: " + names(rc[rc.c1n >= 20].sort_values("c1w").head(6), lambda x: f"{x['name']} {pct(x.c1w)}（{x.c1n}走）"),
      "- 注: フライング持ち・出走回数の残りはデータに無い（確認できず）。"]

L += ["", f"## 3. 場の傾向（{VN}）", "", "### コース別の1着率（かっこ内は3連率）", "",
      "| 期間・条件 | レース | 1コース | 2コース | 3コース | 4コース | 5コース | 6コース |", "|---|---|---|---|---|---|---|---|"]
for keys, label in [(y1.index, "直近1年・全レース"), (y3.index, "直近3年・全レース"), (g1.index, "直近3年・A1が5人以上（記念級）"), (g1all.index, "2015年〜・A1が5人以上"),
                    (y3[y3.month == mm].index, f"直近3年・{int(mm)}月"),
                    (y3[y3.wind_speed <= 2].index, "直近3年・風0〜2m"), (y3[y3.wind_speed.between(3, 4)].index, "直近3年・風3〜4m"), (y3[y3.wind_speed >= 5].index, "直近3年・風5m以上"),
                    (y3[y3.wave_height >= 5].index, "直近3年・波5cm以上")]:
    s = course_table(set(keys), label)
    if s:
        L.append(s)
L += ["", "### 風向きごと（直近3年、風3m以上のレース）", "", "| 風向 | レース | 1コース | 2コース | 3コース | 4コース | 5コース | 6コース |", "|---|---|---|---|---|---|---|---|"]
for k, nm in COMPASS.items():
    s = course_table(set(y3[(y3.wind_direction == k) & (y3.wind_speed >= 3)].index), nm)
    if s:
        L.append(s)
L += ["", "風向は方角（成績ファイルの表記）。追い風・向かい風のどちらに当たるかは場の向きによる（この資料では方角のまま）。", "",
      "### 3連単の配当", "", "| 条件 | レース | 1コース1着 | 中央値 | 平均 | 万舟の割合 | 1,500円未満 | 1号艇が頭 |", "|---|---|---|---|---|---|---|---|"]
for rs_, label in [(y1, "直近1年・全レース"), (y3, "直近3年・全レース"), (g1, "直近3年・A1が5人以上（記念級）"), (g1all, "2015年〜・A1が5人以上"),
                   (y3[y3.wind_speed >= 5], "直近3年・風5m以上")]:
    s = pay_line(rs_, label)
    if s:
        L.append(s)
L += ["", "### 過去の記念級の節（A1が5人以上のレースが1日8本以上の日が4日以上続いた節。開催名はデータに無いので日付で示す）", "",
      "| 節（初日〜最終日） | レース | 1コース1着 | 3連単 中央値 | 平均 | 万舟の割合 | 最高配当 | 最終日12R（優勝戦）の3連単 |", "|---|---|---|---|---|---|---|---|"]
for m in meets[-10:]:
    g = race[race.hd.isin(m)]
    p = g.payout.dropna()
    fin = g[(g.hd == m[-1]) & (g.rno == 12)]
    ft = f"{fin.combination.iloc[0]} {fin.payout.iloc[0]:,.0f}円" if len(fin) and pd.notna(fin.payout.iloc[0]) else "-"
    L.append(f"| {m[0]}〜{m[-1][4:]} | {len(g)} | {pct((g.win_course == 1).mean())} | {p.median():,.0f}円 | {p.mean():,.0f}円 | {pct((p >= 10000).mean())} | {p.max():,.0f}円 | {ft} |")
off = Path(f"notes/reads/sources/stadium_{a.jcd}.md")
if off.exists():
    L += ["", "### 公式の場データ（決まり手など）", "", off.read_text().strip()]

L += ["", "## 4. 節間の気配（毎晩更新）", ""]
if t.empty:
    L += ["まだこの節の成績を取り込んでいない。取り込み後に作り直すと、ここに選手・モーターの気配が出る。"]
else:
    t["exr"] = t.groupby(["hd", "rno"]).ex.rank(method="min")
    t["stn"] = t.groupby(["hd", "rno"]).st.rank(method="min")
    Xt = np.column_stack([(t.lane == k).astype(float) for k in range(1, 7)] + [t.nwr.fillna(hr.nwr.mean() if len(hr) else 6.5)])
    t["resid"] = t.position.fillna(6).values - Xt @ beta
    dd_ = sorted(t.hd.unique())
    L += [f"取り込み済み: {', '.join(x[4:6] + '/' + x[6:] for x in dd_)}（{t.groupby(['hd', 'rno']).ngroups}レース）。「残り」は着順を枠と全国勝率で説明した残り（マイナスが良い）。"
          "展示順位・ST順位は6艇中の順位の平均。数走の数字なので気配の目安。", "",
          "| 選手 | 機 | 着順（日ごと） | 平均着 | 残り | 展示順位 | 平均展示 | ST順位 | 平均ST | 進入 |", "|---|---|---|---|---|---|---|---|---|---|"]
    agg = []
    for (rn, nm, mo), g in t.groupby(["racer_no", "racer_name", "motor_no"]):
        g = g.sort_values(["hd", "rno"])
        pos = " ".join("".join(str(int(p)) if pd.notna(p) else "・" for p in g[g.hd == x].position) for x in dd_ if (g.hd == x).any())
        agg.append(dict(name=nm, motor=mo, pos=pos, n=len(g), avg=g.position.mean(), resid=g.resid.mean(), exr=g.exr.mean(), ex=g.ex.mean(), stn=g.stn.mean(), st=g.st.mean(),
                        crs="".join(str(int(c)) if pd.notna(c) else "・" for c in g.course)))
    ag = pd.DataFrame(agg)
    ag["score"] = ag.resid.rank() + ag.exr.rank() + 0.5 * ag.stn.rank()
    for _, x in ag.sort_values("score").iterrows():
        L.append(f"| {x['name']} | {x.motor} | {x.pos} | {f2(x.avg)} | {x.resid:+.2f} | {f2(x.exr, 1)} | {f2(x.ex)} | {f2(x.stn, 1)} | {f2(x.st)} | {x.crs} |")
    s = ag.sort_values("score")
    L += ["", "- 気配が良い側（残り・展示順位・ST順位の合計が上位）: " + "、".join(f"{x['name']}({x.motor})" for _, x in s.head(8).iterrows()),
          "- 気配が悪い側: " + "、".join(f"{x['name']}({x.motor})" for _, x in s.tail(8).iloc[::-1].iterrows()),
          "- 展示は出るが着が悪い（展示順位2.5以内・残り+0.5以上）: " + ("、".join(f"{x['name']}({x.motor})" for _, x in ag[(ag.exr <= 2.5) & (ag.resid >= 0.5)].iterrows()) or "なし"),
          "- 展示は出ないが着が良い（展示順位4.5以上・残り-0.5以下）: " + ("、".join(f"{x['name']}({x.motor})" for _, x in ag[(ag.exr >= 4.5) & (ag.resid <= -0.5)].iterrows()) or "なし")]
    if not bi.empty:
        bi["d"] = bi.data.map(json.loads)
        ch = bi[bi.d.map(lambda x: bool(x.get("parts_changed")))]
        tl = bi[pd.to_numeric(bi.tilt, errors="coerce") >= 0.5]
        ent = t.set_index(["hd", "rno", "lane"])
        def who(r):
            k = (r.hd, r.rno, r.lane)
            return ent.loc[k, "racer_name"] if k in ent.index else f"{r.lane}号艇"
        if len(ch):
            L.append("- 部品交換: " + "、".join(f"{r.hd[4:6]}/{r.hd[6:]} {r.rno}R {who(r)}（{'・'.join(map(str, r.d['parts_changed']))}）" for r in ch.itertuples()))
        if len(tl):
            L.append("- チルト0.5以上: " + "、".join(f"{r.hd[4:6]}/{r.hd[6:]} {r.rno}R {who(r)}（{r.tilt}）" for r in tl.itertuples()))
    cw = t[t.position == 1].groupby("hd").course.apply(lambda s_: "".join(str(int(c)) for c in s_))
    L += ["", "### 日ごとの1着のコース（1R→12R）"] + [f"- {k[4:6]}/{k[6:]}: {v_}（1コース {sum(c == '1' for c in v_)}/{len(v_)}）" for k, v_ in cw.items()]
nx = e[~e.hd.isin(set(t.hd))].hd.min() if not e.empty else None
if nx is not None and pd.notna(nx):
    en = e[e.hd == nx].sort_values(["rno", "lane"])
    L += ["", f"## 番組（{nx[4:6]}/{nx[6:]}）", "", "| R | 締切 | 種別 | 1 | 2 | 3 | 4 | 5 | 6 |", "|---|---|---|---|---|---|---|---|---|"]
    for rno, g in en.groupby("rno"):
        L.append(f"| {rno} | {g.close_time.iloc[0]} | {g.race_name.iloc[0]} | " + " | ".join(f"{n}({m})" for n, m in zip(g.racer_name, g.motor_no)) + " |")
Path(a.out).write_text("\n".join(L) + "\n")
print(f"wrote {a.out}: 選手 {len(field)}, モーター {len(mot)}, 現モーター開始 {since}, 節間レース {0 if t.empty else t.groupby(['hd', 'rno']).ngroups}")
