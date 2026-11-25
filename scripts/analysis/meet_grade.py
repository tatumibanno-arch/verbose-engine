"""boatrace.jp の月間スケジュールから節のグレード表を作る -> data/meta/meets.csv"""
import urllib.request,re,datetime as dt,csv,time
rows=[]
y,m=2023,10
while (y,m)<=(2026,12):
    h=urllib.request.urlopen(f"https://www.boatrace.jp/owpc/pc/race/monthlyschedule?ym={y}{m:02d}",timeout=30).read().decode()
    for c,g,j,hd,t in re.findall(r'colspan ="(\d+)" class="is-gradeColor(\w+)"[^>]*><a href="/owpc/pc/race/raceindex\?jcd=(\d+)&hd=(\d+)">([^<]*)<',h):
        rows.append((j,hd,int(c),g,t))
    m+=1
    if m>12: y,m=y+1,1
    time.sleep(0.5)
seen=set();out=[]
for j,hd,c,g,t in rows:
    if (j,hd) in seen: continue
    seen.add((j,hd)); out.append((j,hd,c,g,t))
with open('data/meta/meets.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['jcd','start','days','grade','title']); w.writerows(out)
print(len(out))
