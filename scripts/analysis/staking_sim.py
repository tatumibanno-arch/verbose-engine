import numpy as np
rng=np.random.default_rng(0)
P=0.077            # 2連単の買い目の的中率（実測 7.7%）
BETS=300; MONTHS=12; SIMS=4000; BANK=100_000; UNIT=100
def sim(odds, system):
    res=[]; bust=0; posm=0; worst=[]
    for _ in range(SIMS):
        bank=BANK; months=[]; dead=False
        loss=0; seq=[UNIT,UNIT]  # martingale の累積損 / ココモの直前2回
        for m in range(MONTHS):
            start=bank
            for _ in range(BETS):
                if system=='flat': st=UNIT
                elif system=='martin': st=int(np.ceil((loss+UNIT)/(odds-1)/100))*100   # 当たれば負け分＋100円取り返す額
                elif system=='cocomo': st=seq[-1]+seq[-2] if loss>0 else UNIT
                if st>bank: dead=True; break
                bank-=st
                if rng.random()<P:
                    bank+=st*odds; loss=0; seq=[UNIT,UNIT]
                else:
                    loss+=st; seq.append(st) if system=='cocomo' else None
            months.append(bank-start)
            if dead: break
        bust+=dead; res.append(bank-BANK); posm+=sum(x>0 for x in months); worst.append(min(months))
    return np.mean(res), posm/(SIMS*MONTHS), bust/SIMS, np.median(worst)
for odds,lab in [(9.7,'回収率0.75（ふつうに買う）'),(14.0,'回収率1.08（わずかに有利）')]:
    print(f"\n### 2連単 的中7.7%・平均{odds}倍 = {lab}、元手10万円、月300点、1年")
    for s,n in [('flat','毎回100円（定額）'),('martin','マーチンゲール型（外れたら負け分を取り返す額）'),('cocomo','ココモ法（直前2回の合計を賭ける）')]:
        mean,pm,b,w=sim(odds,s)
        print(f"{n}: 1年の損益 平均{mean:+,.0f}円 / プラスの月 {pm:.0%} / 1年で元手が尽きる {b:.0%} / 最悪月(中央値) {w:+,.0f}円")
