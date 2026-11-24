import numpy as np
exec(open('scripts/analysis/par_ori_deep.py').read().split("# ---- A")[0])
m=S1m&ok&has; i=np.where(m)[0]
pa=np.where(res[i]=='1-2-3',pay[i],0); pb=np.where(res[i]=='1-3-2',pay[i],0)
print('n',len(i),'1-2-3 only',pa.sum()/100/len(i),'1-3-2 only',pb.sum()/100/len(i),'hits 123/132',(pa>0).sum(),(pb>0).sum())
rng=np.random.default_rng(0); sims=[]
for _ in range(5000):
    c=rng.random(len(i))<.5; sims.append(np.where(c,pa,pb).sum()/100/len(i))
sims=np.array(sims)
for nm,t in (('et',pair_ticket(h2,h3,et)),('直線',pair_ticket(h2,h3,np.where(np.isnan(cho),9,cho)))):
    pp=np.where(res[i]==t[i],pay[i],0).sum()/100/len(i); print(nm,round(pp,3),'ランダム順で超える確率',(sims>=pp).mean(),'平均',sims.mean().round(3))
# 直線順とet順が一致する割合
a=pair_ticket(h2,h3,et)[i]; b=pair_ticket(h2,h3,np.where(np.isnan(cho),9,cho))[i]; print('一致率',(a==b).mean())
