import json, numpy as np, torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
ROOT="/home/alphabridge/Research/MoodRisk/data"
d=json.load(open(f"{ROOT}/260317_8736_data.json"))
def col(c):
    v=d[c]; return [v[k] for k in sorted(v,key=lambda x:int(x))]
postlists=col("post_id"); y=np.array(col("trans_730_y")); split=col("train_test_1")
tr=np.array([s=="train" for s in split])
print(f"rows={len(y)} pos={y.mean():.3f} train={tr.sum()} test={(~tr).sum()}")
print(f"{'layer':>5} | {'test-acc':>8} {'bal-acc':>8}")
best=(0,None)
for L in [0,4,8,12,16,20,24,28,31]:
    emb=torch.load(f"{ROOT}/mental_mistral_7b_layer_{L}_mean.pt",weights_only=False)
    X=[]; keep=[]
    for i,pl in enumerate(postlists):
        vecs=[emb[p] for p in pl if p in emb]
        if vecs: X.append(np.mean(vecs,0)); keep.append(i)
    X=np.stack(X); yy=y[keep]; t=tr[keep]
    sc=StandardScaler().fit(X[t]); clf=LogisticRegression(max_iter=400,C=0.5).fit(sc.transform(X[t]),yy[t])
    Xte,yte=sc.transform(X[~t]),yy[~t]; pred=clf.predict(Xte)
    acc=(pred==yte).mean(); bal=0.5*((pred[yte==1]==1).mean()+(pred[yte==0]==0).mean())
    print(f"{L:>5} | {acc:>8.3f} {bal:>8.3f}")
    if bal>best[0]: best=(bal,L)
print(f"BEST risk-direction layer L{best[1]}: balanced-acc={best[0]:.3f} (chance 0.50) | LCF logic best=0.82")
