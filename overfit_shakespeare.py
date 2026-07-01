#!/usr/bin/env python3
"""RayTention overfit — memorize a Shakespeare passage word-for-word."""

import torch, torch.nn as nn, torch.nn.functional as F, time

D=128; L=4; FH=512; CTX=32; BS=16; STEPS=8000; LR=0.005
DEV=torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load and slice
# Load from same directory, fall back to original path
import os
DATA = "shakespeare.txt"
if not os.path.exists(DATA):
    DATA = "../helios_raytention/data/shakespeare.txt"
with open(DATA) as f: raw=f.read()
text=raw[:2000]  # 2K chars — force memorization
chars=sorted(list(set(text)))
V=len(chars); c2i={c:i for i,c in enumerate(chars)}; i2c={i:c for i,c in enumerate(chars)}
data=torch.tensor([c2i[c] for c in text],dtype=torch.long)
print(f"Target ({V} chars):\n{text[:300]}\n...")

# Model
class Swish(nn.Module):
    def forward(self,x): return x*torch.sigmoid(x)

class RTB(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1=nn.LayerNorm(D); self.ln2=nn.LayerNorm(D)
        sd=D*5+2; self.af=nn.Sequential(nn.Linear(sd,FH),Swish(),nn.Linear(FH,D))
        self.ff=nn.Sequential(nn.Linear(D,FH),Swish(),nn.Linear(FH,D))
    def forward(self,x,keys,ci):
        r=x; xn=self.ln1(x); ck=keys[ci]; n=ck.shape[1]
        diff=xn.unsqueeze(1)-ck; dist=(diff*diff).sum(-1).sqrt()
        sc=-dist; w=F.softmax(sc-sc.max(-1,True)[0],-1)
        cent=(w.unsqueeze(-1)*ck).sum(1)
        pw=0.7**torch.arange(n-1,-1,-1,device=DEV).float()
        tw=w*pw; ts=tw.sum(-1,True).clamp(1e-8); tcent=(tw.unsqueeze(-1)*ck).sum(1)/ts
        pred=ck[:,-1,:]
        _,ti=torch.topk(sc,min(2,n),-1)
        B=x.shape[0]; t1=ck[torch.arange(B),ti[:,0]]
        t2=ck[torch.arange(B),ti[:,1]] if n>=2 else t1
        sp=(w*dist).sum(-1,True)
        wc=w.clamp(1e-10); ent=-(wc*wc.log()).sum(-1,True)
        return r+self.af(torch.cat([cent,tcent,pred,t1,t2,sp,ent],-1))+self.ff(self.ln2(r))

class RTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb=nn.Embedding(V,D); self.bs=nn.ModuleList([RTB() for _ in range(L)])
        self.ln=nn.LayerNorm(D); self.hd=nn.Linear(D,V,bias=False)
    def forward(self,ci):
        x=self.emb(ci[:,0])
        for b in self.bs: x=b(x,self.emb.weight,ci)
        return self.hd(self.ln(x))

torch.manual_seed(42)
model=RTM().to(DEV)
opt=torch.optim.AdamW(model.parameters(),lr=LR)
print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

# Training — deterministic cycling for memorization
n=len(data)-CTX
t0=time.time()
# Pre-build all context/target pairs
all_ci=torch.stack([data[i:i+CTX] for i in range(n)])
all_tg=data[CTX:CTX+n]
steps_per_epoch=n
print(f"Epochs: {STEPS*BS//n}  pairs: {n}")

for step in range(STEPS):
    idx=[(step*BS+j)%n for j in range(BS)]
    ci=all_ci[idx].to(DEV); tg=all_tg[idx].to(DEV)
    opt.zero_grad()
    loss=F.cross_entropy(model(ci),tg)
    loss.backward()
    opt.step()
    if step%500==0 or step==STEPS-1:
        tps=(step+1)/(time.time()-t0) if step>0 else 0
        print(f"step {step:>5}/{STEPS}  ce={loss.item():.4f}  {tps:.0f} s/s")

print(f"Done in {time.time()-t0:.0f}s")

# Teacher-forcing
model.eval()
correct=0; total=0
with torch.no_grad():
    for i in range(0,n,BS):
        end=min(i+BS,n); ci=all_ci[i:end].to(DEV); tg=all_tg[i:end].to(DEV)
        pred=torch.argmax(model(ci),-1)
        correct+=(pred==tg).sum().item(); total+=tg.numel()
print(f"\n  TEACHER-FORCING: {correct}/{total} = {correct/total*100:.1f}% memorized")

# Autoregressive generation
torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(DEV)
start_pos=500
ctx=data[start_pos:start_pos+CTX].unsqueeze(0).to(DEV)
original=text[start_pos+CTX:start_pos+CTX+400]
seed_text=text[start_pos:start_pos+CTX]

generated=[]
with torch.no_grad():
    for _ in range(400):
        logits=model(ctx)
        next_id=torch.argmax(logits[0],-1).item()
        generated.append(i2c[next_id])
        ctx=torch.cat([ctx[:,1:],torch.tensor([[next_id]],device=DEV)],dim=1)

torch.cuda.synchronize()
peak_vram=torch.cuda.max_memory_allocated(DEV)/1e6
steady_vram=torch.cuda.memory_allocated(DEV)/1e6

gen_text=''.join(generated)
matches=sum(1 for a,b in zip(gen_text,original) if a==b)
# Find where it first diverges
diverge_at=len(gen_text)
for i in range(min(len(gen_text),len(original))):
    if gen_text[i]!=original[i]: diverge_at=i; break

print(f"\n  AUTOREGRESSIVE (greedy, 400 chars):")
print(f"  {'─'*60}")
print(f"  GEN:  {gen_text}")
print(f"  {'─'*60}")
print(f"  ORIG: {original}")
print(f"  {'─'*60}")
if diverge_at<len(gen_text):
    print(f"  First divergence at char {diverge_at}:")
    print(f"    GEN:  ...{gen_text[max(0,diverge_at-20):diverge_at+30]}...")
    print(f"    ORIG: ...{original[max(0,diverge_at-20):diverge_at+30]}...")
print(f"  Match: {matches}/{len(original)} = {matches/len(original)*100:.1f}%")
print(f"  VRAM: peak={peak_vram:.0f} MB  steady={steady_vram:.0f} MB")
