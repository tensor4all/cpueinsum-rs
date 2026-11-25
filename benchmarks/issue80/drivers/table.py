import sys, glob, statistics
from collections import defaultdict
B=sys.argv[1]
cell=defaultdict(list); route={}; eq={}
for f in sorted(glob.glob(B+'/raw-t*.tsv')):
    for l in open(f):
        if l.startswith('#'): continue
        p=l.rstrip('\n').split('\t')
        rnd,arm,name,dt,T,med=p[0],p[1],p[2],p[3],int(p[4]),float(p[5])
        cell[(name,dt,T,arm)].append(med)
        if arm!='native': route[(name,dt,T,arm)]=p[8]; eq[(name,dt,T,arm)]=p[9]
names=[]
for k in cell:
    if (k[0],k[1]) not in names: names.append((k[0],k[1]))
Ts=sorted({k[2] for k in cell})
print("| workload | dtype | T | native µs | GP before µs | GP after µs | GP/native before | GP/native after | after route | equal to serial |")
print("|---|---|---|---|---|---|---|---|---|---|")
for (n,d) in sorted(names, key=lambda x:(x[1],x[0])):
    for T in Ts:
        if (n,d,T,'native') not in cell: continue
        m=lambda a: statistics.mean(cell[(n,d,T,a)])
        r=lambda a: "/".join(f"{x/min(cell[(n,d,T,'native')]) if False else x:.0f}" for x in cell[(n,d,T,a)])
        nat,ba,ca=m('native'),m('base'),m('cand')
        rb=[b/a for a,b in zip(cell[(n,d,T,'native')],cell[(n,d,T,'base')])]
        rc=[b/a for a,b in zip(cell[(n,d,T,'native')],cell[(n,d,T,'cand')])]
        print(f"| {n} | {d.lower()} | {T} | {nat:.0f} | {ba:.0f} | {ca:.0f} | {statistics.mean(rb):.2f} ({min(rb):.2f}–{max(rb):.2f}) | {statistics.mean(rc):.2f} ({min(rc):.2f}–{max(rc):.2f}) | `{route[(n,d,T,'cand')]}` | {eq[(n,d,T,'cand')].split('=')[1]}/{eq[(n,d,T,'base')].split('=')[1]} |")
