"""Summarises gpu_final.sh: per row and precision, the median over rounds of each framework's host-to-host
time, first-call time and peak GPU memory, and jitNN's ratio to the faster of PyTorch and ONNX Runtime
and to the fastest of all. jitNN's time is its best configuration's median.

  python gpu_summary.py <out dir>/final.txt [summary.json]"""
import re,collections,statistics as st,json
import sys
L=open(sys.argv[1]).read().split('\n')
mem={}
for l in L:
    m=re.match(r"\[mem\] (\S+) (\d+) (\d+)",l)
    if m: mem[m.group(1)]=int(m.group(2))-int(m.group(3))
inf=collections.defaultdict(list); setup={}; jmem={}; jcfg=collections.defaultdict(list)
for l in L:
    m=re.match(r"r(\d) \[gsweep\] (\S+)\s+(\S+)\s+(\S+)\s+(\S+) host\s+([\d.]+).*setup\s+([\d.]+)",l)
    if m:
        r,fw,v,mod,p,h,s=m.groups()
        f={'pytorch':'PyTorch','trt':'TRT'}.get(fw,'ORT-TRT' if v=='trt' else 'ORT')
        inf[(mod,p,f)].append(float(h)); setup.setdefault((mod,p,f),float(s)); continue
    m=re.match(r"r(\d) (\S+) \[t1\] jitnn (\S+?)_(fp32|fp16h|fp16)\s+median\s+([\d.]+).*first call (\d+) ms",l)
    if m:
        r,cfg,mod,mode,ms,fc=m.groups()
        if mode=='fp16': continue
        p='fp16' if mode=='fp16h' else ('fp32' if cfg.startswith('strict') else 'tf32')
        # one name per configuration: round 1 ran resnet strict on jitNN's kernels, bert strict with libraries, fp16 with the tf32 settings
        c={'strict':'strict-k' if mod.startswith('resnet') else 'strict-n','fp16':'tf32'}.get(cfg,cfg)
        jcfg[(mod,p,c)].append(float(ms))
        if r=='1': setup[(mod,p,'jitNN')]=int(fc)/1000; jmem[(mod,p)]=mem.get(f"r1-jitnn-{cfg}-{mod}_{mode}")
best={}
for (mo,p,c),v in jcfg.items():
    if (mo,p) not in best or st.median(v)<st.median(jcfg[best[(mo,p)]]): best[(mo,p)]=(mo,p,c)
for k,b in best.items(): inf[(k[0],k[1],'jitNN')]=jcfg[b]
order=['resnet50_b1','resnet50_b32','resnet50_b128','bert_b1_s128','bert_b32_s128','bert_b8_s512']
memkey={'PyTorch':'pytorch','ORT':'ort','TRT':'trt-dynamo','ORT-TRT':'ort-trt'}
def fmem(f,mo,p):
    k,b=(mo.split('_b')[0],mo.split('_b')[1]); bb=b.split('_s')[0]; t=b.split('_s')[1] if '_s' in b else '0'
    kind='resnet50' if k=='resnet50' else 'bert'
    for name,v in mem.items():
        if not name.startswith('r1-'): continue
        if f=='ORT' and (name.startswith('r1-ort-trt')): continue
        if name.startswith('r1-'+memkey[f]) and name.endswith(f"-{kind}-{bb}-{t}-{p}"): return v
res={}
for p in ['fp32','tf32','fp16']:
    print(f"\n== {p}: median of 3 rounds, ms host to host [first call s] {{peak MB}}")
    for mo in order:
        row={f:(st.median(inf[(mo,p,f)]) if inf.get((mo,p,f)) else None) for f in ['jitNN','PyTorch','ORT','TRT','ORT-TRT']}
        bp=min(x for x in [row['PyTorch'],row['ORT']] if x); ba=min(x for f,x in row.items() if x and f!='jitNN')
        cells=[]
        for f,x in row.items():
            if x is None: cells.append(' '*24); continue
            mm=jmem.get((mo,p)) if f=='jitNN' else fmem(f,mo,p)
            cells.append(f"{x:7.2f} [{setup.get((mo,p,f),0):4.1f}] {{{mm}}}".ljust(24))
        print(f"{mo:14s} "+' '.join(cells)+f" {row['jitNN']/bp:5.2f}x {row['jitNN']/ba:5.2f}x {best[(mo,p)][2]} n={len(inf[(mo,p,'jitNN')])}")
        res[f"{mo}|{p}"]={**row,'jcfg':best[(mo,p)][2],'setup':{f:setup.get((mo,p,f)) for f in row},'mem':{f:(jmem.get((mo,p)) if f=='jitNN' else fmem(f,mo,p)) for f in row}}
if len(sys.argv) > 2:
    json.dump(res, open(sys.argv[2], 'w'), indent=1)
