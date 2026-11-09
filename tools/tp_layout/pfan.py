import gzip, json, os, re, sys, statistics
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
path, label = sys.argv[1], sys.argv[2]
with gzip.open(path, "rt") as f:
    d = json.load(f)
ev = [e for e in d.get("traceEvents", []) if e.get("ph") == "X" and "dur" in e]
GPU_CATS = {"kernel", "gpu_memcpy", "gpu_memset"}
gpu = [e for e in ev if e.get("cat") in GPU_CATS]
rt = [e for e in ev if e.get("cat") in ("cuda_runtime", "cuda_driver")]
ann = [e for e in ev if e.get("cat") == "user_annotation"]
cpuop = [e for e in ev if e.get("cat") in ("cpu_op", "user_annotation", "cuda_runtime", "python_function")]
def corr(e):
    a = e.get("args", {}); return a.get("correlation", a.get("correlation_id"))
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "rules.py")).read())
gl = sorted([e for e in rt if "graphlaunch" in e["name"].lower()], key=lambda e: e["ts"])
print(f"[{label}] graph launches {len(gl)}; annotations: {sorted(set(a['name'] for a in ann))[:40]}")
# graph launch intervals (decode steps)
if len(gl) > 1:
    iv = [b["ts"] - a["ts"] for a, b in zip(gl, gl[1:])]
    print(f"[{label}] hipGraphLaunch start-to-start us: {[round(x) for x in iv[:30]]}")
gcorr = set(corr(e) for e in gl)
steps = sorted([a for a in ann if a["name"].startswith("step[")], key=lambda a: a["ts"])
for s in steps[:6]:
    print(f"[{label}] STEP annotation {s['name']} cpu ts {s['ts']:.0f} dur {s['dur']/1000:.2f} ms")
# prefill = first step annotation containing EXTEND/PREFILL, else window before first graph launch
pf = [s for s in steps if "DECODE" not in s["name"]]
if pf:
    s0 = pf[0]; cs, ce = s0["ts"], s0["ts"] + s0["dur"]
else:
    cs = min(e["ts"] for e in cpuop); ce = gl[0]["ts"] if gl else max(e["ts"] + e["dur"] for e in cpuop)
# GPU kernels launched by runtime calls inside the prefill CPU window
rc = {corr(e): e for e in rt if cs <= e["ts"] <= ce}
k = [e for e in gpu if corr(e) in rc]
ngl = sum(1 for e in rt if cs <= e["ts"] <= ce and "graphlaunch" in e["name"].lower())
kg = [e for e in k if corr(e) in gcorr]
print(f"[{label}] prefill window: hipGraphLaunch calls {ngl}; kernels from graph replays {len(kg)} sum {sum(e['dur'] for e in kg)/1000:.2f} ms; eager kernels {len(k)-len(kg)} sum {sum(e['dur'] for e in k if corr(e) not in gcorr)/1000:.2f} ms")
if not k:
    print("no prefill kernels found"); sys.exit(0)
ks, ke = min(e["ts"] for e in k), max(e["ts"] + e["dur"] for e in k)
def union(iv):
    iv = sorted(iv); tot = 0; a = b = None
    for s, t in iv:
        if a is None or s > b:
            if a is not None: tot += b - a
            a, b = s, t
        else: b = max(b, t)
    return tot + (b - a if a is not None else 0)
busy = union([(e["ts"], e["ts"] + e["dur"]) for e in k])
print(f"[{label}] PREFILL CPU step {(ce-cs)/1000:.2f} ms; GPU first->last kernel {(ke-ks)/1000:.2f} ms; GPU busy {busy/1000:.2f} ms ({100*busy/(ke-ks):.0f}%); kernels {len(k)}; launches(runtime calls) {len(rc)}")
tot = sum(e["dur"] for e in k)
bc = defaultdict(float); bn = defaultdict(int)
for e in k:
    c = cat_of(e); bc[c] += e["dur"]; bn[c] += 1
for c, v in sorted(bc.items(), key=lambda x: -x[1]):
    print(f"  PCAT {c:12s} {v/1000:8.2f} ms {100*v/tot:5.1f}%  n {bn[c]}")
bk = defaultdict(lambda: [0.0, 0])
for e in k:
    x = bk[e["name"][:100]]; x[0] += e["dur"]; x[1] += 1
for n, (v, c) in sorted(bk.items(), key=lambda x: -x[1][0])[:25]:
    print(f"  PK {v/1000:7.2f} ms n {c:5d} avg {v/c:7.1f} us [{cat_of({'name':n,'cat':'kernel'})}] {n}")
# GPU idle gaps inside prefill: biggest, and the CPU runtime call that launched the kernel after each gap
ks_sorted = sorted(k, key=lambda e: e["ts"])
gaps = []; end = ks_sorted[0]["ts"] + ks_sorted[0]["dur"]
for e in ks_sorted[1:]:
    if e["ts"] > end:
        gaps.append((e["ts"] - end, e))
    end = max(end, e["ts"] + e["dur"])
gs = [g for g, _ in gaps]
print(f"[{label}] GPU idle gaps: n {len(gs)} total {sum(gs)/1000:.2f} ms; >50us: n {sum(1 for g in gs if g>50)} total {sum(g for g in gs if g>50)/1000:.2f} ms; >1ms: n {sum(1 for g in gs if g>1000)} total {sum(g for g in gs if g>1000)/1000:.2f} ms")
for g, e in sorted(gaps, key=lambda x: -x[0])[:12]:
    print(f"  GAP {g/1000:7.2f} ms before {e['name'][:70]}")
# CPU: top ops inside prefill window, and synchronizing calls
cw = [e for e in cpuop if cs <= e["ts"] <= ce]
b2 = defaultdict(lambda: [0.0, 0])
for e in cw:
    x = b2[(e["cat"], e["name"][:80])]; x[0] += e["dur"]; x[1] += 1
for (c, n), (v, m) in sorted(b2.items(), key=lambda x: -x[1][0])[:35]:
    print(f"  PC {v/1000:8.2f} ms n {m:5d} [{c}] {n}")
sync = [e for e in rt if cs <= e["ts"] <= ce and re.search(r"synchron|memcpy(?!async)|memcpydtoh|hipmemcpy$", e["name"].lower())]
print(f"[{label}] sync-ish runtime calls in prefill: n {len(sync)} total {sum(e['dur'] for e in sync)/1000:.2f} ms; top: {[(e['name'], round(e['dur'])) for e in sorted(sync, key=lambda e:-e['dur'])[:8]]}")
lk = [e for e in rt if cs <= e["ts"] <= ce and "launch" in e["name"].lower()]
print(f"[{label}] launch calls n {len(lk)} total {sum(e['dur'] for e in lk)/1000:.2f} ms")

