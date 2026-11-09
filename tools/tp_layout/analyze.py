import gzip, json, re, sys, statistics
from collections import defaultdict

path, label = sys.argv[1], sys.argv[2]
cpu_mode = len(sys.argv) > 3 and sys.argv[3] == "cpu"
with gzip.open(path, "rt") as f:
    d = json.load(f)
ev = [e for e in d.get("traceEvents", []) if e.get("ph") == "X" and "dur" in e]
cats = defaultdict(int)
for e in ev:
    cats[e.get("cat")] += 1
print("event cats:", dict(cats))

GPU_CATS = {"kernel", "gpu_memcpy", "gpu_memset"}
gpu = [e for e in ev if e.get("cat") in GPU_CATS]
rt = [e for e in ev if e.get("cat") in ("cuda_runtime", "cuda_driver")]
if not gpu:
    print("NO GPU EVENTS"); sys.exit(0)

RULES = [
    ("allreduce", r"nccl|rccl|allreduce|all_reduce|cross_device|reduce_scatter|allgather|all_gather"),
    ("moe", r"fused_moe|moe_align|moe_sum|count_and_sort|topk|moe|expert|align_block|grouped_gemm"),
    ("gdn", r"gated_delta|delta_rule|causal_conv1d|conv1d|gdn|recurrent|l2norm|kkt|solve_tril|wy_fast|cumsum|chunk_local|chunk_fwd|chunk_scaled|sigmoid_gating|mamba|fused_gate|chunk_o|chunk_h|merge_16x16|solve"),
    ("attn", r"_fwd_kernel|decode_att|stage1|stage2|stage_1|stage_2|extend|flash|attention|attn|rope|rotary|_fwd_grouped|softmax_reducev|set_kv|store_kv|kv_buffer"),
    ("norm", r"norm"),
    ("act", r"silu|gelu|act_and_mul|swiglu|sigmoid"),
    ("gemm", r"cijk|gemm|gemv|matmul|mfma|wvsplit|hipblaslt|tensile|splitk|ck_|_mm_|addmm|bmm|skinny|llgemm|ordered_reduce"),
    ("sample_misc", r"argmax|sampl|reduce_kernel|index|gather|scatter|cat|copy|fill|elementwise|vectorized|unrolled|arange|embedding|where|compare|cumsum"),
]


def cat_of(e):
    if e.get("cat") == "gpu_memcpy":
        return "memcpy"
    if e.get("cat") == "gpu_memset":
        return "memset"
    n = e.get("name", "").lower()
    for c, rx in RULES:
        if re.search(rx, n):
            return c
    return "other"


def union(iv):
    iv = sorted(iv)
    tot = 0; cs = ce = None
    for s, t in iv:
        if cs is None or s > ce:
            if cs is not None:
                tot += ce - cs
            cs, ce = s, t
        else:
            ce = max(ce, t)
    if cs is not None:
        tot += ce - cs
    return tot


def corr(e):
    a = e.get("args", {})
    return a.get("correlation", a.get("correlation_id"))


glaunch = [e for e in rt if "graphlaunch" in e.get("name", "").lower()]
gcorr = {corr(e): e for e in glaunch}
by_corr = defaultdict(list)
for e in gpu:
    by_corr[corr(e)].append(e)
graphs = []
for c, l in gcorr.items():
    ks = by_corr.get(c, [])
    if ks:
        graphs.append({"launch": l, "k": ks, "s": min(k["ts"] for k in ks), "e": max(k["ts"] + k["dur"] for k in ks)})
graphs.sort(key=lambda g: g["s"])
if not graphs and len(gpu) > 2000:
    # fallback: kernels not correlated to hipGraphLaunch; segment steps at GPU idle gaps >= GAP us
    GAP = 150
    gs = sorted(gpu, key=lambda e: e["ts"])
    cur = [gs[0]]; end = gs[0]["ts"] + gs[0]["dur"]; segs = []
    for e in gs[1:]:
        if e["ts"] - end >= GAP:
            segs.append(cur); cur = []
        cur.append(e); end = max(end, e["ts"] + e["dur"])
    segs.append(cur)
    print(f"FALLBACK segmentation at idle>={GAP}us: {len(segs)} segments, sizes median {statistics.median(len(x) for x in segs)}")
    big = statistics.median(len(x) for x in segs) * 0.5
    fake = {"ts": 0, "dur": 0}
    graphs = [{"launch": fake, "k": x, "s": min(k["ts"] for k in x), "e": max(k["ts"] + k["dur"] for k in x)} for x in segs if len(x) >= big]
print(f"runtime events {len(rt)}; graph launches {len(glaunch)}; graphs with kernels {len(graphs)}")
if graphs:
    nk = [len(g["k"]) for g in graphs]
    print(f"kernels per graph replay: min {min(nk)} median {statistics.median(nk)} max {max(nk)}")
    # keep only the dominant graph (decode bs) replays
    ws, we = graphs[0]["s"], graphs[-1]["e"]
    nsteps = len(graphs)
else:
    ws = min(e["ts"] for e in gpu); we = max(e["ts"] + e["dur"] for e in gpu)
    nsteps = 1
win = [e for e in gpu if e["ts"] >= ws - 1 and e["ts"] + e["dur"] <= we + 1]
W = we - ws
busy = union([(e["ts"], e["ts"] + e["dur"]) for e in win])
gk = set(id(k) for g in graphs for k in g["k"])
ingraph = [e for e in win if id(e) in gk]
outgraph = [e for e in win if id(e) not in gk]
print(f"window {W/1000:.2f} ms, steps(graph replays) {nsteps}")
if graphs and nsteps > 1:
    per = [b["s"] - a["s"] for a, b in zip(graphs, graphs[1:])]
    span = [g["e"] - g["s"] for g in graphs]
    gbusy = [union([(k["ts"], k["ts"] + k["dur"]) for k in g["k"]]) for g in graphs]
    gap = [b["s"] - a["e"] for a, b in zip(graphs, graphs[1:])]
    print(f"STEP period start-to-start us: p50 {statistics.median(per):.0f} mean {statistics.mean(per):.0f} p90 {sorted(per)[int(.9*len(per))]:.0f} min {min(per):.0f} max {max(per):.0f}")
    print(f"STEP graph span us (first kernel start->last kernel end): p50 {statistics.median(span):.0f} mean {statistics.mean(span):.0f}")
    print(f"STEP graph busy (union) us: p50 {statistics.median(gbusy):.0f} mean {statistics.mean(gbusy):.0f}; idle inside graph span mean {statistics.mean(s-b for s,b in zip(span,gbusy)):.0f}")
    print(f"STEP gap between graph replays us: p50 {statistics.median(gap):.0f} mean {statistics.mean(gap):.0f} p90 {sorted(gap)[int(.9*len(gap))]:.0f} max {max(gap):.0f}")
    ob = union([(e["ts"], e["ts"] + e["dur"]) for e in outgraph])
    print(f"STEP non-graph GPU work per step us: sum {sum(e['dur'] for e in outgraph)/(nsteps):.0f} union {ob/nsteps:.0f} n/step {len(outgraph)/nsteps:.1f}")
print(f"GPU busy(union) {busy/1000:.2f} ms of {W/1000:.2f} ms = {100*busy/W:.1f}%; idle {100*(W-busy)/W:.1f}%  (per step: busy {busy/nsteps:.0f} us, idle {(W-busy)/nsteps:.0f} us)")
tot = sum(e["dur"] for e in win)
bc = defaultdict(float); bcn = defaultdict(int)
for e in win:
    c = cat_of(e); bc[c] += e["dur"]; bcn[c] += 1
print(f"CATEGORY (sum of kernel durations; total {tot/nsteps:.0f} us/step, {tot/W*100:.1f}% of window):")
for c, v in sorted(bc.items(), key=lambda x: -x[1]):
    print(f"  CAT {c:12s} {v/nsteps:9.0f} us/step  {100*v/tot:5.1f}% of kernel time  {100*v/W:5.1f}% of wall  n/step {bcn[c]/nsteps:7.1f}")
def grid(e):
    a = e.get("args", {})
    return str(a.get("grid", "")).replace(" ", "") + "/" + str(a.get("block", "")).replace(" ", "")
bk = defaultdict(lambda: [0.0, 0])
for e in win:
    k = bk[e.get("name", "")[:100]]; k[0] += e["dur"]; k[1] += 1
print("TOP kernels (by name):")
for n, (v, c) in sorted(bk.items(), key=lambda x: -x[1][0])[:45]:
    print(f"  K {v/nsteps:8.0f} us/step {100*v/tot:5.1f}%  n/step {c/nsteps:6.1f}  avg {v/c:7.1f} us  [{cat_of({'name':n,'cat':'kernel'})}]  {n}")
bg = defaultdict(lambda: [0.0, 0, []])
for e in win:
    k = bg[(e.get("name", "")[:70], grid(e))]; k[0] += e["dur"]; k[1] += 1; k[2].append(e["dur"])
print("TOP kernels (by name+grid/block):")
for (n, g), (v, c, l) in sorted(bg.items(), key=lambda x: -x[1][0])[:60]:
    print(f"  KG {v/nsteps:8.0f} us/step n/step {c/nsteps:6.1f} avg {v/c:7.1f} p50 {statistics.median(l):7.1f} us [{cat_of({'name':n,'cat':'kernel'})}] {n} grid/block {g}")
if graphs:  # also with one correlated replay (j153 traces matched only one)
    nk = [len(g["k"]) for g in graphs]
    mode = statistics.mode(nk)
    same = [sorted(g["k"], key=lambda e: e["ts"]) for g in graphs if len(g["k"]) == mode]
    print(f"POSITION table over {len(same)} replays with {mode} kernels (idx avg_us p50_us cat name grid/block):")
    ref = same[0]
    for i in range(mode):
        ds = [s[i]["dur"] for s in same]
        if i < 230 or i >= mode - 25:
            print(f"  P {i:4d} {statistics.mean(ds):7.1f} {statistics.median(ds):7.1f} [{cat_of(ref[i])}] {ref[i].get('name','')[:80]} {grid(ref[i])}")
    # gaps between consecutive kernels inside a replay (launch bubbles inside the graph)
    bub = []
    for s in same:
        bub += [b["ts"] - (a["ts"] + a["dur"]) for a, b in zip(s, s[1:])]
    print(f"INTRA-GRAPH bubbles: mean {statistics.mean(bub):.2f} us, sum per step {sum(bub)/len(same):.0f} us, n>5us {sum(1 for x in bub if x>5)/len(same):.0f}/step")

    # Plan 26: the whole sparse-MoE block per layer, whatever path ran it. A block is every kernel
    # after the post-attention norm and before the all-reduce that follows the routed experts:
    # router GEMV, top-k, routing, routed experts, activation, shared expert and its gate, combine.
    # With the TP=2 megakernel that is the one-launch block; at TP=4/8 it is the per-op path.
    ANCH = re.compile(r"topk|fused_moe|moe_block|decode_moe|moe_align|moe_sum")

    def moe_spans(seq):
        cs = [cat_of(e) for e in seq]
        nm = [e.get("name", "") for e in seq]
        anchors = [i for i, n in enumerate(nm) if ANCH.search(n.lower())]
        spans = []; i = 0
        while i < len(anchors):
            a0 = anchors[i]; j = i
            while j + 1 < len(anchors) and not any(c == "allreduce" for c in cs[anchors[j]:anchors[j + 1]]):
                j += 1
            a1 = anchors[j]
            s0 = a0
            while s0 - 1 >= 0 and cs[s0 - 1] not in ("norm", "allreduce"):
                s0 -= 1
            e1 = a1 + 1
            while e1 < len(seq) and cs[e1] != "allreduce":
                e1 += 1
            spans.append((s0, e1)); i = j + 1
        return spans

    blk = []; nsp = []
    for s in same:
        sp = moe_spans(s)
        nsp.append(len(sp))
        blk.append(sum(s[k]["dur"] for a, b in sp for k in range(a, b)))
    gb = [union([(k["ts"], k["ts"] + k["dur"]) for k in s]) for s in same]
    ksum = [sum(k["dur"] for k in s) for s in same]
    print(f"MOEBLOCK spans/step {statistics.median(nsp)}; MoE block kernel time p50 {statistics.median(blk):.0f} us/step; "
          f"step busy p50 {statistics.median(gb):.0f} us; step kernel-sum p50 {statistics.median(ksum):.0f} us; "
          f"non-MoE (busy - block) p50 {statistics.median([g - b for g, b in zip(gb, blk)]):.0f} us/step")
    sp0 = moe_spans(same[0])
    if sp0:
        a, b = sp0[min(1, len(sp0) - 1)]
        print("MOEBLOCK kernels of one layer (avg_us cat name):")
        for k in range(a, b):
            ds = [s[k]["dur"] for s in same]
            print(f"  MB {statistics.mean(ds):7.1f} [{cat_of(same[0][k])}] {same[0][k].get('name','')[:90]}")

if cpu_mode:
    cpu = [e for e in ev if e.get("cat") in ("cpu_op", "user_annotation", "python_function", "cuda_runtime", "cuda_driver")]
    if graphs and nsteps > 1:
        lead = []
        for a, b in zip(graphs, graphs[1:]):
            l = b["launch"]
            lead.append(a["e"] - (l["ts"] + l["dur"]))
        print(f"LAUNCH lead (prev graph GPU end - next hipGraphLaunch return) us: p50 {statistics.median(lead):.0f} min {min(lead):.0f} max {max(lead):.0f}  (positive => CPU is ahead, GPU-bound)")
        ld = [g["launch"]["dur"] for g in graphs]
        print(f"LAUNCH hipGraphLaunch CPU call duration us: p50 {statistics.median(ld):.0f} max {max(ld):.0f}")
        cw = [e for e in cpu if e["ts"] >= ws and e["ts"] <= we]
    else:
        cw = cpu
        ls = [e for e in rt if "launch" in e.get("name", "").lower()]
        print(f"LAUNCH calls {len(ls)} sum {sum(e['dur'] for e in ls)/1000:.2f} ms; CPU window first->last launch {(max(e['ts'] for e in ls)-min(e['ts'] for e in ls))/1000 if ls else 0:.2f} ms")
    bn = defaultdict(lambda: [0.0, 0])
    for e in cw:
        k = bn[(e.get("cat"), e.get("name", "")[:90])]; k[0] += e["dur"]; k[1] += 1
    print("TOP CPU events (inclusive, per step):")
    for (c, n), (v, k) in sorted(bn.items(), key=lambda x: -x[1][0])[:30]:
        print(f"  C {v/nsteps:9.0f} us/step n/step {k/nsteps:6.1f}  [{c}] {n}")

