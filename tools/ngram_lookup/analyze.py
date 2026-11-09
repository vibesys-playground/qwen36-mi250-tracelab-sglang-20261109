"""Plan 27 stage 0, phase C (CPU only): prompt-lookup accept length, verify-window expert
overlap and the projected decode speedup per k, with the plan's kill gates.

Draft source (the only thing a server would run): for one request, the context is its own
prompt ids + the output ids emitted so far. Find the longest suffix of n tokens (n from NMAX
down to NMIN) that also ended earlier in the context, take the most recent such earlier
occurrence and draft the up-to-k tokens that followed it. No match: no draft, the row verifies
alone (M=1). Nothing else is read: no other request, no trace, no token pool.

Greedy verification with exact numerics accepts the longest draft prefix equal to the target's
own greedy tokens, plus one bonus token. Since the replay captured those greedy tokens, the
simulation is exact for acceptance.

usage: analyze.py REPLAY.json ROUTES.npz [REPLAY2.json]
"""
import base64, json, math, random, sys, zlib

import numpy as np

KS = [1, 2, 3, 5, 7]
NE, TOPK, NL = 256, 8, 40

# ---------------- head per-op costs, j138 (5215874, one TP=2 replica, bs=2 decode graph) -------
# us per step: moe 2649 (40 x hip_decode_moe::moe_block_kernel<2>, 66.2 us each), skf2 GEMVs
# 838+611, lm_head GEMM 541, allreduce 634, norm 480, attn 510 (10 layers), gdn 409 (30 layers),
# sample/misc/act/memcpy 87+41+18+3. Window 6733 us, GPU 100% busy.
J138 = dict(moe=2649, gemv=1449, lm_head=541, ar=634, norm=480, attn=510, gdn=409, misc=149)
J138_STEP = 6733.0
# j151 (MI210 gfx90a, moe_block_bench, head kernel 'base'): us per layer and MB streamed per
# layer at M = 1/2/4 rows with random routing.
J151_US = {1: 39.38, 2: 62.75, 4: 123.5}
J151_MB = {1: 29.364224, 2: 53.743616, 4: 100.929536}


def e_unique_random(m):
    return NE * (1 - (1 - TOPK / NE) ** m)


# MB streamed per layer as a function of unique routed experts u: fit fixed + per-expert on j151.
_PER_EXPERT_MB = (J151_MB[4] - J151_MB[1]) / (e_unique_random(4) - TOPK)
_FIXED_MB = J151_MB[1] - TOPK * _PER_EXPERT_MB
# us per MB: M<=2 rows -> j151 M=2 rate; M>=3 -> the M=4 rate (slower per byte).
_US_PER_MB = {2: J151_US[2] / J151_MB[2], 4: J151_US[4] / J151_MB[4]}


def moe_layer_us(u, m):
    return (_FIXED_MB + u * _PER_EXPERT_MB) * (_US_PER_MB[2] if m <= 2 else _US_PER_MB[4])


def nonmoe_us(rows, mode):
    """rows: list of rows per sequence (1 + draft len). Non-MoE cost of one step."""
    m = sum(rows)
    nseq = len(rows)
    extra = m - nseq  # draft positions beyond the plain decode
    base = J138.copy()
    if nseq == 1:  # bs=1: per-sequence ops (attn, gdn, sampling) roughly halve
        base["attn"] *= 0.6
        base["gdn"] *= 0.6
    if mode == "opt":
        # GEMV flat within 5% to M=16 (plan 20); lm_head and AR latency bound at M<=16 (+1%/row);
        # attention: k+1 queries share the row's KV read (+10% per extra query);
        # GDN: per-position intermediate states written for verify (+50% per extra position:
        # one state write on top of the read+write of a plain step); +60 us host/accept.
        q_extra = extra / nseq
        return (base["gemv"] * (1 + 0.05 * max(0, m - 2) / 14)
                + (base["lm_head"] + base["ar"]) * (1 + 0.01 * max(0, m - 2))
                + base["norm"] + base["misc"]
                + base["attn"] * (1 + 0.10 * q_extra)
                + base["gdn"] * (1 + 0.50 * q_extra)
                + (60.0 if extra else 0.0))
    # pess: every extra row costs what an extra request costs per replica in j153
    # (bs 2 -> 4: 6.9 -> 9.9 ms, of which ~2.2 ms is MoE by this model): ~0.4 ms per row.
    return sum(base.values()) - base["moe"] + 400.0 * extra


# ------------------------------------------------------------------ prompt lookup ----------
class Index:
    def __init__(self, nmin, nmax):
        self.nmin, self.nmax = nmin, nmax
        self.ctx = []
        self.d = {n: {} for n in range(nmin, nmax + 1)}

    def copy(self):
        o = Index(self.nmin, self.nmax)
        o.ctx = list(self.ctx)
        o.d = {n: dict(v) for n, v in self.d.items()}
        return o

    def append(self, t):
        c = self.ctx
        c.append(t)
        L = len(c)
        for n in range(self.nmin, self.nmax + 1):
            if L >= n:
                key = tuple(c[L - n:L])
                prev = self.d[n].get(key)
                self.d[n][key] = (L, prev[0] if prev else None)

    def draft(self, k):
        c = self.ctx
        L = len(c)
        for n in range(self.nmax, self.nmin - 1, -1):
            if L < n:
                continue
            ent = self.d[n].get(tuple(c[L - n:L]))
            if ent is None:
                continue
            e = ent[1] if ent[0] == L else ent[0]
            if e is not None:
                return c[e:e + k]
        return []


def simulate(rec, k, idx0):
    """Steps (draft_len, emitted) for one request. out[0] comes from prefill."""
    out = rec["out"]
    T = len(out)
    ix = idx0.copy()
    ix.append(out[0])
    pos, steps = 1, []
    while pos < T:
        d = ix.draft(min(k, T - pos - 1)) if T - pos - 1 > 0 else []
        acc = 0
        while acc < len(d) and d[acc] == out[pos + acc]:
            acc += 1
        emit = acc + 1
        steps.append((len(d), emit))
        for t in out[pos:pos + emit]:
            ix.append(t)
        pos += emit
    return steps


def lookup_tables(recs, nmin, nmax, label):
    res = {}
    for k in KS:
        res[k] = []
    for r in recs:
        if len(r["out"]) < 2:
            continue
        idx0 = Index(nmin, nmax)
        for t in r["prompt"]:
            idx0.append(t)
        for k in KS:
            res[k].append((r, simulate(r, k, idx0)))
    print(f"\n=== a(k) [{label}] suffix n={nmax}..{nmin}, most recent occurrence; "
          f"a = tokens per verify step incl. bonus, over all decode steps of the row ===")
    groups = [("all", lambda r: True), ("round0", lambda r: r["round"] == 0),
              ("round>=1", lambda r: r["round"] >= 1),
              ("out<64", lambda r: len(r["out"]) < 64), ("out64-255", lambda r: 64 <= len(r["out"]) < 256),
              ("out>=256", lambda r: len(r["out"]) >= 256)]
    print(f"{'group':10s} {'n_req':>5s} " + " ".join(f"{'a(' + str(k) + ')':>7s}" for k in KS)
          + "   draft%(k=3) a|draft(k=3) mean_d(k=3) acc_rate(k=3)")
    summary = {}
    for g, f in groups:
        row = []
        for k in KS:
            st = [s for r, ss in res[k] if f(r) for s in ss]
            ntok = sum(e for _, e in st)
            row.append(ntok / max(1, len(st)))
            if k == 3:
                dr = [s for s in st if s[0] > 0]
                pct = 100 * len(dr) / max(1, len(st))
                ad = sum(e for _, e in dr) / max(1, len(dr))
                md = sum(d for d, _ in st) / max(1, len(st))
                accr = sum(e - 1 for _, e in dr) / max(1, sum(d for d, _ in dr))
        nreq = sum(1 for r, _ in res[KS[0]] if f(r))
        print(f"{g:10s} {nreq:5d} " + " ".join(f"{x:7.3f}" for x in row)
              + f"   {pct:6.1f}%      {ad:6.3f}      {md:5.2f}       {accr:5.3f}")
        summary[g] = dict(zip(KS, row))
    return res, summary


# ------------------------------------------------------------------ expert overlap ---------
def overlap_tables(routes, rng):
    print("\n=== unique routed experts per layer in a verify window (real routing, teacher-forced) ===")
    print("w = k+1 consecutive tokens of one request; bs=2: union with an independent window of "
          "another request (M = 2w rows). random = 256(1-(1-8/256)^M).")
    print(f"{'k':>2s} {'w':>2s} {'1row_u':>7s} {'1row_rand':>9s} {'bs2_M':>5s} {'bs2_u':>7s} {'bs2_rand':>8s} {'bs2_u/rand':>10s}")
    out = {}
    longs = [r for r in routes if r.shape[1] >= 8]
    for k in [0] + KS:
        w = k + 1
        u1, u2 = [], []
        for _ in range(3000):
            a = longs[rng.randrange(len(longs))]
            b = longs[rng.randrange(len(longs))]
            l = rng.randrange(NL)
            sa = rng.randrange(a.shape[1] - w + 1)
            sb = rng.randrange(b.shape[1] - w + 1)
            ea = set(a[l, sa:sa + w].ravel().tolist())
            eb = set(b[l, sb:sb + w].ravel().tolist())
            u1.append(len(ea))
            u2.append(len(ea | eb))
        m1, m2 = np.mean(u1), np.mean(u2)
        print(f"{k:2d} {w:2d} {m1:7.2f} {e_unique_random(w):9.2f} {2 * w:5d} {m2:7.2f} "
              f"{e_unique_random(2 * w):8.2f} {m2 / e_unique_random(2 * w):10.3f}")
        out[k] = (m1, m2)
    return out


def step_cost(rows, windows, mode):
    """rows: per-seq row counts; windows: per-seq (route array, start). Returns us."""
    m = sum(rows)
    moe = 0.0
    step_cost.u = 0
    for l in range(NL):
        s = set()
        for (rt, st), w in zip(windows, rows):
            s.update(rt[l, st:st + w].ravel().tolist())
        moe += moe_layer_us(len(s), m)
        step_cost.u += len(s) / NL
    return moe + nonmoe_us(rows, mode), moe


def project(res, routes, rng, label):
    longs = [r for r in routes if r.shape[1] >= 16]
    print(f"\n=== projected decode speedup S(k) [{label}] (MC over row-steps x real routing windows) ===")
    print("S = (tokens/step / cost/step) / (plain tokens/step / plain cost/step); "
          "cost calibrated so the plain bs=2 step equals j138's 6733 us")
    print(f"{'k':>2s} {'bs':>2s} {'mode':>4s} {'a/row':>6s} {'E[M]':>5s} {'E[u]':>6s} {'moe_us':>7s} {'step_us':>8s} {'c':>5s} {'S':>5s} {'wall_gain':>9s}")
    N = 3000
    # plain cost by bs
    plain = {}
    for bs in (1, 2):
        cs, ms = [], []
        for _ in range(N):
            ws = []
            for _ in range(bs):
                rt = longs[rng.randrange(len(longs))]
                ws.append((rt, rng.randrange(rt.shape[1])))
            c, mo = step_cost([1] * bs, ws, "opt")
            cs.append(c); ms.append(mo)
        plain[bs] = (np.mean(cs), np.mean(ms))
    scale = J138_STEP / plain[2][0]
    print(f"plain model: bs=1 {plain[1][0]:.0f} us (moe {plain[1][1]:.0f}), bs=2 {plain[2][0]:.0f} us "
          f"(moe {plain[2][1]:.0f}; j138 moe 2649); calibration x{scale:.3f}")
    best = {}
    for k in KS:
        steps = [s for _, ss in res[k] for s in ss]
        for bs in (1, 2):
            for mode in ("opt", "pess"):
                tok, cost, mm, uu, mo_s = 0.0, 0.0, 0.0, 0.0, 0.0
                for _ in range(N):
                    rows, ws = [], []
                    for _ in range(bs):
                        d, e = steps[rng.randrange(len(steps))]
                        tok += e
                        w = d + 1
                        rt = longs[rng.randrange(len(longs))]
                        while rt.shape[1] < w:
                            rt = longs[rng.randrange(len(longs))]
                        rows.append(w)
                        ws.append((rt, rng.randrange(rt.shape[1] - w + 1)))
                    c, mo = step_cost(rows, ws, mode)
                    cost += c; mo_s += mo; mm += sum(rows)
                    uu += step_cost.u
                pc = plain[bs][0]
                c_rel = (cost / N) / pc
                S = (tok / N / bs) / c_rel
                wall = 77.0 / (13.8 + 63.0 / S) - 1
                print(f"{k:2d} {bs:2d} {mode:>4s} {tok / N / bs:6.3f} {mm / N:5.1f} {uu / N:6.1f} {mo_s / N * scale:7.0f} "
                      f"{cost / N * scale:8.0f} {c_rel:5.2f} {S:5.2f} {100 * wall:8.1f}%")
                best[(k, bs, mode)] = S
    return best


def main():
    rp = json.load(open(sys.argv[1]))
    recs = [r for r in rp["recs"] if r["out"]]
    print(f"replay: {len(recs)} requests with output, {sum(len(r['out']) for r in recs)} output tokens, "
          f"nonce {rp['nonce']}")
    res, summ = lookup_tables(recs, 2, 4, "main n=2..4")
    lookup_tables(recs, 1, 4, "variant n=1..4")
    if len(sys.argv) > 3:
        rp2 = json.load(open(sys.argv[3]))
        r2 = [r for r in rp2["recs"] if r["out"]]
        print(f"\nnonce check: second replay (round 0 only) nonce {rp2['nonce']}, {len(r2)} requests; "
              "compare with round0 of the main replay above")
        lookup_tables(r2, 2, 4, "nonce-2 round0, n=2..4")
        same = 0
        m = {(r["sid"], r["round"]): r for r in recs}
        for r in r2:
            o = m.get((r["sid"], r["round"]))
            if o and len(o["out"]) == len(r["out"]):
                same += sum(1 for x, y in zip(o["out"], r["out"]) if x == y)
        print(f"nonce check: tokens equal at the same position across nonces: {same} of "
              f"{sum(len(r['out']) for r in r2)} (expected low: ids differ with the nonce)")
    # samples of the text, so a reader can see what the model writes
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-35B-A3B", revision="995ad96eacd98c81ed38be0c5b274b04031597b0")
        for r in recs[:2] + [x for x in recs if x["round"] >= 2][:2]:
            print(f"\nSAMPLE sess {r['sid']} round {r['round']} prompt tail: {tok.decode(r['prompt'][-60:])!r}")
            print(f"SAMPLE out head: {tok.decode(r['out'][:120])!r}")
    except Exception as e:
        print("sample decode failed", e)

    z = np.load(sys.argv[2])
    routes = [z[f"r{i}"] for i in range(len(z["keys"]))]
    print(f"\nrouting: {len(routes)} requests, shapes e.g. {[r.shape for r in routes[:3]]}, "
          f"id range {min(int(r.min()) for r in routes)}..{max(int(r.max()) for r in routes)}")
    routes = [r[:, :, :TOPK] for r in routes]
    rng = random.Random(27)
    overlap_tables(routes, rng)
    best = project(res, routes, rng, "bs=1/bs=2, opt/pess non-MoE")

    a3 = summ["all"][3]
    s_opt = {k: best[(k, 2, "opt")] for k in KS}
    s_pess = {k: best[(k, 2, "pess")] for k in KS}
    bk = max(KS, key=lambda k: s_opt[k])
    print("\n=== GATES (bs=2) ===")
    print(f"a(3) = {a3:.3f}  (kill if < 1.9) -> {'KILL' if a3 < 1.9 else 'pass'}")
    print(f"best S (opt) = {s_opt[bk]:.2f} at k={bk}; best S (pess) = {max(s_pess.values()):.2f} at "
          f"k={max(KS, key=lambda k: s_pess[k])}  (kill if best < 1.25)")
    pays = [k for k in KS if s_opt[k] >= 1.25]
    print(f"k with S(opt) >= 1.25: {pays}  (park if only k<=1)")

    # compact data for re-analysis off the node
    outs = {f"{r['sid']}_{r['round']}": r["out"] for r in recs}
    blob = base64.b64encode(zlib.compress(json.dumps({"nonce": rp["nonce"], "outs": outs}).encode(), 9)).decode()
    print(f"\nOUTS_B64 {len(blob)} {blob}")


if __name__ == "__main__":
    main()
