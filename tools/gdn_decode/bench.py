"""Microbenchmark and correctness check: GDN decode body on gfx90a (plan 15).

Graph-replays 30 chained GDN layers at the real per-GCD TP=2 shapes, each layer
with its own weights, Conv1D state and SSM state, and reports us per layer for:
  A  today: split -> causal_conv1d_update -> packed decode (BV32, w1) -> gated norm
  B  level 1: fused unpack+Conv1D -> packed decode (BV32, w1) -> gated norm
  C  level 1 + packed decode retiled (BV, num_warps sweep)
  D  level 2: fused unpack+Conv1D -> recurrence+gated norm (num_warps sweep)
  E  level 3: one kernel for the whole body (num_warps sweep)
Correctness: fp32 torch reference over 8 steps (output and state), bitwise
parity of B's unpack/conv with A's, bitwise repeatability, row (batch)
invariance, and padding slots. Not on any serving path.
"""
import json
import os
import sys
import time
import traceback
from types import SimpleNamespace

import torch

LAYERS = 30
BATCHES = (1, 2, 4, 8)
RESULTS = {}


def log(*a):
    print(*a, flush=True)


from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))

import triton

from sglang.kernels.ops.attention.fla.fused_recurrent import (
    fused_recurrent_gated_delta_rule_packed_decode,
    fused_recurrent_gated_delta_rule_packed_decode_kernel,
)
from sglang.kernels.ops.attention.fla.gdn_decode_fused import (
    gdn_decode_fused_body,
    gdn_packed_decode_norm,
)
from sglang.kernels.ops.attention.fla.layernorm_gated import RMSNorm as RMSNormGated
from sglang.kernels.ops.attention.triton_gdn_fused_proj import (
    fused_qkvzba_causal_conv1d_update_contiguous,
    fused_qkvzba_split_reshape_cat_contiguous,
)
from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_update
from sglang.srt.configs.mamba_utils import mamba2_state_dtype

dev = torch.device("cuda")
MODEL = "Qwen/Qwen3.6-35B-A3B"
REV = "995ad96eacd98c81ed38be0c5b274b04031597b0"
TP = 2


def _ns(o):
    if isinstance(o, dict):
        return SimpleNamespace(**{k: _ns(v) for k, v in o.items()})
    return o


def load_config():
    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(MODEL, "config.json", revision=REV, local_files_only=True)
        with open(path) as f:
            raw = json.load(f)
        log(f"config: {path}")
        return raw
    except Exception as e:  # noqa: BLE001
        log(f"WARNING: model config not found ({e!r}); using plan shapes, fp32 state")
        return None


raw = load_config()
if raw is not None:
    tc = raw.get("text_config", raw)
    NK = tc["linear_num_key_heads"] // TP
    NV = tc["linear_num_value_heads"] // TP
    DK = tc["linear_key_head_dim"]
    DV = tc["linear_value_head_dim"]
    KW = tc["linear_conv_kernel_dim"]
    EPS = tc["rms_norm_eps"]
    ACT = tc.get("output_gate_type") or "swish"
    dts = mamba2_state_dtype(_ns(raw))
    SSM_DTYPE, CONV_DTYPE = dts.temporal, dts.conv
    log(f"config text mamba_ssm_dtype={tc.get('mamba_ssm_dtype')!r} "
        f"root={raw.get('mamba_ssm_dtype')!r} env={os.environ.get('SGLANG_MAMBA_SSM_DTYPE')!r}")
else:
    NK, NV, DK, DV, KW, EPS, ACT = 8, 16, 128, 128, 4, 1e-6, "swish"
    SSM_DTYPE, CONV_DTYPE = torch.float32, torch.bfloat16
QKV = 2 * NK * DK + NV * DV
VD = NV * DV
SCALE = DK ** -0.5
ACT_DT = torch.bfloat16
log(f"per-GCD shapes: H={NK} HV={NV} K={DK} V={DV} conv_width={KW} qkv_dim={QKV} "
    f"qkvz={QKV + VD} ba={2 * NV} eps={EPS} norm_act={ACT} ssm_dtype={SSM_DTYPE} "
    f"conv_dtype={CONV_DTYPE} device={torch.cuda.get_device_name()} "
    f"arch={getattr(torch.cuda.get_device_properties(0), 'gcnArchName', '?')}")


class Layer:
    def __init__(self, slots, gen):
        r = lambda *s, sc=1.0: (torch.randn(*s, generator=gen, device=dev) * sc)
        self.conv_w = r(QKV, KW, sc=0.5).to(ACT_DT)
        self.conv_b = None  # Qwen3.5 GDN conv1d has no bias
        self.A_log = r(NV, sc=0.5).float()
        self.dt_bias = r(NV, sc=0.5).to(ACT_DT)
        self.norm = RMSNormGated(DV, eps=EPS, group_size=None, norm_before_gate=True,
                                 dtype=ACT_DT, activation=ACT).to(dev)
        with torch.no_grad():
            self.norm.weight.copy_(1.0 + 0.1 * r(DV))
        self.conv_state = (r(slots, QKV, KW - 1)).to(CONV_DTYPE)
        self.ssm = (r(slots, NV, DV, DK, sc=0.05)).to(SSM_DTYPE)


def run_A(L, qkvz, ba, idx):
    mixed, z, b, a = fused_qkvzba_split_reshape_cat_contiguous(qkvz, ba, NK, NV, DK, DV)
    mixed = causal_conv1d_update(mixed, L.conv_state, L.conv_w, L.conv_b, "silu",
                                 conv_state_indices=idx)
    out = mixed.new_empty(mixed.shape[0], 1, NV, DV)
    fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv=mixed, a=a, b=b, A_log=L.A_log, dt_bias=L.dt_bias, scale=SCALE,
        initial_state=L.ssm, out=out, ssm_state_indices=idx,
        use_qk_l2norm_in_kernel=True)
    o = out.transpose(0, 1).reshape(-1, DV)
    return L.norm(o, z.reshape(-1, DV)).reshape(-1, NV, DV)


def fused_conv(L, qkvz, ba, idx):
    return fused_qkvzba_causal_conv1d_update_contiguous(
        qkvz, ba, L.conv_state, L.conv_w, L.conv_b, idx, qkv_dim=QKV, v_dim=VD,
        num_v_heads=NV, head_v_dim=DV, activation="silu")


def packed_tiled(mixed, a, b, L, idx, BV, warps):
    B = mixed.shape[0]
    out = mixed.new_empty(B, 1, NV, DV)
    grid = (triton.cdiv(DV, BV), B * NV)
    fused_recurrent_gated_delta_rule_packed_decode_kernel[grid](
        mixed_qkv=mixed, a=a, b=b, A_log=L.A_log, dt_bias=L.dt_bias, o=out,
        h0=L.ssm, ht=L.ssm, ssm_state_indices=idx, scale=SCALE,
        stride_mixed_qkv_tok=mixed.stride(0), stride_a_tok=a.stride(0),
        stride_b_tok=b.stride(0), stride_init_state_token=L.ssm.stride(0),
        stride_final_state_token=L.ssm.stride(0), stride_indices_seq=idx.stride(0),
        H=NK, HV=NV, K=DK, V=DV, BK=triton.next_power_of_2(DK), BV=BV,
        SOFTPLUS_THRESHOLD=20.0, USE_QK_L2NORM_IN_KERNEL=True,
        num_warps=warps, num_stages=3)
    return out


def make_C(BV, warps):
    def run(L, qkvz, ba, idx):
        mixed, z, b, a = fused_conv(L, qkvz, ba, idx)
        out = packed_tiled(mixed, a, b, L, idx, BV, warps)
        return L.norm(out.reshape(-1, DV), z.reshape(-1, DV)).reshape(-1, NV, DV)
    return run


def make_D(warps):
    def run(L, qkvz, ba, idx):
        mixed, z, b, a = fused_conv(L, qkvz, ba, idx)
        return gdn_packed_decode_norm(
            mixed, a, b, z, A_log=L.A_log, dt_bias=L.dt_bias,
            norm_weight=L.norm.weight, norm_eps=EPS, norm_activation=ACT,
            scale=SCALE, ssm_states=L.ssm, cache_indices=idx, num_warps=warps)
    return run


def make_E(warps):
    def run(L, qkvz, ba, idx):
        return gdn_decode_fused_body(
            qkvz, ba, conv_states=L.conv_state, conv_weight=L.conv_w,
            conv_bias=L.conv_b, conv_activation="silu", A_log=L.A_log,
            dt_bias=L.dt_bias, norm_weight=L.norm.weight, norm_eps=EPS,
            norm_activation=ACT, scale=SCALE, ssm_states=L.ssm, cache_indices=idx,
            num_warps=warps)
    return run


VARIANTS = {"A": run_A, "B": make_C(32, 1)}
for bv in (8, 16, 32, 64):
    for w in (1, 2, 4):
        if (bv, w) != (32, 1):
            VARIANTS[f"C_bv{bv}_w{w}"] = make_C(bv, w)
for w in (4, 8, 16):
    VARIANTS[f"D_w{w}"] = make_D(w)
    VARIANTS[f"E_w{w}"] = make_E(w)


# ---------------- fp32 reference ----------------
def ref_step(L, conv_state, ssm, qkvz, ba, idx):
    """fp32 math on fresh copies; conv_state/ssm (full pools) updated in place."""
    B = qkvz.shape[0]
    outs = torch.zeros(B, NV, DV, device=dev)
    for n in range(B):
        s = int(idx[n])
        if s < 0:
            continue
        x = qkvz[n, :QKV].float()
        st = conv_state[s].float()  # [QKV, KW-1]
        win = torch.cat([st, x[:, None]], 1)  # [QKV, KW]
        c = (win * L.conv_w.float()).sum(1)
        c = c * torch.sigmoid(c)
        conv_state[s] = win[:, 1:].to(conv_state.dtype)
        q = c[: NK * DK].view(NK, DK)
        k = c[NK * DK: 2 * NK * DK].view(NK, DK)
        v = c[2 * NK * DK:].view(NV, DV)
        q = q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6) * SCALE
        k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)
        bb = ba[n, :NV].float()
        aa = ba[n, NV:].float()
        g = -torch.exp(L.A_log.float()) * torch.nn.functional.softplus(aa + L.dt_bias.float())
        beta = torch.sigmoid(bb)
        zz = qkvz[n, QKV:].float().view(NV, DV)
        for hv in range(NV):
            h = ssm[s, hv].float() * torch.exp(g[hv])  # [V, K]
            kh, qh = k[hv // (NV // NK)], q[hv // (NV // NK)]
            vv = (v[hv] - h @ kh) * beta[hv]
            h = h + vv[:, None] * kh[None, :]
            o = h @ qh
            ssm[s, hv] = h.to(ssm.dtype)
            y = o * torch.rsqrt((o * o).mean() + EPS) * L.norm.weight.float()
            gate = torch.sigmoid(zz[hv]) if ACT == "sigmoid" else zz[hv] * torch.sigmoid(zz[hv])
            outs[n, hv] = y * gate
    return outs


def rel(a, b):
    return ((a.float() - b.float()).abs().max() / b.float().abs().max().clamp_min(1e-30)).item()


def inputs(B, gen):
    qkvz = torch.randn(B, QKV + VD, generator=gen, device=dev).to(ACT_DT)
    ba = torch.randn(B, 2 * NV, generator=gen, device=dev).to(ACT_DT)
    return qkvz, ba


def correctness():
    log("\n===== correctness =====")
    gen = torch.Generator(device=dev).manual_seed(1)
    B, slots = 8, 12
    L = Layer(slots, gen)
    idx = torch.tensor([9, 2, 11, 0, 5, 7, 3, 10], dtype=torch.int32, device=dev)
    steps = [inputs(B, gen) for _ in range(8)]
    conv0, ssm0 = L.conv_state.clone(), L.ssm.clone()

    def reset():
        L.conv_state.copy_(conv0)
        L.ssm.copy_(ssm0)

    # B's unpack+conv vs today's split+conv, bitwise.
    qkvz, ba = steps[0]
    reset()
    m_a, z_a, b_a, a_a = fused_qkvzba_split_reshape_cat_contiguous(qkvz, ba, NK, NV, DK, DV)
    m_a = causal_conv1d_update(m_a, L.conv_state, L.conv_w, L.conv_b, "silu",
                               conv_state_indices=idx)
    cs_a = L.conv_state.clone()
    reset()
    m_b, z_b, b_b, a_b = fused_conv(L, qkvz, ba, idx)
    eq = all(torch.equal(x, y) for x, y in
             ((m_a, m_b), (z_a, z_b), (b_a, b_b), (a_a, a_b), (cs_a, L.conv_state)))
    log(f"level-1 unpack+conv bitwise == today (mixed_qkv, z, b, a, conv_state): {eq}")
    RESULTS["l1_bitwise"] = eq

    # fp32 reference over 8 steps.
    rc, rs = conv0.float().to(CONV_DTYPE).clone(), ssm0.clone().float()
    ref_outs = [ref_step(L, rc, rs, q, b_, idx) for q, b_ in steps]
    today_outs = None
    table = {}
    for name, fn in VARIANTS.items():
        try:
            reset()
            outs = [fn(L, q, b_, idx).clone() for q, b_ in steps]
            torch.cuda.synchronize()
            e_out = max(rel(o, r) for o, r in zip(outs, ref_outs))
            e_state = rel(L.ssm[idx.long()], rs[idx.long()])
            e_conv = torch.equal(L.conv_state, rc)
            if name == "A":
                today_outs = outs
                today_state = L.ssm.clone()
            same_today = all(torch.equal(o, t) for o, t in zip(outs, today_outs))
            same_state = torch.equal(L.ssm, today_state)
            # repeatability (100 runs of step 0 from the same state)
            ok_rep = True
            first = None
            for _ in range(100):
                reset()
                o = fn(L, *steps[0], idx)
                s = L.ssm[idx.long()]
                if first is None:
                    first = (o.clone(), s.clone())
                elif not (torch.equal(o, first[0]) and torch.equal(s, first[1])):
                    ok_rep = False
                    break
            # row invariance: each prefix batch gives the same rows bitwise
            ok_rows = True
            for bsub in range(1, B + 1):
                reset()
                o = fn(L, steps[0][0][:bsub].contiguous(), steps[0][1][:bsub].contiguous(),
                       idx[:bsub].contiguous())
                if not (torch.equal(o, first[0][:bsub])
                        and torch.equal(L.ssm[idx[:bsub].long()], first[1][:bsub])):
                    ok_rows = False
            # padding: rows 1,3 padded; their outputs zero, no other slot touched
            reset()
            pidx = idx.clone()
            pidx[1] = -1
            pidx[3] = -1
            o = fn(L, *steps[0], pidx)
            live = pidx[pidx >= 0].long()
            untouched = torch.ones(slots, dtype=torch.bool, device=dev)
            untouched[live] = False
            ok_pad = (o[1].abs().max().item() == 0 and o[3].abs().max().item() == 0
                      and torch.equal(L.ssm[untouched], ssm0[untouched])
                      and torch.equal(L.conv_state[untouched], conv0[untouched])
                      and torch.equal(o[0], first[0][0]))
            table[name] = dict(out_rel=e_out, state_rel=e_state, conv_state_eq_ref=e_conv,
                               bitwise_today=same_today, state_bitwise_today=same_state,
                               repeat100=ok_rep, row_invariant=ok_rows, padding=ok_pad)
            log(f"{name:12s} out_rel={e_out:.2e} state_rel={e_state:.2e} "
                f"conv_eq_ref={e_conv} ==today(out/state)={same_today}/{same_state} "
                f"repeat100={ok_rep} rows={ok_rows} pad={ok_pad}")
        except Exception:
            traceback.print_exc()
            table[name] = "ERROR"
    RESULTS["correctness"] = table


def timing():
    log("\n===== timing: us per layer, 30-layer graph replay =====")
    times = {}
    for B in BATCHES:
        gen = torch.Generator(device=dev).manual_seed(100 + B)
        slots = B + 3
        layers = [Layer(slots, gen) for _ in range(LAYERS)]
        perm = torch.randperm(slots, generator=gen, device=dev)[:B].to(torch.int32)
        ins = [inputs(B, gen) for _ in range(LAYERS)]
        log(f"B={B}: ssm state {LAYERS * B * NV * DV * DK * layers[0].ssm.element_size() / 2**20:.0f} MiB touched per replay")
        for name, fn in VARIANTS.items():
            try:
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    for _ in range(2):
                        for L, (q, b_) in zip(layers, ins):
                            fn(L, q, b_, perm)
                torch.cuda.current_stream().wait_stream(s)
                torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    for L, (q, b_) in zip(layers, ins):
                        fn(L, q, b_, perm)
                for _ in range(10):
                    g.replay()
                torch.cuda.synchronize()
                st = [torch.cuda.Event(enable_timing=True) for _ in range(51)]
                for i in range(50):
                    st[i].record()
                    g.replay()
                st[50].record()
                torch.cuda.synchronize()
                per = sorted(st[i].elapsed_time(st[i + 1]) for i in range(50))
                us = per[len(per) // 2] * 1000 / LAYERS
                times[(name, B)] = us
                log(f"  B={B} {name:12s} {us:7.2f} us/layer (min {per[0] * 1000 / LAYERS:.2f})")
                del g
            except Exception:
                traceback.print_exc()
                times[(name, B)] = float("nan")
        del layers
        torch.cuda.empty_cache()
    RESULTS["times"] = times
    hdr = "variant      " + "".join(f"  B={B:<5d}" for B in BATCHES)
    log("\n" + hdr)
    for name in VARIANTS:
        log(f"{name:12s} " + "".join(f"  {times.get((name, B), float('nan')):7.2f}" for B in BATCHES))
    best = min((n for n in VARIANTS if n != "A"), key=lambda n: times.get((n, 2), 1e9))
    for fam in ("C", "D", "E"):
        fb = min((n for n in VARIANTS if n.startswith(fam)), key=lambda n: times.get((n, 2), 1e9))
        log(f"best {fam} at B=2: {fb} {times[(fb, 2)]:.2f} us")
    a2 = times[("A", 2)]
    log(f"\nGATE (B=2): today A={a2:.2f} us; best={best} {times[(best, 2)]:.2f} us")
    if a2 < 25:
        log("GATE: A under 25 us -> j78 attribution does not reproduce; re-derive the bound")
    b2 = times[(best, 2)]
    log("GATE: " + ("PASS (<= 12 us)" if b2 <= 12 else "PARK (> 17 us)" if b2 > 17
                    else "BETWEEN 12 and 17 us"))


if __name__ == "__main__":
    t = time.time()
    try:
        correctness()
    except Exception:
        traceback.print_exc()
    timing()
    log(f"\n[done in {time.time() - t:.0f}s]")
