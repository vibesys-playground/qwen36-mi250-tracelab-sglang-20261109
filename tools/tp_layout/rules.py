import re
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



