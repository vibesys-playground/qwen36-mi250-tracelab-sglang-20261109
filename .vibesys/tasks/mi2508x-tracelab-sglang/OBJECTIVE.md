# Qwen3.6 TraceLab serving with SGLang on one AMD MI250 node

Adapt and optimize upstream SGLang to serve `Qwen/Qwen3.6-35B-A3B` behind an
OpenAI-compatible API on **one AMD MI250 node: 4 MI250 cards, 8 ROCm devices**.
Each MI250 card holds two graphics compute dies (GCDs); each GCD is a separate
ROCm device (gfx90a, CDNA2) with its own 64 GB of HBM2e, so the allocation is
8 devices and 512 GB of aggregate HBM, never one 128 GB device per card.
SGLang's ROCm support centers on MI300-series and newer GPUs; its gfx90a
kernels, model path and parallel modes for this model may be incomplete or
slow. Making the serving path work and fast on MI250 is the task. No validated
AMD baseline is supplied. The candidate lives in the project root, outside
trusted `.vibesys/` inputs; see "Starting point" below. The Transformers
baseline engine at `engine/` serves this project's other tasks and is not this
task's starting point.

Partition BF16 weights across the eight GCDs. Explore tensor, expert or
pipeline parallelism over all eight, or replicas such as two of four GCDs or
four of two GCDs. The two GCDs of one card are joined by an in-package
Infinity Fabric connection; cards are joined by Infinity Fabric links whose
layout depends on the platform. Inspect the actual topology (for example
`rocm-smi --showtopo`) and measure collective bandwidth rather than assuming
a uniform mesh; the communication-heaviest groups may fit best on GCD pairs.
Include collective and dispatch costs, and preserve session affinity for
replicas.

## Model and hardware contract

Use the BF16 checkpoint in `reference/meta.json`, including its pinned tokenizer
and chat template. The model has 35B total and about 3B active parameters,
256 experts with eight routed and one shared expert, and 40 hybrid layers:
30 Gated DeltaNet layers and ten gated-attention layers. Preserve routing,
convolution and recurrent state, positional encoding, trained weights, and the
262,144-token context capacity. Do not quantize, substitute a smaller model,
skip experts, or change precision to improve the measurement.

Serve the text-only coding workload; no vision throughput is scored. Preserve
all text-model weights and the checkpoint's tokenizer/template semantics.
PyTorch and Transformers may supply weight loading and reference operations.
Serve from the workspace SGLang code; do not proxy a separately installed or
prebuilt server or engine.
Do not use extra GPUs, another machine, or a hosted model. Host-side cache
offload is allowed within the allocation, with its costs inside measurement.
See [the project README](../../../README.md) for AMD environment setup.

## Correctness and service interface

Expose streaming `/v1/completions` with exact requested token lengths,
`ignore_eos`, and accurate prompt, cached-prompt and output token usage for
Request Factory's session replay. Expose `/v1/chat/completions` for the service
checker's short chat probes, using a non-thinking default so answers fit their
small output budgets. Honor explicit template options, preserve supplied
reasoning/tool history, and make temperature-zero completions repeatable.

Run the configured service checker before benchmarking. It is shared with the
existing H100 task and checks sentinel echo, known answers, and greedy
repeatability. These are behavioral probes, **not numerical parity or proof of
checkpoint identity**. Correct hybrid state and routing remain requirements;
a passing behavioral probe does not authorize altered model execution.
No full-size run has validated the SGLang checkout on MI250 yet.

## TraceLab recorded workload and objectives

Replay the trusted recorded sample: 48 sessions and 303 rounds, published as
[`multi_session_large.csv`](https://github.com/uw-syfi/request-factory/blob/89dce4a64ae12e7084fbe464ddda882fa0a4c482/examples/multi_session_large.csv) in Request Factory revision
`89dce4a64ae12e7084fbe464ddda882fa0a4c482`, materialized from TraceLab rounds.
This is the recorded sample, not the larger statistically resampled Qwen trace
or the complete TraceLab public release. The adapter verifies its SHA256.

Preserve every row, session dependency, growing prefix, output length and tool
wait. Saturated replay ignores arrival times and permits eight active sessions;
no sessions are dropped on context overflow. Preserve real prefix-cache usage
accounting so the runner's prefix-cache preflight can pass. The token pool is
synthetic coding text because these sanitized traces describe lengths and
topology rather than private prompt content. Setup occurs before measurement;
throughput and percentiles come from the runner's native measured interval.
No separate warmup is supplied beyond the runner's preflight; keep that policy
fixed when comparing candidates. This small sample can be sensitive to cold
kernels; it is a separate frontier from the 15-minute AgentX task.

Optimize output tokens/s (maximize), TTFT p50/p90 and TPOT p50/p90 in milliseconds
(minimize), with 3% relative noise tolerance. A failed round, missing metric,
wrong token count or incomplete replay fails measurement. No dry-run result is
accepted. See [the TraceLab adapter](../../shared/tracelab/README.md).

## Starting point

The workspace contains a checkout of upstream **SGLang** (`sgl-project/sglang`
main) at `sglang/`, pinned via `[[workspace.sources]]`; provenance is in
`_vibesys_sources.json`. Do not write the serving system from scratch: start
from SGLang's OpenAI-compatible server, scheduler, radix prefix cache, and its
support for hybrid linear-attention MoE models, adapting them to gfx90a and
this workload. Its kernels, build steps and dependencies may assume another GPU
architecture; port, replace or disable them as needed without changing model
semantics. Vendored SGLang code counts as the candidate's own code, and you
may edit, prune, or rebuild any part of the checkout freely. It is ordinary
mutable workspace code.
