"""Plan 27 s2: GpuLookupDrafter (device drafts, no sync) vs PromptLookupDrafter (host).

Simulates decode steps of several requests sharing a batch: each step the request's
context grows by 1..W accepted tokens; the host drafter sees the committed ids plus the
pending ones, the device drafter sees its history row plus the step's accept row, with
the committed/pending split varied (result processed or not, appends repeated). Drafts
must agree exactly for T = W and T = 1."""
import os
import random
import sys

import numpy as np
import torch

from sglang.srt.speculative.decode_verify import GpuLookupDrafter, PromptLookupDrafter


class FakeKv:
    def __init__(self, slot):
        self.req_pool_idx = slot


class FakeReq:
    def __init__(self, rid, slot, prompt):
        self.rid = rid
        self.kv = FakeKv(slot)
        self.origin_input_ids_unpadded = list(prompt)
        self.output_ids = []


class Info:
    def __init__(self, acc_tok, acc_len):
        self.accept_tokens = acc_tok
        self.accept_lens = acc_len


def main():
    dev = os.environ.get("P27_DEV", "cuda")
    W = 3
    rng = random.Random(7)
    fails = 0
    checks = 0
    for trial in range(40):
        vocab = rng.choice([5, 20, 200, 5000])
        nreq = rng.choice([1, 2, 3, 8])
        gpu = GpuLookupDrafter(k_max=W - 1, num_slots=10, max_len=20000, device=dev, vocab_size=vocab)
        cpu = PromptLookupDrafter(W - 1, window=gpu.window, max_ngram=gpu.max_n, min_ngram=gpu.min_n)
        cpu1 = PromptLookupDrafter(0, window=gpu.window, max_ngram=gpu.max_n, min_ngram=gpu.min_n)
        slots = rng.sample(range(10), nreq)
        seqs, reqs, L = [], [], []
        for i in range(nreq):
            plen = rng.choice([1, 2, 5, 50, 700, 9000])
            period = rng.choice([0, 3, 17])
            s = [rng.randrange(vocab) for _ in range(plen + 4000)]
            if period:
                for j in range(period, len(s)):
                    if rng.random() < 0.85:
                        s[j] = s[j - period]
            seqs.append(s)
            reqs.append(FakeReq(f"r{trial}_{i}", slots[i], s[:plen]))
            L.append(plen + 1)  # first token from prefill
        last_acc = [1] * nreq
        for step in range(60):
            T = 1 if rng.random() < 0.25 else W
            acc_tok = torch.zeros((nreq, W), dtype=torch.int64)
            acc_len = torch.zeros((nreq,), dtype=torch.int32)
            pend = []
            for i, r in enumerate(reqs):
                a = last_acc[i]
                Li = L[i]
                acc_tok[i, :a] = torch.tensor(seqs[i][Li - a : Li])
                acc_len[i] = a
                # committed outputs: everything before Li, or before Li - a (pending)
                processed = rng.random() < 0.5
                c = Li if processed else Li - a
                plen = len(r.origin_input_ids_unpadded)
                r.output_ids = seqs[i][plen:c]
                pend.append(seqs[i][c:Li])
            out = torch.empty(nreq * T, dtype=torch.int64, device=dev)
            seq_lens = torch.tensor([x - 1 for x in L], dtype=torch.int64, device=dev)
            rpi = torch.tensor(slots, dtype=torch.int64, device=dev)
            gpu.draft(reqs, rpi, slots, seq_lens, Info(acc_tok.flatten().to(dev), acc_len.to(dev)), out, T)
            if dev != "cpu":
                torch.cuda.synchronize()
            got = out.view(nreq, T).cpu().numpy()
            for i, r in enumerate(reqs):
                ref = (cpu if T == W else cpu1).draft(r, np.asarray(pend[i], dtype=np.int64))
                checks += 1
                if not np.array_equal(ref[:T], got[i]):
                    fails += 1
                    if fails <= 10:
                        print(f"MISMATCH trial {trial} step {step} req {i} T {T} L {L[i]}: cpu {ref[:T].tolist()} gpu {got[i].tolist()}")
            if rng.random() < 0.1:
                continue  # the same accept rows again next step: appends are idempotent
            for i in range(nreq):
                a = rng.randint(1, T)
                L[i] += a
                last_acc[i] = a
    print(f"GPU drafts vs host: {checks - fails}/{checks} equal")
    print("ALL OK" if fails == 0 else "FAILED")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
