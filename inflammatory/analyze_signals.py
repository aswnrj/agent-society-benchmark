import json, csv, os, statistics as st

# Qwen2.5-7B-Instruct (GQA, bf16)
LAYERS, KV_HEADS, HEAD_DIM, DTYPE_BYTES = 28, 4, 128, 2
KV_PER_TOKEN_BYTES = 2 * LAYERS * KV_HEADS * HEAD_DIM * DTYPE_BYTES
MODEL_WEIGHTS_GB = 15.2
GPU_MEMORY_GB = 80
GPU_UTIL = 0.90
NUM_GPUS = int(os.environ.get("NUM_GPUS", 1))  # tensor-parallel size

SIGNALS_FILE = "llm_signals.jsonl"
PROMPTS_FILE = "prompts_head.jsonl"


def percentile(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p / 100 * len(xs)))]


def longest_common_prefix(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def main():
    sig = [json.loads(l) for l in open(SIGNALS_FILE)]
    prm = [json.loads(l) for l in open(PROMPTS_FILE)]

    steps = sorted({r["step"] for r in sig})
    n_steps = len(steps)
    n_agents = sig[0]["n_agents"]
    n_citizens = sig[0].get("n_citizens", n_agents)

    ins = [r["in_tokens"] for r in sig]
    outs = [r["out_tokens"] for r in sig]
    lat = [r["latency_s"] for r in sig]
    calls_per_step = len(sig) / n_steps
    calls_per_agent = calls_per_step / n_agents
    calls_per_citizen = calls_per_step / n_citizens

    mean_seq = st.mean(ins) + st.mean(outs)
    p95_seq = percentile(ins, 95) + percentile(outs, 95)
    kv_req_mean = mean_seq * KV_PER_TOKEN_BYTES
    kv_req_p95 = p95_seq * KV_PER_TOKEN_BYTES
    kv_budget_gb = NUM_GPUS * GPU_MEMORY_GB * GPU_UTIL - MODEL_WEIGHTS_GB
    max_conc_mean = int(kv_budget_gb * 1024 ** 3 / kv_req_mean)
    max_conc_p95 = int(kv_budget_gb * 1024 ** 3 / kv_req_p95)

    heads = [r["head"] for r in prm]
    families = {}
    for h in heads:
        families[h[:60]] = families.get(h[:60], 0) + 1
    n_families = len(families)
    top1 = max(families.values()) / len(heads)
    top5 = sum(sorted(families.values(), reverse=True)[:5]) / len(heads)
    adj = []
    for s in steps:
        hs = sorted(p["head"] for p in prm if p["step"] == s)
        adj += [longest_common_prefix(hs[i], hs[i + 1]) for i in range(len(hs) - 1)]
    dup_rate = 1 - len(set(heads)) / len(heads)

    out = []
    def w(s=""):
        out.append(s)

    w("Many-agent LLM execution characterization")
    w(f"serving: Qwen2.5-7B-Instruct, tensor-parallel x{NUM_GPUS} (A100 {GPU_MEMORY_GB}GB)")
    w(f"agents per lock-step: {n_agents} ({n_citizens} citizens + "
      f"{n_agents - n_citizens} institution agents)")
    w(f"steps: {n_steps} | total LLM calls captured: {len(sig)}")
    w("")
    w("Generations per agent per lock-step")
    w(f"  LLM calls per lock-step  : {calls_per_step:.1f}")
    w(f"  LLM calls per agent      : {calls_per_agent:.2f}")
    w(f"  LLM calls per citizen    : {calls_per_citizen:.2f}")
    w("")
    w("Token distribution (per LLM call)")
    w(f"  {'':8}{'mean':>7}{'p50':>7}{'p90':>7}{'p95':>7}{'p99':>7}{'max':>7}")
    w(f"  {'input':8}{st.mean(ins):>7.0f}{percentile(ins, 50):>7}{percentile(ins, 90):>7}"
      f"{percentile(ins, 95):>7}{percentile(ins, 99):>7}{max(ins):>7}")
    w(f"  {'output':8}{st.mean(outs):>7.0f}{percentile(outs, 50):>7}{percentile(outs, 90):>7}"
      f"{percentile(outs, 95):>7}{percentile(outs, 99):>7}{max(outs):>7}")
    w(f"  latency seconds: mean {st.mean(lat):.2f}  p95 {percentile(lat, 95):.2f}")
    w("")
    w("KV-cache memory")
    w(f"  KV per token         : {KV_PER_TOKEN_BYTES / 1024:.0f} KiB")
    w(f"  mean seq (in+out)    : {mean_seq:.0f} tokens -> {kv_req_mean / 1024 ** 2:.1f} MiB/request")
    w(f"  p95 seq  (in+out)    : {p95_seq:.0f} tokens -> {kv_req_p95 / 1024 ** 2:.1f} MiB/request")
    w(f"  KV budget (TPx{NUM_GPUS}) : {kv_budget_gb:.1f} GB after model weights")
    w(f"  max concurrent requests @ mean seq : {max_conc_mean:,}")
    w(f"  max concurrent requests @ p95 seq  : {max_conc_p95:,}")
    w("")
    w("Prompt templates / prefix reuse")
    w(f"  distinct template families (60-char) : {n_families}")
    w(f"  top-1 family share                   : {100 * top1:.0f}%")
    w(f"  top-5 family share                   : {100 * top5:.0f}%")
    w(f"  adjacent common prefix (chars)       : mean {st.mean(adj):.0f}  p95 {percentile(adj, 95)}")
    w(f"  exact-duplicate prompt rate          : {100 * dup_rate:.1f}%")

    report = "\n".join(out)
    print(report)
    with open("characterization_report.txt", "w", encoding="utf-8") as f:
        f.write(report + "\n")
    with open("token_distribution.csv", "w", newline="") as f:
        cw = csv.writer(f)
        cw.writerow(["metric", "mean", "p50", "p90", "p95", "p99", "max"])
        cw.writerow(["input_tokens", round(st.mean(ins)), percentile(ins, 50),
                     percentile(ins, 90), percentile(ins, 95), percentile(ins, 99), max(ins)])
        cw.writerow(["output_tokens", round(st.mean(outs)), percentile(outs, 50),
                     percentile(outs, 90), percentile(outs, 95), percentile(outs, 99), max(outs)])


if __name__ == "__main__":
    main()
