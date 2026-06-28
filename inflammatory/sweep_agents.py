import json, csv, glob, re, statistics as st

LAYERS, KV_HEADS, HEAD_DIM, DTYPE_BYTES = 28, 4, 128, 2
KV_PER_TOKEN_BYTES = 2 * LAYERS * KV_HEADS * HEAD_DIM * DTYPE_BYTES
MODEL_WEIGHTS_GB = 15.2
GPU_MEMORY_GB = 80
GPU_UTIL = 0.90


def percentile(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p / 100 * len(xs)))]


def main():
    files = sorted(glob.glob("llm_signals_n*.jsonl"),
                   key=lambda f: int(re.search(r"n(\d+)", f).group(1)))
    if not files:
        print("no llm_signals_n*.jsonl files found")
        return

    kv_budget_gb = GPU_MEMORY_GB * GPU_UTIL - MODEL_WEIGHTS_GB
    rows = []
    for f in files:
        sig = [json.loads(l) for l in open(f)]
        n_citizens = sig[0].get("n_citizens", sig[0]["n_agents"])
        n_steps = len({r["step"] for r in sig})
        ins = [r["in_tokens"] for r in sig]
        outs = [r["out_tokens"] for r in sig]
        calls_per_citizen = len(sig) / n_steps / n_citizens
        mean_seq = st.mean(ins) + st.mean(outs)
        kv_req_mb = mean_seq * KV_PER_TOKEN_BYTES / 1024 ** 2
        kv_lockstep_gb = kv_req_mb * len(sig) / n_steps / 1024
        max_conc = int(kv_budget_gb * 1024 ** 3 / (mean_seq * KV_PER_TOKEN_BYTES))
        rows.append({
            "citizens": n_citizens,
            "calls_per_citizen": round(calls_per_citizen, 2),
            "in_mean": round(st.mean(ins)),
            "out_mean": round(st.mean(outs)),
            "seq_mean": round(mean_seq),
            "kv_per_req_MiB": round(kv_req_mb, 1),
            "kv_per_lockstep_GiB": round(kv_lockstep_gb, 2),
            "max_agents_per_lockstep": max_conc,
        })

    cols = list(rows[0].keys())
    width = {c: max(len(c), max(len(str(r[c])) for r in rows)) for c in cols}
    line = "  ".join(c.rjust(width[c]) for c in cols)
    print(line)
    print("-" * len(line))
    for r in rows:
        print("  ".join(str(r[c]).rjust(width[c]) for c in cols))

    with open("kv_vs_agents.csv", "w", newline="") as f:
        cw = csv.DictWriter(f, fieldnames=cols)
        cw.writeheader()
        cw.writerows(rows)


if __name__ == "__main__":
    main()
