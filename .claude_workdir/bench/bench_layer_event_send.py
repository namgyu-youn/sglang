"""TTFT A/B driver for the layer-event KV send prototype.

Sends the same prompt set through the PD router and records server-reported
e2e_latency (max_new_tokens=1 -> ~= TTFT = prefill + KV transfer + 1 decode).
"""

import json
import random
import statistics
import sys
import urllib.request

ROUTER = "http://127.0.0.1:30002/generate"
N_WARMUP = 3
N_MEASURE = 10
TARGET_TOKENS = 6000

WORDS = (
    "system model layer tensor cache prefill decode transfer event stream "
    "kernel batch token page index buffer memory schedule pipeline overlap"
).split()


def make_prompt(seed: int) -> str:
    rng = random.Random(seed)
    # ~1.3 tokens/word for llama tokenizers on plain words + numbers
    n_words = int(TARGET_TOKENS / 1.3)
    words = [f"{rng.choice(WORDS)}{rng.randint(0, 999)}" for _ in range(n_words)]
    return "Document: " + " ".join(words) + "\nSummary:"


def send(prompt: str) -> dict:
    body = json.dumps(
        {
            "text": prompt,
            "sampling_params": {"temperature": 0, "max_new_tokens": 1},
        }
    ).encode()
    req = urllib.request.Request(
        ROUTER, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.load(resp)


def main() -> None:
    label = sys.argv[1]
    results = []
    for i in range(N_WARMUP + N_MEASURE):
        d = send(make_prompt(seed=1000 + i))
        meta = d["meta_info"]
        row = {
            "i": i,
            "warmup": i < N_WARMUP,
            "prompt_tokens": meta["prompt_tokens"],
            "e2e_latency": meta["e2e_latency"],
            "text": d["text"],
        }
        results.append(row)
        kind = "warmup " if row["warmup"] else "measure"
        print(
            f"[{label}] {kind} #{i}: prompt_tokens={row['prompt_tokens']} "
            f"e2e={row['e2e_latency']*1000:.1f}ms text={row['text']!r}",
            flush=True,
        )

    lat = [r["e2e_latency"] for r in results if not r["warmup"]]
    print(
        f"[{label}] SUMMARY n={len(lat)} mean={statistics.mean(lat)*1000:.1f}ms "
        f"median={statistics.median(lat)*1000:.1f}ms "
        f"stdev={statistics.stdev(lat)*1000:.1f}ms "
        f"min={min(lat)*1000:.1f}ms max={max(lat)*1000:.1f}ms",
        flush=True,
    )
    out = f"/tmp/claude-0/-workspace-sglang/58734390-9c93-47a7-bb6a-b87820e9954c/scratchpad/bench_{label}.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"[{label}] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
