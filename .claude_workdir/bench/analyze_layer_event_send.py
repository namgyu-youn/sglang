#!/usr/bin/env python3
"""Summarize bench_layer_event_send.sh results into one comparison table.

Separates the three time components the #23515 dispute is about:
  1. e2e TTFT (mean / p50 / p99)      <- bench_serving jsonl
  2. exposed-transfer estimate        <- TTFT delta between modes at fixed
                                         (cps, isl); this is what per-layer
                                         overlap can actually save
  3. theoretical wire time            <- kv_bytes / --link-gbps, the number
                                         cctry's "<20ms ceiling" argument is
                                         based on; on chunked prefill only the
                                         LAST chunk's share should be exposed

Anything decode-side poll-loop related shows up as a constant additive offset
in BOTH modes, so mode deltas cancel it — that is the separation trick. If the
measured delta exceeds the last chunk's theoretical wire time, the difference
is scheduler/poll artifact, not transfer overlap (this was the PR author's own
admission for part of the original numbers).
"""

import argparse
import json
import re
import statistics
from pathlib import Path

TAG_RE = re.compile(r"bench_(?P<mode>[\w-]+)_cps(?P<cps>\d+)_isl(?P<isl>\d+)\.jsonl")


def load_ttfts(path: Path):
    ttfts = []
    with path.open() as f:
        for line in f:
            rec = json.loads(line)
            # bench_serving writes one summary record (has "ttfts" or
            # per-request records with "ttft"); handle both layouts.
            if "ttfts" in rec:
                ttfts.extend(v for v in rec["ttfts"] if v is not None)
            elif "ttft" in rec and rec["ttft"] is not None:
                ttfts.append(rec["ttft"])
    return ttfts


def theoretical_wire_ms(isl, cps, args):
    kv_bytes_per_token = (
        2 * args.num_layers * args.kv_heads * args.head_dim * args.dtype_bytes
    )
    full = isl * kv_bytes_per_token / (args.link_gbps * 1e9 / 8) * 1e3
    last_chunk_tokens = isl % cps or min(cps, isl)
    last_chunk = last_chunk_tokens * kv_bytes_per_token / (args.link_gbps * 1e9 / 8) * 1e3
    return full, last_chunk


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out_dir", type=Path)
    p.add_argument("--link-gbps", type=float, default=100.0, help="RDMA link speed")
    # Llama-3.1-8B defaults; override per model.
    p.add_argument("--num-layers", type=int, default=32)
    p.add_argument("--kv-heads", type=int, default=8)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--dtype-bytes", type=int, default=2)
    args = p.parse_args()

    rows = {}
    for f in sorted(args.out_dir.glob("bench_*.jsonl")):
        m = TAG_RE.match(f.name)
        if not m:
            continue
        ttfts = load_ttfts(f)
        if not ttfts:
            print(f"warn: no TTFT samples in {f.name}")
            continue
        key = (int(m["cps"]), int(m["isl"]))
        rows.setdefault(key, {})[m["mode"]] = ttfts

    hdr = (
        f"{'cps':>6} {'isl':>7} {'mode':>10} {'n':>4} "
        f"{'ttft_mean':>10} {'ttft_p50':>9} {'ttft_p99':>9} "
        f"{'d_vs_base':>10} {'wire_full':>10} {'wire_last':>10}"
    )
    print(hdr)
    print("-" * len(hdr))
    for (cps, isl), modes in sorted(rows.items()):
        base_mean = (
            statistics.mean(modes["baseline"]) if "baseline" in modes else None
        )
        wire_full, wire_last = theoretical_wire_ms(isl, cps, args)
        for mode, ttfts in sorted(modes.items()):
            mean = statistics.mean(ttfts)
            qs = statistics.quantiles(ttfts, n=100) if len(ttfts) >= 2 else [ttfts[0]] * 99
            delta = "" if base_mean is None or mode == "baseline" else f"{(mean - base_mean) * 1e3:+9.1f}ms"
            print(
                f"{cps:>6} {isl:>7} {mode:>10} {len(ttfts):>4} "
                f"{mean * 1e3:>8.1f}ms {qs[49] * 1e3:>7.1f}ms {qs[98] * 1e3:>7.1f}ms "
                f"{delta:>10} {wire_full:>8.1f}ms {wire_last:>8.1f}ms"
            )
    print(
        "\nReading guide: a real overlap win shows d_vs_base ≈ -(wire_full - "
        "wire_last); a delta much larger than that is poll/scheduler artifact, "
        "not wire time. Single-node loopback runs overstate link speed — pass "
        "the loopback bandwidth via --link-gbps or treat wire columns as N/A."
    )


if __name__ == "__main__":
    main()
