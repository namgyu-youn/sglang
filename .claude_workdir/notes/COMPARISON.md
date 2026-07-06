# Event-reuse per-layer KV send — comparison prototype for PR #23515

Branch: `pd-layer-event-send-prototype` (local, based on `67361ff91b`, 2026-07-04).
Purpose: help resolve the open review questions on #23515 by prototyping the
alternative @cctry suggested (reuse the HiCache `LayerDoneCounter` per-layer event
machinery instead of the `run_batch_pipelined` scheduler path) and by providing a
benchmark methodology for the disputed TTFT numbers. This is **not** a competing PR;
the transport layer is #23515's own code, vendored unchanged with attribution.

## What was built

Two commits:

1. `[Prototype] Vendor transport layer from PR #23515 unchanged (attribution)` —
   +363/−2 across `disaggregation/{base,common,fake,mooncake}/conn.py` and
   `common/utils.py`. This is the scheduler-mode-independent subset of #23515:
   `send_kvcache_layer` / `_send_kvcache_layer_head_slice`, `KVSender.send_layer` /
   `send_final_metadata`, `TransferKVChunk.layer_id/cuda_event`, and the
   `transfer_worker` per-layer branch. All of it is @michael7193's work. It applied
   cleanly onto current main and references nothing from `run_batch_pipelined` —
   which is itself useful review evidence: **#23515's transport and its scheduler
   path are separable.**

2. `[Prototype] Per-layer PD KV send via LayerDoneCounter events` — +208/−9 across
   five files (of which ~70 lines are the new `layer_event_send.py`, roughly half
   docstrings/comments). The producer design:

   - The normal `run_batch` prefill forward is the producer. `KVCache.set_kv_buffer`
     (MHA; MLA `set_kv_buffer`/`set_mla_kv_buffer`) fires an optional per-layer
     notifier while the forward is being *submitted* on the scheduler thread.
   - The notifier records that layer's CUDA event on the compute stream via
     `LayerDoneCounter` / `LayerLoadingEvent` — the exact primitive HiCache's
     host→device loads use, with the producer/consumer roles flipped — and enqueues
     that layer's send through #23515's `send_layer`.
   - The consumer is #23515's own transfer-worker branch: `cuda_event.synchronize()`
     then `send_kvcache_layer`. Transfer of layer *i* overlaps with GPU compute of
     layers *i+1..N−1*.
   - Arm/disarm happens around `run_batch` in `event_loop_normal_disagg_prefill`;
     requests finishing their prefill in this batch send per-layer, middle chunks
     keep the existing `send_kv_chunk` path, and `process_batch_result` sends only
     the final metadata for per-layer requests (mirroring #23515's flow).

   Total new code (excluding the vendored transport): **~215 lines including
   comments/docstrings and the env-var registration; the logic itself is within
   @cctry's <150-line estimate.**

## Against the three disputed points

### Point 3 — "reuse the existing per-layer event machinery, <150 lines"

Feasible, confirmed at the prototype level. Key differences vs #23515's scheduler
path, on identical transport:

| | #23515 (`run_batch_pipelined`) | this prototype (event reuse) |
|---|---|---|
| Per-model changes | needs `forward_split_prefill` (exists for many models already; #23515 adds Qwen3.5, Falcon-H1) | none — any model writing through the standard MHA/MLA pools participates |
| Scheduler path | new `run_batch_pipelined` + split init/layer/sample tp_worker plumbing | normal `run_batch`, arm/disarm wrapper only |
| Bypassed machinery | split path skips `prepare_mlp_sync_batch`, EPLB recorder, multimodal, input_embeds → 10 launch-time fallback guards | run_batch machinery intact; guards reduce to transport-side ones (PP, staging, compressed MLA, hybrid state, spec decode) |
| Overlap granularity | per layer-*group* (adaptive group size, 6 env knobs) | per layer (no tuning knobs) |
| Event per | group (one `torch.cuda.Event` per group) | layer (pre-allocated `LayerDoneCounter` slots, 3-deep rotation) |
| Chunked prefill | middle chunks: normal path; final chunk: pipelined | same |

Caveats we carry rather than hide:

- **Hybrid linear attention (Mamba conv/ssm, SWA, DSA state) is guarded off, not
  supported.** That state does not flow through `set_kv_buffer`; "send once at
  end-of-forward" is the likely answer but is not implemented or tested here.
  (#23515 handles these via `pipelined_state_indices`; the event-reuse design would
  need an equivalent.)
- Only the normal (non-overlap) disagg prefill loop is wired. #23515 has the same
  effective restriction (it force-disables the overlap scheduler); a real PR would
  need a deliberate answer for overlap mode rather than a silent fallback.
- Pool subclasses that override `set_kv_buffer` (FP4, NoOp) are excluded by an
  exact-method check so a missed notifier can't silently strand the decode side.
- If `run_batch` raises mid-forward, the notifier stays armed for that batch; the
  scheduler is about to crash anyway, but a real PR should add try/finally.
- `LayerDoneCounter.update_producer()` asserts the slot's finish event completed;
  safe here because the non-overlap loop synchronizes on sampling each batch, but
  this is an invariant to re-check if extended to overlap mode.

### Point 2 — the TTFT claim (−48%…−68% vs <20ms ceiling)

**No numbers are reported here because this environment has no GPU and no RDMA
fabric.** What is provided instead (`.claude_workdir/bench/`):

- `bench_layer_event_send.sh` — sweeps mode × chunked-prefill-size × input-length
  on a PD pair, low request rate, radix cache off, and records TTFT via
  `bench_serving` plus both server logs.
- `analyze_layer_event_send.py` — reduces results to one table with the separation
  logic the dispute needs: decode-side poll-loop latency is a constant additive
  offset in *both* modes, so the mode-delta cancels it; the delta is then compared
  against the *theoretical* wire time of the terminal chunk
  (`kv_bytes/link_bw`). A measured delta that exceeds the terminal chunk's wire
  time is scheduler/poll artifact, not transfer overlap — which is exactly the
  distinction @cctry asked for, and matches the author's own admission that part
  of the original measurement was a poll-loop artifact.

Prediction to verify, not assert: with correct chunked-prefill config, both
pipelined designs can only hide the transfer of everything *before* the terminal
chunk's last layer (group), so the achievable TTFT gain should approach
`wire(full) − wire(last chunk)` and the residual exposure stays ≈ one layer
(group) drain — i.e. the data will either confirm the <20ms ceiling or show where
the extra claimed gain actually comes from.

### Point 1 — `page_first` layout / HiCache compatibility

Not addressed by this prototype (it is orthogonal to the producer-side question).
One relevant data point: the prototype creates its **own** `LayerDoneCounter`
instance; it does not touch `HiCacheController.layer_done_counter`, so the event
*machinery* reuse does not collide with HiCache's use of the same class. Actual
interaction with HiCache write-through during prefill (device→host backup racing
per-layer RDMA reads of the same pages) is untested and worth flagging on the PR
regardless of which producer design lands.

## Verification status

- `python -m py_compile` passes on all touched files; the full `sglang` import
  chain was also verified on a CPU-only host (system torch 2.11, pinned
  `transformers==5.12.1`).
- CPU unit test added and passing: `test/registered/unit/disaggregation/test_layer_event_send.py`
  (coordinator fan-out, `is_last` placement, notifier offset/disarm; registered
  `base-a-test-cpu`). 7/7 pass on a CPU-only host (2026-07-05); no GPU required
  (`LayerDoneCounter` is mocked).
- **E2E (GPU host, 1P1D, `mooncake_tcp`, Llama-3.2-1B, temperature 0): found a
  correctness bug, now root-caused, fix applied but not yet re-verified on
  hardware.** Cache-miss requests (no radix-cache prefix hit) match the
  flag-off baseline bit-exact. Any request with a cache-hit prefix diverges
  from baseline under the flag.

  Root cause: `maybe_begin_layer_event_send()` (`prefill.py`) is called
  *before* `run_batch()` in the event loop, but `run_batch()` itself calls
  `maybe_send_cached_prefix_chunk()` as its first step (`scheduler.py:3204-3206`,
  gated by `SGLANG_DISAGG_PREFILL_EARLY_SEND_CACHED_PREFIX`, default `True`).
  For a cache-hit request, that call sends the device-resident cached prefix
  early via the normal `.send()` chunk path and, as a side effect, advances
  both `req.start_send_idx` and the sender's internal `curr_idx` page counter.
  Because our arm function read `req.start_send_idx` *before* that advance
  happened, its per-layer send plan wrongly included the already-separately-
  sent cached-prefix pages on top of the real new tokens; the subsequent
  `send_layer()` calls then computed a destination offset from a `curr_idx`
  that had *already* moved past those pages, producing a shifted, duplicated
  write on the decode side. Cache-miss requests have `decode_prefix_len=0` and
  no cached prefix, so `maybe_send_cached_prefix_chunk` is a no-op for them —
  which is why they were unaffected.

  Fix applied (`prefill.py`, `maybe_begin_layer_event_send`): call
  `self.maybe_send_cached_prefix_chunk(req)` inside our own arm loop, before
  reading `req.start_send_idx`, for each candidate request. This is safe to
  call twice per request — the function already guards on
  `cached_end <= req.start_send_idx`, so `run_batch`'s later call becomes a
  no-op once ours has run.

  **Status: root-caused and fixed at the code-review level (`py_compile`
  clean); not yet re-verified against the GPU e2e repro** (this environment
  has no 2-node/multi-GPU setup in this session). The existing CPU unit test
  (`test_layer_event_send.py`) does not cover this code path — it only
  exercises `LayerEventSendCoordinator`/`KVCache` notifier plumbing, not
  `prefill.py`'s arm function — so it cannot validate this fix either way.
  Re-running the cache-hit repro from the "1P1D e2e reproduction" section
  above (flag=1 vs flag=0 diff) on GPU hardware is the next step before this
  can be called resolved.

## Code-anchor drift noted during the work

`LayerDoneCounter` registration moved: it is wired from
`HiCacheController.__init__` (`cache_controller.py:267`), not `tp_worker.py` as
the task notes said — i.e. today the counter only exists when HiCache is enabled,
which is itself a mild argument for the separate-instance approach taken here.
