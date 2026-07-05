"""Prototype: per-layer PD KV send driven by HiCache's LayerDoneCounter events.

Comparison prototype for the PR #23515 review discussion (@cctry's suggestion):
instead of #23515's run_batch_pipelined scheduler path, the normal prefill
forward is the producer. KVCache.set_kv_buffer notifies this coordinator once
per layer while the forward is being submitted on the scheduler thread; we
record that layer's CUDA event on the compute stream (via LayerLoadingEvent,
the same primitive HiCache's host->device loads use) and enqueue the layer's
RDMA send. The consumer is the Mooncake transfer worker thread (transport
vendored unchanged from #23515), which synchronizes on the event before
issuing the transfer, so transfer of layer i overlaps with GPU compute of
layers i+1..N-1.

Requires no per-model changes and no separate scheduler batch path: any model
whose KV writes go through MHATokenToKVPool.set_kv_buffer or
MLATokenToKVPool.set_kv_buffer / set_mla_kv_buffer participates.
"""

from __future__ import annotations

from typing import List, Tuple

from sglang.srt.managers.cache_controller import LayerDoneCounter

# (kv_sender, page_indices) pairs armed for the current forward. All armed
# requests are in their final prefill chunk, so the last layer's send carries
# is_last=True to trigger the sender's transfer-index bookkeeping.
LayerSendPlan = List[Tuple[object, "npt.NDArray"]]


class LayerEventSendCoordinator:
    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        # Reuse HiCache's per-layer event machinery: 3 rotating slots of
        # per-layer device events. Slot rotation is safe here because the
        # normal (non-overlap) prefill loop synchronizes on sampling before
        # the next batch launches, so a slot's finish event is always done
        # before update_producer() reuses it.
        self.counter = LayerDoneCounter(num_layers)
        self._event = None
        self._plan: LayerSendPlan = []

    def begin_batch(self, plan: LayerSendPlan) -> None:
        """Arm the coordinator for one disagg prefill forward."""
        producer_id = self.counter.update_producer()
        self._event = self.counter.events[producer_id]
        self._plan = plan

    def on_layer_written(self, layer_idx: int) -> None:
        """KVCache notifier callback; runs on the scheduler thread during
        forward submission, right after layer ``layer_idx``'s KV write was
        submitted to the compute stream."""
        self._event.complete(layer_idx)
        cuda_event = self._event.load_events[layer_idx]
        is_final_layer = layer_idx == self.num_layers - 1
        for sender, page_indices in self._plan:
            sender.send_layer(
                page_indices,
                layer_id=layer_idx,
                cuda_event=cuda_event,
                is_last=is_final_layer,
            )

    def end_batch(self) -> None:
        self._event = None
        self._plan = []
