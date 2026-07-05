"""Unit tests for srt/disaggregation/layer_event_send (PR #23515 review prototype)."""

import unittest
from unittest.mock import MagicMock, call, patch

from sglang.srt.disaggregation.layer_event_send import LayerEventSendCoordinator
from sglang.srt.mem_cache.memory_pool import KVCache
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _StubPool(KVCache):
    """Minimal concrete KVCache exposing only the notifier plumbing."""

    def __init__(self, start_layer: int):
        self.layer_send_notifier = None
        self.start_layer = start_layer

    def get_key_buffer(self, layer_id):
        raise NotImplementedError()

    def get_value_buffer(self, layer_id):
        raise NotImplementedError()

    def get_kv_buffer(self, layer_id):
        raise NotImplementedError()

    def set_kv_buffer(self, layer, loc, cache_k, cache_v):
        raise NotImplementedError()


class TestLayerSendNotifier(CustomTestCase):
    def test_unarmed_notify_is_noop(self):
        pool = _StubPool(start_layer=0)
        pool._notify_layer_send(3)  # must not raise

    def test_notify_applies_start_layer_offset(self):
        pool = _StubPool(start_layer=2)
        notifier = MagicMock()
        pool.register_layer_send_notifier(notifier)
        pool._notify_layer_send(5)
        notifier.assert_called_once_with(3)

    def test_disarm_stops_notification(self):
        pool = _StubPool(start_layer=0)
        notifier = MagicMock()
        pool.register_layer_send_notifier(notifier)
        pool.register_layer_send_notifier(None)
        pool._notify_layer_send(0)
        notifier.assert_not_called()


@patch("sglang.srt.disaggregation.layer_event_send.LayerDoneCounter")
class TestLayerEventSendCoordinator(CustomTestCase):
    def _armed_coordinator(self, num_layers, plan):
        coordinator = LayerEventSendCoordinator(num_layers=num_layers)
        coordinator.begin_batch(plan)
        return coordinator

    def test_fans_out_layer_to_every_sender(self, _mock_counter):
        sender_a, sender_b = MagicMock(), MagicMock()
        pages_a, pages_b = object(), object()
        coordinator = self._armed_coordinator(
            4, [(sender_a, pages_a), (sender_b, pages_b)]
        )

        coordinator.on_layer_written(0)

        event = coordinator._event.load_events[0]
        sender_a.send_layer.assert_called_once_with(
            pages_a, layer_id=0, cuda_event=event, is_last=False
        )
        sender_b.send_layer.assert_called_once_with(
            pages_b, layer_id=0, cuda_event=event, is_last=False
        )
        coordinator._event.complete.assert_called_once_with(0)

    def test_is_last_only_on_final_layer(self, _mock_counter):
        sender = MagicMock()
        coordinator = self._armed_coordinator(3, [(sender, object())])

        for layer_idx in range(3):
            coordinator.on_layer_written(layer_idx)

        is_last_flags = [
            kwargs["is_last"] for _, kwargs in sender.send_layer.call_args_list
        ]
        self.assertEqual(is_last_flags, [False, False, True])
        layer_ids = [
            kwargs["layer_id"] for _, kwargs in sender.send_layer.call_args_list
        ]
        self.assertEqual(layer_ids, [0, 1, 2])

    def test_end_batch_clears_plan(self, _mock_counter):
        sender = MagicMock()
        coordinator = self._armed_coordinator(2, [(sender, object())])
        coordinator.end_batch()

        self.assertEqual(coordinator._plan, [])
        self.assertIsNone(coordinator._event)

    def test_rearming_uses_fresh_producer_slot(self, _mock_counter):
        coordinator = LayerEventSendCoordinator(num_layers=2)
        coordinator.begin_batch([(MagicMock(), object())])
        coordinator.end_batch()
        coordinator.begin_batch([(MagicMock(), object())])

        self.assertEqual(coordinator.counter.update_producer.call_count, 2)


if __name__ == "__main__":
    unittest.main()
