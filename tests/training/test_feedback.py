"""Tests for FeedbackCollector and InMemoryFeedbackBackend."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from synapsekit.training.feedback import FeedbackCollector, InMemoryFeedbackBackend
from synapsekit.training.types import FeedbackSample

# ── Helpers ────────────────────────────────────────────────────────────────────


def _collector(queue_maxsize: int = 100) -> FeedbackCollector:
    return FeedbackCollector(backend=InMemoryFeedbackBackend(), queue_maxsize=queue_maxsize)


# ── FeedbackSample construction ────────────────────────────────────────────────


class TestFeedbackSampleConstruction:
    def test_record_returns_feedback_sample(self) -> None:
        c = _collector()
        sample = c.record("q", "r", "positive")
        assert isinstance(sample, FeedbackSample)

    def test_record_assigns_unique_id(self) -> None:
        c = _collector()
        a = c.record("q", "r", "positive")
        b = c.record("q", "r", "positive")
        assert a.id != b.id

    def test_record_preserves_query_and_response(self) -> None:
        c = _collector()
        s = c.record("What is 2+2?", "4", "positive")
        assert s.query == "What is 2+2?"
        assert s.response == "4"

    def test_record_positive_feedback(self) -> None:
        c = _collector()
        s = c.record("q", "r", "positive")
        assert s.feedback == "positive"

    def test_record_negative_feedback(self) -> None:
        c = _collector()
        s = c.record("q", "r", "negative")
        assert s.feedback == "negative"

    def test_record_with_correction(self) -> None:
        c = _collector()
        s = c.record("q", "r", "negative", corrected_response="better r")
        assert s.corrected_response == "better r"

    def test_record_with_context(self) -> None:
        c = _collector()
        ctx = ["chunk 1", "chunk 2"]
        s = c.record("q", "r", "positive", context=ctx)
        assert s.context == ctx

    def test_record_with_metadata(self) -> None:
        c = _collector()
        s = c.record("q", "r", "positive", metadata={"session": "abc"})
        assert s.metadata == {"session": "abc"}

    def test_record_with_latency_ms(self) -> None:
        c = _collector()
        s = c.record("q", "r", "positive", latency_ms=123.4)
        assert s.latency_ms == 123.4

    def test_record_with_cost_usd(self) -> None:
        c = _collector()
        s = c.record("q", "r", "positive", cost_usd=0.0015)
        assert s.cost_usd == 0.0015

    def test_record_latency_cost_default_none(self) -> None:
        c = _collector()
        s = c.record("q", "r", "positive")
        assert s.latency_ms is None
        assert s.cost_usd is None

    def test_record_timestamp_is_utc(self) -> None:
        c = _collector()
        s = c.record("q", "r", "positive")
        assert s.timestamp.tzinfo is not None


# ── Non-blocking behaviour ─────────────────────────────────────────────────────


class TestNonBlocking:
    def test_record_is_synchronous(self) -> None:
        """record() must not be a coroutine."""
        c = _collector()
        result = c.record("q", "r", "positive")
        assert not asyncio.iscoroutine(result)

    @pytest.mark.asyncio
    async def test_record_does_not_block_event_loop(self) -> None:
        """record() completes within a single event-loop tick."""
        c = _collector()
        c.start()
        try:
            before = asyncio.get_event_loop().time()
            for _ in range(50):
                c.record("q", "r", "positive")
            after = asyncio.get_event_loop().time()
            assert after - before < 0.1
        finally:
            await c.stop()


# ── Persistence ────────────────────────────────────────────────────────────────


class TestPersistence:
    @pytest.mark.asyncio
    async def test_drain_persists_samples(self) -> None:
        """stop() must flush all queued samples to the backend."""
        c = _collector()
        c.start()
        for i in range(5):
            c.record(f"q{i}", f"r{i}", "positive")
        await c.stop()
        assert await c.stored_count() == 5

    @pytest.mark.asyncio
    async def test_get_samples_returns_persisted(self) -> None:
        c = _collector()
        c.start()
        c.record("q1", "r1", "positive")
        c.record("q2", "r2", "negative")
        await c.stop()
        samples = await c.get_samples()
        assert len(samples) == 2

    @pytest.mark.asyncio
    async def test_get_samples_since_filters(self) -> None:
        backend = InMemoryFeedbackBackend()
        c = FeedbackCollector(backend=backend)
        c.start()

        early = FeedbackSample(
            query="old",
            response="r",
            feedback="positive",
            timestamp=datetime(2020, 1, 1, tzinfo=timezone.utc),
        )
        await backend.save(early)

        c.record("new", "r", "positive")
        await c.stop()

        cutoff = datetime(2024, 1, 1, tzinfo=timezone.utc)
        recent = await c.get_samples(since=cutoff)
        assert all(s.timestamp >= cutoff for s in recent)

    @pytest.mark.asyncio
    async def test_pending_count_reflects_queue(self) -> None:
        """pending_count() returns queue size before draining."""
        c = _collector()
        # Do NOT call start() — items accumulate in queue
        c.record("q", "r", "positive")
        c.record("q", "r", "negative")
        assert await c.pending_count() == 2

    @pytest.mark.asyncio
    async def test_stored_count_zero_before_drain(self) -> None:
        c = _collector()
        c.record("q", "r", "positive")
        # No start() → backend untouched
        assert await c.stored_count() == 0


# ── flush() ───────────────────────────────────────────────────────────────────


class TestFlush:
    @pytest.mark.asyncio
    async def test_flush_waits_for_drain(self) -> None:
        """flush() must not return until all queued items are persisted."""
        c = _collector()
        c.start()
        for i in range(10):
            c.record(f"q{i}", f"r{i}", "positive")
        await c.flush()
        assert await c.stored_count() == 10
        await c.stop()

    @pytest.mark.asyncio
    async def test_flush_without_start_returns_immediately(self) -> None:
        """flush() with no running drainer must return immediately."""
        c = _collector()
        c.record("q", "r", "positive")
        # No start() — should not hang
        await c.flush()

    @pytest.mark.asyncio
    async def test_collector_continues_after_flush(self) -> None:
        """After flush(), the collector can still accept and drain new records."""
        c = _collector()
        c.start()
        c.record("q1", "r1", "positive")
        await c.flush()
        c.record("q2", "r2", "negative")
        await c.stop()
        assert await c.stored_count() == 2


# ── Queue-full safety ─────────────────────────────────────────────────────────


class TestQueueFull:
    @pytest.mark.asyncio
    async def test_queue_full_drops_without_raising(self) -> None:
        """When the queue is full, record() must not raise — it drops silently."""
        c = FeedbackCollector(backend=InMemoryFeedbackBackend(), queue_maxsize=2)
        c.record("q", "r", "positive")
        c.record("q", "r", "positive")
        # Third record should be silently dropped
        dropped = c.record("q", "r", "positive")
        assert isinstance(dropped, FeedbackSample)


# ── Error handling in drainer ─────────────────────────────────────────────────


class TestDrainerErrorHandling:
    @pytest.mark.asyncio
    async def test_backend_save_exception_is_logged_not_raised(self) -> None:
        """A failing backend.save() must be logged and not crash the drainer."""
        backend = InMemoryFeedbackBackend()
        c = FeedbackCollector(backend=backend)

        call_count = 0
        # Capture original bound method BEFORE replacing it
        _original_save = backend.save

        async def _failing_save(sample: FeedbackSample) -> None:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("disk full")
            await _original_save(sample)

        c._backend.save = _failing_save  # type: ignore[method-assign]
        c.start()
        c.record("q1", "r1", "positive")  # will fail
        c.record("q2", "r2", "negative")  # should succeed
        await c.stop()
        # Both calls were attempted; drainer survived the first failure.
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_backend_save_exception_does_not_kill_drainer(self) -> None:
        """Drainer loop continues processing after a backend failure."""

        class _BrokenBackend(InMemoryFeedbackBackend):
            def __init__(self) -> None:
                super().__init__()
                self.fail_count = 0

            async def save(self, sample: FeedbackSample) -> None:
                if self.fail_count < 3:
                    self.fail_count += 1
                    raise OSError("transient error")
                await super().save(sample)

        backend = _BrokenBackend()
        c = FeedbackCollector(backend=backend)
        c.start()
        for i in range(5):
            c.record(f"q{i}", "r", "positive")
        await c.stop()
        # First 3 fail, last 2 succeed
        assert await backend.count() == 2


# ── InMemoryFeedbackBackend ────────────────────────────────────────────────────


class TestInMemoryBackend:
    @pytest.mark.asyncio
    async def test_save_and_load_all(self) -> None:
        b = InMemoryFeedbackBackend()
        s = FeedbackSample(query="q", response="r", feedback="positive")
        await b.save(s)
        all_samples = await b.load_all()
        assert len(all_samples) == 1
        assert all_samples[0].id == s.id

    @pytest.mark.asyncio
    async def test_count_matches_saves(self) -> None:
        b = InMemoryFeedbackBackend()
        for _ in range(7):
            await b.save(FeedbackSample(query="q", response="r", feedback="positive"))
        assert await b.count() == 7

    @pytest.mark.asyncio
    async def test_clear_resets_count(self) -> None:
        b = InMemoryFeedbackBackend()
        await b.save(FeedbackSample(query="q", response="r", feedback="positive"))
        await b.clear()
        assert await b.count() == 0

    @pytest.mark.asyncio
    async def test_load_since_filters(self) -> None:
        b = InMemoryFeedbackBackend()
        old = FeedbackSample(
            query="old",
            response="r",
            feedback="positive",
            timestamp=datetime(2020, 6, 1, tzinfo=timezone.utc),
        )
        new = FeedbackSample(
            query="new",
            response="r",
            feedback="positive",
            timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc),
        )
        await b.save(old)
        await b.save(new)
        cutoff = datetime(2024, 1, 1, tzinfo=timezone.utc)
        recent = await b.load_since(cutoff)
        assert len(recent) == 1
        assert recent[0].query == "new"
