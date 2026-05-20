"""Non-blocking feedback collection with pluggable storage backends."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from datetime import datetime
from typing import Any, Literal, Protocol, runtime_checkable

from .types import FeedbackSample

_log = logging.getLogger(__name__)


@runtime_checkable
class FeedbackBackend(Protocol):
    """Persistence contract for feedback samples.

    Implement this to plug in SQLite, Redis, Postgres, etc.
    """

    async def save(self, sample: FeedbackSample) -> None: ...
    async def load_all(self) -> list[FeedbackSample]: ...
    async def load_since(self, since: datetime) -> list[FeedbackSample]: ...
    async def count(self) -> int: ...


class InMemoryFeedbackBackend:
    """Thread-safe in-process backend suitable for development and testing."""

    def __init__(self) -> None:
        self._samples: list[FeedbackSample] = []
        self._lock = asyncio.Lock()

    async def save(self, sample: FeedbackSample) -> None:
        async with self._lock:
            self._samples.append(sample)

    async def load_all(self) -> list[FeedbackSample]:
        async with self._lock:
            return list(self._samples)

    async def load_since(self, since: datetime) -> list[FeedbackSample]:
        async with self._lock:
            return [s for s in self._samples if s.timestamp >= since]

    async def count(self) -> int:
        async with self._lock:
            return len(self._samples)

    async def clear(self) -> None:
        async with self._lock:
            self._samples.clear()


class FeedbackCollector:
    """
    Non-blocking production feedback collector.

    Writes samples to an internal asyncio.Queue that is drained by a
    background task.  Callers (RAG, agents, voice hooks) are never
    blocked waiting for storage I/O.

    Lifecycle
    ---------
    1. ``collector = FeedbackCollector()``    — safe to call anywhere
    2. ``collector.start()``                  — call once inside async context
    3. ``collector.record(...)``              — synchronous, call from anywhere
    4. ``await collector.flush()``            — wait for all queued items to persist
    5. ``await collector.stop()``             — flush and shut down

    Parameters
    ----------
    backend:
        Storage backend.  Defaults to InMemoryFeedbackBackend.
    queue_maxsize:
        Maximum unprocessed items held in memory.  When full, new
        items are dropped with a warning rather than blocking the caller.
    """

    def __init__(
        self,
        backend: FeedbackBackend | None = None,
        queue_maxsize: int = 10_000,
    ) -> None:
        self._backend: FeedbackBackend = backend or InMemoryFeedbackBackend()
        self._queue: asyncio.Queue[FeedbackSample | None] = asyncio.Queue(maxsize=queue_maxsize)
        self._task: asyncio.Task[None] | None = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the background drainer task.  Requires a running event loop."""
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._drain_loop(), name="synapsekit-feedback-drainer")

    async def flush(self) -> None:
        """Block until all currently queued items have been persisted (or failed)."""
        if self._task is not None and not self._task.done():
            await self._queue.join()

    async def stop(self) -> None:
        """Flush all pending samples to the backend, then stop the drainer."""
        await self._queue.put(None)  # sentinel
        if self._task and not self._task.done():
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        self._task = None

    # ── Public API ────────────────────────────────────────────────────────────

    def record(
        self,
        query: str,
        response: str,
        feedback: Literal["positive", "negative"],
        *,
        corrected_response: str | None = None,
        context: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        latency_ms: float | None = None,
        cost_usd: float | None = None,
    ) -> FeedbackSample:
        """
        Record a feedback sample.

        Enqueues immediately and returns the sample.  Persistence happens
        asynchronously in the background drainer — this method never blocks.
        """
        sample = FeedbackSample(
            query=query,
            response=response,
            feedback=feedback,
            corrected_response=corrected_response,
            context=context,
            metadata=metadata,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
        )
        try:
            self._queue.put_nowait(sample)
        except asyncio.QueueFull:
            _log.warning("FeedbackCollector queue full — dropping sample %s", sample.id)
        return sample

    async def get_samples(self, since: datetime | None = None) -> list[FeedbackSample]:
        """Return persisted samples, optionally filtered by timestamp."""
        if since is not None:
            return await self._backend.load_since(since)
        return await self._backend.load_all()

    async def pending_count(self) -> int:
        """Items waiting in the in-memory queue (not yet persisted)."""
        return self._queue.qsize()

    async def stored_count(self) -> int:
        """Items already persisted to the backend."""
        return await self._backend.count()

    # ── Internal ─────────────────────────────────────────────────────────────

    async def _drain_loop(self) -> None:
        while True:
            item = await self._queue.get()
            try:
                if item is None:
                    break
                try:
                    await self._backend.save(item)
                except Exception:
                    _log.exception(
                        "FeedbackCollector: backend.save() failed for sample %s",
                        item.id,
                    )
            finally:
                self._queue.task_done()
