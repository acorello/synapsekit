"""Fine-tuning provider abstraction and implementations."""

from __future__ import annotations

import asyncio
import io
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from .types import FineTuneJob

# ── Base provider ─────────────────────────────────────────────────────────────


class BaseFineTuneProvider(ABC):
    """
    Provider-agnostic fine-tuning interface.

    All orchestration logic in ContinuousTrainer talks only to this interface,
    keeping provider-specific code fully isolated behind each implementation.
    """

    @abstractmethod
    async def upload_dataset(self, jsonl_path: str) -> str:
        """Upload a JSONL file and return a provider-specific file ID."""
        ...

    @abstractmethod
    async def start_job(
        self,
        file_id: str,
        base_model: str,
        *,
        hyperparams: dict[str, Any] | None = None,
        suffix: str | None = None,
    ) -> FineTuneJob:
        """Start a fine-tuning job and return the job descriptor."""
        ...

    @abstractmethod
    async def status(self, job_id: str) -> FineTuneJob:
        """Return the current state of a fine-tuning job."""
        ...

    @abstractmethod
    async def list_jobs(self, limit: int = 20) -> list[FineTuneJob]:
        """Return recent fine-tuning jobs in reverse-chronological order."""
        ...

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a running fine-tuning job.  Returns True on success."""
        return False

    async def wait_for_completion(
        self,
        job_id: str,
        poll_interval_s: float = 60.0,
        timeout_s: float | None = None,
    ) -> FineTuneJob:
        """
        Poll *job_id* until it reaches a terminal state.

        Raises
        ------
        TimeoutError
            If *timeout_s* is exceeded before the job finishes.
        """
        elapsed = 0.0
        while True:
            job = await self.status(job_id)
            if job.status in ("succeeded", "failed", "cancelled"):
                return job
            await asyncio.sleep(poll_interval_s)
            elapsed += poll_interval_s
            if timeout_s is not None and elapsed >= timeout_s:
                raise TimeoutError(f"Fine-tune job {job_id} did not complete within {timeout_s}s")


# ── OpenAI provider ───────────────────────────────────────────────────────────


class OpenAIFineTuneProvider(BaseFineTuneProvider):
    """
    OpenAI fine-tuning API adapter.

    Requires: ``pip install synapsekit[training]``

    Parameters
    ----------
    api_key:
        OpenAI API key.  Falls back to ``OPENAI_API_KEY`` env var.
    organization:
        Optional OpenAI organization ID.
    """

    def __init__(
        self,
        api_key: str | None = None,
        organization: str | None = None,
    ) -> None:
        self._api_key = api_key
        self._organization = organization
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import openai
            except ImportError:
                raise ImportError(
                    "openai is required for OpenAIFineTuneProvider. "
                    "Install with: pip install synapsekit[training]"
                ) from None
            kwargs: dict[str, Any] = {}
            if self._api_key:
                kwargs["api_key"] = self._api_key
            if self._organization:
                kwargs["organization"] = self._organization
            self._client = openai.AsyncOpenAI(**kwargs)
        return self._client

    async def upload_dataset(self, jsonl_path: str) -> str:
        client = self._get_client()
        # Read file bytes in a thread to avoid blocking the event loop.
        data = await asyncio.to_thread(Path(jsonl_path).read_bytes)
        response = await client.files.create(
            file=(Path(jsonl_path).name, io.BytesIO(data)),
            purpose="fine-tune",
        )
        return str(response.id)

    async def start_job(
        self,
        file_id: str,
        base_model: str,
        *,
        hyperparams: dict[str, Any] | None = None,
        suffix: str | None = None,
    ) -> FineTuneJob:
        client = self._get_client()
        kwargs: dict[str, Any] = {
            "training_file": file_id,
            "model": base_model,
        }
        if suffix:
            kwargs["suffix"] = suffix
        if hyperparams:
            kwargs["hyperparameters"] = hyperparams
        raw = await client.fine_tuning.jobs.create(**kwargs)
        return self._parse_job(raw)

    async def status(self, job_id: str) -> FineTuneJob:
        client = self._get_client()
        raw = await client.fine_tuning.jobs.retrieve(job_id)
        return self._parse_job(raw)

    async def list_jobs(self, limit: int = 20) -> list[FineTuneJob]:
        client = self._get_client()
        page = await client.fine_tuning.jobs.list(limit=limit)
        return [self._parse_job(j) for j in page.data]

    async def cancel_job(self, job_id: str) -> bool:
        client = self._get_client()
        try:
            await client.fine_tuning.jobs.cancel(job_id)
            return True
        except Exception:
            return False

    def _parse_job(self, raw: Any) -> FineTuneJob:
        def _ts(ts: int | None) -> datetime | None:
            if ts is None:
                return None
            return datetime.fromtimestamp(ts, tz=timezone.utc)

        status_map: dict[str, str] = {
            "validating_files": "queued",
            "queued": "queued",
            "running": "running",
            "succeeded": "succeeded",
            "failed": "failed",
            "cancelled": "cancelled",
        }
        raw_status = getattr(raw, "status", "queued")
        status = status_map.get(raw_status, "queued")
        error_obj = getattr(raw, "error", None)
        error_msg = getattr(error_obj, "message", None) if error_obj else None

        return FineTuneJob(
            job_id=str(raw.id),
            provider="openai",
            base_model=str(getattr(raw, "model", "")),
            status=status,  # type: ignore[arg-type]
            created_at=_ts(getattr(raw, "created_at", None)) or datetime.now(timezone.utc),
            finished_at=_ts(getattr(raw, "finished_at", None)),
            fine_tuned_model=getattr(raw, "fine_tuned_model", None),
            error=error_msg,
            training_file_id=getattr(raw, "training_file", None),
        )


# ── Anthropic provider ────────────────────────────────────────────────────────


class AnthropicFineTuneProvider(BaseFineTuneProvider):
    """
    Anthropic fine-tuning adapter.

    Anthropic's fine-tuning API is available to enterprise customers.
    This implementation provides the correct interface contract so the
    orchestration layer remains unchanged when the public API stabilises.

    Parameters
    ----------
    api_key:
        Anthropic API key (for future use).
    """

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key
        self._jobs: dict[str, FineTuneJob] = {}

    async def upload_dataset(self, jsonl_path: str) -> str:
        # Return the local path as file_id — Anthropic does not yet have a
        # public dataset upload endpoint; implementations can override this.
        return jsonl_path

    async def start_job(
        self,
        file_id: str,
        base_model: str,
        *,
        hyperparams: dict[str, Any] | None = None,
        suffix: str | None = None,
    ) -> FineTuneJob:
        import uuid as _uuid

        job = FineTuneJob(
            job_id=str(_uuid.uuid4()),
            provider="anthropic",
            base_model=base_model,
            status="queued",
            training_file_id=file_id,
            metadata={"suffix": suffix, "hyperparams": hyperparams},
        )
        self._jobs[job.job_id] = job
        return job

    async def status(self, job_id: str) -> FineTuneJob:
        if job_id not in self._jobs:
            raise KeyError(f"Unknown Anthropic fine-tune job: {job_id}")
        return self._jobs[job_id]

    async def list_jobs(self, limit: int = 20) -> list[FineTuneJob]:
        jobs = list(self._jobs.values())
        return jobs[-limit:]

    def _update_status(
        self,
        job_id: str,
        status: Literal["queued", "running", "succeeded", "failed", "cancelled"],
        fine_tuned_model: str | None = None,
    ) -> None:
        """For testing / simulation — update a job's status in place."""
        if job_id in self._jobs:
            self._jobs[job_id].status = status
            if fine_tuned_model:
                self._jobs[job_id].fine_tuned_model = fine_tuned_model
            if status in ("succeeded", "failed", "cancelled"):
                self._jobs[job_id].finished_at = datetime.now(timezone.utc)
