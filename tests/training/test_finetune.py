"""Tests for fine-tuning providers (OpenAI and Anthropic)."""

from __future__ import annotations

import os
import tempfile
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from synapsekit.training.finetune import (
    AnthropicFineTuneProvider,
    OpenAIFineTuneProvider,
)
from synapsekit.training.types import FineTuneJob

# ── Helpers ────────────────────────────────────────────────────────────────────


def _job(status: str = "queued", model: str | None = None) -> Any:
    """Build a mock OpenAI fine-tune job object."""
    raw = MagicMock()
    raw.id = "ft-abc123"
    raw.model = "gpt-4o-mini"
    raw.status = status
    raw.created_at = 1_700_000_000
    raw.finished_at = 1_700_001_000 if status in ("succeeded", "failed") else None
    raw.fine_tuned_model = model
    raw.error = None
    raw.training_file = "file-xyz"
    return raw


# ── OpenAIFineTuneProvider ────────────────────────────────────────────────────


class TestOpenAIFineTuneProvider:
    def _provider(self) -> OpenAIFineTuneProvider:
        return OpenAIFineTuneProvider(api_key="test-key")

    @pytest.mark.asyncio
    async def test_upload_dataset_returns_file_id(self) -> None:
        p = self._provider()
        mock_file = MagicMock()
        mock_file.id = "file-uploaded"
        mock_client = MagicMock()
        mock_client.files.create = AsyncMock(return_value=mock_file)
        p._client = mock_client

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
            f.write('{"messages": []}\n')
            path = f.name
        try:
            fid = await p.upload_dataset(path)
            assert fid == "file-uploaded"
        finally:
            os.unlink(path)

    @pytest.mark.asyncio
    async def test_upload_dataset_uses_to_thread(self) -> None:
        """upload_dataset must not open files synchronously in the async path."""
        import asyncio

        p = self._provider()
        mock_file = MagicMock()
        mock_file.id = "file-async"
        mock_client = MagicMock()
        mock_client.files.create = AsyncMock(return_value=mock_file)
        p._client = mock_client

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
            f.write('{"messages": []}\n')
            path = f.name
        try:
            # Should complete without stalling the event loop
            fid = await asyncio.wait_for(p.upload_dataset(path), timeout=5.0)
            assert fid == "file-async"
        finally:
            os.unlink(path)

    @pytest.mark.asyncio
    async def test_start_job_returns_fine_tune_job(self) -> None:
        p = self._provider()
        raw = _job("queued")
        mock_client = MagicMock()
        mock_client.fine_tuning.jobs.create = AsyncMock(return_value=raw)
        p._client = mock_client

        job = await p.start_job("file-xyz", "gpt-4o-mini")
        assert isinstance(job, FineTuneJob)
        assert job.job_id == "ft-abc123"
        assert job.provider == "openai"
        assert job.status == "queued"

    @pytest.mark.asyncio
    async def test_status_parses_succeeded(self) -> None:
        p = self._provider()
        raw = _job("succeeded", model="ft:gpt-4o-mini:custom")
        mock_client = MagicMock()
        mock_client.fine_tuning.jobs.retrieve = AsyncMock(return_value=raw)
        p._client = mock_client

        job = await p.status("ft-abc123")
        assert job.status == "succeeded"
        assert job.fine_tuned_model == "ft:gpt-4o-mini:custom"

    @pytest.mark.asyncio
    async def test_status_parses_validating_files_as_queued(self) -> None:
        p = self._provider()
        raw = _job("validating_files")
        mock_client = MagicMock()
        mock_client.fine_tuning.jobs.retrieve = AsyncMock(return_value=raw)
        p._client = mock_client

        job = await p.status("ft-abc123")
        assert job.status == "queued"

    @pytest.mark.asyncio
    async def test_list_jobs_returns_list(self) -> None:
        p = self._provider()
        page = MagicMock()
        page.data = [_job("succeeded"), _job("running")]
        mock_client = MagicMock()
        mock_client.fine_tuning.jobs.list = AsyncMock(return_value=page)
        p._client = mock_client

        jobs = await p.list_jobs(limit=10)
        assert len(jobs) == 2
        assert all(isinstance(j, FineTuneJob) for j in jobs)

    @pytest.mark.asyncio
    async def test_cancel_returns_true_on_success(self) -> None:
        p = self._provider()
        mock_client = MagicMock()
        mock_client.fine_tuning.jobs.cancel = AsyncMock()
        p._client = mock_client

        result = await p.cancel_job("ft-abc123")
        assert result is True

    @pytest.mark.asyncio
    async def test_cancel_returns_false_on_error(self) -> None:
        p = self._provider()
        mock_client = MagicMock()
        mock_client.fine_tuning.jobs.cancel = AsyncMock(side_effect=RuntimeError("not found"))
        p._client = mock_client

        result = await p.cancel_job("ft-abc123")
        assert result is False

    @pytest.mark.asyncio
    async def test_raises_import_error_without_openai(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import builtins

        real = builtins.__import__

        def _block(name: str, *args: Any, **kw: Any) -> Any:
            if name == "openai":
                raise ImportError("openai")
            return real(name, *args, **kw)

        monkeypatch.setattr(builtins, "__import__", _block)
        p = OpenAIFineTuneProvider(api_key="test")
        with pytest.raises(ImportError, match="openai"):
            p._get_client()


# ── wait_for_completion ────────────────────────────────────────────────────────


class TestWaitForCompletion:
    @pytest.mark.asyncio
    async def test_returns_on_succeeded(self) -> None:
        p = AnthropicFineTuneProvider()
        job = await p.start_job("file", "claude-3-haiku-20240307")
        p._update_status(job.job_id, "succeeded", fine_tuned_model="claude-3-haiku-ft")

        result = await p.wait_for_completion(job.job_id, poll_interval_s=0.01)
        assert result.status == "succeeded"
        assert result.fine_tuned_model == "claude-3-haiku-ft"

    @pytest.mark.asyncio
    async def test_raises_timeout_error(self) -> None:
        p = AnthropicFineTuneProvider()
        job = await p.start_job("file", "claude-3-haiku-20240307")
        # Never update — stays in "queued"

        with pytest.raises(TimeoutError):
            await p.wait_for_completion(job.job_id, poll_interval_s=0.01, timeout_s=0.05)


# ── AnthropicFineTuneProvider ─────────────────────────────────────────────────


class TestAnthropicFineTuneProvider:
    @pytest.mark.asyncio
    async def test_start_job_returns_job(self) -> None:
        p = AnthropicFineTuneProvider()
        job = await p.start_job("file.jsonl", "claude-3-haiku-20240307")
        assert isinstance(job, FineTuneJob)
        assert job.provider == "anthropic"
        assert job.status == "queued"
        assert job.training_file_id == "file.jsonl"

    @pytest.mark.asyncio
    async def test_upload_dataset_returns_path(self) -> None:
        p = AnthropicFineTuneProvider()
        fid = await p.upload_dataset("/tmp/dataset.jsonl")
        assert fid == "/tmp/dataset.jsonl"

    @pytest.mark.asyncio
    async def test_status_tracks_job(self) -> None:
        p = AnthropicFineTuneProvider()
        job = await p.start_job("f", "claude-3-haiku-20240307")
        retrieved = await p.status(job.job_id)
        assert retrieved.job_id == job.job_id

    @pytest.mark.asyncio
    async def test_status_raises_for_unknown_job(self) -> None:
        p = AnthropicFineTuneProvider()
        with pytest.raises(KeyError):
            await p.status("nonexistent-id")

    @pytest.mark.asyncio
    async def test_list_jobs_respects_limit(self) -> None:
        p = AnthropicFineTuneProvider()
        for _ in range(5):
            await p.start_job("f", "claude-3-haiku-20240307")
        jobs = await p.list_jobs(limit=3)
        assert len(jobs) == 3

    @pytest.mark.asyncio
    async def test_update_status_changes_job(self) -> None:
        p = AnthropicFineTuneProvider()
        job = await p.start_job("f", "claude-3-haiku-20240307")
        p._update_status(job.job_id, "succeeded", fine_tuned_model="claude-ft")
        updated = await p.status(job.job_id)
        assert updated.status == "succeeded"
        assert updated.fine_tuned_model == "claude-ft"
