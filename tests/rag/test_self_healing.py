from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

from synapsekit.rag.self_healing import SelfHealingRAG


class _Strategy:
    def __init__(self, chunks):
        self.retrieve = AsyncMock(return_value=chunks)


@pytest.mark.asyncio
async def test_self_healing_single_strategy_success():
    llm = MagicMock()
    llm.tokens_used = {"input": 1, "output": 1}
    llm.generate_with_messages = AsyncMock(return_value="answer")

    metric = MagicMock()
    metric.evaluate = AsyncMock(return_value=MagicMock(score=0.9))

    rag = SelfHealingRAG(
        llm=llm,
        strategies=[_Strategy(["ctx"])],
        quality_threshold=0.75,
        max_retries=2,
        metric=metric,
    )

    answer = await rag.ask("q")
    assert answer == "answer"
    assert rag.last_report is not None
    assert rag.last_report.success is True
    assert rag.last_report.retries == 0


@pytest.mark.asyncio
async def test_self_healing_fallback_triggered():
    llm = MagicMock()
    llm.tokens_used = {"input": 1, "output": 1}
    llm.generate_with_messages = AsyncMock(side_effect=["bad", "good"])

    metric = MagicMock()
    metric.evaluate = AsyncMock(
        side_effect=[MagicMock(score=0.2), MagicMock(score=0.8)]
    )

    primary = _Strategy(["ctx1"])
    fallback = _Strategy(["ctx2"])

    rag = SelfHealingRAG(
        llm=llm,
        strategies=[primary, fallback],
        quality_threshold=0.75,
        max_retries=2,
        metric=metric,
    )

    answer = await rag.ask("q")
    assert answer == "good"
    assert rag.last_report is not None
    assert rag.last_report.success is True
    assert rag.last_report.retries == 1


@pytest.mark.asyncio
async def test_self_healing_max_retries_cap():
    llm = MagicMock()
    llm.tokens_used = {"input": 1, "output": 1}
    llm.generate_with_messages = AsyncMock(return_value="ans")

    metric = MagicMock()
    metric.evaluate = AsyncMock(return_value=MagicMock(score=0.1))

    strategies = [_Strategy(["ctx1"]), _Strategy(["ctx2"]), _Strategy(["ctx3"])]

    rag = SelfHealingRAG(
        llm=llm,
        strategies=strategies,
        quality_threshold=0.9,
        max_retries=1,
        metric=metric,
    )

    answer = await rag.ask("q")
    assert answer == "ans"
    assert rag.last_report is not None
    # max_retries=1 => at most 2 attempts
    assert rag.last_report.attempts == 2
    assert rag.last_report.success is False
