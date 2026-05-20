import asyncio

import pytest
from pydantic import BaseModel

from synapsekit.structured_output import (
    StructuredOutput,
    StructuredOutputRetryStrategy,
    StructuredOutputValidationError,
    build_corrective_prompt,
)


class DocumentSummary(BaseModel):
    title: str
    page_count: int


class FakeLLM:
    provider = "fake"
    model = "fake-primary"

    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    async def ask(self, prompt):
        self.prompts.append(prompt)
        return self.responses.pop(0)


class FakeStreamingLLM(FakeLLM):
    async def stream(self, prompt):
        self.prompts.append(prompt)
        response = self.responses.pop(0)
        for chunk in response:
            yield chunk


class FakeCostTracker:
    def __init__(self):
        self.records = []

    def record_call(self, **kwargs):
        self.records.append(kwargs)
        return {"recorded": True, "record_index": len(self.records)}


def test_retries_after_json_decode_error_and_records_cost():
    llm = FakeLLM(
        [
            "not json",
            '{"title": "Board report", "page_count": 12}',
        ]
    )
    tracker = FakeCostTracker()
    structured = StructuredOutput(
        llm,
        DocumentSummary,
        retry_strategy=StructuredOutputRetryStrategy(max_attempts=2),
        cost_tracker=tracker,
    )

    result = asyncio.run(structured.generate("Extract the document summary."))

    assert result.output == DocumentSummary(title="Board report", page_count=12)
    assert len(result.attempts) == 2
    assert result.metadata["structured_output"]["attempt_count"] == 2
    assert len(tracker.records) == 2
    assert tracker.records[0]["operation"] == "structured_output"
    assert "JSON Schema" in llm.prompts[1]
    assert "Validation error" in llm.prompts[1]
    assert "Expecting value" in llm.prompts[1]
    assert "Previous response" in llm.prompts[1]
    assert "not json" in llm.prompts[1]


def test_retries_after_pydantic_validation_error_with_schema_context():
    llm = FakeLLM(
        [
            '{"title": "Board report", "page_count": "many"}',
            '{"title": "Board report", "page_count": 12}',
        ]
    )
    structured = StructuredOutput(
        llm,
        DocumentSummary,
        retry_strategy=StructuredOutputRetryStrategy(max_attempts=2),
    )

    result = asyncio.run(structured.generate("Extract the document summary."))

    assert result.output.page_count == 12
    corrective_prompt = llm.prompts[1]
    assert "DocumentSummary" in corrective_prompt
    assert "page_count" in corrective_prompt
    assert "int_parsing" in corrective_prompt
    assert '{"title": "Board report", "page_count": "many"}' in corrective_prompt


def test_uses_fallback_provider_for_retry_attempts():
    primary = FakeLLM(["not json"])
    fallback = FakeLLM(['{"title": "Fallback report", "page_count": 3}'])
    fallback.provider = "fallback"
    fallback.model = "fallback-model"

    structured = StructuredOutput(
        primary,
        DocumentSummary,
        retry_strategy=StructuredOutputRetryStrategy(
            max_attempts=2,
            fallback_provider=fallback,
            fallback_after_attempt=2,
        ),
    )

    result = asyncio.run(structured.generate("Extract the document summary."))

    assert result.output.title == "Fallback report"
    assert len(primary.prompts) == 1
    assert len(fallback.prompts) == 1
    assert result.attempts[1].provider == "fallback"
    assert result.attempts[1].model == "fallback-model"


def test_stream_buffers_chunks_and_yields_validated_result():
    llm = FakeStreamingLLM(
        [
            ['{"title": ', '"Streamed report"', ', "page_count": 8}'],
        ]
    )
    structured = StructuredOutput(
        llm,
        DocumentSummary,
        retry_strategy=StructuredOutputRetryStrategy(max_attempts=1),
    )

    async def collect():
        return [event async for event in structured.stream("Extract summary.")]

    events = asyncio.run(collect())

    chunks = [event.content for event in events if event.type == "chunk"]
    result_events = [event for event in events if event.type == "result"]
    assert chunks == ['{"title": ', '"Streamed report"', ', "page_count": 8}']
    assert len(result_events) == 1
    assert result_events[0].output == DocumentSummary(
        title="Streamed report",
        page_count=8,
    )
    assert result_events[0].metadata["structured_output"]["attempt_count"] == 1


def test_stream_retries_with_corrective_prompt():
    llm = FakeStreamingLLM(
        [
            ["not ", "json"],
            ['{"title": "Recovered", "page_count": 2}'],
        ]
    )
    structured = StructuredOutput(
        llm,
        DocumentSummary,
        retry_strategy=StructuredOutputRetryStrategy(max_attempts=2),
    )

    async def collect():
        events = []
        async for event in structured.stream("Extract summary."):
            events.append(event)
        return events

    events = asyncio.run(collect())

    retry_events = [event for event in events if event.type == "retry"]
    result_events = [event for event in events if event.type == "result"]
    assert len(retry_events) == 1
    assert "JSON Schema" in retry_events[0].content
    assert "not json" in retry_events[0].content
    assert result_events[0].output.title == "Recovered"
    assert len(llm.prompts) == 2


def test_raises_with_attempt_metadata_after_exhaustion():
    llm = FakeLLM(["not json"])
    structured = StructuredOutput(
        llm,
        DocumentSummary,
        retry_strategy=StructuredOutputRetryStrategy(max_attempts=1),
    )

    with pytest.raises(StructuredOutputValidationError) as exc_info:
        asyncio.run(structured.generate("Extract summary."))

    assert len(exc_info.value.attempts) == 1
    assert exc_info.value.metadata["structured_output"]["success"] is False


def test_build_corrective_prompt_mentions_schema_and_error():
    prompt = build_corrective_prompt(
        "Extract summary.",
        DocumentSummary,
        "bad",
        "page_count is required",
    )

    assert "DocumentSummary" in prompt
    assert "JSON Schema" in prompt
    assert "page_count is required" in prompt
    assert "bad" in prompt
    assert "Extract summary." in prompt
