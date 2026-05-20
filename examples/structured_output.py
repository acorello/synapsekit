"""Structured output validation with automatic correction retries.

Run with an LLM provider that implements SynapseKit's async ``ask`` interface,
for example ``OpenAILLM`` from ``synapsekit.llms``.
"""

import asyncio
from typing import Literal

from pydantic import BaseModel, Field

from synapsekit.structured_output import (
    StructuredOutput,
    StructuredOutputRetryStrategy,
)


class InvoiceExtraction(BaseModel):
    invoice_id: str = Field(description="Invoice identifier from the document")
    vendor: str
    total_usd: float
    due_date: str


class APIResponseCheck(BaseModel):
    status: Literal["ok", "error"]
    request_id: str
    retryable: bool
    message: str


async def main() -> None:
    from synapsekit.llms import OpenAILLM

    llm = OpenAILLM("gpt-4o-mini")
    retry_strategy = StructuredOutputRetryStrategy(
        max_attempts=3,
        backoff_seconds=0.25,
    )

    invoice_extractor = StructuredOutput(
        llm,
        InvoiceExtraction,
        retry_strategy=retry_strategy,
    )
    invoice = await invoice_extractor.generate(
        "Extract invoice fields from this text as JSON only:\n"
        "Invoice INV-1042 from Northwind Labs, total $1842.50, due 2026-06-15."
    )
    print(invoice.output.model_dump())
    print(invoice.metadata["structured_output"]["attempt_count"])

    api_validator = StructuredOutput(
        llm,
        APIResponseCheck,
        retry_strategy=retry_strategy,
    )
    api_response = await api_validator.generate(
        "Normalize this upstream API response as JSON only:\n"
        '{"request_id":"req_123","state":"temporarily_unavailable","detail":"try later"}'
    )
    print(api_response.output.model_dump())


if __name__ == "__main__":
    asyncio.run(main())
