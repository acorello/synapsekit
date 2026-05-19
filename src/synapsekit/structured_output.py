"""Provider-agnostic structured output validation for LLM responses."""

from __future__ import annotations

import asyncio
import inspect
import json
import time
from collections.abc import AsyncIterator, Callable, Iterable
from dataclasses import asdict, dataclass, field
from json import JSONDecodeError
from typing import Any, Generic, TypeVar, cast

try:  # Pydantic is optional until this module is used.
    from pydantic import BaseModel, ValidationError
except ImportError as exc:  # pragma: no cover - exercised only without pydantic.
    BaseModel = object  # type: ignore[assignment,misc]
    ValidationError = Exception  # type: ignore[assignment,misc]
    _PYDANTIC_IMPORT_ERROR: ImportError | None = exc
else:
    _PYDANTIC_IMPORT_ERROR = None


SchemaT = TypeVar("SchemaT", bound=Any)
PromptBuilder = Callable[[str, type[SchemaT], str, str], str]


@dataclass
class StructuredOutputRetryStrategy:
    """Retry policy for structured output generation.

    ``max_attempts`` counts every provider call, including fallback calls.
    When a fallback provider or model is configured, it is used starting at
    ``fallback_after_attempt`` after the previous attempt failed.
    """

    max_attempts: int = 3
    backoff_seconds: float = 0.0
    backoff_multiplier: float = 2.0
    fallback_provider: Any | None = None
    fallback_model: str | None = None
    fallback_after_attempt: int = 2

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if self.backoff_seconds < 0:
            raise ValueError("backoff_seconds must be greater than or equal to 0")
        if self.backoff_multiplier < 1:
            raise ValueError("backoff_multiplier must be greater than or equal to 1")
        if self.fallback_after_attempt < 1:
            raise ValueError("fallback_after_attempt must be at least 1")


@dataclass
class StructuredOutputAttempt:
    """Metadata for one LLM attempt."""

    attempt: int
    provider: str
    model: str | None
    prompt: str
    raw_output: str
    success: bool
    streamed: bool = False
    error_type: str | None = None
    error_message: str | None = None
    duration_ms: float = 0.0
    prompt_tokens_estimate: int = 0
    completion_tokens_estimate: int = 0
    cost_metadata: dict[str, Any] = field(default_factory=dict)

    def to_metadata(self) -> dict[str, Any]:
        data = asdict(self)
        data["prompt"] = _truncate(data["prompt"])
        data["raw_output"] = _truncate(data["raw_output"])
        return data


@dataclass
class StructuredOutputResult(Generic[SchemaT]):
    """Validated structured output plus the raw response and attempt metadata."""

    output: SchemaT
    raw_output: str
    attempts: list[StructuredOutputAttempt]
    metadata: dict[str, Any]


@dataclass
class StructuredOutputStreamEvent(Generic[SchemaT]):
    """Event yielded by ``StructuredOutput.stream``."""

    type: str
    content: str = ""
    output: SchemaT | None = None
    metadata: dict[str, Any] | None = None
    attempt: int = 1


class StructuredOutputError(Exception):
    """Base exception for structured output failures."""


class StructuredOutputValidationError(StructuredOutputError):
    """Raised when all attempts fail to produce valid structured output."""

    def __init__(
        self,
        message: str,
        *,
        attempts: list[StructuredOutputAttempt],
        last_error: Exception,
    ) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.last_error = last_error
        self.metadata = _build_metadata(attempts)


class IncrementalJSONBuffer:
    """Small helper for buffering streamed JSON and detecting completion."""

    def __init__(self) -> None:
        self._parts: list[str] = []
        self._decoder = json.JSONDecoder()
        self.complete = False

    @property
    def text(self) -> str:
        return "".join(self._parts)

    def append(self, chunk: str) -> bool:
        self._parts.append(chunk)
        stripped = self.text.strip()
        if not stripped:
            self.complete = False
            return False

        try:
            _, end = self._decoder.raw_decode(stripped)
        except JSONDecodeError:
            self.complete = False
            return False

        self.complete = not stripped[end:].strip()
        return self.complete


class StructuredOutput(Generic[SchemaT]):
    """Wrap an LLM/provider and validate its JSON response with Pydantic v2.

    The wrapped provider only needs one of the common async generation methods:
    ``ask(prompt)``, ``generate(prompt)``, ``complete(prompt)``, or
    ``__call__(prompt)``. Streaming uses ``stream(prompt)`` when available and
    falls back to the non-streaming generation path otherwise.
    """

    def __init__(
        self,
        llm: Any,
        schema: type[SchemaT],
        *,
        retry_strategy: StructuredOutputRetryStrategy | None = None,
        cost_tracker: Any | None = None,
        corrective_prompt_builder: PromptBuilder[SchemaT] | None = None,
    ) -> None:
        _ensure_pydantic_schema(schema)
        self.llm = llm
        self.schema = schema
        self.retry_strategy = retry_strategy or StructuredOutputRetryStrategy()
        self.cost_tracker = cost_tracker
        self.corrective_prompt_builder = corrective_prompt_builder or build_corrective_prompt

    async def generate(self, prompt: str) -> StructuredOutputResult[SchemaT]:
        """Generate, validate, and return a Pydantic model instance."""

        result, _ = await self._generate_with_retry(prompt, streamed=False)
        return result

    async def ask(self, prompt: str) -> StructuredOutputResult[SchemaT]:
        """Alias for ``generate`` to match SynapseKit LLM naming."""

        return await self.generate(prompt)

    async def stream(self, prompt: str) -> AsyncIterator[StructuredOutputStreamEvent[SchemaT]]:
        """Stream raw JSON chunks and yield a final validated result event.

        Each attempt buffers provider chunks while forwarding them as ``chunk``
        events. Validation happens once the provider stream finishes, or once a
        complete JSON payload has been observed and no further non-whitespace
        content arrives. Failed attempts yield a ``retry`` event before the
        corrective prompt is sent.
        """

        attempts: list[StructuredOutputAttempt] = []
        current_prompt = prompt
        last_error: Exception | None = None

        for attempt_number in range(1, self.retry_strategy.max_attempts + 1):
            llm, model_override = self._select_llm(attempt_number)
            started = time.perf_counter()
            buffer = IncrementalJSONBuffer()

            async for chunk in self._stream_text(llm, current_prompt, model_override):
                buffer.append(chunk)
                yield StructuredOutputStreamEvent(
                    type="chunk",
                    content=chunk,
                    attempt=attempt_number,
                )

            raw_output = buffer.text
            duration_ms = (time.perf_counter() - started) * 1000

            try:
                output = self._validate_raw(raw_output)
            except (JSONDecodeError, ValidationError) as exc:
                last_error = exc
                attempt = await self._record_attempt(
                    llm=llm,
                    model_override=model_override,
                    attempt_number=attempt_number,
                    prompt=current_prompt,
                    raw_output=raw_output,
                    success=False,
                    error=exc,
                    duration_ms=duration_ms,
                    streamed=True,
                )
                attempts.append(attempt)
                if attempt_number >= self.retry_strategy.max_attempts:
                    break

                current_prompt = self.corrective_prompt_builder(
                    prompt, self.schema, raw_output, _format_validation_error(exc)
                )
                yield StructuredOutputStreamEvent(
                    type="retry",
                    content=current_prompt,
                    metadata=_build_metadata(attempts),
                    attempt=attempt_number + 1,
                )
                await self._sleep_before_retry(attempt_number)
                continue

            attempt = await self._record_attempt(
                llm=llm,
                model_override=model_override,
                attempt_number=attempt_number,
                prompt=current_prompt,
                raw_output=raw_output,
                success=True,
                error=None,
                duration_ms=duration_ms,
                streamed=True,
            )
            attempts.append(attempt)
            metadata = _build_metadata(attempts)
            yield StructuredOutputStreamEvent(
                type="result",
                output=output,
                metadata=metadata,
                attempt=attempt_number,
            )
            return

        assert last_error is not None
        raise StructuredOutputValidationError(
            "LLM response did not match the requested structured output schema",
            attempts=attempts,
            last_error=last_error,
        )

    async def _generate_with_retry(
        self, prompt: str, *, streamed: bool
    ) -> tuple[StructuredOutputResult[SchemaT], list[StructuredOutputAttempt]]:
        attempts: list[StructuredOutputAttempt] = []
        current_prompt = prompt
        last_error: Exception | None = None

        for attempt_number in range(1, self.retry_strategy.max_attempts + 1):
            llm, model_override = self._select_llm(attempt_number)
            started = time.perf_counter()
            raw_output = await self._call_text(llm, current_prompt, model_override)
            duration_ms = (time.perf_counter() - started) * 1000

            try:
                output = self._validate_raw(raw_output)
            except (JSONDecodeError, ValidationError) as exc:
                last_error = exc
                attempt = await self._record_attempt(
                    llm=llm,
                    model_override=model_override,
                    attempt_number=attempt_number,
                    prompt=current_prompt,
                    raw_output=raw_output,
                    success=False,
                    error=exc,
                    duration_ms=duration_ms,
                    streamed=streamed,
                )
                attempts.append(attempt)
                if attempt_number >= self.retry_strategy.max_attempts:
                    break

                current_prompt = self.corrective_prompt_builder(
                    prompt, self.schema, raw_output, _format_validation_error(exc)
                )
                await self._sleep_before_retry(attempt_number)
                continue

            attempt = await self._record_attempt(
                llm=llm,
                model_override=model_override,
                attempt_number=attempt_number,
                prompt=current_prompt,
                raw_output=raw_output,
                success=True,
                error=None,
                duration_ms=duration_ms,
                streamed=streamed,
            )
            attempts.append(attempt)
            metadata = _build_metadata(attempts)
            return (
                StructuredOutputResult(
                    output=output,
                    raw_output=raw_output,
                    attempts=attempts,
                    metadata=metadata,
                ),
                attempts,
            )

        assert last_error is not None
        raise StructuredOutputValidationError(
            "LLM response did not match the requested structured output schema",
            attempts=attempts,
            last_error=last_error,
        )

    def _select_llm(self, attempt_number: int) -> tuple[Any, str | None]:
        strategy = self.retry_strategy
        use_fallback = attempt_number >= strategy.fallback_after_attempt
        if use_fallback and strategy.fallback_provider is not None:
            return strategy.fallback_provider, strategy.fallback_model
        if use_fallback and strategy.fallback_model is not None:
            return self.llm, strategy.fallback_model
        return self.llm, None

    async def _call_text(self, llm: Any, prompt: str, model_override: str | None) -> str:
        method = _find_generation_method(llm)
        response = await _invoke_provider(method, prompt, model_override)
        return _coerce_text(response)

    async def _stream_text(
        self, llm: Any, prompt: str, model_override: str | None
    ) -> AsyncIterator[str]:
        stream_method = getattr(llm, "stream", None)
        if callable(stream_method):
            stream = _invoke_provider(stream_method, prompt, model_override)
            stream = await stream if inspect.isawaitable(stream) else stream
            async for chunk in _aiter(stream):
                yield _coerce_text(chunk)
            return

        yield await self._call_text(llm, prompt, model_override)

    def _validate_raw(self, raw_output: str) -> SchemaT:
        data = json.loads(raw_output)
        return self.schema.model_validate(data)  # type: ignore[attr-defined,no-any-return]

    async def _record_attempt(
        self,
        *,
        llm: Any,
        model_override: str | None,
        attempt_number: int,
        prompt: str,
        raw_output: str,
        success: bool,
        error: Exception | None,
        duration_ms: float,
        streamed: bool,
    ) -> StructuredOutputAttempt:
        prompt_tokens = _estimate_tokens(prompt)
        completion_tokens = _estimate_tokens(raw_output)
        attempt = StructuredOutputAttempt(
            attempt=attempt_number,
            provider=_provider_name(llm),
            model=model_override or _model_name(llm),
            prompt=prompt,
            raw_output=raw_output,
            success=success,
            streamed=streamed,
            error_type=type(error).__name__ if error else None,
            error_message=_format_validation_error(error) if error else None,
            duration_ms=duration_ms,
            prompt_tokens_estimate=prompt_tokens,
            completion_tokens_estimate=completion_tokens,
        )
        tracker = self.cost_tracker or getattr(llm, "cost_tracker", None)
        if tracker is not None:
            attempt.cost_metadata = await _record_cost_tracker_attempt(
                tracker,
                attempt,
                operation="structured_output",
                schema_name=self.schema.__name__,
            )
        return attempt

    async def _sleep_before_retry(self, failed_attempt_number: int) -> None:
        base = self.retry_strategy.backoff_seconds
        if base == 0:
            return
        delay = base * (self.retry_strategy.backoff_multiplier ** (failed_attempt_number - 1))
        await asyncio.sleep(delay)


def structured_output(
    llm: Any,
    schema: type[SchemaT],
    *,
    retry_strategy: StructuredOutputRetryStrategy | None = None,
    cost_tracker: Any | None = None,
) -> StructuredOutput[SchemaT]:
    """Convenience factory for ``StructuredOutput``."""

    return StructuredOutput(
        llm,
        schema,
        retry_strategy=retry_strategy,
        cost_tracker=cost_tracker,
    )


def build_corrective_prompt(
    original_prompt: str,
    schema: type[SchemaT],
    previous_output: str,
    validation_error: str,
) -> str:
    """Build the default corrective prompt for a failed validation attempt."""

    schema_json = json.dumps(schema.model_json_schema(), indent=2, sort_keys=True)  # type: ignore[attr-defined]
    return (
        "Your previous response could not be parsed as the requested structured "
        "output.\n\n"
        "Return only a complete JSON value that conforms to this JSON Schema. "
        "Do not include markdown fences, prose, comments, or trailing text.\n\n"
        f"Schema name: {schema.__name__}\n"
        f"JSON Schema:\n{schema_json}\n\n"
        f"Validation error:\n{validation_error}\n\n"
        f"Previous response:\n{_truncate(previous_output, limit=4000)}\n\n"
        f"Original prompt:\n{original_prompt}"
    )


async def _record_cost_tracker_attempt(
    tracker: Any,
    attempt: StructuredOutputAttempt,
    *,
    operation: str,
    schema_name: str,
) -> dict[str, Any]:
    record = {
        "operation": operation,
        "schema_name": schema_name,
        "attempt": attempt.attempt,
        "provider": attempt.provider,
        "model": attempt.model,
        "prompt": attempt.prompt,
        "completion": attempt.raw_output,
        "prompt_tokens": attempt.prompt_tokens_estimate,
        "completion_tokens": attempt.completion_tokens_estimate,
        "input_tokens": attempt.prompt_tokens_estimate,
        "output_tokens": attempt.completion_tokens_estimate,
        "total_tokens": (attempt.prompt_tokens_estimate + attempt.completion_tokens_estimate),
        "cost_usd": 0.0,
        "success": attempt.success,
        "metadata": {
            "structured_output": True,
            "streamed": attempt.streamed,
            "error_type": attempt.error_type,
            "error_message": attempt.error_message,
        },
    }

    for method_name in (
        "record_attempt",
        "record_llm_call",
        "record_call",
        "track_call",
        "track",
        "add",
    ):
        method = getattr(tracker, method_name, None)
        if not callable(method):
            continue
        try:
            result = await _invoke_tracker_method(method, record)
        except TypeError:
            continue
        return _serialize_tracker_result(result)

    calls = getattr(tracker, "calls", None)
    if isinstance(calls, list):
        calls.append(record)
        return {"recorded": True, "method": "calls.append"}

    attempts = getattr(tracker, "attempts", None)
    if isinstance(attempts, list):
        attempts.append(record)
        return {"recorded": True, "method": "attempts.append"}

    return {"recorded": False, "reason": "no compatible CostTracker hook found"}


async def _invoke_tracker_method(method: Callable[..., Any], record: dict[str, Any]) -> Any:
    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError):
        result = method(record)
        return await result if inspect.isawaitable(result) else result

    params = signature.parameters
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()):
        result = method(**record)
        return await result if inspect.isawaitable(result) else result

    kwargs: dict[str, Any] = {}
    for name, param in params.items():
        if name == "self":
            continue
        if name in record:
            kwargs[name] = record[name]
        elif param.default is inspect.Parameter.empty:
            kwargs = {}
            break

    if kwargs:
        result = method(**kwargs)
    else:
        positional = [
            param
            for param in params.values()
            if param.kind
            in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            and param.default is inspect.Parameter.empty
            and param.name != "self"
        ]
        if len(positional) <= 1:
            result = method(record)
        else:
            raise TypeError("CostTracker method signature is not compatible")

    return await result if inspect.isawaitable(result) else result


def _serialize_tracker_result(result: Any) -> dict[str, Any]:
    if result is None:
        return {"recorded": True}
    if isinstance(result, dict):
        return cast(dict[str, Any], result)
    if hasattr(result, "model_dump"):
        return cast(dict[str, Any], result.model_dump())
    if hasattr(result, "__dict__"):
        return cast(dict[str, Any], dict(result.__dict__))
    return {"recorded": True, "result": repr(result)}


async def _invoke_provider(
    method: Callable[..., Any], prompt: str, model_override: str | None
) -> Any:
    kwargs = _provider_kwargs(method, model_override)
    result = method(prompt, **kwargs)
    return await result if inspect.isawaitable(result) else result


def _provider_kwargs(method: Callable[..., Any], model_override: str | None) -> dict[str, Any]:
    if model_override is None:
        return {}
    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError):
        return {}
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return {"model": model_override}
    if "model" in signature.parameters:
        return {"model": model_override}
    if "model_name" in signature.parameters:
        return {"model_name": model_override}
    return {}


def _find_generation_method(llm: Any) -> Callable[..., Any]:
    for name in ("ask", "generate", "complete", "predict"):
        method = getattr(llm, name, None)
        if callable(method):
            return cast(Callable[..., Any], method)
    if callable(llm):
        return cast(Callable[..., Any], llm)
    raise TypeError(
        "StructuredOutput requires an LLM/provider with ask(), generate(), "
        "complete(), predict(), stream(), or __call__()."
    )


async def _aiter(value: Any) -> AsyncIterator[Any]:
    if hasattr(value, "__aiter__"):
        async for item in value:
            yield item
        return
    if isinstance(value, Iterable):
        for item in value:
            yield item
        return
    raise TypeError("Provider stream() must return an async or sync iterable")


def _coerce_text(response: Any) -> str:
    if response is None:
        return ""
    if isinstance(response, str):
        return response
    for attr in ("content", "text", "message", "delta"):
        value = getattr(response, attr, None)
        if isinstance(value, str):
            return value
    if isinstance(response, dict):
        for key in ("content", "text", "message", "delta"):
            value = response.get(key)
            if isinstance(value, str):
                return value
    return str(response)


def _ensure_pydantic_schema(schema: type[Any]) -> None:
    if _PYDANTIC_IMPORT_ERROR is not None:
        raise ImportError(
            "StructuredOutput requires pydantic>=2. Install SynapseKit with the "
            "appropriate extra or add pydantic to your environment."
        ) from _PYDANTIC_IMPORT_ERROR
    if not inspect.isclass(schema) or not issubclass(schema, BaseModel):
        raise TypeError("schema must be a Pydantic BaseModel subclass")
    if not hasattr(schema, "model_validate") or not hasattr(schema, "model_json_schema"):
        raise TypeError("schema must be a Pydantic v2 BaseModel subclass")


def _format_validation_error(error: Exception | None) -> str:
    if error is None:
        return ""
    if hasattr(error, "errors"):
        try:
            return json.dumps(error.errors(), indent=2, sort_keys=True)
        except Exception:
            pass
    return str(error)


def _build_metadata(attempts: list[StructuredOutputAttempt]) -> dict[str, Any]:
    prompt_tokens = sum(attempt.prompt_tokens_estimate for attempt in attempts)
    completion_tokens = sum(attempt.completion_tokens_estimate for attempt in attempts)
    return {
        "structured_output": {
            "attempt_count": len(attempts),
            "success": bool(attempts and attempts[-1].success),
            "attempts": [attempt.to_metadata() for attempt in attempts],
            "total_prompt_tokens_estimate": prompt_tokens,
            "total_completion_tokens_estimate": completion_tokens,
            "total_tokens_estimate": prompt_tokens + completion_tokens,
        }
    }


def _provider_name(llm: Any) -> str:
    provider = getattr(llm, "provider", None) or getattr(llm, "provider_name", None)
    if provider:
        return str(provider)
    name = str(llm.__class__.__name__)
    return name[:-3] if name.lower().endswith("llm") else name


def _model_name(llm: Any) -> str | None:
    for attr in ("model", "model_name", "model_id", "deployment", "engine"):
        value = getattr(llm, attr, None)
        if value is not None:
            return str(value)
    return None


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text.split()))


def _truncate(text: str, *, limit: int = 1000) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 15] + "...<truncated>"


__all__ = [
    "IncrementalJSONBuffer",
    "StructuredOutput",
    "StructuredOutputAttempt",
    "StructuredOutputError",
    "StructuredOutputResult",
    "StructuredOutputRetryStrategy",
    "StructuredOutputStreamEvent",
    "StructuredOutputValidationError",
    "build_corrective_prompt",
    "structured_output",
]
