"""Convert feedback samples into training datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .types import FeedbackSample, PreferencePair, TrainingExample


class TrainingDataGenerator:
    """
    Converts FeedbackSample records into training datasets.

    Supported output formats
    ------------------------
    * Instruction-tuning JSONL — compatible with OpenAI fine-tuning API and
      Anthropic dataset format (chat message arrays).
    * Preference pairs — chosen / rejected format for DPO and RLHF pipelines.

    Conversion rules
    ----------------
    * Positive feedback → query becomes user turn, accepted response becomes
      assistant turn.
    * Negative feedback with correction → query + corrected_response used.
    * Negative feedback without correction → skipped by default.

    Context injection
    -----------------
    RAG context is merged into a single system message rather than appended
    as a second system turn.  OpenAI fine-tuning only allows one system
    message per example; a second system turn causes validation failure.

    Parameters
    ----------
    max_pairs_per_query:
        Cap on preference pairs generated per unique query string.
        Prevents O(P x N) blow-up when one query has many positives and
        negatives.  Default 500.
    """

    def __init__(
        self,
        system_prompt: str | None = None,
        max_pairs_per_query: int = 500,
    ) -> None:
        self._system_prompt = (
            system_prompt or "You are a helpful, accurate, and concise AI assistant."
        )
        self._max_pairs_per_query = max_pairs_per_query

    # ── Public API ────────────────────────────────────────────────────────────

    def generate_examples(self, samples: list[FeedbackSample]) -> list[TrainingExample]:
        """Convert feedback samples to chat-format training examples."""
        examples: list[TrainingExample] = []
        for sample in samples:
            ex = self._to_example(sample)
            if ex is not None:
                examples.append(ex)
        return examples

    def generate_jsonl(self, samples: list[FeedbackSample]) -> list[str]:
        """
        Return JSONL lines compatible with the OpenAI fine-tuning API.

        Each line is a JSON-serialised object with a ``messages`` key
        containing the conversation in chat format.
        """
        lines: list[str] = []
        for ex in self.generate_examples(samples):
            obj: dict[str, Any] = {"messages": ex.messages}
            if ex.source_feedback_id:
                obj["_source_id"] = ex.source_feedback_id
            lines.append(json.dumps(obj, ensure_ascii=False))
        return lines

    def generate_preference_pairs(self, samples: list[FeedbackSample]) -> list[PreferencePair]:
        """
        Build DPO / RLHF preference pairs.

        Two strategies are used and deduplicated:
        1. Correction pairs — negative sample with a correction becomes a
           (corrected, original) chosen/rejected pair.
        2. Cross-sample pairs — for queries that appear in both positive and
           negative samples, pair the best accepted response against each
           rejected response.
        """
        pairs: list[PreferencePair] = []
        seen: set[tuple[str, str, str]] = set()

        def _add(pair: PreferencePair) -> None:
            key = (pair.prompt, pair.chosen[:64], pair.rejected[:64])
            if key not in seen:
                seen.add(key)
                pairs.append(pair)

        # Strategy 1: negative-with-correction pairs
        for s in samples:
            if s.feedback == "negative" and s.corrected_response:
                _add(
                    PreferencePair(
                        prompt=s.query,
                        chosen=s.corrected_response,
                        rejected=s.response,
                        source_ids=(s.id, s.id),
                    )
                )

        # Strategy 2: cross-sample pairing by query (capped per query)
        by_query: dict[str, list[FeedbackSample]] = {}
        for s in samples:
            by_query.setdefault(s.query, []).append(s)

        for query, group in by_query.items():
            positives = [s for s in group if s.feedback == "positive"]
            negatives = [s for s in group if s.feedback == "negative" and not s.corrected_response]
            added = 0
            for pos in positives:
                if added >= self._max_pairs_per_query:
                    break
                for neg in negatives:
                    if added >= self._max_pairs_per_query:
                        break
                    _add(
                        PreferencePair(
                            prompt=query,
                            chosen=pos.response,
                            rejected=neg.response,
                            source_ids=(pos.id, neg.id),
                        )
                    )
                    added += 1

        return pairs

    def write_jsonl(self, samples: list[FeedbackSample], path: str) -> int:
        """
        Write instruction-tuning JSONL to *path*.

        Returns the number of examples written.
        """
        lines = self.generate_jsonl(samples)
        Path(path).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        return len(lines)

    def write_preference_jsonl(self, samples: list[FeedbackSample], path: str) -> int:
        """
        Write preference pairs to *path* in JSONL format.

        Each line contains ``prompt``, ``chosen``, and ``rejected`` keys,
        compatible with TRL / Axolotl DPO trainers.

        Returns the number of pairs written.
        """
        pairs = self.generate_preference_pairs(samples)
        lines = [
            json.dumps(
                {
                    "prompt": p.prompt,
                    "chosen": p.chosen,
                    "rejected": p.rejected,
                    "_source_ids": list(p.source_ids),
                },
                ensure_ascii=False,
            )
            for p in pairs
        ]
        Path(path).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        return len(lines)

    # ── Internals ─────────────────────────────────────────────────────────────

    def _to_example(self, sample: FeedbackSample) -> TrainingExample | None:
        if sample.feedback == "positive":
            target = sample.response
        elif sample.corrected_response:
            target = sample.corrected_response
        else:
            return None

        # Merge context into the system prompt so there is exactly one system
        # message — the OpenAI fine-tuning API rejects examples with multiple
        # system turns.
        system_content = self._system_prompt
        if sample.context:
            ctx_block = "\n\n".join(sample.context)
            system_content = f"{self._system_prompt}\n\nContext:\n{ctx_block}"

        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": sample.query},
            {"role": "assistant", "content": target},
        ]

        return TrainingExample(messages=messages, source_feedback_id=sample.id)
