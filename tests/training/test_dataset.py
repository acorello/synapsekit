"""Tests for TrainingDataGenerator — JSONL and preference pair generation."""

from __future__ import annotations

import json
import os
import tempfile

from synapsekit.training.dataset import TrainingDataGenerator
from synapsekit.training.types import FeedbackSample

# ── Helpers ────────────────────────────────────────────────────────────────────


def _pos(query: str = "q", response: str = "r") -> FeedbackSample:
    return FeedbackSample(query=query, response=response, feedback="positive")


def _neg(
    query: str = "q",
    response: str = "bad r",
    corrected: str | None = None,
) -> FeedbackSample:
    return FeedbackSample(
        query=query,
        response=response,
        feedback="negative",
        corrected_response=corrected,
    )


# ── Example generation ────────────────────────────────────────────────────────


class TestExampleGeneration:
    def test_positive_sample_becomes_example(self) -> None:
        g = TrainingDataGenerator()
        examples = g.generate_examples([_pos("What is Python?", "A language.")])
        assert len(examples) == 1

    def test_positive_response_is_target(self) -> None:
        g = TrainingDataGenerator()
        ex = g.generate_examples([_pos("q", "accepted answer")])[0]
        assistant_msgs = [m for m in ex.messages if m["role"] == "assistant"]
        assert assistant_msgs[0]["content"] == "accepted answer"

    def test_negative_with_correction_uses_corrected(self) -> None:
        g = TrainingDataGenerator()
        s = _neg("q", "bad", "good answer")
        ex = g.generate_examples([s])[0]
        assert ex.messages[-1]["content"] == "good answer"

    def test_negative_no_correction_is_skipped(self) -> None:
        g = TrainingDataGenerator()
        examples = g.generate_examples([_neg("q", "bad")])
        assert examples == []

    def test_example_has_system_user_assistant_roles(self) -> None:
        g = TrainingDataGenerator()
        ex = g.generate_examples([_pos()])[0]
        roles = [m["role"] for m in ex.messages]
        assert "system" in roles
        assert "user" in roles
        assert "assistant" in roles

    def test_source_feedback_id_set(self) -> None:
        g = TrainingDataGenerator()
        s = _pos()
        ex = g.generate_examples([s])[0]
        assert ex.source_feedback_id == s.id

    def test_context_injected_into_single_system_message(self) -> None:
        """Context must be merged into the system prompt — not added as a second system turn."""
        g = TrainingDataGenerator()
        s = FeedbackSample(
            query="q", response="r", feedback="positive", context=["chunk A", "chunk B"]
        )
        ex = g.generate_examples([s])[0]
        system_msgs = [m for m in ex.messages if m["role"] == "system"]
        # Exactly one system message
        assert len(system_msgs) == 1
        # Context content is inside that single message
        assert "Context:" in system_msgs[0]["content"]
        assert "chunk A" in system_msgs[0]["content"]
        assert "chunk B" in system_msgs[0]["content"]

    def test_no_context_produces_exactly_three_messages(self) -> None:
        g = TrainingDataGenerator()
        ex = g.generate_examples([_pos()])[0]
        assert len(ex.messages) == 3

    def test_context_produces_exactly_three_messages(self) -> None:
        """After merge-fix, context examples still have 3 messages (system, user, assistant)."""
        g = TrainingDataGenerator()
        s = FeedbackSample(query="q", response="r", feedback="positive", context=["ctx"])
        ex = g.generate_examples([s])[0]
        assert len(ex.messages) == 3

    def test_custom_system_prompt(self) -> None:
        g = TrainingDataGenerator(system_prompt="Be concise.")
        ex = g.generate_examples([_pos()])[0]
        assert ex.messages[0]["content"].startswith("Be concise.")

    def test_mixed_samples_correct_count(self) -> None:
        g = TrainingDataGenerator()
        samples = [
            _pos(),
            _neg("q2", corrected="fixed"),
            _neg("q3"),  # no correction → skipped
        ]
        examples = g.generate_examples(samples)
        assert len(examples) == 2


# ── JSONL output ──────────────────────────────────────────────────────────────


class TestJSONLOutput:
    def test_jsonl_lines_are_valid_json(self) -> None:
        g = TrainingDataGenerator()
        lines = g.generate_jsonl([_pos("q", "r")])
        for line in lines:
            obj = json.loads(line)
            assert isinstance(obj, dict)

    def test_jsonl_has_messages_key(self) -> None:
        g = TrainingDataGenerator()
        lines = g.generate_jsonl([_pos()])
        for line in lines:
            obj = json.loads(line)
            assert "messages" in obj

    def test_jsonl_messages_is_list_of_dicts(self) -> None:
        g = TrainingDataGenerator()
        lines = g.generate_jsonl([_pos()])
        obj = json.loads(lines[0])
        assert isinstance(obj["messages"], list)
        assert all(isinstance(m, dict) for m in obj["messages"])

    def test_jsonl_skips_negative_without_correction(self) -> None:
        g = TrainingDataGenerator()
        lines = g.generate_jsonl([_neg()])
        assert lines == []

    def test_write_jsonl_creates_file(self) -> None:
        g = TrainingDataGenerator()
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            n = g.write_jsonl([_pos(), _neg("q2", corrected="fix")], path)
            assert n == 2
            assert os.path.exists(path)
            with open(path, encoding="utf-8") as fh:
                raw_lines = [line for line in fh.read().splitlines() if line.strip()]
            assert len(raw_lines) == 2
        finally:
            os.unlink(path)

    def test_write_jsonl_returns_count(self) -> None:
        g = TrainingDataGenerator()
        samples = [_pos(), _pos(), _neg()]  # 2 valid
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            n = g.write_jsonl(samples, path)
            assert n == 2
        finally:
            os.unlink(path)

    def test_jsonl_with_context_has_single_system_role(self) -> None:
        """OpenAI fine-tuning schema: only one system message per example."""
        g = TrainingDataGenerator()
        s = FeedbackSample(query="q", response="r", feedback="positive", context=["ctx chunk"])
        lines = g.generate_jsonl([s])
        obj = json.loads(lines[0])
        system_msgs = [m for m in obj["messages"] if m["role"] == "system"]
        assert len(system_msgs) == 1


# ── Preference pairs ──────────────────────────────────────────────────────────


class TestPreferencePairs:
    def test_correction_creates_preference_pair(self) -> None:
        g = TrainingDataGenerator()
        s = _neg("q", "bad", "good")
        pairs = g.generate_preference_pairs([s])
        assert len(pairs) == 1
        p = pairs[0]
        assert p.prompt == "q"
        assert p.chosen == "good"
        assert p.rejected == "bad"

    def test_cross_sample_pair_positive_vs_negative(self) -> None:
        g = TrainingDataGenerator()
        pos = _pos("same query", "great answer")
        neg = _neg("same query", "bad answer")  # no correction
        pairs = g.generate_preference_pairs([pos, neg])
        cross = [p for p in pairs if p.chosen == "great answer" and p.rejected == "bad answer"]
        assert len(cross) == 1

    def test_no_pairs_when_only_positives(self) -> None:
        g = TrainingDataGenerator()
        samples = [_pos("q1"), _pos("q2")]
        pairs = g.generate_preference_pairs(samples)
        assert pairs == []

    def test_no_pairs_when_only_negatives_no_correction(self) -> None:
        g = TrainingDataGenerator()
        samples = [_neg("q1"), _neg("q2")]
        pairs = g.generate_preference_pairs(samples)
        assert pairs == []

    def test_deduplication_prevents_duplicate_pairs(self) -> None:
        g = TrainingDataGenerator()
        s1 = _pos("q", "answer")
        s2 = _pos("q", "answer")
        neg = _neg("q", "bad")
        pairs = g.generate_preference_pairs([s1, s2, neg])
        matching = [p for p in pairs if p.chosen == "answer" and p.rejected == "bad"]
        assert len(matching) == 1

    def test_source_ids_are_set(self) -> None:
        g = TrainingDataGenerator()
        s = _neg("q", "bad", "good")
        pairs = g.generate_preference_pairs([s])
        assert pairs[0].source_ids != ("", "")

    def test_write_preference_jsonl(self) -> None:
        g = TrainingDataGenerator()
        samples = [_neg("q", "bad", "good")]
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            n = g.write_preference_jsonl(samples, path)
            assert n == 1
            with open(path, encoding="utf-8") as fh:
                obj = json.loads(fh.readline())
            assert "prompt" in obj
            assert "chosen" in obj
            assert "rejected" in obj
        finally:
            os.unlink(path)

    def test_max_pairs_per_query_cap(self) -> None:
        """max_pairs_per_query must limit cross-sample pair explosion."""
        g = TrainingDataGenerator(max_pairs_per_query=3)
        # 10 positives x 10 negatives = 100 possible pairs per query
        positives = [_pos("same-q", f"good-{i}") for i in range(10)]
        negatives = [_neg("same-q", f"bad-{i}") for i in range(10)]
        pairs = g.generate_preference_pairs(positives + negatives)
        # Should be capped at 3 cross-sample pairs plus any correction pairs
        cross_pairs = [p for p in pairs if p.chosen.startswith("good-")]
        assert len(cross_pairs) <= 3
