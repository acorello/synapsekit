"""SynapseKit continuous fine-tuning and feedback learning pipeline."""

from .ab_testing import ABTestRouter
from .cost_analysis import (
    ANTHROPIC_TRAINING_RATE_PER_M_TOKENS,
    OPENAI_TRAINING_RATE_PER_M_TOKENS,
    PROVIDER_PRICING,
    CostBenefitAnalyzer,
    InferenceProfile,
)
from .dataset import TrainingDataGenerator
from .feedback import (
    FeedbackBackend,
    FeedbackCollector,
    InMemoryFeedbackBackend,
)
from .finetune import (
    AnthropicFineTuneProvider,
    BaseFineTuneProvider,
    OpenAIFineTuneProvider,
)
from .orchestrator import ContinuousTrainer
from .rollout import AutoRolloutManager
from .types import (
    ABTestMetrics,
    ABTestResult,
    CostBenefitReport,
    FeedbackSample,
    FineTuneJob,
    PreferencePair,
    RolloutPolicy,
    RolloutState,
    TrainingExample,
)

__all__ = [
    # Types
    "FeedbackSample",
    "TrainingExample",
    "PreferencePair",
    "FineTuneJob",
    "ABTestResult",
    "ABTestMetrics",
    "RolloutPolicy",
    "RolloutState",
    "CostBenefitReport",
    # Feedback
    "FeedbackBackend",
    "FeedbackCollector",
    "InMemoryFeedbackBackend",
    # Dataset
    "TrainingDataGenerator",
    # Fine-tuning providers
    "BaseFineTuneProvider",
    "OpenAIFineTuneProvider",
    "AnthropicFineTuneProvider",
    # A/B testing
    "ABTestRouter",
    # Rollout
    "AutoRolloutManager",
    # Cost analysis
    "CostBenefitAnalyzer",
    "InferenceProfile",
    "PROVIDER_PRICING",
    "OPENAI_TRAINING_RATE_PER_M_TOKENS",
    "ANTHROPIC_TRAINING_RATE_PER_M_TOKENS",
    # Orchestrator
    "ContinuousTrainer",
]
