from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .config import EvaluationConfig


@dataclass(slots=True)
class EvaluationResult:
    """Placeholder result structure for future model evaluation."""

    metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class LLMEvaluator:
    """Future evaluator for comparing baseline and compressed inference."""

    def __init__(self, config: EvaluationConfig) -> None:
        self.config = config

    def evaluate(self) -> EvaluationResult:
        return EvaluationResult(
            metadata={
                "status": "not_implemented",
                "model_name_or_path": self.config.model_name_or_path,
            }
        )
