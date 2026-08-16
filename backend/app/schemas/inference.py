from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class CustomInferenceRequest(BaseModel):
    feature_values: dict[str, float] = Field(default_factory=dict)


class InferenceResponse(BaseModel):
    domain: str
    player: str | None = None
    suitability_score: float
    predicted_probability: float
    cluster_id: int | None = None
    cluster_label: str | None = None
    archetype_role: str | None = None
    feature_values: dict[str, Any]
