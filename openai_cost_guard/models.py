from datetime import UTC, datetime
from uuid import uuid4

from pydantic import BaseModel, Field


class ModelPricing(BaseModel):
    model: str
    input_per_million: float   # USD per 1M input tokens
    output_per_million: float  # USD per 1M output tokens


# Prices are USD per 1M tokens, Azure OpenAI Global Standard, verified against the Azure
# pricing page on 2026-06-08. Regional and Data Zone deployments cost roughly 10% more -
# override those via CostTracker(pricing={...}) / tracker.add_pricing(...). Prices DRIFT;
# re-verify periodically. (See the README roadmap for the planned Azure Retail Prices API
# live source that would remove this manual step.)
DEFAULT_PRICING: dict[str, ModelPricing] = {
    "gpt-4o": ModelPricing(
        model="gpt-4o",
        input_per_million=2.50,
        output_per_million=10.00,
    ),
    "gpt-4o-mini": ModelPricing(
        model="gpt-4o-mini",
        input_per_million=0.15,
        output_per_million=0.60,
    ),
    "gpt-4.1": ModelPricing(
        model="gpt-4.1",
        input_per_million=2.00,
        output_per_million=8.00,
    ),
    "gpt-4.1-mini": ModelPricing(
        model="gpt-4.1-mini",
        input_per_million=0.40,
        output_per_million=1.60,
    ),
    "gpt-4.1-nano": ModelPricing(
        model="gpt-4.1-nano",
        input_per_million=0.10,
        output_per_million=0.40,
    ),
    "gpt-4-turbo": ModelPricing(
        model="gpt-4-turbo",
        input_per_million=11.00,
        output_per_million=33.00,
    ),
    "gpt-35-turbo": ModelPricing(
        model="gpt-35-turbo",
        input_per_million=0.55,   # gpt-35-turbo-0125
        output_per_million=1.65,
    ),
    "text-embedding-3-small": ModelPricing(
        model="text-embedding-3-small",
        input_per_million=0.022,
        output_per_million=0.0,
    ),
    "text-embedding-3-large": ModelPricing(
        model="text-embedding-3-large",
        input_per_million=0.143,
        output_per_million=0.0,
    ),
    "text-embedding-ada-002": ModelPricing(
        model="text-embedding-ada-002",
        input_per_million=0.11,
        output_per_million=0.0,
    ),
}


class UsageRecord(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    input_cost: float   # USD
    output_cost: float  # USD
    total_cost: float   # USD
    endpoint: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)


class CostReport(BaseModel):
    records: list[UsageRecord] = Field(default_factory=list)
    total_cost: float = 0.0
    total_tokens: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0

    @classmethod
    def from_records(cls, records: list[UsageRecord]) -> "CostReport":
        return cls(
            records=records,
            total_cost=sum(r.total_cost for r in records),
            total_tokens=sum(r.total_tokens for r in records),
            total_prompt_tokens=sum(r.prompt_tokens for r in records),
            total_completion_tokens=sum(r.completion_tokens for r in records),
        )


class BudgetConfig(BaseModel):
    limit_usd: float
    warn_at_percent: float = 80.0  # emit a warning log when this % of budget is consumed
