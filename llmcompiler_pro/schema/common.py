import uuid
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Language(Enum):
    Korean = "Korean"
    English = "English"


class LLMCompilerProRequest(BaseModel):
    # tools: Sequence[],
    model_name: str
    max_replan: int
    session_id: str = Field(default=str(uuid.uuid4()))
    language: Language = (Field(default=Language.Korean),)
    callbacks: list = (None,)
    browsing_tool_callbacks: list = (None,)


class ModelType(Enum):
    """Model type of LLM"""

    openai = "openai"
    claude = "claude"
    gemini = "gemini"


class Step(BaseModel):
    """Step."""

    value: str
    """The value."""


class Plan(BaseModel):
    """Plan."""

    steps: list[Step]
    """The steps."""


class StepResponse(BaseModel):
    """Step response."""

    response: str
    """The response."""


class CostSetting(str, Enum):
    large = "large"
    medium = "medium"
    small = "small"


class PersonalizationSetting(BaseModel):
    name: str
    age: int
    age_range: str = Field(..., description="age range like '20대")
    job: str
    gender: str
    interest: Optional[str | list[str]] = Field(
        None, description="interest like '마케팅', '프로그래밍'"
    )
    country: Optional[str] = Field(None, description="country name like 'Korean'")
    region: Optional[str] = Field(None, description="region like 'North East Asia'")
    city: Optional[str] = Field(None, description="country name like 'Daejeon'")
