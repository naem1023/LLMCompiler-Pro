import uuid
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Language(Enum):
    Korean = "Korean"
    English = "English"

    @classmethod
    def from_value(cls, value: str):
        if value == cls.Korean.value:
            return cls.Korean
        elif value == cls.English.value:
            return cls.English
        else:
            raise ValueError(f"Unsupported language: {value}")


class GoogleDomainEnum(str, Enum):
    kr = "google.co.kr"
    en = "google.com"

    @classmethod
    def from_language(cls, language: Language):
        if language == Language.Korean:
            return cls.kr
        elif language == Language.English:
            return cls.en
        else:
            raise ValueError(f"Unsupported language: {language}")


class CityEnum(Enum):
    seoul = "Seoul"
    california = "California"

    @classmethod
    def from_language(cls, language: Language):
        if language == Language.Korean:
            return cls.seoul
        elif language == Language.English:
            return cls.california
        else:
            raise ValueError(f"Unsupported language: {language}")


class LanguageHLEnum(str, Enum):
    """
    Supported Google HL Parameters Reference: https://serpapi.com/google-languages
    """

    ko = "ko"
    en = "en"

    @classmethod
    def from_language(cls, language: Language):
        if language == Language.Korean:
            return cls.ko
        elif language == Language.English:
            return cls.en
        else:
            raise ValueError(f"Unsupported language: {language}")


class RegionEnum(str, Enum):
    """
    Reference: https://serpapi.com/google-countries
    """

    kr = "kr"
    us = "us"

    @classmethod
    def from_language(cls, language: Language):
        if language == Language.Korean:
            return cls.kr
        elif language == Language.English:
            return cls.us
        else:
            raise ValueError(f"Unsupported language: {language}")


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
