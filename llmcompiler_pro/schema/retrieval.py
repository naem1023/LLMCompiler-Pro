from enum import Enum

from pydantic import BaseModel


class ToolCallType(Enum):
    """
    Define the provider type of the tool parameter.
    """

    openai = "OpenAI"
    anthropic = "Anthropic"

    @classmethod
    def from_value(cls, value):
        """
        Check if the value exists in the enum and return the corresponding enum member.

        :param value: The value to check in the enum.
        :return: The enum member if the value exists, else None.
        """
        for member in cls:
            if member.value == value:
                return member
        return None


class RetrievedAPI(BaseModel):
    """
    Define the retrieved content of APIs from the hybrid searching of ElasticSearch.
    """

    function_name: str
    function_description: str
    content: dict
    embedding: list[float]
    type: ToolCallType
    score: float
