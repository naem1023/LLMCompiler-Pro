from typing import Optional

from pydantic import BaseModel, Field


class OpenAIFunctionDefinition(BaseModel):
    name: str
    description: str

    # Can't define the type of parameters because OpenAI and Anthropic SDKs don't have a type of parameters.
    # OpenAI refernece: https://github.com/openai/openai-node/blob/22cf0362c4a72873433d10452248934672bd65ac/src/resources/shared.ts#L47
    parameters: Optional[object]


class OpenAIChatCompletionTool(BaseModel):
    """Tool Parameters of OpenAI
    This type can be used in "tool message" in OpenAI.
    """

    type: str = Field("function")
    function: OpenAIFunctionDefinition


class AnthropicInputSchema(BaseModel):
    type: str

    # Can't define the type of parameters because OpenAI and Anthropic SDKs don't have a type of parameters.
    # Anthropic reference: https://github.com/anthropics/anthropic-sdk-python/blob/main/src/anthropic/types/tool_param.py#L14
    properties: Optional[object]


class AnthropicToolParam(BaseModel):
    """Tool Parameters of Anthropic
    This type can be used in "tool message" in Anthropic.
    """

    input_schema: AnthropicInputSchema
    name: str
    description: str


class SeparatedParams(BaseModel):
    """Separated parameters of llm and human.
    - "llm" is the parameters for llm.
    - "human" is the parameters for human like secret key, content type.
    """

    llm: list
    human: list


class OpenAPIDocument(BaseModel):
    """OpenAPI 3.1 Document type"""

    method: str
    path: str
    name: str
    parameters: list[dict]
    separated: SeparatedParams
    output: dict | None
    description: str | None
