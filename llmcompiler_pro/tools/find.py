from typing import Any, Dict, Optional

from logzero import logger

from llmcompiler_pro.schema.common import ModelType
from llmcompiler_pro.schema.tool_calls import OpenAPIDocument

from .api.apis import get_openai_request_json, get_openapi_documents_and_tool_schema
from .browse import WebContentAnalyzer
from .tool_interface import Tool


async def fetch_api_documentation(
    api_document_url: str,
) -> Dict[str, Dict[str, Dict | OpenAPIDocument]]:
    """
    Fetch and process API documentation for a given identifier.

    :param api_document_url: The URL to fetch the API documentation from, like swagger or openapi.
    :return: A dictionary containing processed API documentation and tool schemas.
    """
    raw_specs: list[Dict] = await get_openai_request_json(api_document_url)
    logger.debug(f"Number of API specifications retrieved: {len(raw_specs)}")
    return get_openapi_documents_and_tool_schema(raw_specs)


def construct_tool(
    identifier: str,
    spec: Dict[str, Any],
    documentation: OpenAPIDocument,
    model_type: ModelType = ModelType.openai,
) -> Tool:
    """
    Construct a Tool object based on the provided specifications.

    :param identifier: The identifier of the tool.
    :param spec: The tool specification dictionary.
    :param documentation: The OpenAPI documentation for the tool.
    :param model_type: The type of language model to use for the tool.
    :return: A Tool object constructed from the provided specifications.

    Example of spec:
    {
      "type": "function",
      "function": {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {
              "type": "string",
              "description": "The city and state, e.g. San Francisco, CA"
            },
            "unit": {
              "type": "string",
              "enum": ["celsius", "fahrenheit"]
            }
          },
          "required": ["location"]
        }
      }
    }
    """
    if model_type == ModelType.openai:
        logger.debug(f"Documentation: {documentation}")
        # TODO: Prepare the real tool constructor, not abstract interface of tool.
        return Tool()
    else:
        raise ValueError(f"Model type {model_type} is not supported")


async def find_tool(
    identifier: str, api_document_url: str | None = None
) -> Optional[Tool]:
    """
    Find and construct a Tool object based on the provided identifier.

    This function searches for the tool in the API documentation and returns
    the corresponding Tool object. If the identifier matches a special case
    (e.g., web search and browse), it returns a specialized tool.

    :param identifier: The identifier of the tool to find, like api path or api name.
    :param api_document_url: The URL to fetch the API documentation from, like swagger or openapi.
    :return: A Tool object if found, None otherwise.
    """
    if identifier == WebContentAnalyzer.name:
        return WebContentAnalyzer()

    # TODO: Implement the real tool finding logic.

    if api_document_url:
        api_docs = await fetch_api_documentation(api_document_url)

        if identifier in api_docs:
            tool_spec = api_docs[identifier]["tool_schema"]
            documentation = api_docs[identifier]["doc"]
            return construct_tool(identifier, tool_spec, documentation)
    # If api_document_url is None and identifier isn't the WebContentAnalyzer, return None.
    else:
        logger.warning(f"No tool found for identifier: {identifier}")
        return None
