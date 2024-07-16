from llmcompiler_pro.schema.retrieval import RetrievedAPI
from llmcompiler_pro.schema.tool_calls import (
    AnthropicToolParam,
    OpenAIChatCompletionTool,
)


def find_apis(
    request_docs: list[OpenAIChatCompletionTool | AnthropicToolParam],
    candidates: list[RetrievedAPI],
) -> list[OpenAIChatCompletionTool | AnthropicToolParam]:
    """
    Find Open API 3.1 documents for requesting api from retrieved apis.

    :param request_docs: The list of Open API 3.1 documents for requesting to apis.
    :param candidates: The list of retrieved apis. These only include the name and description.
    :return: The list of Open API 3.1 documents for requesting apis.
    """
    results: list[OpenAIChatCompletionTool | AnthropicToolParam] = []
    for candidate in candidates:
        for request_doc in request_docs:
            if candidate.function_name == request_doc.function.name:
                results.append(request_doc)

    return results


def transform_to_dictionary(
    operations: list[OpenAIChatCompletionTool | AnthropicToolParam],
) -> list[dict]:
    """
    Transform an array of api documents to dictionaries.

    :param operations:
    :return:

    Example Returns:
    [
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
      ],
    """
    results = []
    for operation in operations:
        if isinstance(operation, OpenAIChatCompletionTool):
            results.append(
                {
                    "type": operation.type,
                    "function": {
                        "name": operation.function.name,
                        "description": operation.function.description,
                        "parameters": operation.function.parameters,
                    },
                }
            )
        elif isinstance(operation, AnthropicToolParam):
            # Not implemented for Anthropic
            raise NotImplementedError(f"Anthropic is not yet implemented: {operation}")
        else:
            raise ValueError(f"Invalid type: {operation}")

    return results
