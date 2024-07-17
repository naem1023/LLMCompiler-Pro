import aiohttp

from llmcompiler_pro.schema.tool_calls import (
    OpenAIChatCompletionTool,
    OpenAIFunctionDefinition,
    OpenAPIDocument,
    SeparatedParams,
)


async def fetch_json(url):
    """Fetch a JSON file from the given URL."""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.json()


async def get_openai_request_json(url: str) -> list[dict]:
    """Get the OpenAI request JSON from the APIs."""
    data = await fetch_json(url)
    operations_list = data.get("functions", [])
    return operations_list


def transform_function_name(name: str) -> str:
    """Transform the function name to a valid OpenAI function name."""
    return name.replace(".", "-")


def pop_useless_elements(properties: dict) -> dict:
    """Pop the useless elements from the properties."""
    useless_keys = ["title", "description"]

    for key in useless_keys:
        if key in properties:
            properties.pop(key)

    return properties


def _validate_parameters(properties: list) -> dict | None:
    """Validate the parameters.

    :returns if dictionary is empty, return None. Otherwise, return the first element.
    """

    # TODO: Find a better way to handle this. It's very simple way to validate the swagger.
    # TODO: Don't ignore the operations without parameters.
    if len(properties) == 0:
        return None
    else:
        # TODO: Handle the case when there are multiple parameters.
        for p in properties:
            if len(p) != 0:
                return p


def validate_openapi_combinations(properties: dict) -> bool:
    """Function Calling of LLM API provider can't handle OpenAPI combinations.
    Before handling this issue, ignore the "oneOf" and "allOf" properties in the root level.

    Reference: https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/function-calling?hl=ko#function-parameters

    Sample Error Code of OpenAI
        openai.BadRequestError: Error code: 400 - {'error': {'message': "Invalid schema for function '...': schema must have type 'object' and not have 'oneOf'/'anyOf'/'allOf'/'enum'/'not' at the top level.", 'type': 'invalid_request_error', 'param': 'tools[0].function.parameters', 'code': 'invalid_function_parameters'}}
        ../../miniconda3/envs/llmcompiler_pro/lib/python3.12/site-packages/openai/_base_client.py:1020: BadRequestError
    """
    if "oneOf" in properties or "allOf" in properties:
        return False
    return True


def validate_property_type(properties: dict) -> bool:
    """Return True if the property type is valid.
    - OpenAI Function Calling can't handle the property type of "string" in the root level.
    openai.BadRequestError: Error code: 400 - {'error': {'message': 'Invalid schema for function \'...\': schema must be a JSON Schema of \'type: "object"\', got \'type: "string"\'.', 'type': 'invalid_request_error', 'param': 'tools[2].function.parameters', 'code': 'invalid_function_parameters'}}
    """
    if properties.get("type") == "string":
        return False
    return True


def validate_properties(properties: list) -> bool | list:
    """Validate the properties can be used in the Function Calling of LLM API provider.
    # TODO: Should accept all the apis, not ignore the exceptional cases.
    """
    if (valid_properties := _validate_parameters(properties)) is None:
        return False

    valid_properties: dict = pop_useless_elements(valid_properties)

    if not validate_openapi_combinations(
        valid_properties
    ) or not validate_property_type(valid_properties):
        return False
    return valid_properties


def transform_to_openai_function_calling_type(
    operations_list: list[dict],
) -> list[OpenAIChatCompletionTool]:
    """Transform the operations list to OpenAI function calling type.

    It only considers to provide the information of function to LLM

    Non-Goal:
    - It doesn't consider the parameters of human like secret key, content type.

    :param operations_list: The list of operations from the OpenAPI document.
    :return: The list of parameters for llm.
    """
    result = []

    for operation in operations_list:
        # Transform the function name into a valid OpenAI function name
        name: str = operation.get("name")
        properties: list = operation.get("separated").get("llm")

        # Validate the parameters
        if valid_properties := validate_properties(properties) is False:
            continue

        result.append(
            OpenAIChatCompletionTool(
                type="function",
                function=OpenAIFunctionDefinition(
                    name=name,
                    description=operation.get("description")[:1024],
                    parameters=valid_properties,
                ),
            )
        )
    return result


def get_openapi_documents_and_tool_schema(
    operations_list: list[dict],
) -> dict[str, dict[str, dict | OpenAPIDocument]]:
    """Transform the original operations list to OpenAPI Document type.

    :param operations_list: The list of operations from the OpenAPI document.
    :return : The dictionary of OpenAPI Document type include the parameters of llm and human properties.
    """
    result: dict = {}

    for operation in operations_list:
        # Transform the function name into a valid OpenAI function name
        llm_properties: list[dict] = operation.get("separated").get("llm")

        # Validate the parameters, temporarily.
        if (validation_parameters := validate_properties(llm_properties)) is False:
            continue

        result[operation.get("name")] = {
            "tool_schema": validation_parameters,
            "doc": OpenAPIDocument(
                method=operation.get("method", None),
                path=operation.get("path"),
                name=operation.get("name", None),
                parameters=operation.get("parameters", None),
                separated=SeparatedParams(
                    llm=llm_properties,
                    human=operation.get("separated").get("human", None),
                ),
                output=operation.get("output", None),
                description=operation.get("description", None),
            ),
        }
    return result
