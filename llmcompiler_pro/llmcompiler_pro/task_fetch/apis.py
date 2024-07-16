from llmcompiler_pro.infra.elastic_search.hybrid_search import ElasticsearchHybridSearch
from llmcompiler_pro.schema.retrieval import RetrievedAPI
from llmcompiler_pro.schema.tool_calls import OpenAIChatCompletionTool
from llmcompiler_pro.tools.api.apis import (
    get_openai_request_json,
    transform_to_openai_function_calling_type,
)
from llmcompiler_pro.tools.api.utils import find_apis, transform_to_dictionary


async def _get_openai_json() -> list[dict]:
    return await get_openai_request_json()


async def hybrid_searching(
    query: str, index_name: str = "test-local-index"
) -> list[RetrievedAPI]:
    es_client = ElasticsearchHybridSearch()
    return await es_client.search(query, index_name)


async def get_relevant_apis(query: str) -> list[dict]:
    """
    Get the relevant apis for the query.
    It only considers OpenAI function calling schema and client sdk.

    # TODO: Return Langchain Tool, not a list of dictionary.
    :param query: A freeform string to define the subtask. e.g., {task instruction} + {task thought}
    :return: A list of apis that are relevant to the query.
    """
    raw_tools: list[dict] = await _get_openai_json()
    tools: list[OpenAIChatCompletionTool] = transform_to_openai_function_calling_type(
        raw_tools
    )
    top_k_tools: list[RetrievedAPI] = await hybrid_searching(query)
    top_k_tools: list[OpenAIChatCompletionTool] = find_apis(tools, top_k_tools)

    openai_tools: list[dict] = transform_to_dictionary(top_k_tools)
    return openai_tools
