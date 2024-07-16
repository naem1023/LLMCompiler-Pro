from llmcompiler_pro.infra.elastic_search.hybrid_search import ElasticsearchHybridSearch
from llmcompiler_pro.schema.retrieval import RetrievedAPI


async def retrieve_top_k_apis(query: str, index: str) -> list[RetrievedAPI]:
    """Retrieve the top k APIs from the operation list.

    Features:
    - Using embedding similarity to retrieve the top k APIs.
    - Request bulk request for generating embeddings.

    """
    es_client = ElasticsearchHybridSearch()
    return await es_client.search(query, index)
