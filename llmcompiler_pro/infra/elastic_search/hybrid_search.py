from itertools import chain

from elastic_transport._response import ObjectApiResponse
from logzero import logger
from openai import AsyncOpenAI
from tqdm.auto import tqdm

from llmcompiler_pro.schema.retrieval import RetrievedAPI, ToolCallType
from llmcompiler_pro.tools.api.apis import (
    get_openai_request_json,
    transform_to_openai_function_calling_type,
)
from llmcompiler_pro.utils.batch import get_batches

from ...schema.tool_calls import AnthropicToolParam, OpenAIChatCompletionTool
from .elastic_search_interface import ElasticsearchInterface


def get_es_query(query: str, query_embedding: list[float], top_k: int) -> dict:
    return {
        "query": {
            "bool": {
                "should": [
                    {
                        "multi_match": {
                            "query": query,
                            "fields": ["function_name", "function_description"],
                            "type": "most_fields",
                        }
                    }
                ]
            }
        },
        "knn": {
            "field": "embedding",
            "query_vector": query_embedding,
            "k": top_k,
            "num_candidates": top_k * 10,
            "boost": 0.8,
        },
    }
    # es_query["rank"] = {"rrf": {"window_size": max(topk, 100)}}


class ElasticsearchHybridSearch(ElasticsearchInterface):
    """
    ElasticSearch with hybrid search capabilities for APIs.
    """

    @staticmethod
    def transform_to_es_document(
        docs: list[OpenAIChatCompletionTool | AnthropicToolParam],
    ) -> list[dict]:
        """
        Objective: Transform the documents to only contain searchable fields.
        - Only extract "function" key.
        - Remove parameter information
        """
        # TODO: How to check all the type of documents?
        if isinstance(docs[0], OpenAIChatCompletionTool):
            result = []
            for doc in docs:
                function_dict: dict = dict(
                    doc.function
                )  # Convert pydantic to dictionary
                if function_dict.get("parameters"):
                    function_dict.pop("parameters")
                result.append(function_dict)
            return result
        elif isinstance(docs[0], AnthropicToolParam):
            raise NotImplementedError("AnthropicToolParam is not implemented yet.")
        else:
            raise ValueError(f"Unknown type of document is given: {type(docs[0])}")

    @staticmethod
    def transform_docs_to_string(docs: list[dict]) -> list[str]:
        """
        Transform the documents to a list of strings.

        :param docs: A list of dictionaries containing the documents.
        :return: A list of strings.
        """
        return [str(doc) for doc in docs]

    async def generate_embeddings(
        self, tools: list[dict], batch_size: int = 2048
    ) -> list[list[float]]:
        """Generate embeddings for the given tools."""
        tools_str: list[str] = self.transform_docs_to_string(tools)

        client = AsyncOpenAI()

        embeddings: list[list[float]] = []
        for i in tqdm(range(0, len(tools), batch_size)):
            res = await client.embeddings.create(
                model=self._embedding_model,
                input=tools_str[i : min(i + batch_size, len(tools_str))],
            )
            embeddings.extend([d.embedding for d in res.data])
        return embeddings

    async def store_apis(self, index_name: str):
        """
        Store openai type documents of apis to ElasticSearch
        1. Load OpenAI Json schema of APIs
        2. Convert OpenAI Json schema to dictionary
        3. Generate the embeddings of OpenAI Json Schema using title and description.
        4. Insert the documents to ElasticSearch with the embeddings.

        TODO: It only consider OpenAIChatCompletionTool. Need to consider AnthropicToolParam.
        """
        apis_openai_json_schema: list[dict] = await get_openai_request_json()
        tools: list[
            OpenAIChatCompletionTool
        ] = transform_to_openai_function_calling_type(apis_openai_json_schema)
        tools_dict: list[dict] = self.transform_to_es_document(tools)

        logger.debug(f"Generating embeddings for {len(tools)} apis.")
        embeddings = await self.generate_embeddings(tools_dict)

        logger.debug(f"Inserting {len(tools)} apis to {index_name}.")

        def get_operations_of_insert_documents():
            operations = []
            for i in range(len(tools)):
                logger.debug(
                    f"type of tools_dict[i]: {type(tools_dict[i])}, {tools_dict[i]}"
                )
                operations.append(
                    [
                        {"index": {"_index": index_name}},
                        {
                            "function_name": tools[i].function.name,
                            "function_description": tools[i].function.description,
                            "content": tools_dict[i],
                            "embedding": embeddings[i],
                            "type": ToolCallType.openai.value,  # Using openai embedding as default embedding model
                        },
                    ]
                )
            return operations

        for operation in get_batches(get_operations_of_insert_documents(), 200):
            await self.client.bulk(refresh=False, operations=list(chain(*operation)))

    async def create_index(self, index_name: str):
        """Create an index with the given name for apis."""
        await self.client.indices.create(
            index=index_name,
            mappings={
                "properties": {
                    "function_name": {"type": "text"},
                    "function_description": {"type": "text"},
                    "content": {"type": "object"},
                    "embedding": {
                        "type": "dense_vector",
                        "dims": self._dims,
                        "index": True,
                        "similarity": "cosine",
                        "index_options": {
                            "type": "hnsw",
                            "m": 16,
                            "ef_construction": 100,
                        },
                    },
                    "type": {"type": "keyword"},
                }
            },
            settings={
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "analysis": {
                    "analyzer": {
                        "nori_analyzer": {"type": "nori"},
                        "kuromoji_analyzer": {"type": "kuromoji"},
                    }
                },
            },
        )

    async def get_query_embedding(self, query: str) -> list[float]:
        client = AsyncOpenAI()
        res = await client.embeddings.create(model=self._embedding_model, input=query)
        return res.data[0].embedding

    async def search(
        self, query: str, index_name: str, top_k: int = 20
    ) -> list[RetrievedAPI]:
        """
        Perform a hybrid search in Elasticsearch.

        :param query: A dictionary containing Elasticsearch query DSL.
        :param index_name: Name of the index to search.
        :param top_k: Number of results to return.
        :return: A dictionary containing the search result.
        """

        # Valid if the index is already exists.
        if not await self.index_exists(index_name):
            # If the index does not exist, make a new index and upload the data.
            await self.create_index(index_name)
            await self.store_apis(index_name)
            logger.debug(f"Index {index_name} is created and stored the apis.")
        else:
            logger.debug(f"Index {index_name} already exists.")

        query_embedding: list[float] = await self.get_query_embedding(query)

        es_query: dict = get_es_query(query, query_embedding, top_k)

        # Start hybrid searching
        res: ObjectApiResponse = await self.client.search(index=index_name, **es_query)
        resp = res["hits"]["hits"]

        results = []
        for hit in resp:
            llm_type = hit["_source"]["type"]
            if llm_type == ToolCallType.openai.value:
                results.append(
                    RetrievedAPI(
                        function_name=hit["_source"]["function_name"],
                        function_description=hit["_source"]["function_description"],
                        content=hit["_source"]["content"],
                        embedding=hit["_source"]["embedding"],
                        type=ToolCallType.from_value(llm_type),
                        score=hit["_score"],
                    )
                )
            else:
                raise ValueError(f"Unknown type of document is given: {llm_type}")
        return results
