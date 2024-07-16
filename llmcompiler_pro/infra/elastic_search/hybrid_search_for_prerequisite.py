import uuid
from itertools import chain
from typing import Any

from elastic_transport._response import ObjectApiResponse
from logzero import logger
from openai import AsyncOpenAI
from tqdm.auto import tqdm

from llmcompiler_pro.schema.retrieval import ToolCallType
from llmcompiler_pro.tools.api.prerequisite.interface import (
    PrerequisiteCommonElement,
    PrerequisiteOutput,
)
from llmcompiler_pro.utils.batch import get_batches

from .elastic_search_interface import ElasticsearchInterface


def get_es_query(query: str, query_embedding: list[float], top_k: int) -> dict:
    return {
        "query": {
            "bool": {
                "should": [
                    {
                        "multi_match": {
                            "query": query,
                            "fields": ["title", "label"],
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


class ElasticsearchHybridSearchForPrerequisite(ElasticsearchInterface):
    """
    ElasticSearch with hybrid search capabilities for Prerequisite response of APIs.
    """

    async def generate_embeddings(
        self, data: PrerequisiteOutput, batch_size: int = 2048
    ) -> list[list[float]]:
        """Generate embeddings for the given response of prerequisite.
        PrerequisiteElement is transformed to string and then generate the embeddings.
        """
        client = AsyncOpenAI()
        embeddings: list[list[float]] = []

        def transform_to_string(element: PrerequisiteCommonElement) -> str:
            return element.to_string()

        target_data = [transform_to_string(element) for element in data.array]

        for i in tqdm(range(0, len(data.array), batch_size)):
            res = await client.embeddings.create(
                model=self._embedding_model,
                input=target_data[i : min(i + batch_size, len(target_data))],
            )
            embeddings.extend([d.embedding for d in res.data])
        return embeddings

    async def insert(self, index_name: str, data: PrerequisiteOutput, data_id: str):
        """
        Store Prerequisite response to the ElasticSearch.
        1. Transform prerequisite output to a string.
        2. Generate a list of embeddings of the prerequisite response of 1.
        3. Insert prerequisite response to the index with the embeddings.
        """
        logger.debug(f"Generating embeddings for prerequisite data: {len(data.array)}")
        embeddings: list[list[float]] = await self.generate_embeddings(data)

        logger.debug(
            f"Inserting {len(data.array)} prerequisite response to {index_name}."
        )

        def get_operations_of_insert_documents():
            operations = []
            for i in range(len(data.array)):
                operations.append(
                    [
                        {"index": {"_index": index_name}},
                        {
                            "title": data.array[i].title,
                            "label": data.array[i].label,
                            "id": data.array[i].id,
                            "__id": data_id,
                            "content": data.array[i].content,
                            "embedding": embeddings[i],
                            "type": ToolCallType.openai.value,  # Using openai embedding as default embedding model
                        },
                    ]
                )
            return operations

        for operation in get_batches(get_operations_of_insert_documents(), 200):
            await self.client.bulk(refresh=False, operations=list(chain(*operation)))

    async def create_index(self, index_name: str):
        """Create an index with the given name for apis index"""
        await self.client.indices.create(
            index=index_name,
            mappings={
                "properties": {
                    "title": {"type": "text"},
                    "label": {"type": "text"},
                    "id": {"type": "keyword", "index": False},
                    "__id": {"type": "keyword", "index": False},
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

    async def _prepare_to_search(
        self, index_name: str, data: PrerequisiteOutput, data_id: str
    ):
        """
        Prepare to search by creating an temporal index and storing the response of the prerequisite apis.

        :param index_name: Name of the index to search.
        :param data: PrerequisiteOutput data.
        :param data_id: Unique ID for the data.
        :return:
        """
        # Valid if the index is already exists.
        if not await self.index_exists(index_name):
            # If the index does not exist, make a new index and upload the data.
            await self.create_index(index_name)

            # Generate embeddings and insert to the index
            await self.insert(index_name, data, data_id)
            logger.debug(f"Index {index_name} is created and stored the apis")
        else:
            logger.debug(f"Index {index_name} already exists.")

    async def _delete_upload_data(self, index_name: str, data_id: str):
        """
        Delete all the data which __id is the same as the given data_id.

        :param index_name: Name of the index to search.
        :param data_id: Unique ID for the data.
        """
        await self.client.delete_by_query(
            index=index_name,
            body={"query": {"match": {"__id": data_id}}},
            refresh=True,
        )

    async def search(self, query: str, index_name: str, top_k: int = 20) -> Any:
        raise NotImplementedError("This method is not implemented.")

    async def search_from_target_data(
        self, query: str, index_name: str, data: PrerequisiteOutput, top_k: int = 20
    ) -> PrerequisiteOutput:
        """
        Perform a hybrid search in Elasticsearch.

        :param query: A dictionary containing Elasticsearch query DSL.
        :param index_name: Name of the index to search.
        :param data: The data to search.
        :param top_k: Number of results to return.
        :return: A dictionary containing the search result.

        # TODO: uuid can't be unique. Implement the unique id for the data.
        """
        data_id: str = str(uuid.uuid4())

        # Valid the index and insert the data
        await self._prepare_to_search(index_name, data, data_id)

        # Generate the embeddings for the data.
        # The order of the data is the same as the order of the embeddings.
        query_embedding = await self.get_query_embedding(query)

        es_query = get_es_query(query, query_embedding, top_k)

        # Start hybrid searching
        res: ObjectApiResponse = await self.client.search(index=index_name, **es_query)
        resp = res["hits"]["hits"]

        results = []
        for hit in resp:
            llm_type = hit["_source"]["type"]
            if llm_type == ToolCallType.openai.value:
                results.append(
                    PrerequisiteCommonElement(
                        id=hit["_source"]["id"],
                        title=hit["_source"]["title"],
                        label=hit["_source"]["label"],
                        content=hit["_source"]["content"],
                    )
                )
            else:
                raise ValueError(f"Unknown type of document is given: {llm_type}")

        # Remove the index
        await self._delete_upload_data(index_name, data_id)

        return PrerequisiteOutput(array=results)
