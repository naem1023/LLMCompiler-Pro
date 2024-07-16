import os
from abc import ABC, abstractmethod
from typing import Any

from elasticsearch import AsyncElasticsearch
from elasticsearch.exceptions import NotFoundError
from logzero import logger


class ElasticsearchInterface(ABC):
    _embedding_model: str = "text-embedding-3-large"
    _dims: int = 3072

    def __init__(
        self,
        user: str = None,
        password: str = None,
        host: str = None,
        port: int = None,
        embedding_model: str = None,
        dims: int = None,
    ):
        """
        Interface of ElasticSearch to define how to search.

        :param user: Elasticsearch user. If not provided, it will be read from the environment variable ES_USER.
        :param password: Elasticsearch password. If not provided, it will be read from the environment variable ES_PASSWORD.
        :param host: Elasticsearch host. If not provided, it will be read from the environment variable ES_HOST.
        """
        user = user if user else os.environ.get("ES_USER")
        password = password if password else os.environ.get("ES_PASSWORD")
        host = host if host else os.environ.get("ES_URL")
        self._embedding_model = (
            embedding_model if embedding_model else self._embedding_model
        )
        self._dims = dims if dims else self._dims

        assert user is not None, "Elasticsearch user is not provided."
        assert password is not None, "Elasticsearch password is not provided."
        assert host is not None, "Elasticsearch host is not provided."

        logger.debug(
            f"Connecting to Elasticsearch at {host} with user {user} /w pwd: {password}"
        )

        self.client = AsyncElasticsearch([host], basic_auth=(user, password))

    async def index_exists(self, index_name: str) -> bool:
        """
        Check if the given index exists in Elasticsearch.

        :param index_name: Name of the index to check.
        :return: True if the index exists, False otherwise.
        """
        try:
            res = await self.client.indices.exists(index=index_name)
            return res.body
        except NotFoundError:
            return False
        except Exception as e:
            logger.error(f"Unexpected error on index_exists: {e}")
            raise e

    @abstractmethod
    async def search(
        self, query: str, index_name: str, top_k: int = 20
    ) -> Any:  # Can't define the specific type of return
        """
        Perform a search in Elasticsearch.

        :param query: A string containing the query to search.
        :param index_name: Name of the index to search.
        :param top_k: Number of results to return.
        :return: Any data type containing the search result.
        """
        raise NotImplementedError("This method needs to be implemented by subclasses.")

    @abstractmethod
    async def search_from_target_data(
        self, query: str, index_name: str, data: Any, top_k: int = 20
    ) -> Any:  # Can't define the specific type of return
        """
        Perform a search in Elasticsearch from the target data.

        :param query: A string containing the query to search.
        :param index_name: Name of the index to search.
        :param data: The data to search.
        :param top_k: Number of results to return.
        :return: Any data type containing the search result.
        """
