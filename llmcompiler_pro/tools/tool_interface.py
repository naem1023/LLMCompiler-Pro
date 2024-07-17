from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from llmcompiler_pro.schema.common import Language
from llmcompiler_pro.schema.tool_calls import OpenAPIDocument


class Tool(ABC):
    """
    Abstract base class defining the common interface for all tools.

    This class provides a standardized structure for creating tools that can be
    used within a larger system. Each tool should have a name, optional OpenAPI
    documentation, and a defined schema.
    """

    name: str
    description: str
    _tool_schema: Dict[str, Any]
    doc: Optional[OpenAPIDocument] = None
    language: Language

    @property
    def tool_schema(self) -> Dict[str, Any]:
        """
        Get the schema definition for this tool.

        :return: A dictionary representing the tool's schema.
        """
        return self._tool_schema

    @tool_schema.setter
    def tool_schema(self, schema: Dict[str, Any]):
        """
        Set the schema definition for this tool.

        :param schema: A dictionary representing the tool's schema.
        """
        self._tool_schema = schema

    @abstractmethod
    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """
        Execute the tool's main functionality.

        This method should be implemented by all concrete tool classes.
        The specific parameters and return type may vary depending on the tool.

        :param args: Positional arguments passed to the tool.
        :param kwargs: Keyword arguments passed to the tool.
        :return: The result of the tool's operation. The specific type depends on the tool implementation.
        """
        ...

    def __str__(self) -> str:
        """
        Return a string representation of the tool.

        :return: A string containing the tool's name and schema.
        """
        return f"Tool(name='{self.name}', schema={self._tool_schema})"

    def __repr__(self) -> str:
        """
        Return a detailed string representation of the tool.

        :return: A string containing detailed information about the tool.
        """
        return f"Tool(name='{self.name}', doc={self.doc}, schema={self._tool_schema})"
