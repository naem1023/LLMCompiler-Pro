from langchain.tools import BaseTool

from llmcompiler_pro.tools.browse.code_interpreter import CodeInterpreterTool
from llmcompiler_pro.tools.browse.searching_and_browsing import SearchAndBrowsingTool


def get_tools() -> list[BaseTool]:
    """Get tools."""
    return [SearchAndBrowsingTool(), CodeInterpreterTool()]
