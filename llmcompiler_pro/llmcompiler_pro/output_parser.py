import ast
import re
from typing import Any, Sequence, Union

from langchain.agents.agent import AgentOutputParser
from langchain.tools.base import StructuredTool, Tool
from langchain_core.exceptions import OutputParserException

from llmcompiler_pro.schema.common import Language

from .constants import ACTION_PATTERN, ID_PATTERN, THOUGHT_PATTERN
from .task_fetch import Task

# $1 or ${1} -> 1


def default_dependency_rule(idx, args: str):
    matches = re.findall(ID_PATTERN, args)
    numbers = [int(match) for match in matches]
    return idx in numbers


class PlanParser(AgentOutputParser, extra="allow"):
    """Planning output parser."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def parse(self, text: str) -> dict[int, Task]:
        """
        1. search("Ronaldo number of kids") -> 1, "search", '"Ronaldo number of kids"'
        """
        pattern = rf"(?:{THOUGHT_PATTERN}\n)?{ACTION_PATTERN}"
        matches = re.findall(pattern, text)

        graph_dict = {}

        for match in matches:
            # idx = 1, function = "search", args = "Ronaldo number of kids"
            # thought will be the preceding thought, if any, otherwise an empty string
            thought, idx, instruction = match
            idx = int(idx)

            task = instantiate_task(idx=idx, thought=thought, instruction=instruction)

            graph_dict[idx] = task
            if task.is_join:
                break

        return graph_dict


# Helper functions


def _parse_llm_compiler_action_args(args: str) -> list[Any]:
    """Parse arguments from a string."""
    # This will convert the string into a python object
    # e.g. '"Ronaldo number of kids"' -> ("Ronaldo number of kids", )
    # '"I can answer the question now.", [3]' -> ("I can answer the question now.", [3])
    if args == "":
        return ()
    try:
        args = ast.literal_eval(args)
    except Exception:
        args = args
    if not isinstance(args, list) and not isinstance(args, tuple):
        args = (args,)
    return args


def _find_tool(
    tool_name: str, tools: Sequence[Union[Tool, StructuredTool]]
) -> Union[Tool, StructuredTool]:
    """Find a tool by name.

    Args:
        tool_name: Name of the tool to find.

    Returns:
        Tool or StructuredTool.
    """
    for tool in tools:
        if tool.name == tool_name:
            return tool
    raise OutputParserException(f"Tool {tool_name} not found.")


def _get_dependencies_from_graph(idx: int, instruction: str) -> list[int]:
    """Get dependencies from a graph."""
    # define dependencies based on the dependency rule in tool_definitions.py
    dependencies = [i for i in range(1, idx) if default_dependency_rule(i, instruction)]
    return dependencies


def instantiate_task(
    idx: int,
    thought: str,
    instruction: str,
) -> Task:
    dependencies = _get_dependencies_from_graph(idx, instruction)
    return Task(
        idx=idx,
        instruction=instruction,
        dependencies=dependencies,
        thought=thought,
        language=Language.Korean,
    )
