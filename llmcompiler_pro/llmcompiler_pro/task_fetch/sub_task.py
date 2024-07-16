from __future__ import annotations

import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

from logzero import logger

from llmcompiler_pro.tools.tool_interface import Tool


@dataclass
class SubTask:
    """
    SubTask is a sub plan of the Task.
    Task has multiple subtasks and each subtask has a tool and arguments to run the tool.
    """

    tool: Optional[Tool]
    args: Optional[Union[List, Tuple, Dict]]
    observation: Optional[Any] = None  # Can't expect the type of tool response
    id: Optional[str] = None

    async def __call__(self, **kwargs) -> Any:
        """Run the tool and save the observation."""
        if self.tool is None:
            return None

        result = None
        try:
            logger.debug(f"Running subtask {self.id} with args: {self.args}")
            response = await self.tool(**self.args)
            logger.debug(f"Subtask {self.id} response: {len(response)}")
            self.observation = response
            result = response
        except Exception as e:
            logger.error(
                f"Error in running tool {self.tool.name}: {traceback.format_exc()}"
            )
            self.observation = str(e)
            result = str(e)
        return result
