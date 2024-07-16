from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Collection, Optional

from langchain_community.adapters.openai import convert_message_to_dict
from langchain_core.messages import BaseMessage, HumanMessage
from logzero import logger
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
)

from llmcompiler_pro.prompt.llmcompiler_pro.subtask import subtask_query
from llmcompiler_pro.prompt_render.jinja2_render import Jinja2Render
from llmcompiler_pro.schema.common import Language, ModelType
from llmcompiler_pro.tools.find import find_tool
from llmcompiler_pro.tools.tool_interface import Tool

from . import SubTask
from .apis import get_relevant_apis
from .generate_subtasks import generate_subtasks, get_subtask_tool_prompt


def get_task_fetching_unit_prompt(observations: str) -> str:
    return Jinja2Render("llmcompiler_pro/prompt").render(
        "llmcompiler_pro",
        "task_fetching_unit_task_result.jinja2",
        context=observations,
    )


@dataclass
class Task:
    """
    Task is a single task of the plan.

    - Each task has an instruction to execute.
    - Each task has dependencies to execute.
    - After function calling execution, each task can generate subtasks. Subtasks are the subplans of the task.
    """

    idx: int
    instruction: str
    dependencies: Collection[int]
    language: Language
    thought: Optional[str] = None
    observation: Optional[str] = None
    is_join: bool = False
    sub_tasks: Optional[Collection[SubTask]] = None

    # Parameters for Code Interpreter.
    user_input: Optional[str] = None
    chat_history: Optional[list[BaseMessage]] = None
    plan: Optional[str] = None
    current_goal: Optional[str] = None  # Same as instruction
    model_name: Optional[str] = None

    def set_observation(self) -> None:
        """Build the combine observation of sub_tasks.

        Each subtask has own observation.
        Because task dependency pattern of planning, the observation should consider on other subtask execution.
        This function builds the observation for other subtasks, not using tool call id.
        """
        if self.observation is None:
            if self.sub_tasks is not None:
                observations = ""
                for i, sub_task in enumerate(self.sub_tasks):
                    # Langchain tool is based on pydantic v1, so don't use json serialization because of unicode decode.
                    logger.debug(f"Subtask {i}: {sub_task}")
                    observations += (
                        f"\nPlan task number={self.idx}, Subtask_number_{i}_Tool: {sub_task.tool.name}\nSubtask_number_{i}_Description: {sub_task.tool.description}\n"
                        f"Subtask_number_{i}_Arguments:{sub_task.args}\nSubtask_number_{i}_Observation:{sub_task.observation}\n######\n\n"
                    )
                self.observation = observations
            else:
                raise ValueError(
                    f"{self.__class__.__name__} doesn't have self.sub_tasks."
                )

    def get_though_action_observation(
        self, include_action=True, include_thought=True, include_action_idx=False
    ) -> str:
        """Return the thought, action, and observation of the task."""
        thought_action_observation = ""
        if self.thought and include_thought:
            thought_action_observation = f"Thought: {self.thought}\n"
        if include_action:
            idx = f"{self.idx}. " if include_action_idx else ""
            thought_action_observation += f"{idx}{self.instruction}"
        if self.observation is not None:
            thought_action_observation += f"Observation: \n{self.observation}"
        return thought_action_observation

    async def _build_sub_tasks(
        self,
        instruction: str,
        tool_response: ChatCompletionMessageToolCall,
        lm_type: ModelType,
    ) -> SubTask:
        """
        Build SubTask from the tool response and return it.

        :param tool_response: Single tool call response
        """
        # Find the tool using the tool name
        searched_tool: Tool | None = await find_tool(tool_response.function.name)
        if searched_tool is None:
            # Fail to find the tool using the tool name.
            # Run the fallback generation to regenerate the function calling response.
            return await self.generate_fallback_for_subtask()

        if ModelType.openai == lm_type:
            return SubTask(
                tool=searched_tool,
                args=json.loads(tool_response.function.arguments),
                observation=None,
                id=tool_response.id,
            )
        else:
            raise ValueError(f"ModelType {lm_type} is not supported")

    async def build_subtasks(
        self,
        language: Language,
        model_name: str,
        tools: list[dict],
        lm_type: ModelType,
        top_k: int = 2,
    ) -> None:
        """Build the subtasks of the task."""

        # Request function calling to generate subtasks.
        res: ChatCompletion = await generate_subtask_via_function_calling(
            self, language, model_name, tools
        )
        logger.debug(f"Generated subtasks for task {self.idx}: {res}")

        response_message = res.choices[0].message

        try:
            if hasattr(response_message, "tool_calls"):
                # Limit the parallel function calling can only execute top_k function.
                sub_tasks: list[SubTask] = [
                    await self._build_sub_tasks(self.instruction, tool, lm_type=lm_type)
                    for tool in response_message.tool_calls[
                        : min(len(response_message.tool_calls), top_k)
                    ]
                ]
                self.sub_tasks: Collection[SubTask] = sub_tasks
                logger.debug(
                    f"Generated subtasks for task {self.idx}: {self.sub_tasks}"
                )
            else:
                # If function calling decided not to execute the function, generate the fallback response.
                self.sub_tasks: Collection[SubTask] = [
                    SubTask(
                        tool=None,
                        args=None,
                        observation=response_message.content,
                    )
                ]
                logger.debug(
                    f"Generated fallback subtasks for task {self.idx}: {self.sub_tasks}"
                )
        except Exception as e:
            logger.error(f"Error in building subtasks for task {self.idx}: {e}")
            raise e

    async def generate_fallback_for_subtask(
        self,
    ) -> SubTask:
        """
        Fallback generation for generating subtasks.

        - If function calling generate the freeform string response, not tool response,
            generate the fallback response using this function.
        - This function doesn't use function calling, using the normal string response to structure the subtasks.

        TODO: Implement the retry logic or advanced fallback generation.
        """
        # _messages = [m.dict() for m in self.chat_history]
        # res: str = await generate_fallback_response_of_subtasks(
        #     _messages,
        #     self.model_name,
        #     self.language
        # )
        # TODO: handle the normal response more properly
        return SubTask(
            tool=None,
            args=None,
            observation=None,
        )


async def generate_subtask_via_function_calling(
    task: Task,
    language: Language,
    model_name: str,
    tools: list[dict],
    use_outer_tools: bool = False,
) -> ChatCompletion:
    """Generate the function calling response for generating subtasks.

    It only considers OpenAI model.

    :param task: Task object to generate subtasks.
    :param language: Language object.
    :param model_name: LLM model name to use for generating subtasks.
    :param tools: List of tool schema to use for generating subtasks.
    :param use_outer_tools: If True, use the outer tools for generating subtasks.
    :return: Output Response from Langchain Chat LLM object.
    """
    tool_prompt = get_subtask_tool_prompt(task.instruction, task.thought, language)
    messages: list[BaseMessage] = task.chat_history + [HumanMessage(tool_prompt)]
    messages: list[dict] = [convert_message_to_dict(m) for m in messages]
    if not use_outer_tools:
        tools = tools
    else:
        # Search the proper apis for the task.
        apis: list[dict] = await get_relevant_apis(
            subtask_query.format(instruction=task.instruction, thought=task.thought)
        )
        tools = tools + apis

    # Generate the subtasks for the task using function calling.
    res: ChatCompletion = await generate_subtasks(model_name, messages, tools)
    return res
