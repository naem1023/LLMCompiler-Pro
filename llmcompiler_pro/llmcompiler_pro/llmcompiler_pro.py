from typing import List, Union

from langchain_core.messages.ai import AIMessage
from langchain_core.messages.human import HumanMessage
from langchain_core.messages.system import SystemMessage
from langchain_core.tracers.base import AsyncBaseTracer, BaseTracer
from logzero import logger

from llmcompiler_pro.schema.common import Language, LLMCompilerProRequest

from .callbacks import AsyncStatsCallbackHandler
from .final_answer.generate import generate_final_answer
from .plan_and_fetch import generate_plan_and_fetch
from .planner.planner import Planner


class LLMCompilerPro:
    """LLMCompiler Engine."""

    def __init__(
        self,
        request_params: LLMCompilerProRequest,
        tools: list[dict],
        callbacks: list[BaseTracer | AsyncBaseTracer],
    ) -> None:
        """
        :param request_params: The request parameters to configure the framework.
        :param tools: List of tool schema to be used in the function calling. Only predefined tools are allowed.
        :param callbacks: List of tracer callbacks to handle LLM streaming response.
        """
        self.request_params = request_params
        self.max_replan: int = request_params.max_replan
        self.model_name: str = request_params.model_name
        self.session_id: str = request_params.session_id
        self.language: Language = request_params.language
        self.tools: list[dict] = tools

        self.planner_callback: AsyncStatsCallbackHandler = AsyncStatsCallbackHandler(
            stream=True
        )
        self.executor_callback: AsyncStatsCallbackHandler = AsyncStatsCallbackHandler(
            stream=True
        )
        # self.task_fetching_unit_callback: AsyncStatsCallbackHandler = AsyncStatsCallbackHandler(stream=True)
        self.outer_callbacks: list[BaseTracer | AsyncBaseTracer] = callbacks

        self.planner: Planner = Planner(self.model_name)
        self.agent_scratchpad: str = ""
        self._answers = []

    def prepare_planner_callbacks(self) -> list:
        if self.outer_callbacks:
            return [self.planner_callback] + self.outer_callbacks
        else:
            return [self.planner_callback]

    def prepare_final_answer_callbacks(self) -> list:
        if self.outer_callbacks:
            return [self.executor_callback] + self.outer_callbacks
        else:
            return [self.executor_callback]

    def update_agent_scraptchpad(self, tasks):
        self.agent_scratchpad += "\n\n"
        self.agent_scratchpad += "".join(
            [
                task.get_though_action_observation(
                    include_action=True, include_thought=True
                )
                for task in tasks.values()
                if not task.is_join
            ]
        )
        self.agent_scratchpad = self.agent_scratchpad.strip()

    def update_answer(self, answer: str):
        self._answers.append(answer)

    @property
    def answers(self):
        return self._answers

    async def acall(
        self,
        user_input: str,
        chat_history: List[Union[SystemMessage, HumanMessage, AIMessage]],
    ) -> list[str]:
        """Run the llm compiler.

        Plan:
        1. {first sub plan}
        2. {second sub plan}
        ...
        n. {last sub plan}

        Behaviors:
        1. Generate plan and receive the plan streaming.
        2. When the stream chunk is sub plan, task fetching unit execute the function calling to make subtask.
        3. Subtask run parallel and return the result.
        4. When all the subtask is finished, join the observations and generate the join result.
        5. Determine whether to replan or not via join result.

        """
        logger.info(f"user_input: {user_input}")
        logger.info(f"chat_history: {chat_history}")

        for i in range(self.max_replan):
            # Generate plan and start task fetching
            plan, tasks = await generate_plan_and_fetch(
                self.planner,
                user_input,
                chat_history,
                self.model_name,
                self.tools,
                self.language,
                self.session_id,
                self.prepare_planner_callbacks(),
            )

            self.update_agent_scraptchpad(tasks)

            res = await generate_final_answer(
                self.model_name,
                user_input,
                plan,
                self.agent_scratchpad,
                self.language,
                self.prepare_final_answer_callbacks(),
            )

            if isinstance(res, str):
                self.update_answer(res)
                break
            elif isinstance(res, tuple):
                terminate, answer = res
                self.update_answer(answer)

                if not terminate:
                    logger.info("Break out of replan loop.")
                    break

            if i == self.max_replan - 1:
                logger.info("Reached max replan limit.")

        # Close all the streaming handlers
        if self.outer_callbacks:
            for callback in self.outer_callbacks:
                await callback.on_close()

        return self.answers
