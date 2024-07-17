"""LLM Compiler Planner"""
import asyncio
import traceback
from typing import Any, Optional

from langchain.pydantic_v1 import BaseModel
from langchain_core.callbacks.base import Callbacks
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages.human import HumanMessage
from langchain_core.messages.system import SystemMessage
from langchain_openai import ChatOpenAI
from logzero import logger

from llmcompiler_pro.infra.openai import factory_openai_async_client
from llmcompiler_pro.llmcompiler_pro.constants import END_OF_PLAN
from llmcompiler_pro.llmcompiler_pro.output_parser import PlanParser
from llmcompiler_pro.prompt_render.jinja2_render import Jinja2Render

from .callback_handler import AutoAgentCallback


class Planner:
    def __init__(
        self,
        model_name: str,
    ):
        self.prompt_render = Jinja2Render("llmcompiler_pro/prompt")
        self.llm = ChatOpenAI(
            model_name=model_name,
            async_client=factory_openai_async_client().chat.completions,
            streaming=True,
        )
        self.output_parser = PlanParser()

    @staticmethod
    def _convert_to_json(obj) -> str | None:
        if isinstance(obj, BaseModel):
            # Convert json for pydantic v1 models
            return str(obj.dict())
        else:
            # Not implemented for other types, but not raise error
            return None

    def _system_prompt(self, is_replan: bool = False):
        few_shot_examples = self.prompt_render.render(
            "llmcompiler_pro",
            "planner_few_shot.jinja2",
        )
        return self.prompt_render.render(
            "llmcompiler_pro",
            "planner_system_prompt.jinja2",
            END_OF_PLAN=END_OF_PLAN,
            few_shot_examples=few_shot_examples,
        )

    def _user_prompt(self, user_input: str, context=None):
        return self.prompt_render.render(
            "llmcompiler_pro",
            "planner_user_prompt.jinja2",
            user_input=user_input,
            context=context,
        )

    async def run_llm(
        self,
        inputs: dict[str, Any],
        is_replan: bool = False,
        callbacks: list[Callbacks] = None,
    ) -> str:
        """Run the LLM."""
        if is_replan:
            system_prompt = self._system_prompt(is_replan=True)
            assert "context" in inputs, "If replanning, context must be provided"
            human_prompt = self._user_prompt(inputs["input"], context=inputs["context"])
        else:
            system_prompt = self._system_prompt()
            human_prompt = self._user_prompt(inputs["input"])

        if isinstance(self.llm, BaseChatModel):
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt),
            ]
            llm_response = await self.llm.ainvoke(messages, {"callbacks": callbacks})
            response = llm_response.content
        else:
            print("LLM must be either BaseChatModel")
            raise ValueError("LLM must be either BaseChatModel")

        print(f"LLMCompiler planner response: \n{response}")

        return response

    async def plan(
        self, inputs: dict, is_replan: bool, callbacks: Callbacks = None, **kwargs: Any
    ):
        llm_response = await self.run_llm(
            inputs=inputs, is_replan=is_replan, callbacks=callbacks
        )
        llm_response = llm_response + "\n"
        return self.output_parser.parse(llm_response)

    async def aplan(
        self,
        inputs: dict,
        task_queue: asyncio.Queue[Optional[str]],
        is_replan: bool = False,
        callbacks: list = None,
        **kwargs: Any,
    ) -> str:
        """Given input, asynchronously decide what to do."""
        all_callbacks = [
            AutoAgentCallback(
                queue=task_queue,
            )
        ]
        if callbacks:
            all_callbacks.extend(callbacks)
        logger.info(f"START to generate plan, callbacks: {all_callbacks}")

        try:
            return await self.run_llm(
                inputs=inputs, is_replan=is_replan, callbacks=all_callbacks
            )
        except Exception:
            logger.error(f"Error in running LLM in Planner: {traceback.format_exc()}")
            return None
