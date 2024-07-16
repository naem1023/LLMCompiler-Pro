import asyncio

from langchain_core.messages.base import BaseMessage
from langchain_core.tracers.base import AsyncBaseTracer, BaseTracer
from logzero import logger

from llmcompiler_pro.schema.common import Language

from .task_fetching_unit import TaskFetchingUnit


async def start_fetch(
    user_input: str,
    chat_history: list[BaseMessage],
    model_name: str,
    tools: list[dict],
    language: Language,
    session_id: str,
    planner_callbacks: list[BaseTracer | AsyncBaseTracer],
    queue: asyncio.Queue,
):
    task_fetching_unit = TaskFetchingUnit(
        user_input, chat_history, model_name, tools, language, session_id
    )

    logger.info("Start task fetching unit scheduler")
    await task_fetching_unit.schedule(task_queue=queue)

    return task_fetching_unit
