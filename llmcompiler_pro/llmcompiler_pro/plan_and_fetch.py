import asyncio

from langchain_core.messages.base import BaseMessage
from langchain_core.tracers.base import AsyncBaseTracer, BaseTracer

from llmcompiler_pro.schema.common import Language

from .planner.generate import generate_plan
from .planner.planner import Planner
from .task_fetch.fetch import start_fetch


async def generate_plan_and_fetch(
    planner: Planner,
    user_input: str,
    chat_history: list[BaseMessage],
    model_name: str,
    tools: list[dict],
    language: Language,
    session_id: str,
    planner_callbacks: list[BaseTracer | AsyncBaseTracer],
) -> tuple[str, list]:
    task_queue = asyncio.Queue()

    async_task_of_plan = generate_plan(
        planner,
        {"input": user_input, "messages": chat_history},
        task_queue,
        planner_callbacks,
    )

    task_fetching_unit = await start_fetch(
        user_input,
        chat_history,
        model_name,
        tools,
        language,
        session_id,
        planner_callbacks,
        task_queue,
    )

    # Get plan from previous coroutine
    plan = await asyncio.gather(async_task_of_plan)

    plan = plan[0]
    tasks = task_fetching_unit.tasks

    return plan, tasks
