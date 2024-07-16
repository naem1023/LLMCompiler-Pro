import asyncio
from typing import Optional

from .planner import Planner


def generate_plan(
    planner: Planner,
    inputs: dict,
    task_queue: asyncio.Queue[Optional[str]],
    plan_callbacks: list,
) -> asyncio.Task:
    return asyncio.create_task(
        planner.aplan(
            inputs=inputs,
            task_queue=task_queue,
            callbacks=plan_callbacks,
        )
    )
