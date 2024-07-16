from __future__ import annotations

import asyncio
import traceback
from typing import Dict, Optional

from langchain_core.messages.base import BaseMessage
from logzero import logger

from llmcompiler_pro.prompt_render.jinja2_render import Jinja2Render
from llmcompiler_pro.schema.common import Language

from ...schema.common import ModelType
from .task import Task
from .utils import preprocess_instruction_from_task

SCHEDULING_INTERVAL = 0.01  # seconds


class TaskFetchingUnit:
    """
    Task Fetching Unit is responsible for scheduling and running tasks asynchronously.

    It listens to the task_queue and schedules the tasks as they arrive.
    After scheduling the tasks, it runs them asynchronously and waits for them to finish.
    """

    tasks: Dict[int, Task]
    tasks_done: Dict[int, asyncio.Event]
    remaining_tasks: set[int]

    user_input: str
    chat_history: list[BaseMessage]

    def __init__(
        self,
        user_input: str,
        chat_history: list[BaseMessage],
        model_name: str,
        tools: list[dict],
        language: Language,
        session_id: str = None,
    ):
        self.tasks = {}
        self.tasks_done = {}
        self.remaining_tasks = set()
        self.language: Language = language
        self.tools: list[dict] = tools
        self.prompt_render = Jinja2Render("llmcompiler_pro/prompt")
        self.model_name: str = model_name
        self.session_id: str = session_id

        # Static Parameters for tool like Code Interpreter
        self.user_input: str = user_input
        self.chat_history: list[BaseMessage] = chat_history

        # Status for the planning
        # TODO: Implement the plan add and update, do not use str object directly.
        self.plan = ""  # Gather all the plan in this variable

    def update_tasks(self, tasks: dict[int, Task]) -> None:
        """
        Update the tasks dictionary with new tasks.
        Task Fetching Unit will schedule and run these tasks.

        Args:
            tasks: A dictionary of tasks, where the key is the task index and the value is the task object.
        """
        # Set chat history to task object
        for task_idx in tasks:
            tasks[task_idx].chat_history = self.chat_history

        self.tasks.update(tasks)
        self.tasks_done.update({task_idx: asyncio.Event() for task_idx in tasks})
        self.remaining_tasks.update(set(tasks.keys()))

        # Update the plan(string)
        for task_idx, task in tasks.items():
            self.plan += (
                f"{tasks.get(task_idx).idx}. {tasks.get(task_idx).instruction}\n"
            )

    def _all_tasks_done(self):
        return all(self.tasks_done[d].is_set() for d in self.tasks_done)

    def _get_all_executable_tasks(self):
        """Get all tasks that are ready to be executed."""
        return [
            task_name
            for task_name in self.remaining_tasks
            if all(
                self.tasks_done[d].is_set() for d in self.tasks[task_name].dependencies
            )
        ]

    async def _build_subtasks_for_tasks(self, task: Task) -> None:
        """
        Build the subtasks for the given task.
        Subtasks are generated based on the function calling using the instruction of task object.

        :param task: Target Task object. This function will generate the subtasks for this object.
        """
        await task.build_subtasks(
            self.language, self.model_name, self.tools, ModelType.openai
        )

    async def _run_task(self, task: Task) -> None:
        """
        Run the task asynchronously and set the observation to the task object.
        :param task: Task object to be executed.
        """
        try:
            # Preprocess the task object before running the task
            preprocess_instruction_from_task(task)

            # Prepare the subtasks for the task.
            await self._build_subtasks_for_tasks(task)
            logger.debug(f"Generated subtasks for task {task.idx}")

            # Run each sub_tasks
            for _sub_task in task.sub_tasks:
                res = await _sub_task()
                assert res is not None

            # Join all the observations from the subtasks
            logger.debug(task.sub_tasks[0].observation)
            task.set_observation()

            self.tasks_done[task.idx].set()
        except Exception as e:
            logger.error(f"Error in running task {task.idx}: {traceback.format_exc()}")
            raise e

    async def schedule(self, task_queue: asyncio.Queue[Optional[Task]]):
        """Asynchronously listen to task_queue and schedule tasks as they arrive.
        After scheduling the tasks, run them asynchronously and wait for them to finish.

        :param task_queue: An asyncio.Queue object that contains the tasks to be scheduled from Planner Streaming Callback.
        """
        no_more_tasks = False  # Flag to check if all tasks are received

        try:
            while True:
                if not no_more_tasks:
                    # Wait for a new task to be added to the queue
                    logger.debug("Waiting for new task")
                    task: Task = await task_queue.get()
                    logger.debug(f"Received task {task}")

                    # Check for sentinel value indicating end of tasks
                    if task is None:
                        no_more_tasks = True
                    else:
                        # Parse and set the new tasks
                        self.update_tasks({task.idx: task})
                        logger.debug(f"Find the new task {task.idx}")

                # Find the executable tasks
                executable_tasks = self._get_all_executable_tasks()

                if executable_tasks:
                    for task_idx in executable_tasks:
                        # Execute the task asynchronously
                        asyncio.create_task(self._run_task(self.tasks[task_idx]))

                        # Remove the task from the remaining tasks list
                        self.remaining_tasks.remove(task_idx)
                        logger.debug(f"Create task {task_idx}")
                elif no_more_tasks and self._all_tasks_done():
                    # Exit the loop if no more tasks are expected and all tasks are done
                    break
                else:
                    # If no executable tasks are found, sleep for the SCHEDULING_INTERVAL
                    await asyncio.sleep(SCHEDULING_INTERVAL)
        except Exception as e:
            logger.error(f"Error occurred while scheduling tasks: {e}")
            raise e
