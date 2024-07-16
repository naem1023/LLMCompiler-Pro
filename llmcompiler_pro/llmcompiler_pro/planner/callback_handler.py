import asyncio
import re
from typing import Any, Optional
from uuid import UUID

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.outputs import LLMResult

from llmcompiler_pro.llmcompiler_pro.constants import ACTION_PATTERN, THOUGHT_PATTERN
from llmcompiler_pro.llmcompiler_pro.output_parser import instantiate_task

from ..task_fetch.task import Task


class StreamingGraphParser:
    """Streaming version of the GraphParser."""

    buffer = ""
    thought = ""
    graph_dict = {}

    def __init__(self) -> None:
        ...

    def _match_buffer_and_generate_task(self, suffix: str) -> Optional[Task]:
        """Runs every time "\n" is encountered in the input stream or at the end of the stream.
        Matches the buffer against the regex patterns and generates a task if a match is found.
        Match patterns include:
        1. Thought: <thought>\n<idx>. <instruction>
          - the thought is then used as the thought for the next action.
          - instruction is the freeform text.
          - idx is the number of the plan.
        """
        pattern = rf"(?:{THOUGHT_PATTERN}\n)?{ACTION_PATTERN}"
        if match := re.match(pattern, self.buffer):
            # if action is parsed, return the task, and clear the buffer
            thought, idx, instruction = match.groups()
            idx = int(idx)
            task = instantiate_task(
                idx=idx,
                thought=thought,
                instruction=instruction,
            )
            return task
        return None

    def ingest_token(self, token: str) -> Optional[Task]:
        # Append token to buffer
        if "\n" in token:
            prefix, suffix = token.split("\n", 1)
            prefix = prefix.strip()
            self.buffer += prefix + "\n"
            matched_item = self._match_buffer_and_generate_task(suffix)
            self.buffer = suffix
            return matched_item
        else:
            self.buffer += token

        return None

    def finalize(self):
        self.buffer = self.buffer + "\n"
        return self._match_buffer_and_generate_task("")


class AutoAgentCallback(AsyncCallbackHandler):
    _queue: asyncio.Queue[Optional[Task]]
    _parser: StreamingGraphParser

    def __init__(
        self,
        queue: asyncio.Queue[Optional[str]],
    ):
        """
        LLM Streaming Callback Handler for the fetching the tasks from the Planner Streaming Output.

        :param queue: Asyncio queue to put the parsed data into. This queue is consumed by the Task Fetching Unit.
        """
        self._queue = queue
        self._parser = StreamingGraphParser()

    async def on_llm_start(self, serialized, prompts, **kwargs: Any) -> Any:
        """Run when LLM starts running."""

    async def on_llm_new_token(
        self,
        token: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        parsed_data = self._parser.ingest_token(token)
        # logger.info(f"on_llm_new_token: new token is put into the queue: {parsed_data}")
        if parsed_data:
            await self._queue.put(parsed_data)
            if parsed_data.is_join:
                await self._queue.put(None)

    async def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        parsed_data = self._parser.finalize()
        # logger.info(f"on_llm_end: new token is put into the queue: {parsed_data}")
        if parsed_data:
            await self._queue.put(parsed_data)
        await self._queue.put(None)
