import re
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

import chainlit as cl
from chainlit.step import Step
from langchain.callbacks.tracers.base import BaseTracer
from langchain.callbacks.tracers.schemas import Run
from langchain.schema import BaseMessage
from langchain_core.outputs import ChatGenerationChunk, GenerationChunk, LLMResult
from langchain_core.tracers.base import AsyncBaseTracer

from llmcompiler_pro.llmcompiler_pro.constants import ACTION_PATTERN, THOUGHT_PATTERN


class StreamingGraphParser:
    """Streaming version of the GraphParser for LLMCompiler Pro.
    It doesn't make the Task object for Task Fetching Unit.
    """

    buffer = ""
    thought = ""
    graph_dict = {}

    def __init__(self) -> None:
        ...

    def _match_buffer_and_generate_task(self, suffix: str) -> Optional[tuple]:
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
            return thought, idx, instruction
        return None

    def ingest_token(self, token: str) -> tuple | None:
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


class Role(Enum):
    planner = "PLANNER"
    code_interpreter = "CODE_INTERPRETER"
    task_fetching_unit = "SUB_TASK_RUNNER"


class LLMCompilerProTracer(AsyncBaseTracer):
    """
    Langchain callback handler for handling streaming results of LLMCompilerPro.

    TODO: It only accept the unidirectional and non-parallel chunk.
    TODO: Demonstration should accept parallel streaming from multiple callback handler.

    """

    steps: Dict[str, Step]
    parent_id_map: Dict[str, str]
    ignored_runs: set

    cur_step: Optional[cl.Message] = None
    cur_attachment_list: List = []
    cur_post_status: str = "Updating"
    cur_message: str = ""
    cur_message_is_end: bool = False
    cur_message_sent: bool = False

    def __init__(
        self,
        answer_prefix_tokens: Optional[List[str]] = None,
        stream_final_answer: bool = False,
        force_stream_final_answer: bool = False,
        to_ignore: Optional[List[str]] = None,
        to_keep: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        BaseTracer.__init__(self, **kwargs)
        super().__init__(**kwargs)
        self.reset_cur_step()

        self.steps = {}
        self.parent_id_map = {}
        self.ignored_runs = set()

        # self.context = context_var.get()
        # if self.context.current_step:
        #     self.root_parent_id = self.context.current_step.id
        # elif self.context.session.root_message:
        #     self.root_parent_id = self.context.session.root_message.id
        # else:
        #     self.root_parent_id = None

        self.task_list = cl.TaskList()
        self.task_list.status = "Init"

        self.planner_parser = StreamingGraphParser()

    def reset_cur_step(self):
        # Set Chainlit variables to default values
        self.cur_step: Optional[cl.Message] = None
        self.cur_attachment_list = []
        self.cur_post_status: str = "Updating"
        self.cur_message: str = ""
        self.cur_message_is_end: bool = False
        self.cur_message_sent: bool = False

    async def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        *,
        run_id: "UUID",
        parent_run_id: Optional["UUID"] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        # Assumption: A first message of messages is the user input.
        messages[0]
        self.reset_cur_step()
        # self.cur_step = cl.Step(
        #     type="llm", name=Role.planner.value, show_input=True
        # )
        self.cur_step = cl.Message(content="", author="Planner")

        return await super().on_chat_model_start(
            serialized,
            messages,
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags,
            metadata=metadata,
            **kwargs,
        )

    async def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: "UUID",
        tags: Optional[List[str]] = None,
        parent_run_id: Optional["UUID"] = None,
        metadata: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> Run:
        await super().on_llm_start(
            serialized,
            prompts,
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags,
            metadata=metadata,
            name=name,
            **kwargs,
        )

    async def on_llm_new_token(
        self,
        token: str,
        *,
        chunk: Optional[Union[GenerationChunk, ChatGenerationChunk]] = None,
        run_id: "UUID",
        parent_run_id: Optional["UUID"] = None,
        **kwargs: Any,
    ) -> Run:
        # Update the current message
        self.cur_message += token

        # Stream the chunk into the current step
        # cl.run_sync(self.cur_step.stream_token(self.cur_message, ))
        await self.cur_step.stream_token(self.cur_message, True)

        # Check the cur_message is the plan using planner parser
        parsed_data = self.planner_parser.ingest_token(token)

        await self.cur_step.send()

        # If the cur_message is the plan, then generate the task and add to the task list
        if parsed_data:
            thought, idx, instruction = parsed_data
            self.task_list.status = "Planning"
            task_title = f"{idx}. {thought if thought else ''}\n{instruction}"
            _task = cl.Task(title=task_title, status=cl.TaskStatus.RUNNING)
            # cl.run_sync(self.task_list.add_task(_task))
            await self.task_list.add_task(_task)

            # Update the task list
            # cl.run_sync(self.task_list.send())
            await self.task_list.send()

    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        """Run when LLM ends running.

        Similar with BaseTracer._end_trace() but on_llm_end handle the end of the streaming.
        """
        assert self.cur_step is not None
        # cl.run_sync(self.cur_step.stream_token(content, True))
        # cl.run_sync(self.cur_step.__aexit__(None, None, None))  # type: ignore
        # await self.cur_step.__aexit__(None, None, None)  # type: ignore
        self.reset_cur_step()

        # TODO: Callback handler can accept a task fetching unit's streaming. It should be considered.
        if self.task_list.status == "Planning":
            self.task_list.status = "Complete"
            for _task in self.task_list.tasks:
                _task.status = cl.TaskStatus.DONE

    async def on_close(self):
        # if self.task_list.status == "Planning":
        #     self.task_list.status = "Complete"
        for _task in self.task_list.tasks:
            _task.status = cl.TaskStatus.DONE
        await self.task_list.send()

    async def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        """Run when LLM errors."""

    # def _run_sync(self, co):  # TODO: WHAT TO DO WITH THIS?
    #     context_var.set(self.context)
    #     self.context.loop.create_task(co)

    async def _persist_run(self, run: Run) -> None:
        pass

    async def _get_run_parent_id(self, run: Run):
        parent_id = str(run.parent_run_id) if run.parent_run_id else self.root_parent_id

        return parent_id

    async def _get_non_ignored_parent_id(self, current_parent_id: Optional[str] = None):
        if not current_parent_id:
            return self.root_parent_id

        if current_parent_id not in self.parent_id_map:
            return None

        while current_parent_id in self.parent_id_map:
            # If the parent id is in the ignored runs, we need to get the parent id of the ignored run
            if current_parent_id in self.ignored_runs:
                current_parent_id = self.parent_id_map[current_parent_id]
            else:
                return current_parent_id

        return self.root_parent_id

    async def _should_ignore_run(self, run: Run):
        ...

    async def _is_annotable(self, run: Run):
        return run.run_type in ["retriever", "llm"]

    async def _start_trace(self, run: Run) -> None:
        ...

    async def _on_run_update(self, run: Run) -> None:
        """Process a run upon update.
        It is called in BaseTracer._end_trace(): End a trace for a run.
        """
        ...
