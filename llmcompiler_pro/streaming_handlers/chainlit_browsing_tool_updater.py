from typing import Any, Dict, List, Optional, Union
from uuid import UUID

import chainlit as cl
from chainlit.context import context_var
from chainlit.step import Step
from langchain.callbacks.tracers.schemas import Run
from langchain_core.outputs import ChatGenerationChunk, GenerationChunk
from langchain_core.tracers.base import BaseTracer


class BrowsingToolTracer(BaseTracer):
    """
    Langchain callback handler for handling streaming results of the browsing tool.

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
        # Token sequence that prefixes the answer
        answer_prefix_tokens: Optional[List[str]] = None,
        # Should we stream the final answer?
        stream_final_answer: bool = False,
        # Should force stream the first response?
        force_stream_final_answer: bool = False,
        # Runs to ignore to enhance readability
        to_ignore: Optional[List[str]] = None,
        # Runs to keep within ignored runs
        to_keep: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        BaseTracer.__init__(self, **kwargs)
        self.reset_cur_step()

        self.context = context_var.get()
        self.steps = {}
        self.parent_id_map = {}
        self.ignored_runs = set()

        if self.context.current_step:
            self.root_parent_id = self.context.current_step.id
        elif self.context.session.root_message:
            self.root_parent_id = self.context.session.root_message.id
        else:
            self.root_parent_id = None

        self.task_list = cl.TaskList()
        self.task_list.status = "Init"

    def reset_cur_step(self):
        # Set Chainlit variables to default values
        self.cur_step: Optional[cl.Message] = None
        self.cur_attachment_list = []
        self.cur_post_status: str = "Updating"
        self.cur_message: str = ""
        self.cur_message_is_end: bool = False
        self.cur_message_sent: bool = False

    def on_start(self):
        self.reset_cur_step()
        self.cur_step = cl.Step(name="Web Browsing Tool", type="tool")

    def on_llm_start(
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
        return super().on_llm_start(
            serialized,
            prompts,
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags,
            metadata=metadata,
            name=name,
            **kwargs,
        )

    def on_llm_new_token(
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
        cl.run_sync(self.cur_step.stream_token(self.cur_message, True))

    def on_llm_end(self, **kwargs: Any) -> Any:
        """Run when LLM ends running.

        Similar with BaseTracer._end_trace() but on_llm_end handle the end of the streaming.
        """
        assert self.cur_step is not None
        # cl.run_sync(self.cur_step.stream_token(content, True))
        cl.run_sync(self.cur_step.send())
        self.reset_cur_step()

    def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        """Run when LLM errors."""

    def _run_sync(self, co):  # TODO: WHAT TO DO WITH THIS?
        context_var.set(self.context)
        self.context.loop.create_task(co)

    def _persist_run(self, run: Run) -> None:
        pass

    def _get_run_parent_id(self, run: Run):
        parent_id = str(run.parent_run_id) if run.parent_run_id else self.root_parent_id

        return parent_id

    def _get_non_ignored_parent_id(self, current_parent_id: Optional[str] = None):
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

    def _should_ignore_run(self, run: Run):
        ...

    def _is_annotable(self, run: Run):
        return run.run_type in ["retriever", "llm"]

    def _start_trace(self, run: Run) -> None:
        ...

    def _on_run_update(self, run: Run) -> None:
        """Process a run upon update.
        It is called in BaseTracer._end_trace(): End a trace for a run.
        """
        ...
