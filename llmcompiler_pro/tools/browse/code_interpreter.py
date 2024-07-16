from typing import Any, Dict

from code_interpreter_libs.code_interpreter_manager import (
    CodeInterpreterManager,
    code_interpreter_consumer,
)

from llmcompiler_pro.schema.tool_calls import OpenAPIDocument

from ..tool_interface import Tool


class CodeExecutorTool(Tool):
    """
    A tool for executing and interpreting code based on given instructions.

    This tool provides functionality to generate, run, and analyze Python code,
    create visualizations, and process data based on user instructions.
    """

    name = "code_interpreter"
    description = """Code Generator, Execution and Interpretation Tool
    Features:
    - Data visualization and analysis using Python
    - Creation of plots and charts
    - Generation and execution of Python code
    - Step-by-step problem-solving through code generation
    - Visualization of tables, images, and plots
    - Return of code execution results as text and images
    """
    _tool_schema = {
        "type": "object",
        "properties": {
            "instruction": {
                "type": "string",
                "description": "The instruction representing the task for code execution and interpretation.",
            }
        },
        "required": ["instruction"],
    }
    doc: OpenAPIDocument = None

    async def __call__(self, instruction: str, **kwargs: Dict[str, Any]) -> Any:
        """
        Execute the code interpreter based on the given instruction.

        :param instruction: The instruction representing the task for code execution.
        :param kwargs: Additional parameters required for code execution.
        :return: The result of code execution.
        """
        params = {
            "user_input": kwargs.get("user_input"),
            "chat_history": kwargs.get("chat_history"),
            "plan": kwargs.get("plan"),
            "current_goal": kwargs.get("current_goal"),
            "model_name": kwargs.get("model_name"),
        }

        code_interpreter_manager = CodeInterpreterManager()

        if code_interpreter_consumer.consumer_thread is None:
            code_interpreter_consumer.run_consumer()

        await code_interpreter_manager.enqueue(params)
        return await code_interpreter_manager.result()

    def __str__(self) -> str:
        return f"CodeExecutorTool(name='{self.name}', schema={self._tool_schema})"

    def __repr__(self) -> str:
        return f"CodeExecutorTool(name='{self.name}', doc={self.doc}, schema={self._tool_schema})"
