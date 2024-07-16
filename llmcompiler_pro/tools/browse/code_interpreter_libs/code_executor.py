from abc import ABC, abstractmethod

from e2b_code_interpreter import CodeInterpreter as E2BCodeInterpreter

from .jupyter_session import CodeExecutionResult, JupyterSession


class CodeValidator:
    def __init__(self):
        ...

    def validate(self, code: str) -> bool:
        """Validate the code
        - Check for syntax errors
        - Check the expected running time
        """
        # TODO: Implement the code validation logic.
        return True


class CodeExecutor(ABC):
    @abstractmethod
    async def execute(self, code: str):
        ...

    @abstractmethod
    async def close(self):
        ...


class CodeExecutorLocal(CodeExecutor):
    def __init__(self):
        self.jupyter_session = JupyterSession()
        self.code_validator = CodeValidator()

    async def execute(self, code: str) -> CodeExecutionResult:
        if not self.code_validator.validate(code):
            # TODO: Define the error type.
            return CodeExecutionResult(
                result="Invalid code",
                images=[],
                is_error=True,
            )

        return await self.jupyter_session.add_and_run(code)

    async def close(self):
        await self.jupyter_session.close()


class CodeExecutorE2B(CodeExecutor):
    def __init__(self):
        self.executor = E2BCodeInterpreter()

    async def execute(self, code: str) -> dict:
        execution = self.executor.notebook.exec_cell(code)
        return {
            "results": execution.results,
            "stdout": execution.logs.stdout,
            "stderr": execution.logs.stderr,
            "error": execution.error,
        }
