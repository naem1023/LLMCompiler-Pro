from auto_agent_deprecated.ports.neural_net.containers import LanguageModelContainer
from dependency_injector import providers
from dependency_injector.containers import DeclarativeContainer

from .code_executor import CodeExecutor, CodeValidator
from .code_generator import CodeGenerator
from .code_interpreter import CodeInterpreter
from .jupyter_session import JupyterSession


class CodeInterpreterConatainer(DeclarativeContainer):
    lc_containers = providers.Container(LanguageModelContainer)

    jupyter_session = providers.Factory(JupyterSession)
    code_validator = providers.Factory(CodeValidator)
    code_executor = providers.Factory(
        CodeExecutor, jupyter_session=jupyter_session, code_validator=code_validator
    )
    code_generator = providers.Factory(CodeGenerator, language_model=lc_containers.lm)

    code_interpreter = providers.Factory(
        CodeInterpreter, executor=code_executor, generator=code_generator
    )
