import asyncio
import re
import subprocess
import sys
import threading
import traceback
from typing import Any, TypedDict

from jupyter_client import AsyncKernelClient, AsyncKernelManager
from logzero import logger


def change_markdown_image(text: str):
    modified_text = re.sub(r"!\[(.*?)\]\(\'(.*?)\'\)", r"![\1](/file=\2)", text)
    return modified_text


class CodeExecutionResult(TypedDict):
    result: str
    images: list[str]  # a list of base64 string
    is_error: bool


class JupyterSession:
    """
    Manager class to intializae Jupyter session and run the code.

    Original code from: https://github.com/SeungyounShin/Llama2-Code-Interpreter/blob/main/code_interpreter/JuypyterClient.py
    """

    def __init__(self, timeout: int = 10):
        self.km = AsyncKernelManager()
        self.kc: AsyncKernelClient = None
        self.timeout = timeout

        thread = threading.Thread(target=self.start_kernel)
        thread.start()
        thread.join()

        self.install_packages()

    def start_kernel(self):
        async def _start_kernel():
            await self.km.start_kernel()
            self.kc = self.km.client()

        asyncio.run(_start_kernel())

    def install_packages(self):
        # TODO: check the modules are already installed.
        packages = ["numpy", "pandas", "matplotlib"]
        for package in packages:
            try:
                # 패키지가 이미 설치되어 있는지 시도
                subprocess.check_call([sys.executable, "-m", "pip", "show", package])
                logger.info(f"{package} is already installed.")
            except subprocess.CalledProcessError:
                # 패키지가 설치되어 있지 않다면 설치
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                logger.info(f"Installed {package} successfully.")

    def clean_output(self, outputs: Any):
        outputs_only_str = list()
        for i in outputs:
            if type(i) == dict:
                if "text/plain" in list(i.keys()):
                    outputs_only_str.append(i["text/plain"])
            elif type(i) == str:
                outputs_only_str.append(i)
            elif type(i) == list:
                error_msg = "\n".join(i)
                error_msg = re.sub(r"\x1b\[.*?m", "", error_msg)
                outputs_only_str.append(error_msg)

        return "\n".join(outputs_only_str).strip()

    async def add_and_run(self, code_string: str) -> CodeExecutionResult:
        # This inner function will be executed in a separate thread
        async def run_code_in_thread():
            _outputs = []
            _images = []
            _error_flag = False

            try:
                async with asyncio.timeout(self.timeout):
                    # Execute the code and get the execution count
                    self.kc.execute(code_string)

                    while True:
                        msg = await self.kc.get_iopub_msg(timeout=self.timeout)

                        msg_type = msg["header"]["msg_type"]
                        content = msg["content"]

                        if msg_type == "execute_result":
                            _outputs.append(content["data"])
                        elif msg_type == "execute_result" or msg_type == "display_data":
                            data = content.get("data", {})
                            if "image/png" in data:
                                image_data = data["image/png"]
                                _images.append(image_data)
                        elif msg_type == "stream":
                            _outputs.append(content["text"])
                        elif msg_type == "error":
                            _error_flag = True
                            _outputs.append(content["traceback"])

                        # If the execution state of the kernel is idle, it means the cell finished executing
                        if (
                            msg_type == "status"
                            and content["execution_state"] == "idle"  # noqa: W503
                        ):
                            break
            except TimeoutError as e:
                logger.error(f"Timeout after {self.timeout} seconds, {e}")
                _error_flag = True
            except Exception as e:
                logger.error(
                    f"Error while running the code: {e}\n\n{traceback.format_exc()}"
                )
                print(f"Error while running the code: {e}\n\n{traceback.format_exc()}")

            return _outputs, _images, _error_flag

        outputs, images, error_flag = await asyncio.create_task(run_code_in_thread())
        return CodeExecutionResult(
            result=self.clean_output(outputs), images=images, is_error=error_flag
        )

    async def close(self):
        """Shutdown the kernel."""
        await self.km.shutdown_kernel()
