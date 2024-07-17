import openai
from promptlayer import PromptLayer


def factory_openai_async_client() -> openai.AsyncOpenAI:
    return PromptLayer().openai.AsyncOpenAI()
