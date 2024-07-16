import openai
from promptlayer import PromptLayer


def factory_openai_client() -> openai.AsyncOpenAI:
    return PromptLayer().openai.AsyncOpenAI()
