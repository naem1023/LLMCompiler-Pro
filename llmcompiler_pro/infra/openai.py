import os

import openai
from openai import AsyncOpenAI
from promptlayer import PromptLayer


def factory_openai_async_client() -> openai.AsyncOpenAI:
    if os.getenv("PROMPTLAYER_API_KEY") is None:
        return AsyncOpenAI()
    else:
        return PromptLayer().openai.AsyncOpenAI()
