from openai.types.chat.chat_completion import ChatCompletion

from llmcompiler_pro.infra.openai import factory_openai_async_client
from llmcompiler_pro.prompt_render.jinja2_render import Jinja2Render
from llmcompiler_pro.schema.common import Language


def get_subtask_tool_prompt(instruction: str, thought: str, language: Language):
    """Build the tool message prompt for generating subtasks."""
    return Jinja2Render("llmcompiler_pro/prompt").render(
        "llmcompiler_pro",
        "tool_prompt.jinja2",
        instruction=instruction,
        thought=thought,
        language=language.value,
    )


async def generate_subtasks(
    model_name: str,
    messages: list[dict],
    tools: list[dict],
) -> ChatCompletion:
    """Get the Langchain Chat LLM client with binding tools for generating subtasks.

    :param model_name: The model name of LLM to generate subtasks.
    :param messages: The list of messages to bind with the LLM client.
    :param tools: The list of tools to bind with the LLM client. It only considers OpenAI at this time
    :return
    """
    from logzero import logger

    logger.debug(f"Messages: {messages}")
    logger.debug(f"Tools: {tools}")
    return await factory_openai_async_client().chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.2,
        tool_choice="auto",
        tools=tools,
    )
