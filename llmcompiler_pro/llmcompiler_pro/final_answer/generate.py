import json

from langchain_core.messages.human import HumanMessage
from langchain_core.messages.system import SystemMessage
from langchain_openai import ChatOpenAI
from logzero import logger

from llmcompiler_pro.prompt_render.jinja2_render import Jinja2Render
from llmcompiler_pro.prompt_render.prompt_render_interface import PromptRenderInterface
from llmcompiler_pro.schema.common import Language


def get_system_prompt(prompt_render: PromptRenderInterface, language: Language) -> str:
    return prompt_render.render(
        "llmcompiler_pro",
        "join_system_prompt.jinja2",
        language=language,
    )


def get_user_prompt(
    prompt_render: PromptRenderInterface, plan: str, query: str, context: str
) -> str:
    return prompt_render.render(
        "llmcompiler_pro",
        "join_user_prompt.jinja2",
        plan=plan,
        query=query,
        context=context,
    )


def get_chat_messages(
    prompt_render: PromptRenderInterface,
    plan: str,
    query: str,
    context: str,
    language: Language,
) -> list:
    system_prompt = get_system_prompt(prompt_render, language)
    user_prompt = get_user_prompt(prompt_render, plan, query, context)
    return [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]


async def request_llm_generation(model_name, messages, callbacks):
    llm = ChatOpenAI(
        model_name=model_name,
        streaming=True,
        model_kwargs={"response_format": {"type": "json_object"}},
    )
    res = await llm.ainvoke(messages, {"callbacks": callbacks})
    return res.content


async def generate_final_answer(
    model_name: str,
    user_input: str,
    plan: str,
    agent_scratchpad: str,
    language: Language,
    callbacks,
) -> tuple[bool, str]:
    prompt_render = Jinja2Render("llmcompiler_pro/prompt")
    messages = get_chat_messages(
        prompt_render, plan, user_input, agent_scratchpad, language
    )

    res = await request_llm_generation(model_name, messages, callbacks)

    try:
        response = json.loads(res)
    except Exception:
        logger.error(f"Failed to parse response as JSON at Join Generation: {res}")
        return res
    return response.get("terminate"), response.get("thought")
