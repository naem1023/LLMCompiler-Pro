import chainlit as cl
from chainlit.input_widget import Select
from dotenv import load_dotenv

from gui_demo.starter import demo_starter
from llmcompiler_pro.llmcompiler_pro.llmcompiler_pro import LLMCompilerPro
from llmcompiler_pro.schema.common import Language, LLMCompilerProRequest
from llmcompiler_pro.streaming_handlers.chainlit_updater import LLMCompilerProTracer
from llmcompiler_pro.tools import get_tools

load_dotenv()


class AIAssistantManager:
    """
    Manages the creation and interaction with an AI assistant.

    This class encapsulates the logic for initializing and using an AI assistant
    powered by LLMCompilerPro.
    """

    def __init__(self):
        """
        Initializes the AIAssistantManager.
        """
        self.assistant = None

    async def initialize_assistant(self, language: Language = Language.Korean):
        """
        Initializes the AI assistant with specific configurations.

        :return: None
        """
        print(f"Set language: {language}")
        configuration = LLMCompilerProRequest(
            max_replan=1,
            model_name="gpt-4o",
            session_id="123",
            language=language,
        )
        self.assistant = LLMCompilerPro(
            configuration,
            tools=get_tools(),
            callbacks=[LLMCompilerProTracer()],
        )

    async def process_message(self, content: str) -> str:
        """
        Processes a user message using the AI assistant.

        :param content: The content of the user's message
        :return: The AI assistant's response
        """
        if not self.assistant:
            raise ValueError("Assistant not initialized")

        response = await self.assistant.acall(content, [])
        return response[-1]


@cl.set_starters
async def set_starters():
    return demo_starter


@cl.on_chat_start
async def initiate_chat():
    """
    Initializes the chat session by creating and storing an AI assistant.

    :return: None
    """
    settings = await cl.ChatSettings(
        [
            Select(
                id="Agent Language",
                label="The primary language used in the tool",
                values=[Language.Korean.value, Language.English.value],
                initial_index=0,
            )
        ]
    ).send()

    language: Language = Language.from_value(settings.get("Agent Language"))

    manager = AIAssistantManager()
    await manager.initialize_assistant(language)
    cl.user_session.set("assistant_manager", manager)


@cl.on_settings_update
async def update_settings(settings):
    """
    Handles updates to chat settings.

    :param settings: The updated settings
    :return: None
    """
    print(f"Settings updated: {settings}")
    manager = AIAssistantManager()
    language: Language = Language.from_value(settings.get("Agent Language"))
    await manager.initialize_assistant(language)


@cl.on_message
async def handle_message(message: cl.Message):
    """
    Processes incoming user messages and sends AI-generated responses.

    :param message: The user's message object
    :return: None
    """
    manager = cl.user_session.get("assistant_manager")
    if not manager:
        raise ValueError("Assistant manager not found in session")

    response = await manager.process_message(message.content)
    await cl.Message(author="AI Assistant", content=response).send()
