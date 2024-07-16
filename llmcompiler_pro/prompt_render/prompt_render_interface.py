from abc import ABC, abstractmethod


class PromptRenderInterface(ABC):
    def __init__(self):
        ...

    @abstractmethod
    def render(self, planner: str, template_name: str, **kwargs) -> str:
        """
        Render the prompt_render.
        :param planner: The planner name.
        :param template_name: The template name.
        """
        ...
