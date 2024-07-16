import os
from functools import lru_cache

from jinja2 import Template

from .prompt_render_interface import PromptRenderInterface


class Jinja2Render(PromptRenderInterface):
    """
    Prompt render using Jinja2 template engine.
    """

    def __init__(self, template_dir: str):
        """
        :param template_dir: The base directory of the template files. e.g., "prompt/"
        """
        super().__init__()
        self.template_dir = template_dir

    def render(self, planner: str, template_name: str, **kwargs) -> str:
        """
        Based on the planner and template name, render the prompt with the given arguments.

        :param planner: The category of planner.
        :param template_name: The name of the template file.

        e.g., planner="ReAct", template_name="default.jinja2"
            redner function render the text using {self.template_dir}/ReAct/default.jinja2
        """
        target_path = os.path.abspath(
            os.path.join(self.template_dir, planner, template_name)
        )
        assert os.path.exists(target_path), f"Template file not found: {target_path}"

        @lru_cache()
        def _prompt_template():
            with open(target_path) as f:
                return f.read()

        rendered_prompt = Template(
            _prompt_template(),
            trim_blocks=True,
            keep_trailing_newline=True,
        ).render(**kwargs)

        return rendered_prompt
