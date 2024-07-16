from .browse import SearchAndBrowsingTool


def get_tools() -> list[dict]:
    """Get predefined tool schema for OpenAI."""
    return [SearchAndBrowsingTool().tool_schema]
