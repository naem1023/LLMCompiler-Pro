from .browse import WebContentAnalyzer


def get_tools() -> list[dict]:
    """Get predefined tool schema for OpenAI."""
    return [WebContentAnalyzer().tool_schema]
