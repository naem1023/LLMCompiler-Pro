import json
from typing import Any, Dict


class CustomError(Exception):
    """
    A custom base exception class that allows for detailed error information.

    This class extends the built-in Exception class to provide additional
    functionality for error handling and serialization.
    """

    def __init__(self, error_info: Any):
        """
        Initialize the CustomError instance.

        :param error_info: Detailed information about the error. Can be of any type.
        """
        self.error_info = error_info

    def __dict__(self) -> Dict[str, Any]:
        """
        Convert the error information to a dictionary.

        :return: A dictionary containing the error information.
        """
        return {"error_details": self.error_info}

    def __str__(self) -> str:
        """
        Provide a string representation of the error.

        This method serializes the error information to a JSON string.

        :return: A JSON string representation of the error information.
        """
        return json.dumps(self.to_dict(), default=str)

    def __repr__(self) -> str:
        """
        Provide a detailed string representation of the CustomError instance.

        :return: A string representation of the CustomError instance.
        """
        return f"CustomError(error_info={self.error_info!r})"
