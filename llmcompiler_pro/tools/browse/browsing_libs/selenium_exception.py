"""
Custom exceptions for web scraping operations.
"""

from .exception import CustomError


class BrowserInitializationError(CustomError):
    """Exception raised when there's an error launching the browser."""

    def __init__(self):
        super().__init__(
            {"description": "Failed to initialize the browser. Please try again later."}
        )


class PageCreationTimeoutError(CustomError):
    """Exception raised when creating a new page times out."""

    def __init__(self):
        super().__init__(
            {
                "description": "Timed out while creating a new page. The server might be overloaded."
            }
        )


class NavigationTimeoutError(CustomError):
    """Exception raised when navigation to a page times out."""

    def __init__(self):
        super().__init__(
            {
                "description": "Timed out while navigating to the requested page. Please try again later."
            }
        )


class UnexpectedScrapingError(CustomError):
    """Exception raised when an unexpected error occurs during scraping."""

    def __init__(self):
        super().__init__(
            {
                "description": "An unexpected error occurred during the scraping process. Please try again later."
            }
        )


class AccessRestrictedError(CustomError):
    """Exception raised when access to a page is restricted or forbidden."""

    def __init__(self):
        super().__init__(
            {"description": "Access to the requested page is restricted or forbidden."}
        )


class ContentExtractionError(CustomError):
    """Exception raised when there's an error extracting content from a page."""

    def __init__(self):
        super().__init__(
            {
                "description": "Failed to extract content from the page. The page structure might have changed."
            }
        )
