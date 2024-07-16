import http.client
from typing import Tuple, TypedDict

from playwright.async_api import BrowserContext, Page, Response

from .selenium_exception import (
    AccessRestrictedError,
    BrowserInitializationError,
    ContentExtractionError,
    NavigationTimeoutError,
    PageCreationTimeoutError,
    UnexpectedScrapingError,
)

# Constants
REQUEST_TIMEOUT = 30000  # milliseconds
CONTENT_LIMIT = 5000  # characters
BOT_IDENTIFIER = "WebScraperBot"


class ScrapedContent(TypedDict):
    """
    A TypedDict representing the scraped content from a web page.

    :key text: The extracted text content from the page.
    """

    text: str


async def fetch_page_content(
    browser_ctx: BrowserContext,
    target_url: str,
    max_chars: int = CONTENT_LIMIT,
    user_agent: str = BOT_IDENTIFIER,
    wait_time: int = REQUEST_TIMEOUT,
) -> Tuple[str, str]:
    """
    Fetch and extract content from a web page.

    :param browser_ctx: The browser context to use for the page visit.
    :param target_url: The URL of the page to fetch.
    :param max_chars: Maximum number of characters to extract from the page content.
    :param user_agent: The user agent string to use for the request.
    :param wait_time: Maximum time to wait for page load in milliseconds.
    :return: A tuple containing the extracted text and HTML content of the page.
    :raises BrowserInitializationError: If there's an error setting up the browser.
    :raises PageCreationTimeoutError: If creating a new page times out.
    :raises NavigationTimeoutError: If navigation to the page times out.
    :raises AccessRestrictedError: If access to the page is forbidden.
    :raises ContentExtractionError: If there's an error extracting content from the page.
    :raises UnexpectedScrapingError: For any other unexpected errors during scraping.
    """
    try:
        page = await _prepare_page(browser_ctx, user_agent, wait_time)
        response = await _navigate_to_page(page, target_url)

        if response.status == http.client.FORBIDDEN:
            raise AccessRestrictedError()

        if response.status != http.client.OK:
            raise ContentExtractionError()

        text_content = await _extract_text_content(page, max_chars)
        html_content = await page.content()

        text_content += await _extract_frame_content(response, max_chars)

        return text_content, html_content
    except Exception as e:
        if isinstance(e, (AccessRestrictedError, ContentExtractionError)):
            raise
        elif "context" in str(e).lower():
            raise BrowserInitializationError()
        elif "timeout" in str(e).lower():
            raise NavigationTimeoutError()
        else:
            raise UnexpectedScrapingError()


async def _prepare_page(
    browser_ctx: BrowserContext, user_agent: str, wait_time: int
) -> Page:
    """
    Prepare a page for scraping by setting up headers and timeout.

    :param browser_ctx: The browser context to use.
    :param user_agent: The user agent string to set.
    :param wait_time: The timeout to set for the page.
    :return: A configured Page object.
    :raises PageCreationTimeoutError: If creating a new page times out.
    """
    try:
        if not browser_ctx.pages:
            new_page = await browser_ctx.new_page()
            headers = {
                "User-Agent": f"Mozilla/5.0 AppleWebKit/537.36 (KHTML, like Gecko; compatible; {user_agent})"
            }
            await new_page.set_extra_http_headers(headers)
            new_page.set_default_timeout(wait_time)
            return new_page
        else:
            await browser_ctx.clear_cookies()
            return browser_ctx.pages[0]
    except Exception as e:
        if "timeout" in str(e).lower():
            raise PageCreationTimeoutError()
        raise BrowserInitializationError()


async def _navigate_to_page(page: Page, url: str) -> Response:
    """
    Navigate to the specified URL and wait for the page to load.

    :param page: The Page object to use for navigation.
    :param url: The URL to navigate to.
    :return: The Response object from the navigation.
    :raises NavigationTimeoutError: If navigation to the page times out.
    """
    try:
        response = await page.goto(url, timeout=10000, wait_until="load")
        assert response is not None
        return response
    except Exception as e:
        if "timeout" in str(e).lower():
            raise NavigationTimeoutError()
        raise UnexpectedScrapingError()


async def _extract_text_content(page: Page, max_chars: int) -> str:
    """
    Extract text content from the page body.

    :param page: The Page object to extract content from.
    :param max_chars: Maximum number of characters to extract.
    :return: The extracted text content.
    :raises ContentExtractionError: If there's an error extracting content from the page.
    """
    try:
        await page.wait_for_selector("body")
        return await page.evaluate(
            f"() => document.querySelector('body').innerText.slice(0, {max_chars})"
            if max_chars > 0
            else "() => document.querySelector('body').innerText"
        )
    except Exception:
        raise ContentExtractionError()


async def _extract_frame_content(response: Response, max_chars: int) -> str:
    """
    Extract additional content from frames if present.

    :param response: The Response object from the page navigation.
    :param max_chars: Maximum number of characters to extract.
    :return: The extracted frame content, or an empty string if no relevant frames are found.
    :raises ContentExtractionError: If there's an error extracting content from the frame.
    """
    try:
        frame_content = ""
        if response.frame:
            for child_frame in response.frame.child_frames:
                if child_frame.name.upper() == "MAINFRAME":
                    frame_page = child_frame.page
                    await frame_page.wait_for_selector("body")
                    frame_text = await frame_page.evaluate(
                        f"() => document.querySelector('body').innerText.slice(0, {max_chars})"
                        if max_chars > 0
                        else "() => document.querySelector('body').innerText"
                    )
                    frame_content += " " + frame_text
        return frame_content
    except Exception:
        raise ContentExtractionError()
