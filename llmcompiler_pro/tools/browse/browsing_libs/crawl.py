import asyncio
import traceback
from typing import Any, Dict, List

from logzero import logger
from playwright.async_api import Browser, BrowserContext, async_playwright

from .selenium import (  # Assuming the main function from selenium.py is now called fetch_page
    fetch_page_content,
)
from .selenium_exception import BrowserInitializationError
from .utils import extract_images, get_og_image


async def scrape_page(
    ctx: BrowserContext,
    target_url: str,
    max_length: int,
    user_agent: str,
    wait_time: int,
):
    """
    Asynchronously scrapes a single web page.

    :param ctx: A BrowserContext object for the web scraping session.
    :param target_url: The URL of the page to scrape.
    :param max_length: The maximum content length to fetch.
    :param user_agent: The user agent string to use for the request.
    :param wait_time: The maximum time to wait for the page to load, in milliseconds.
    :return: A tuple containing (url, content, markup) if successful, (url, None, None) otherwise.
    """
    logger.info(f"Initiating scrape for {target_url}")
    try:
        content, markup = await fetch_page_content(
            ctx,
            target_url,
            max_length,
            user_agent,
            wait_time,
        )
        if content.strip():
            return target_url, content, markup
    except Exception as e:
        logger.error(f"Failed to scrape {target_url}: {e}\n{traceback.format_exc()}")
        return target_url, None, None


async def setup_playwright_browser() -> tuple[Browser, BrowserContext]:
    """
    Sets up and returns a Playwright browser and context.

    :return: A tuple containing the Browser and BrowserContext objects.
    """
    playwright = await async_playwright().start()
    browser = await playwright.chromium.launch()
    context = await browser.new_context()
    return browser, context


async def perform_bulk_scraping(
    url_list: List[str],
    context: BrowserContext,
    max_length: int = 3000,
    user_agent: str = "WebScraperBot",
    limit: int = 3,
    wait_time: int = 30000,
) -> List[Dict[str, Any]]:
    """
    Performs bulk scraping of web pages using the provided BrowserContext.

    :param url_list: A list of URLs to scrape.
    :param context: A BrowserContext object for the web scraping session.
    :param max_length: The maximum content length to fetch for each page.
    :param user_agent: The user agent string to use for requests.
    :param limit: The maximum number of URLs to process (0 or negative means no limit).
    :param wait_time: The maximum time to wait for each page to load, in milliseconds.
    :return: A list of dictionaries containing scraped data for each URL.
    """
    assert limit is not None
    target_urls = url_list[:limit] if 0 < limit < len(url_list) else url_list

    tasks = [
        scrape_page(
            context,
            url,
            max_length,
            user_agent,
            wait_time,
        )
        for url in target_urls
    ]
    scraped_data = await asyncio.gather(*tasks, return_exceptions=True)

    results = []
    for url, content, markup in scraped_data:
        if content:
            results.append(
                {
                    "url": url,
                    "text": content,
                    "html": markup,
                    "image": extract_images(markup, url),
                    "thumbnail": get_og_image(markup),
                }
            )
        else:
            results.append(
                {
                    "url": url,
                    "text": None,
                    "html": None,
                    "image": None,
                    "thumbnail": None,
                }
            )

    return results


async def bulk_scrape_with_playwright(
    url_list: List[str],
    max_length: int = 3000,
    user_agent: str = "WebScraperBot",
    limit: int = 3,
    wait_time: int = 30000,
) -> List[Dict[str, Any]]:
    """
    Orchestrates the bulk scraping process using Playwright.

    :param url_list: A list of URLs to scrape.
    :param max_length: The maximum content length to fetch for each page.
    :param user_agent: The user agent string to use for requests.
    :param limit: The maximum number of URLs to process (0 or negative means no limit).
    :param wait_time: The maximum time to wait for each page to load, in milliseconds.
    :return: A list of dictionaries containing scraped data for each URL.
    """
    try:
        browser, context = await setup_playwright_browser()
        results = await perform_bulk_scraping(
            url_list, context, max_length, user_agent, limit, wait_time
        )
        if "context" in locals():
            await context.close()
        if "browser" in locals():
            await browser.close()

        return results
    except asyncio.TimeoutError as e:
        logger.error(f"Scraping process timed out: {str(e)}")
        raise TimeoutError("Scraping process timed out") from e
    except Exception as e:
        logger.error(f"An error occurred during bulk scraping: {str(e)}")
        if "browser" in str(e).lower():
            raise BrowserInitializationError("Failed to initialize the browser") from e
        else:
            raise e
