import asyncio
import os
from enum import Enum
from typing import Dict, List, Tuple

import tiktoken
from logzero import logger
from serpapi.serp_api_client import SerpApiClient

from llmcompiler_pro.schema.common import (
    CityEnum,
    CostSetting,
    GoogleDomainEnum,
    Language,
    LanguageHLEnum,
    RegionEnum,
)
from llmcompiler_pro.streaming_handlers.chainlit_browsing_tool_updater import (
    BrowsingToolTracer,
)

from ..tool_interface import Tool
from .browsing_libs.crawl import bulk_scrape_with_playwright

SEGMENT_SIZE = 200
OVERLAP = 50
TOP_RESULTS = 3

encoder = tiktoken.encoding_for_model("gpt-4o")


class ContentSize(int, Enum):
    large = 10000
    medium = 2500
    small = 800


def limit_content_length(content: str, config: CostSetting) -> str:
    """
    Limit the length of the input content based on the specified configuration.

    :param content: The input content to be limited.
    :param config: The configuration determining the maximum length.
    :return: The length-limited content.
    """
    if not isinstance(content, str):
        return content

    max_length: int = getattr(ContentSize, config).value
    tokens = encoder.encode(content)
    return encoder.decode(tokens[:max_length])


def filter_content_urls(urls: List[str]) -> List[str]:
    """
    Filter out URLs with specific file extensions.

    :param urls: List of URLs to filter.
    :return: Filtered list of URLs.
    """
    excluded_extensions = [".pdf", ".docx", ".pptx", ".xlsx", ".doc", ".ppt", ".xls"]
    return [url for url in urls if not any(ext in url for ext in excluded_extensions)]


def process_search_results(
    search_data: Dict, k: int, is_test: bool
) -> Tuple[List[str], List[Tuple[str, str]]]:
    """
    Process the search results to extract page links and snippets.

    :param search_data: The raw search result dictionary.
    :param k: The number of top results to process.
    :param is_test: Flag to indicate if this is a test run.
    :return: A tuple containing a list of page links and a list of (link, snippet) tuples.
    """
    if is_test:
        search_data["organic_results"] = [
            r
            for r in search_data["organic_results"]
            if "https://huggingface.co/datasets/gaia-benchmark/" not in r["link"]
        ]

    page_links = [result["link"] for result in search_data["organic_results"][:k]]
    snippets = [
        (result["link"], result.get("snippet", ""))
        for result in search_data["organic_results"][:k]
    ]

    return filter_content_urls(page_links), snippets


def stream_analysis_progress(
    links: List[str],
    summaries: List[Tuple[str, str]],
):
    """
    Stream the progress of analysis to the provided trackers.

    :param links: List of links to be analyzed.
    :param summaries: List of (link, summary) tuples.
    """
    trackers = [BrowsingToolTracer()]
    for link, (_, summary) in zip(links, summaries):
        for tracker in trackers or []:
            tracker.on_start()
            tracker.on_llm_new_token(
                token=f"Currently searching for information through web browsing…\n- Analyzing content from: {link}\n- Summary: {summary}..",
                run_id=None,
            )
            tracker.on_llm_end()


async def fetch_page_contents(links: List[str], k: int) -> List[Dict]:
    """
    Fetch the contents of the given pages using bulk scraping.

    :param links: List of links to fetch.
    :param k: The number of top results to process.
    :return: A list of dictionaries containing the fetched data.
    """
    try:
        return await bulk_scrape_with_playwright(
            url_list=links,
            max_length=3000,
            user_agent="web-content-fetcher",
            limit=k,
        )
    except Exception as e:
        logger.error(f"An error occurred during content fetching: {str(e)}")
        return []


def merge_summaries_with_results(
    fetch_result: List[Dict], summaries: List[Tuple[str, str]]
) -> List[Dict]:
    """
    Merge summaries with fetch results for pages where content couldn't be fetched.

    :param fetch_result: The result of web content fetching.
    :param summaries: List of (link, summary) tuples.
    :return: Updated list of dictionaries with merged data.
    """
    for result in fetch_result:
        if result.get("text") is None:
            for url, summary in summaries:
                if url == result.get("url"):
                    result["text"] = summary
    return fetch_result


def create_fallback_results(
    summaries: List[Tuple[str, str]], links: List[str], k: int
) -> List[Dict]:
    """
    Create fallback results using summaries when fetching fails.

    :param summaries: List of (link, summary) tuples.
    :param links: List of links that were attempted to be fetched.
    :param k: The number of top results to process.
    :return: A list of dictionaries containing fallback data.
    """
    return [
        {
            "text": summaries[i][1] if i < len(summaries) else "No summary available",
            "html": "",
            "url": links[i] if i < len(links) else "URL not available",
        }
        for i in range(k)
    ]


async def analyze_web_content(
    search_data: Dict, k: int = 5, is_test: bool = False, is_streaming: bool = True
) -> List[Dict]:
    """
    Analyze web content based on search results.

    :param search_data: The raw search result dictionary.
    :param k: The number of top results to process.
    :param is_test: Flag to indicate if this is a test run.
    :return: A list of dictionaries containing the analyzed data.
    """
    featured_snippet = str(search_data.get("featured_snippet", ""))

    if "organic_results" not in search_data:
        return []

    links, summaries = process_search_results(search_data, k, is_test)

    if is_streaming:
        stream_analysis_progress(links, summaries)

    fetched_content = await fetch_page_contents(links, k)

    if fetched_content:
        fetched_content = merge_summaries_with_results(fetched_content, summaries)
        fetched_content.append({"text": featured_snippet, "html": "", "url": ""})
    else:
        fetched_content = create_fallback_results(summaries, links, k)

    return fetched_content


class WebContentAnalyzer(Tool):
    """
    A tool for analyzing web content based on search queries.
    """

    name: str = "searching_and_browsing_tool"
    description: str = """Searching tool. Browsing tool. Information seeking tool.
    Process:
    1. Execute a web search using the provided query.
    2. Analyze the resulting web pages and extract relevant content.

    Note: This tool focuses on content analysis and does not perform data visualization.
    """
    _tool_schema: dict = {
        "type": "function",
        "function": {
            "name": "searching_and_browsing_tool",
            "description": "Searching tool. Browsing tool. Information seeking tool. Sequence of the tool action. 1. Search the result using the Google Search Engine. 2. Browses the web pages and return the page content like text and image. Remember: This isn't visualize tool! If you want to visualize the data, use the code_interpreter tool.",
            "parameters": {
                "type": "object",
                "properties": {
                    "q": {
                        "type": "string",
                        "description": "Search parameter to input in Google",
                    }
                },
                "required": ["q"],
            },
        },
    }
    language: Language

    async def __call__(self, q: str, is_streaming: bool = True) -> List[Dict]:
        """
        Execute the web content analysis tool.

        :param q: The search query to be analyzed.
        :return: A list of dictionaries containing the analyzed data.
        """
        api_key = os.getenv("SERP_API_KEY")
        if not api_key:
            raise ValueError("SERP_API_KEY is not set in the environment variables.")

        max_results = 3
        is_test = False

        search_params = {
            "q": q,
            "api_key": api_key,
            "engine": "google",
            "location": CityEnum.from_language(self.language).value,
            "gl": RegionEnum.from_language(self.language).value,
            "hl": LanguageHLEnum.from_language(self.language).value,
            "google_domain": GoogleDomainEnum.from_language(self.language).value,
            "safe": "active",
        }

        analysis_config: CostSetting = CostSetting.large

        search_client = SerpApiClient(search_params, timeout=10)
        search_data = search_client.get_dict()

        analysis_results: list[dict] = await analyze_web_content(
            search_data, max_results, is_test, is_streaming
        )

        for result in analysis_results:
            result["content"] = limit_content_length(
                result.get("text", ""), analysis_config
            )
            result.pop("text", None)
            result.pop("html", None)

        return analysis_results


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    analyzer = WebContentAnalyzer()
    analyzer.language = Language.Korean

    print(f"Tool Name: {analyzer.name}")
    print(f"Tool Description: {analyzer.description}")
    print(f"Tool Schema: {analyzer._tool_schema}")

    async def run_analyzer(tool):
        try:
            results = await tool("Latest advancements in AI", False)
            for result in results:
                print(
                    f"Content: {result['content'][:100]}..."
                )  # Print first 100 characters
                print(f"URL: {result['url']}")
                print("=" * 50)
        except Exception as e:
            print(f"An error occurred: {str(e)}")

    asyncio.run(run_analyzer(analyzer))
