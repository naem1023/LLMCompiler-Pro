import os
import re
import urllib.parse
from urllib.parse import urljoin

import html2text
from bs4 import BeautifulSoup

VALID_IMAGE_TYPES = ["jpg", "png", "webp"]


def extract_image_data(content: str) -> list[tuple]:
    """
    Extracts image data from markdown-style image references in the given content.

    :param content: A string containing markdown-style image references.
    :return: A list of tuples, where each tuple contains (alt_text, image_url).
    """
    pattern = r"!\[(.*?)\]\((.*?)\)"
    matches = re.findall(pattern, content)
    return [(alt, url) for alt, url in matches]


def html_to_plain_text(markup: str) -> str:
    """
    Converts HTML markup to plain text.

    :param markup: A string containing HTML markup.
    :return: A string of plain text extracted from the HTML.
    """
    converter = html2text.HTML2Text()
    converter.ignore_links = converter.ignore_emphasis = converter.ignore_tables = True
    converter.ul_item_mark = converter.emphasis_mark = converter.strong_mark = ""
    converter.body_width = 0
    return converter.handle(markup)


def clean_text(content: str) -> str:
    """
    Cleans the input text by removing excessive whitespace and newlines.

    :param content: A string to be cleaned.
    :return: A cleaned string with normalized whitespace.
    """
    return re.sub(
        r"\n+|\t+|\s{4}|\s{2}", lambda m: "\n" if m.group(0) == "\n" else "", content
    )


def get_base_url(full_url: str, include_path: bool = False) -> str:
    """
    Extracts the base URL from a full URL.

    :param full_url: The full URL to be processed.
    :param include_path: A boolean indicating whether to include the path in the result.
    :return: A string containing the base URL.
    """
    parsed = urllib.parse.urlparse(full_url)
    path = "/".join(parsed.path.split("/")[:-1]) if include_path else ""
    cleaned = parsed._replace(path=path, params="", query="", fragment="")
    return cleaned.geturl()


def get_file_extension(url: str) -> str:
    """
    Extracts the file extension from a given URL.

    :param url: The URL from which to extract the file extension.
    :return: A string containing the file extension without the leading dot.
    """
    return os.path.splitext(url)[1].lstrip(".")


def normalize_url(url: str, base: str) -> str:
    """
    Normalizes a URL by ensuring it has a proper scheme and is absolute.

    :param url: The URL to normalize.
    :param base: The base URL to use for relative URLs.
    :return: A normalized, absolute URL.
    """
    if url.startswith("//"):
        return f"https:{url}"
    if url.startswith("/"):
        return urljoin(base, url)
    return url


def get_og_image(markup: str) -> str:
    """
    Extracts the Open Graph image URL from HTML markup.

    :param markup: A string containing HTML markup.
    :return: The URL of the Open Graph image if found, an empty string otherwise.
    """
    soup = BeautifulSoup(markup, "html.parser")
    og_tag = soup.find("meta", property="og:image")
    return str(og_tag.get("content", "")) if og_tag else ""


def prepare_html_for_image_extraction(
    markup: str, base: str, remove_non_alt: bool = True
) -> str:
    """
    Prepares HTML for image extraction by modifying image tags.

    :param markup: A string containing HTML markup.
    :param base: The base URL for normalizing relative URLs.
    :param remove_non_alt: A boolean indicating whether to remove images without alt text.
    :return: A string of processed HTML markup.
    """
    soup = BeautifulSoup(markup, "html.parser")
    for img in soup.find_all("img"):
        src = img.get("llmcompiler_pro", "")
        if img and get_file_extension(src) in VALID_IMAGE_TYPES:
            alt = img.get("alt", "")
            if not alt and remove_non_alt:
                img.extract()
            else:
                img["llmcompiler_pro"] = normalize_url(src, base)
                img["alt"] = alt.replace(" ", "_")
        else:
            img.extract()
    return soup.prettify()


def extract_images(markup: str, base_url: str) -> list[tuple]:
    """
    Extracts image data from HTML markup.

    :param markup: A string containing HTML markup.
    :param base_url: The base URL for normalizing image URLs.
    :return: A list of tuples containing (alt_text, image_url) for each image.
    """
    processed_html = prepare_html_for_image_extraction(markup, base_url)
    plain_text = html_to_plain_text(processed_html)
    clean_content = clean_text(plain_text)
    return extract_image_data(clean_content)
