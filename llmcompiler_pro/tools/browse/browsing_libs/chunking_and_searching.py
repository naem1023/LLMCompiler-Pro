from typing import List

import numpy as np
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import TokenTextSplitter
from logzero import logger
from numpy import dot
from numpy.linalg import norm


def split_text_into_chunks(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """
    Split the input text into overlapping chunks.

    :param text: The input text to be split.
    :param chunk_size: The size of each chunk.
    :param chunk_overlap: The number of overlapping tokens between chunks.
    :return: A list of text chunks.
    """
    text_splitter = TokenTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return text_splitter.split_text(text)


async def generate_embeddings(
    texts: List[str], model: str = "text-embedding-3-large"
) -> np.ndarray:
    """
    Generate embeddings for a list of texts.

    :param texts: A list of texts to embed.
    :param model: The name of the embedding model to use.
    :return: A numpy array of embeddings.
    """
    embeddings = OpenAIEmbeddings(model=model)
    embedded_texts = await embeddings.aembed_documents(texts)
    return np.array(embedded_texts)


def calculate_similarity(
    query_embedding: np.ndarray, chunk_embeddings: np.ndarray
) -> np.ndarray:
    """
    Calculate cosine similarity between a query embedding and chunk embeddings.

    :param query_embedding: The embedding of the query.
    :param chunk_embeddings: The embeddings of the text chunks.
    :return: An array of similarity scores.
    """
    return dot(chunk_embeddings, query_embedding) / (
        norm(chunk_embeddings, axis=1) * norm(query_embedding)
    )


def get_top_k_indices(similarity_scores: np.ndarray, k: int) -> np.ndarray:
    """
    Get the indices of the top k highest similarity scores.

    :param similarity_scores: An array of similarity scores.
    :param k: The number of top scores to return.
    :return: An array of indices for the top k scores.
    """
    return np.argsort(similarity_scores)[::-1][:k]


async def find_similar_chunks(query: str, chunks: List[str], top_k: int) -> List[str]:
    """
    Find the top k chunks most similar to the query.

    :param query: The query string.
    :param chunks: A list of text chunks to search through.
    :param top_k: The number of top similar chunks to return.
    :return: A list of the top k most similar chunks.
    """
    OpenAIEmbeddings(model="text-embedding-3-large")
    chunk_embeddings = await generate_embeddings(chunks)
    query_embedding = await generate_embeddings([query])

    query_embedding = query_embedding.flatten()

    similarity_scores = calculate_similarity(query_embedding, chunk_embeddings)
    top_k_indices = get_top_k_indices(similarity_scores, top_k)

    logger.info(f"Top {top_k} similar chunks indices: {top_k_indices}")
    logger.info(f"Similarity scores: {similarity_scores[top_k_indices]}")

    return [chunks[i] for i in top_k_indices]


async def process_text_and_find_similar(
    query: str, text: str, chunk_size: int, chunk_overlap: int, top_k: int
) -> List[str]:
    """
    Process the input text and find the top k chunks most similar to the query.

    :param query: The query string.
    :param text: The input text to process.
    :param chunk_size: The size of each text chunk.
    :param chunk_overlap: The number of overlapping tokens between chunks.
    :param top_k: The number of top similar chunks to return.
    :return: A list of the top k most similar chunks.
    """
    chunks = split_text_into_chunks(text, chunk_size, chunk_overlap)
    top_k_chunks = await find_similar_chunks(query, chunks, top_k)
    return top_k_chunks


if __name__ == "__main__":
    import asyncio

    async def run_tests():
        test_cases = [
            {
                "name": "Short text about technology",
                "query": "What are the impacts of AI?",
                "text": """Artificial Intelligence (AI) is rapidly transforming various sectors of society. 
                It's enhancing efficiency in industries, revolutionizing healthcare with precise diagnostics, 
                and reshaping education through personalized learning. However, AI also raises concerns about 
                job displacement and ethical considerations in decision-making processes. As AI continues to 
                evolve, it's crucial to balance its benefits with potential risks.""",
                "chunk_size": 50,
                "chunk_overlap": 10,
                "top_k": 2,
            },
            {
                "name": "Longer text about climate change",
                "query": "How does climate change affect biodiversity?",
                "text": """Climate change is one of the most pressing issues of our time, affecting ecosystems 
                worldwide. Rising temperatures are altering habitats, forcing species to migrate or adapt. 
                Polar regions are particularly vulnerable, with melting ice caps threatening species like polar 
                bears and penguins. In tropical areas, coral reefs are bleaching due to warmer ocean temperatures, 
                endangering countless marine species. Forests are also at risk, with changing rainfall patterns 
                and increased frequency of wildfires disrupting delicate balances. The loss of biodiversity due 
                to climate change can have cascading effects on food webs and ecosystem services. It's crucial 
                to implement conservation strategies and reduce greenhouse gas emissions to mitigate these impacts.""",
                "chunk_size": 75,
                "chunk_overlap": 15,
                "top_k": 3,
            },
        ]

        for case in test_cases:
            print(f"\nRunning test: {case['name']}")
            print(f"Query: {case['query']}")
            print(
                f"Parameters: chunk_size={case['chunk_size']}, chunk_overlap={case['chunk_overlap']}, "
                f"top_k={case['top_k']}"
            )

            result = await process_text_and_find_similar(
                case["query"],
                case["text"],
                case["chunk_size"],
                case["chunk_overlap"],
                case["top_k"],
            )

            print("\nTop similar chunks:")
            for i, chunk in enumerate(result, 1):
                print(f"{i}. {chunk}")

            print("\nChunk details:")
            chunks = split_text_into_chunks(
                case["text"], case["chunk_size"], case["chunk_overlap"]
            )
            for i, chunk in enumerate(chunks):
                print(f"Chunk {i + 1}: {chunk[:50]}{'...' if len(chunk) > 50 else ''}")

    asyncio.run(run_tests())
