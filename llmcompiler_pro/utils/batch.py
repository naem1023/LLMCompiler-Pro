from itertools import islice
from typing import Iterable, Iterator, List, TypeVar

ElementT = TypeVar("ElementT")


def create_batch(iterator: Iterator[ElementT], size: int) -> List[ElementT]:
    """
    Creates a single batch of specified size from the given iterator.

    :param iterator: An iterator to draw elements from.
    :param size: The maximum number of elements to include in the batch.
    :return: A list containing the batch elements.
    """
    return list(islice(iterator, size))


def get_batches(
    source: Iterable[ElementT], batch_size: int
) -> Iterator[List[ElementT]]:
    """
    Generates batches of a specified size from the given iterable.

    This function takes an iterable source and yields lists of elements,
    where each list has a maximum length equal to the specified batch size.
    The final batch may contain fewer elements if the source's length
    is not evenly divisible by the batch size.

    :param source: The input iterable to be divided into batches.
    :param batch_size: The maximum number of elements in each batch.
    :return: An iterator yielding lists of elements, each representing a batch.

    Example usage:
        numbers = range(10)
        for batch in generate_batches(numbers, 3):
            print(batch)

    Output:
        [0, 1, 2]
        [3, 4, 5]
        [6, 7, 8]
        [9]
    """
    iterator = iter(source)
    while True:
        batch = create_batch(iterator, batch_size)
        if not batch:
            return
        yield batch
