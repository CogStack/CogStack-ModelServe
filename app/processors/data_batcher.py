from typing import Iterable, List, Any


def mini_batch(data: Iterable[Any], batch_size: Any) -> Iterable[List[Any]]:
    """
    Generates batches from the given iterable data.

    Args:
        data (Iterable[Any]): The input data to be batched.
        batch_size (int): The size of each batch. If batch_size is less than
                          or equal to 0, the entire data is treated as one batch.

    Yields:
        List[Any]: A batch of data with size not greater than the specified.
    """

    if batch_size <= 0:
        yield [item for item in data]
        return
    batch: List = []
    for item in data:
        if len(batch) < batch_size:
            batch.append(item)
        else:
            yield batch
            batch.clear()
            batch.append(item)
    if batch:
        yield batch
        batch.clear()
