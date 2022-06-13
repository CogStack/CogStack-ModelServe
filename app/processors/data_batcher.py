from typing import Iterable, List


def mini_batch(data: Iterable, batch_size: int) -> Iterable[List]:
    if batch_size <= 0:
        yield [text for text in data]
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
