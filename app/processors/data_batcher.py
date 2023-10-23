from typing import Iterable, List, Any


def mini_batch(data: Iterable[Any], batch_size: Any) -> Iterable[List[Any]]:
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
