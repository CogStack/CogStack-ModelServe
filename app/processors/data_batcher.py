import time
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Iterable, List, Any, Dict, Callable, Optional


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


class MicroBatchScheduler:
    """A lightweight micro batch scheduler for grouping compatible requests."""

    def __init__(
        self,
        process_batch_fn: Callable[[List[Dict[str, Any]]], None],
        batch_key_fn: Callable[[Dict[str, Any]], Any],
        executor: ThreadPoolExecutor,
        max_batch_size: int = 8,
        batch_wait_milliseconds: int = 10,
        on_start: Optional[Callable[[int, int], None]] = None,
    ) -> None:
        self._process_batch_fn = process_batch_fn
        self._batch_key_fn = batch_key_fn
        self._executor = executor
        self._max_batch_size = max(max_batch_size, 1)
        self._batch_wait_milliseconds = max(batch_wait_milliseconds, 1)
        self._on_start = on_start
        self._queue: List[Dict[str, Any]] = []
        self._condition = threading.Condition()
        self._worker_started = False
        self._worker_stop = False

    def start(self) -> None:
        """Starts the micro batch scheduler if not already started."""

        if self._worker_started:
            return
        self._worker_stop = False
        self._executor.submit(self._worker_loop)
        self._worker_started = True
        if self._on_start:
            self._on_start(self._max_batch_size, self._batch_wait_milliseconds)

    def submit(self, request: Dict[str, Any]) -> Future:
        """
        Submits a request to the micro batch scheduler

        Args:
            request (Dict[str, Any]): The request as a dictionary to be processed.

        Returns:
            Future: A future that will be set with the result of processing the request.
        """
        self.start()
        future: Future = Future()
        request["future"] = future
        with self._condition:
            self._queue.append(request)
            self._condition.notify()
        return future

    def stop(self) -> None:
        """Stops the micro batch scheduler and waits for the worker to finish."""
        with self._condition:
            self._worker_stop = True
            self._condition.notify_all()

    def _worker_loop(self) -> None:
        while not self._worker_stop:
            batch: List[Dict[str, Any]] = []
            with self._condition:
                while not self._queue and not self._worker_stop:
                    self._condition.wait()
                if self._worker_stop:
                    return

                first = self._queue.pop(0)
                batch_key = self._batch_key_fn(first)
                batch.append(first)
                deadline = time.time() + (self._batch_wait_milliseconds / 1000.0)

                while len(batch) < self._max_batch_size and time.time() < deadline:
                    compatible_index = next(
                        (idx for idx, req in enumerate(self._queue) if self._batch_key_fn(req) == batch_key),
                        None,
                    )
                    if compatible_index is not None:
                        batch.append(self._queue.pop(compatible_index))
                        continue
                    self._condition.wait(timeout=max(deadline - time.time(), 0.0))

            batch = [req for req in batch if not req["future"].done()]
            if not batch:
                continue
            self._process_batch_fn(batch)
