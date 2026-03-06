import time
from concurrent.futures import ThreadPoolExecutor
from app.processors.data_batcher import mini_batch
from app.processors.data_batcher import MicroBatchScheduler


class TestMiniBatcher:

    def test_mini_batch(self):
        data = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
        batches = mini_batch(data, 3)
        assert next(batches) == ["1", "2", "3"]
        assert next(batches) == ["4", "5", "6"]
        assert next(batches) == ["7", "8", "9"]
        assert next(batches) == ["10"]

    def test_non_positive_batch_size(self):
        data = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
        batches1 = mini_batch(data, 0)
        batches2 = mini_batch(data, -1)
        assert next(batches1) == data
        assert next(batches2) == data


class TestMicroBatchScheduler:

    def test_batch_compatible_requests(self):
        processed_batches = []
        executor = ThreadPoolExecutor(max_workers=2)

        def process_batch(batch):
            processed_batches.append([item["value"] for item in batch])
            for item in batch:
                item["future"].set_result(item["value"])

        batcher = MicroBatchScheduler(
            process_batch_fn=process_batch,
            batch_key_fn=lambda request: request["key"],
            executor=executor,
            max_batch_size=4,
            batch_wait_milliseconds=50,
        )

        future1 = batcher.submit({"key": "key", "value": 1})
        future2 = batcher.submit({"key": "key", "value": 2})

        assert future1.result(timeout=2) == 1
        assert future2.result(timeout=2) == 2
        assert processed_batches == [[1, 2]]
        batcher.stop()
        executor.shutdown(wait=True, cancel_futures=True)

    def test_split_incompatible_requests(self):
        processed_batches = []
        executor = ThreadPoolExecutor(max_workers=2)

        def process_batch(batch):
            processed_batches.append([item["value"] for item in batch])
            for item in batch:
                item["future"].set_result(item["value"])

        batcher = MicroBatchScheduler(
            process_batch_fn=process_batch,
            batch_key_fn=lambda request: request["key"],
            executor=executor,
            max_batch_size=4,
            batch_wait_milliseconds=30,
        )

        future1 = batcher.submit({"key": "key_1", "value": 1})
        future2 = batcher.submit({"key": "key_2", "value": 2})

        assert future1.result(timeout=2) == 1
        assert future2.result(timeout=2) == 2
        assert processed_batches == [[1], [2]]
        batcher.stop()
        executor.shutdown(wait=True, cancel_futures=True)

    def test_split_after_wait_window(self):
        processed_batches = []
        executor = ThreadPoolExecutor(max_workers=2)

        def process_batch(batch):
            processed_batches.append([item["value"] for item in batch])
            for item in batch:
                item["future"].set_result(item["value"])

        batcher = MicroBatchScheduler(
            process_batch_fn=process_batch,
            batch_key_fn=lambda request: request["key"],
            executor=executor,
            max_batch_size=4,
            batch_wait_milliseconds=20,
        )

        future1 = batcher.submit({"key": "same", "value": 1})
        time.sleep(0.08)
        future2 = batcher.submit({"key": "same", "value": 2})

        assert future1.result(timeout=2) == 1
        assert future2.result(timeout=2) == 2
        assert processed_batches == [[1], [2]]
        batcher.stop()
        executor.shutdown(wait=True, cancel_futures=True)
