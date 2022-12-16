from app.processors.data_batcher import mini_batch


def test_mini_batch():
    data = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
    batches = mini_batch(data, 3)
    assert next(batches) == ["1", "2", "3"]
    assert next(batches) == ["4", "5", "6"]
    assert next(batches) == ["7", "8", "9"]
    assert next(batches) == ["10"]


def test_non_positive_batch_size():
    data = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
    batches1 = mini_batch(data, 0)
    batches2 = mini_batch(data, -1)
    assert next(batches1) == data
    assert next(batches2) == data
