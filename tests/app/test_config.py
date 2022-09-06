from app.config import Settings


def test_configuration():
    undertest = Settings()
    assert undertest.BASE_MODEL_FILE is not None
    assert undertest.DEVICE is not None
    assert undertest.INCLUDE_SPAN_TEXT is not None
    assert undertest.CONCAT_SIMILAR_ENTITIES is not None
    assert undertest.MLFLOW_TRACKING_URI is not None
