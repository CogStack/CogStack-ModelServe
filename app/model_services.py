from abc import ABC, abstractmethod


class ModelServices(ABC):

    @staticmethod
    @abstractmethod
    def info():
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def load_model(model_file_path, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def _data_iterator(texts):
        for idx, text in enumerate(texts):
            yield idx, text

    @abstractmethod
    def annotate(self, text):
        raise NotImplementedError

    @abstractmethod
    def batch_annotate(self, texts):
        raise NotImplementedError
