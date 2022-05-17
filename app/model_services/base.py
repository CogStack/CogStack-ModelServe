from abc import ABC, abstractmethod


class AbstractModelService(ABC):

    @staticmethod
    @abstractmethod
    def load_model(model_file_path, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def _data_iterator(texts):
        for idx, text in enumerate(texts):
            yield idx, text

    @abstractmethod
    def info(self):
        raise NotImplementedError

    @abstractmethod
    def annotate(self, text):
        raise NotImplementedError

    @abstractmethod
    def batch_annotate(self, texts):
        raise NotImplementedError
