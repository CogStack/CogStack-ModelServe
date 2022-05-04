from abc import ABC, abstractmethod


class ModelServices(ABC):

    @staticmethod
    @abstractmethod
    def info():
        raise NotImplementedError

    @staticmethod
    def _data_iterator(texts):
        for idx, text in enumerate(texts):
            yield idx, text

    @abstractmethod
    def annotate(self, text):
        raise NotImplementedError

    @abstractmethod
    def batchannotate(self, texts):
        raise NotImplementedError
