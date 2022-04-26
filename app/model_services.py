from abc import ABC, abstractmethod


class ModelServices(ABC):

    @abstractmethod
    def annotate(self, text):
        raise NotImplementedError

    @abstractmethod
    def batchannotate(self, texts):
        raise NotImplementedError
    
    def _data_iterator(self, texts):
        for idx, text in enumerate(texts):
            yield (id, text)
