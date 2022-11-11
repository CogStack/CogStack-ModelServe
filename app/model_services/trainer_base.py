from abc import ABC, abstractmethod
from typing import TextIO


class SupervisedTrainer(ABC):

    @abstractmethod
    def train_supervised(self,
                         data_file: TextIO,
                         epochs: int,
                         log_frequency: int,
                         training_id: str,
                         input_file_name: str) -> bool:
        raise NotImplementedError


class UnsupervisedTrainer(ABC):

    @abstractmethod
    def train_unsupervised(self,
                           data_file: TextIO,
                           epochs: int,
                           log_frequency: int,
                           training_id: str,
                           input_file_name: str) -> bool:
        raise NotImplementedError
