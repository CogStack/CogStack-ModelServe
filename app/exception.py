class StartTrainingException(Exception):
    """An exception raised due to training not started"""


class TrainingFailedException(Exception):
    """An exception raised due to failure on training"""


class TrainingCancelledException(Exception):
    """An exception raised due to cancellation of training"""


class ConfigurationException(Exception):
    """An exception raised due to configuration errors"""


class AnnotationException(Exception):
    """An exception raised due to annotation errors"""


class ManagedModelException(Exception):
    """An exception raised due to erroneous models"""


class ClientException(Exception):
    """An exception raised due to generic client errors"""


class DatasetException(Exception):
    """ An exception raised due to dataset errors"""
