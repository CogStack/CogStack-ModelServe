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
    """An exception raised due to dataset errors"""


class DeviceNotAvailableError(RuntimeError):
    """An exception raised when a specificy device is required but not available."""


class ExtraDependencyRequiredException(Exception):
    """An exception raised when an extra dependency is required but not found."""
