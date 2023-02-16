class StartTrainingException(Exception):
    """ An exception raised due to training not started"""


class TrainingFailedException(Exception):
    """ An exception raised due to failure on training"""


class ConfigurationException(Exception):
    """ An exception raised due to configuration errors"""
