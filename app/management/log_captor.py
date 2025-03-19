from typing import Callable, final


@final
class LogCaptor(object):
    """
    The captor used to capture log messages and process them line by line.
    """

    def __init__(self, processor: Callable[[str], None]):
        """
        Initialises a new instance of LogCaptor with a specific log processor function.

        Args:
            processor: A callable that takes a log message and processes it.
        """

        self.buffer = ""
        self.log_processor = processor

    def write(self, buffer: str) -> None:
        """
        Processes log messages which are segmented by newline characters.

        Args:
            buffer (str): A string containing the log message to be processed.
        """

        while buffer:
            try:
                newline_idx = buffer.index("\n")
            except ValueError:
                self.buffer += buffer
                break
            log = self.buffer + buffer[:newline_idx+1]
            self.buffer = ""
            buffer = buffer[newline_idx+1:]
            self.log_processor(log)
