from typing import Callable


class LogCaptor(object):

    def __init__(self, processor: Callable[[str], None]):
        self.buffer = ""
        self.log_processor = processor

    def write(self, buffer: str) -> None:
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
