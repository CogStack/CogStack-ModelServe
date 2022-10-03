import inspect
import os
import sys

current_frame = inspect.currentframe()
if current_frame is None:
    raise Exception("Cannot detect the parent directory!")
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(current_frame))))
sys.path.insert(0, parent_dir)
