import os

os.environ["PYTHONPATH"] = os.path.join(os.path.dirname(__file__), "..", "..")
print(os.environ["PYTHONPATH"])