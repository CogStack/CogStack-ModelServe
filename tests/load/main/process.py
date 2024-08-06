import os
import ijson
from locust import HttpUser, task, constant_throughput

CMS_BASE_URL = os.environ["CMS_BASE_URL"]


class Process(HttpUser):

    wait_time = constant_throughput(1)

    def on_start(self):
        ...

    def on_stop(self):
        ...

    @task
    def process(self):
        with open(os.path.join(os.path.dirname(__file__), "..", "data", "sample_texts.json"), "r") as file:
            texts = ijson.items(file, "item")
            for text in texts:
                self.client.post(f"{CMS_BASE_URL}/process", headers={"Content-Type": "text/plain"}, data=text)
