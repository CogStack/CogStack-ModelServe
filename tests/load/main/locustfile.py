import os
import json
import ijson
from locust import HttpUser, task, between, constant_throughput

CMS_BASE_URL = os.environ["CMS_BASE_URL"]


class MainTest(HttpUser):

    wait_time = constant_throughput(1)

    def on_start(self): ...

    @task
    def info(self):
        self.client.get(f"{CMS_BASE_URL}/info")

    @task
    def process(self):
        with open(os.path.join(os.path.dirname(__file__), "..", "data", "sample_texts.json"), "r") as file:
            texts = ijson.items(file, "item")
            for text in texts:
                self.client.post(f"{CMS_BASE_URL}/process", headers={"Content-Type": "text/plain"}, data=text)

    @task
    def process_bulk(self):
        num_of_doc_per_call = 10
        with open(os.path.join(os.path.dirname(__file__), "..", "data", "sample_texts.json"), "r") as file:
            batch = []
            texts = ijson.items(file, "item")
            for text in texts:
                if len(batch) < num_of_doc_per_call:
                    batch.append(text)
                else:
                    self.client.post(f"{CMS_BASE_URL}/process_bulk", headers={"Content-Type": "application/json"}, data=json.dumps(batch))
                    batch.clear()
                    batch.append(text)
            if batch:
                self.client.post(f"{CMS_BASE_URL}/process_bulk", headers={"Content-Type": "application/json"}, data=json.dumps(batch))
                batch.clear()


    @task
    def train_unsupervised(self):
        with open(os.path.join(os.path.dirname(__file__), "..", "data", "sample_texts.json"), "r") as file:
            self.client.post(f"{CMS_BASE_URL}/train_unsupervised?log_frequency=1000", files={"training_data": file})

    @task
    def train_supervised(self):
        with open(os.path.join(os.path.dirname(__file__), "..", "data", "trainer_export.json"), "r") as file:
            self.client.post(f"{CMS_BASE_URL}/train_supervised?epochs=1&log_frequency=1", files={"trainer_export": file})
