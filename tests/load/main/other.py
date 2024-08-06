import os
from locust import HttpUser, task, constant_throughput

CMS_BASE_URL = os.environ["CMS_BASE_URL"]


class Other(HttpUser):

    wait_time = constant_throughput(1)

    def on_start(self):
        ...

    @task
    def info(self):
        self.client.get(f"{CMS_BASE_URL}/info")

    @task
    def train_unsupervised(self):
        with open(os.path.join(os.path.dirname(__file__), "..", "data", "sample_texts.json"), "r") as file:
            self.client.post(f"{CMS_BASE_URL}/train_unsupervised?log_frequency=1000", files={"training_data": file})

    @task
    def train_supervised(self):
        with open(os.path.join(os.path.dirname(__file__), "..", "data", "trainer_export.json"), "r") as file:
            self.client.post(f"{CMS_BASE_URL}/train_supervised?epochs=1&log_frequency=1", files={"trainer_export": file})
