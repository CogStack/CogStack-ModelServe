import os
from locust import HttpUser, task, between

data_dir = os.path.join(os.path.dirname(__file__), "..", "resources", "fixture")


class MainTest(HttpUser):

    wait_time = between(5, 10)

    def on_start(self): ...

    @task(3)
    def info(self):
        self.client.get("http://localhost:8180/info")

    @task(5)
    def process(self):
        with open(os.path.join(data_dir, "sample_text.txt"), "r") as file:
            self.client.post("http://localhost:8180/process", headers={"Content-Type": "text/plain"}, data=file)

    @task(5)
    def process_bulk(self):
        with open(os.path.join(data_dir, "sample_texts.json"), "r") as file:
            self.client.post("http://localhost:8180/process_bulk", headers={"Content-Type": "application/json"}, data=file)

    @task(2)
    def train_unsupervised(self):
        with open(os.path.join(data_dir, "unsupervised.json"), "r") as file:
            self.client.post("http://localhost:8180/train_unsupervised?log_frequency=1000", files={"training_data": file})

    @task(2)
    def train_supervised(self):
        with open(os.path.join(data_dir, "supervised.json"), "r") as file:
            self.client.post("http://localhost:8180/train_supervised?epochs=1&log_frequency=1", files={"training_data": file})
