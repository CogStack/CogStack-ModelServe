import os
import json
import ijson
from locust import HttpUser, task, constant_throughput

CMS_BASE_URL = os.environ["CMS_BASE_URL"]


class ProcessBulk(HttpUser):

    num_of_doc_per_call = 10
    wait_time = constant_throughput(num_of_doc_per_call*1.5)

    def on_start(self):
        ...

    def on_stop(self):
        ...

    @task
    def process_bulk(self):

        with open(os.path.join(os.path.dirname(__file__), "..", "data", "sample_texts.json"), "r") as file:
            batch = []
            texts = ijson.items(file, "item")
            for text in texts:
                if len(batch) < ProcessBulk.num_of_doc_per_call:
                    batch.append(text)
                else:
                    self.client.post(f"{CMS_BASE_URL}/process_bulk", headers={"Content-Type": "application/json"}, data=json.dumps(batch))
                    batch.clear()
                    batch.append(text)
            if batch:
                self.client.post(f"{CMS_BASE_URL}/process_bulk", headers={"Content-Type": "application/json"}, data=json.dumps(batch))
                batch.clear()
