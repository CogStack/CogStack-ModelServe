import time
import mlflow
from app.config import Settings


def ensure_no_active_run(timeout_seconds: int = 600) -> None:
    active_run = mlflow.active_run()
    if active_run is not None:
        run_id = active_run.info.run_id
        run = mlflow.get_run(run_id)
        start_time = time.monotonic()
        while True:
            if run.info.status != "RUNNING":
                break
            elif time.monotonic() - start_time > timeout_seconds:
                raise TimeoutError(f"Run timed out with ID: {run_id}")
            mlflow.end_run()
            time.sleep(1)


def disable_rate_limits(config: Settings):
    config.PROCESS_RATE_LIMIT = ""
    config.PROCESS_BULK_RATE_LIMIT = ""
    config.GENERATION_RATE_LIMIT = ""


class StringContains(str):
    def __eq__(self, other):
        return self in other
