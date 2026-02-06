from urllib3.exceptions import TimeoutError
from cms_client import TrainingApi
from cms_client.exceptions import ApiException
from ...logger import get_logger


logger = get_logger(__name__)

def get_train_eval_info(train_eval_id: str, training_api: TrainingApi) -> dict:
    """
    Gets training or evaluation status of the running model.

    Args:
        train_eval_id (str): The ID of the training or evaluation job.
        training_api (TrainingApi): The Training API instance.

    Returns:
        Dict: A dictionary containing the training or evaluation status.
    """
    try:
        api_response = training_api.train_eval_info(train_eval_id)
        return api_response[0]  # type: ignore
    except ApiException as e:
        logger.error(f"API Exception when calling train_eval_info: {e}")
        return {
            "status": "error",
            "reason": e.reason,
        }
    except TimeoutError:
        logger.error("Request timed out when calling train_eval_info")
        return {
            "status":  "error",
            "reason": "Request timed out. Retrying or aborting gracefully.",
        }
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return {
            "status":  "error",
            "reason": f"Unexpected error: {e}",
        }
