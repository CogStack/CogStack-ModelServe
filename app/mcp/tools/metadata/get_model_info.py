from urllib3.exceptions import TimeoutError
from cms_client import MetadataApi
from cms_client.exceptions import ApiException
from ...logger import get_logger


logger = get_logger(__name__)

def model_info(metadata_api: MetadataApi) -> dict:
    """
    Gets model metadata information.

    Args:
        metadata_api (MetadataApi): The Metadata API instance.

    Returns:
        Dict: A dictionary containing the model metadata information.
    """
    try:
        api_response = metadata_api.get_model_card()
        return api_response.to_dict()
    except ApiException as e:
        logger.error(f"API Exception when calling get_model_card: {e}")
        return {
            "status": "error",
            "reason": e.reason,
        }
    except TimeoutError:
        logger.error("Request timed out when calling get_model_card")
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
