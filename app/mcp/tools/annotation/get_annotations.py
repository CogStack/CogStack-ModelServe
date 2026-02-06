from urllib3.exceptions import TimeoutError
from cms_client import AnnotationsApi
from cms_client.exceptions import ApiException
from ...logger import get_logger


logger = get_logger(__name__)

def annotate(text: str, annotation_api: AnnotationsApi) -> dict:
    """
    Gets annotations for the provided text.

    Args:
        text (str): The input text to be annotated.
        annotation_api (AnnotationsApi): The Annotations API instance.

    Returns:
        Dict: A dictionary containing the annotations.
    """
    try:
        api_response = annotation_api.get_entities_from_text(text)
        return api_response.to_dict()
    except ApiException as e:
        logger.error(f"API Exception when calling get_entities_from_text: {e}")
        return {
            "status": "error",
            "reason": e.reason,
        }
    except TimeoutError:
        logger.error("Request timed out when calling get_entities_from_text")
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
