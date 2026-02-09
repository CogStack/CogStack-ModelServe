from urllib3.exceptions import TimeoutError
from cms_client import RedactionApi
from cms_client.exceptions import ApiException
from ...logger import get_logger


logger = get_logger(__name__)

def redact(text: str, redaction_api: RedactionApi) -> dict:
    """
    Redacts sensitive information from the provided text.

    Args:
        text (str): The input text to be redacted.
        redaction_api (RedactionApi): The Redaction API instance.

    Returns:
        str: The redacted text.
    """
    try:
        api_response = redaction_api.get_redacted_text(text)
        return { "redact_text": str(api_response) }
    except ApiException as e:
        logger.error(f"API Exception when calling get_redacted_text: {e}")
        return {
            "status": "error",
            "reason": e.reason,
        }
    except TimeoutError:
        logger.error("Request timed out when calling get_redacted_text")
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
