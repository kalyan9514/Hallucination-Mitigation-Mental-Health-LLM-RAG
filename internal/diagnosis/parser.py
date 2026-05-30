"""
internal/diagnosis/parser.py

Parses the LLM response to extract the diagnosed mental disorder
from the generated text.
"""

import re
import logging
from config.config import settings

logger = logging.getLogger(__name__)


def extract_diagnosis(response_text: str) -> str:
    """
    Extract the diagnosed mental disorder from the LLM response.

    Looks for a line containing 'Diagnosed Mental Disorder' and returns
    the matched disorder. Falls back to checking if any known disorder
    label appears directly in the response. Returns 'Unknown' if not found.
    """
    pattern = r"\*\*Diagnosed Mental Disorder\*\*[:\-]*\s*(\w+(?:\s*\-*\w+)*)"
    match = re.search(pattern, response_text, re.IGNORECASE)

    if match:
        diagnosis = match.group(1).strip()
        logger.info(f"Extracted diagnosis: {diagnosis}")
        return diagnosis

    # Fallback: check if any known label appears in the response
    for label in settings.disorder_labels:
        if label.lower() in response_text.lower():
            logger.info(f"Fallback diagnosis match: {label}")
            return label

    logger.warning("Could not extract diagnosis from response.")
    return "Unknown"


def clean_response(response_text: str) -> str:
    """
    Remove repeated newlines and strip whitespace
    from the raw LLM output before displaying to the user.
    """
    cleaned = re.sub(r"\n{3,}", "\n\n", response_text.strip())
    return cleaned