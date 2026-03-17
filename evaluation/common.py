"""
Common evaluation utilities for LHAW AgentCompany tasks.
Based on TheAgentCompany/workspaces/base_image/common.py but adapted for LHAW.
"""

import base64
import functools
import logging
import os
import time

import litellm


# OwnCloud utilities for admin/hr tasks that need them
def check_file_in_owncloud_directory(file_name: str, dir_name: str) -> bool:
    """Check if a file exists in the OwnCloud directory structure."""
    # LHAW mounts OwnCloud data at /workspace/inputs/THE OWNCLOUD FOLDER
    full_path = os.path.join("/workspace/inputs/THE OWNCLOUD FOLDER", dir_name, file_name)
    return os.path.exists(full_path) and os.path.isfile(full_path)


def get_binary_file_content_owncloud(file_name: str, dir_name: str) -> bytes:
    """Get binary content of a file from OwnCloud directory."""
    full_path = os.path.join("/workspace/inputs/THE OWNCLOUD FOLDER", dir_name, file_name)
    try:
        with open(full_path, "rb") as f:
            return f.read()
    except Exception as e:
        logging.warning(f"Failed to read binary file {full_path}: {e}")
        return b""


def get_file_content_owncloud(file_name: str, dir_name: str) -> str:
    """Get text content of a file from OwnCloud directory."""
    full_path = os.path.join("/workspace/inputs/THE OWNCLOUD FOLDER", dir_name, file_name)
    try:
        with open(full_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logging.warning(f"Failed to read file {full_path}: {e}")
        return ""


def grader(func):
    """Decorator for grader functions - exact copy from TAC common.py"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            if result is None:
                logging.warning("Grader returns None, using False instead")
                return False
            return result
        except Exception as e:
            logging.error(f"Error in {func.__name__}: {str(e)}")
            return False

    return wrapper


# LLM evaluation utilities
IMAGE_JPEG = "image/jpeg"
IMAGE_PNG = "image/png"


def llm_complete(messages, max_retries: int = 5):
    """Call LiteLLM for evaluation with retry logic.

    Args:
        messages: The messages to send to the LLM
        max_retries: Maximum number of retries on rate limit errors (default: 5)
    """

    litellm_api_key = os.environ.get("LLM_API_KEY")
    litellm_base_url = os.environ.get("LLM_BASE_URL")
    litellm_model = os.environ.get("LLM_EVAL_MODEL", "gpt-4o")

    for attempt in range(max_retries):
        try:
            return litellm.completion(
                api_key=litellm_api_key,
                base_url=litellm_base_url,
                model=litellm_model,
                messages=messages,
            ).model_dump()
        except litellm.RateLimitError as e:
            wait_time = (2**attempt) + 1  # Exponential backoff: 2, 3, 5, 9, 17 seconds
            logging.warning(
                f"Rate limit hit (attempt {attempt + 1}/{max_retries}), waiting {wait_time}s: {e}"
            )
            if attempt < max_retries - 1:
                time.sleep(wait_time)
            else:
                logging.error(f"Max retries exceeded for rate limit: {e}")
                return None
        except litellm.BadRequestError as e:
            # Image not supported by model - log and fail immediately
            logging.error(
                f"LLM completion failed (bad request, likely image not supported by {litellm_model}): {e}"
            )
            return None
        except Exception as e:
            logging.error(f"LLM completion failed: {e}")
            return None

    return None


def evaluate_with_llm(
    content: str,
    predicate: str,
    additional_prompt: str = "",
    image_path: str = None,
    image_type: str = IMAGE_JPEG,
):
    """
    Evaluates if a predicate can be inferred from the content/image, judged by LLM
    """
    if image_path is not None and image_type not in [IMAGE_JPEG, IMAGE_PNG]:
        logging.warning(f"Invalid image type: {image_type}")
        return False
    if not content and not image_path:
        logging.warning("Both content and image are empty, cannot evaluate")
        return False
    elif content and image_path:
        query = f'Does the content """{content}""" and following picture indicate {predicate}?'
    elif content:
        query = f'Does the content """{content}""" indicate {predicate}?'
    else:
        query = f"Does the following picture indicate {predicate}?"

    query += f' Please answer "yes" if it does, or "no" if it does not. {additional_prompt}'

    msg_content = [{"type": "text", "text": query}]

    if image_path:
        try:
            with open(image_path, "rb") as f:
                base64_image = base64.b64encode(f.read()).decode("utf-8")
        except Exception as e:
            logging.error(f"Failed to read image from {image_path}: {e}")
            return False
        msg_content.append(
            {"type": "image_url", "image_url": {"url": f"data:{image_type};base64,{base64_image}"}}
        )

    try:
        # Construct LLM query
        llm_messages = [{"role": "user", "content": msg_content}]

        # Call LLM for evaluation
        llm_response = llm_complete(llm_messages)
        if not llm_response:
            return False

        logging.info("LLM evaluation completed", extra={"response": llm_response})

        # Extract and process response
        response_content = llm_response["choices"][0]["message"]["content"].lower().strip()

        # Evaluate result
        result = "yes" in response_content
        if result:
            logging.info(f'Predicate "{predicate}" evaluated to "{result}"')
        else:
            logging.warning(f'Predicate "{predicate}" evaluated to "{result}"')

        return result

    except KeyError as e:
        logging.error(f"Invalid LLM response structure: {str(e)}", exc_info=True)
        return False

    except Exception as e:
        logging.error(f"Failed to evaluate message: {str(e)}", exc_info=True)
        return False


def verify_image_file(image_path: str) -> bool:
    """
    Verifies if a file exists and is a valid JPEG/JPG or PNG image.

    Args:
        image_path (str): Path to the image file

    Returns:
        bool: True if file exists and is a valid JPEG/JPG or PNG image, False otherwise
    """
    if not image_path or not os.path.isfile(image_path):
        logging.warning(f"Image file not provided or does not exist: {image_path}")
        return False

    try:
        with open(image_path, "rb") as f:
            # Try reading first few bytes to check if it's a valid image
            header = f.read(8)
            # Check for JPEG/JPG or PNG signatures
            if not (
                header.startswith(b"\xFF\xD8\xFF")  # JPEG/JPG (both use same signature)
                or header.startswith(b"\x89PNG\r\n")
            ):  # PNG
                logging.warning(
                    f"Invalid image format for {image_path}. Only JPEG/JPG and PNG are supported."
                )
                return False
        return True
    except Exception as e:
        logging.error(f"Failed to read image file {image_path}: {e}")
        return False


def compare_images_with_llm(
    image_path1: str = None, image_path2: str = None, query="", additional_prompt: str = ""
):
    """
    Evaluates if  2 images based on the query, judged by LLM
    """

    # Validate both images
    if not verify_image_file(image_path1) or not verify_image_file(image_path2):
        return False

    query += f' Please answer "yes" if it does, or "no" if it does not. {additional_prompt}'

    try:
        with open(image_path1, "rb") as f:
            base64_image1 = base64.b64encode(f.read()).decode("utf-8")
        with open(image_path2, "rb") as f:
            base64_image2 = base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        logging.error(f"Failed to read image: {e}")
        return False

    content = [{"type": "text", "text": query}]
    content.extend(
        [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image1}"}},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image2}"}},
        ]
    )

    try:
        # Construct LLM query
        llm_messages = [{"role": "user", "content": content}]

        # Call LLM for evaluation
        try:
            llm_response = llm_complete(llm_messages)
            logging.info("LLM evaluation completed", extra={"response": llm_response})
        except Exception as e:
            logging.error(f"Failed to evaluate images: {str(e)}", exc_info=True)
            return False

        # Check for None response (llm_complete returns None on errors)
        if not llm_response:
            logging.error("LLM returned None response")
            return False

        # Extract and process response
        response_content = llm_response["choices"][0]["message"]["content"].lower().strip()

        # Evaluate result
        result = "yes" in response_content
        if result:
            logging.info(f'Image comparisons evaluated to "{result}"')
        else:
            logging.warning(f'Image comparisons evaluated to "{result}"')

        return result

    except KeyError as e:
        logging.error(f"Invalid LLM response structure: {str(e)}", exc_info=True)
        return False

    except Exception as e:
        logging.error(f"Failed to evaluate message: {str(e)}", exc_info=True)
        return False
