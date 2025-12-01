"""
Download and load GPT-2 model from openai.
"""
from config.paths import MODELS_DIR
from gpt_download import download_and_load_gpt2
from llm.utils.logging import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)

model_path = MODELS_DIR / "gpt2"

logger.info(str(model_path))

settings, params = download_and_load_gpt2("1558M", model_path)

logger.info(f"Settings: {settings}")
logger.info(f"Param dict keys: {params.keys()}")