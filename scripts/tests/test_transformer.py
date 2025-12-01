import torch

from config.models import GPT2_SMALL
from llm.models.transformer import Transformer
from llm.utils.logging import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)

torch.manual_seed(123)
x = torch.randn(2, 4, 768)
block = Transformer(GPT2_SMALL)
output = block(x)

logger.info(f"Input Shape: {x.shape}")
logger.info(f"Output Shape: {output.shape}")