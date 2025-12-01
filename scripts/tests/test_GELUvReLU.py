import matplotlib.pyplot as plt
import torch

from config.paths import PROJECT_ROOT
from llm.models.feedforward import GELU
from llm.utils.logging import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)

"""
Test the GELU and ReLU activation functions.
"""

gelu, relu = GELU(), torch.nn.ReLU()

x = torch.linspace(-3, 3, 100)
y_gelu, y_relu = gelu(x), relu(x)
plt.figure(figsize=(8,3))
for i, (y, label) in enumerate(zip([y_gelu, y_relu], ["GELU", "ReLU"]), 1):
    plt.subplot(1, 2, i)
    plt.plot(x, y)
    plt.title(f"{label} Activation Function")
    plt.xlabel("x")
    plt.ylabel(f"{label} (x)")
    plt.grid(True)
plt.tight_layout()

# Ensure charts directory exists (relative to project root)
charts_dir = PROJECT_ROOT / "charts"
charts_dir.mkdir(exist_ok=True)

plt.savefig(charts_dir / "gelu_vs_relu.png", dpi=150)
logger.info(f"Saved plot to {charts_dir / 'gelu_vs_relu.png'}")