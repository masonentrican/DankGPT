import matplotlib.pyplot as plt
import torch

from pathlib import Path

from llm.models.feedforward import GELU

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
PROJECT_ROOT = Path(__file__).resolve().parents[1]
charts_dir = PROJECT_ROOT / "charts"
charts_dir.mkdir(exist_ok=True)

plt.savefig(charts_dir / "gelu_vs_relu.png", dpi=150)
print(f"Saved plot to {charts_dir / 'gelu_vs_relu.png'}")