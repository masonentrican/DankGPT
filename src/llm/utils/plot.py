import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import torch

from config.paths import PROJECT_ROOT

def save_plot(plotname):
    # Ensure charts directory exists (relative to project root)
    charts_dir = PROJECT_ROOT / "charts"
    charts_dir.mkdir(exist_ok=True)

    plt.savefig(charts_dir / plotname, dpi=150)
    print(f"Saved plot to {charts_dir / plotname}")

def plot_losses(plotname, epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots(figsize=(5, 3))

    ax1.plot(epochs_seen, train_losses, label='Training Loss')
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation Loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel("Tokens seen")
    fig.tight_layout()

    save_plot(plotname)

def plot_temperature_scaling(scaled_probas, vocab, temperatures):
    x = torch.arange(len(vocab))
    bar_width = 0.15

    fig, ax = plt.subplots(figsize=(5, 3))
    for i, T in enumerate(temperatures):
        rects = ax.bar(x + i * bar_width, scaled_probas[i], bar_width, label=f'Temperature = {T}')

    ax.set_ylabel('Probability')
    ax.set_xticks(x)
    ax.set_xticklabels(vocab.keys(), rotation=90)
    ax.legend()

    plt.tight_layout()

    save_plot("temperature_scaling.png")
