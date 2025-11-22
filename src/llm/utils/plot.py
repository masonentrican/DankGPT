from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

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

    # Ensure charts directory exists (relative to project root)
    project_root = Path(__file__).resolve().parents[3]
    charts_dir = project_root / "scripts" / "charts"
    charts_dir.mkdir(exist_ok=True)

    plt.savefig(charts_dir / plotname, dpi=150)
    print(f"Saved plot to {charts_dir / plotname}")