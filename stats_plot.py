"""
Plot generation for ablation and baseline results.

Can be called programmatically via generate_plots(), or run standalone
with the hardcoded demo values below.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

WIN_COLOR  = "#4CAF50"
DRAW_COLOR = "#FFC107"
LOSS_COLOR = "#F44336"

# ── Demo values (used when running this file directly) ────────────────────────
# Each entry: (label, wins, draws, losses, avg_moves, avg_time_per_move_s, avg_game_duration_s)

_DEMO_ABLATION = [
    ("MCTS base",               35, 5, 60, 28.3, 0.42, 47.2),
    ("+ normalisation",         52, 6, 42, 26.1, 0.43, 44.8),
    ("+ priors heuristique",    65, 5, 30, 24.8, 0.61, 60.5),
    ("+ réutilisation arbre",   72, 4, 24, 23.5, 0.63, 59.1),
]


# ── Core plot functions ───────────────────────────────────────────────────────

def stacked_bar(results, title, output_path):
    labels = [r[0] for r in results]
    wins   = np.array([r[1] for r in results])
    draws  = np.array([r[2] for r in results])
    losses = np.array([r[3] for r in results])

    total    = wins + draws + losses
    win_pct  = wins  / total * 100
    draw_pct = draws / total * 100
    loss_pct = losses / total * 100

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(labels))
    bar_width = 0.5

    ax.bar(x, win_pct,  bar_width, label="Victoire", color=WIN_COLOR)
    ax.bar(x, draw_pct, bar_width, label="Nul",      color=DRAW_COLOR, bottom=win_pct)
    ax.bar(x, loss_pct, bar_width, label="Défaite",  color=LOSS_COLOR, bottom=win_pct + draw_pct)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=10)
    ax.set_ylabel("Pourcentage (%)")
    ax.set_ylim(0, 100)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, fontsize=10, frameon=False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_path}")


def metrics_table(results, output_path):
    col_labels = ["Matchup", "Victoires", "Nuls", "Défaites", "% Victoires",
                  "Coups moy.", "Temps/coup (s)", "Durée partie (s)"]
    col_widths = [0.22, 0.10, 0.07, 0.10, 0.12, 0.12, 0.14, 0.13]

    rows = []
    for label, wins, draws, losses, avg_len, avg_time, avg_duration in results:
        total = wins + draws + losses
        win_pct = f"{wins / total * 100:.1f}%"
        rows.append([label, wins, draws, losses, win_pct,
                     f"{avg_len:.1f}", f"{avg_time:.2f}s", f"{avg_duration:.1f}s"])

    fig, ax = plt.subplots(figsize=(14, len(results) * 0.8 + 1.8))
    ax.axis("off")

    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        colWidths=col_widths,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.2)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#37474F")
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#F5F5F5")
        cell.set_edgecolor("#CCCCCC")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_path}")


# ── Public entry point called by run_stats.py ─────────────────────────────────

def generate_plots(ablation_results, bar_title, output_dir="plots"):
    """
    Generate bar chart + table for one ablation study.

    Parameters
    ----------
    ablation_results : list of (label, wins, draws, losses, avg_moves, avg_time, avg_duration)
    bar_title        : title shown on the bar chart
    output_dir       : folder where PNGs are saved (created if missing)
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    stacked_bar(ablation_results, bar_title, out / "bar.png")
    metrics_table(ablation_results,           out / "table.png")


# ── Standalone demo ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    generate_plots(
        _DEMO_ABLATION,
        "Progression de l'agent MCTS — 20 parties vs Minmax",
        output_dir="plots/mcts",
    )
