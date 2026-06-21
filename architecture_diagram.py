import matplotlib.pyplot as plt

steps = [
    "3D Molecule",
    "Molecular Graph",
    "Atom Features (11D)",
    "GIN Layer 1",
    "GIN Layer 2",
    "GIN Layer 3",
    "Residual Connections",
    "Global Mean Pooling",
    "MLP Head",
    "19 Quantum Properties"
]

fig, ax = plt.subplots(figsize=(6, 10))

ax.axis("off")

for i, step in enumerate(steps):

    y = len(steps) - i

    ax.text(
        0.5,
        y,
        step,
        ha="center",
        va="center",
        fontsize=12,
        bbox=dict(
            boxstyle="round,pad=0.5",
            fc="skyblue",
            ec="black"
        )
    )

    if i < len(steps) - 1:
        ax.annotate(
            "",
            xy=(0.5, y - 0.6),
            xytext=(0.5, y - 0.1),
            arrowprops=dict(
                arrowstyle="->",
                lw=2
            )
        )

ax.set_xlim(0, 1)
ax.set_ylim(0, len(steps) + 1)

plt.title(
    "Residual Multi-Task GIN Architecture",
    fontsize=14,
    pad=20
)

plt.tight_layout()

plt.savefig(
    "results/architecture_diagram.png",
    dpi=300
)

plt.show()