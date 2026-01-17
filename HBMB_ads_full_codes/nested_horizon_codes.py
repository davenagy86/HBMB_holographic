import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, FancyArrowPatch
import os

out_dir = "figs_nested_horizons"
os.makedirs(out_dir, exist_ok=True)

fig, ax = plt.subplots(figsize=(10, 5.5))

# Master screen / code (schematic)
master = Circle((0, 0), 3.25, fill=False, linewidth=2.5)
ax.add_patch(master)
ax.text(0, 3.6, "Master screen / code\n$\\mathcal{H}_{\\mathrm{code}}$", ha="center", va="center")

# Local overlapping horizons/screens (default patch colors, only alpha adjusted)
local_specs = [
    ((-1.35, 0.95), 1.55, "H_1", "U_1"),
    (( 1.35, 0.95), 1.55, "H_2", "U_2"),
    (( 0.00,-0.95), 1.70, "H_3", "U_3"),
]

for (cx, cy), r, Hi, Ui in local_specs:
    circ = Circle((cx, cy), r, alpha=0.18, linewidth=2.0)
    ax.add_patch(circ)
    ax.text(cx, cy, f"${Ui}$", ha="center", va="center")
    ax.text(cx + 0.95*r, cy + 0.95*r, f"${Hi}$", ha="center", va="center")

# Overlap label (redundancy)
ax.text(0, 1.05, "overlap\n(redundancy)", ha="center", va="center")
ax.text(0, 0.25, "$U_1\\cap U_2$", ha="center", va="center", fontsize=10)

# Reconstruction box (bulk slice / accessible algebra)
box_x, box_y = 5.35, 0.40
box_w, box_h = 3.2, 1.3
box = FancyBboxPatch(
    (box_x - box_w/2, box_y - box_h/2),
    box_w, box_h,
    boxstyle="round,pad=0.25,rounding_size=0.08",
    linewidth=2.0,
    fill=False
)
ax.add_patch(box)
ax.text(box_x, box_y, "Reconstruction\n(bulk slice / \naccessible subalgebra)", ha="center", va="center")

# Arrows from local screens to reconstruction box
arrow_starts = [(-0.05, 1.95), (1.75, 0.35), (0.75, -1.55)]
for sx, sy in arrow_starts:
    ax.add_patch(FancyArrowPatch(
        (sx, sy), (box_x - box_w/2, box_y),
        arrowstyle="-|>", mutation_scale=14, linewidth=2.0
    ))

# Small note
ax.text(-3.35, -3.05, "\n \n \n schematic code-structure, not a spacetime map",
        ha="left", va="center", fontsize=9, alpha=0.7)

ax.set_aspect("equal", "box")
ax.set_xlim(-4.2, 7.6)
ax.set_ylim(-4.0, 4.2)
ax.axis("off")

png_path = os.path.join(out_dir, "nested_horizons_overlapping_codes.png")
pdf_path = os.path.join(out_dir, "nested_horizons_overlapping_codes.pdf")
fig.savefig(png_path, dpi=300, bbox_inches="tight")
fig.savefig(pdf_path, bbox_inches="tight")
plt.close(fig)

print("Saved:", png_path, pdf_path)
