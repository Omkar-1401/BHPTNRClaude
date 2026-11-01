"""Non-overlapping labels for Pareto-front points (user, 2026-10-28: the fronts were cluttered with text).

Labels go below-left of their point: by construction no model lies below-left of a front point (it would dominate it),
so that region is empty.  Labels are right-aligned, joined to their point by a thin leader line, and pushed apart
vertically so no two overlap.  The y range is extended downward until the stack fits, and the x range leftward until
the widest label clears the left spine (text extents measured with the renderer, not guessed).
"""
import numpy as np


def _place(ax, pts, dx, dy, gap):
    lo, hi = np.log10(ax.get_ylim())
    ty = []
    for x, y, _ in pts:                                # label i sits at least `gap` below label i-1
        t = (np.log10(y) - lo) / (hi - lo) - dy
        ty.append(min(t, ty[-1] - gap) if ty else t)
    return ty


def label_front(ax, pts, dx=0.02, dy=0.045, gap=0.040, fontsize=8):
    """pts: [(x, y, text)] sorted by x (front order, y decreasing).  Log y axis assumed.  Call after the data is drawn."""
    if not pts:
        return
    for _ in range(8):                                 # y: extend the bottom until the lowest label is inside
        ty = _place(ax, pts, dx, dy, gap)
        if ty[-1] >= 0.03:
            break
        lo, hi = np.log10(ax.get_ylim())
        ax.set_ylim(10 ** (lo - (0.03 - ty[-1] + 0.01) * (hi - lo)), 10 ** hi)
    fig = ax.figure
    for _ in range(8):                                 # x: extend the left edge until every label clears the spine
        for a in getattr(ax, "_front_labels", []):
            a.remove()
        ty = _place(ax, pts, dx, dy, gap)
        ax._front_labels = []
        for (x, y, s), t in zip(pts, ty):
            fx = ax.transLimits.transform(ax.transScale.transform((x, y)))[0]
            ax._front_labels.append(ax.annotate(
                s, (x, y), xytext=(fx - dx, t), textcoords=ax.transAxes, ha="right", va="center", fontsize=fontsize,
                color="#1a1a1a", zorder=5, bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9),
                arrowprops=dict(arrowstyle="-", color="#9a9a9a", lw=0.6, shrinkA=0, shrinkB=3)))
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        left = min(ax.transAxes.inverted().transform(a.get_window_extent(r))[0, 0] for a in ax._front_labels)
        if left >= 0.01:
            break
        x0, x1 = ax.get_xlim()
        ax.set_xlim(x0 - (0.01 - left + 0.01) * (x1 - x0), x1)
