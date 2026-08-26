"""Figure generation.

All figures are produced from the CSVs in ``results/`` -- never hand-authored --
and every figure has a matching table in ``results/processed/`` so that any value
can be read exactly rather than estimated off an axis.

Design rules followed here:

* **One measure per chart, one colour.** A bar chart comparing methods on a
  single metric uses a single hue for every bar; the method names are already on
  the axis, so colouring each bar differently would spend the colour channel on
  information the chart already carries. The optimality reference is the one bar
  drawn in a different, recessive ink.
* **Colour follows the method, not its rank.** The mapping from method to colour
  is fixed globally, so a figure that omits a method does not repaint the others.
* **Series carry a second channel.** Line charts give every series its own marker
  shape as well as its own hue, and label the last point directly, so identity
  never depends on colour alone.
* **Recessive chrome.** Hairline solid gridlines one shade off the surface, no
  top or right spines, generous padding.
* Every bar carries its value as text, which is also the relief required for the
  palette slots that sit below 3:1 contrast on a light surface.

The palette is the validated categorical default (adjacent-pair CVD Delta-E 9.2,
normal-vision 20.8 on a light surface).
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")  # figures are written to disk; no interactive display exists

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from src.utils.io import figure_path  # noqa: E402
from src.utils.metrics import METRIC_LABELS, label_for, sort_methods  # noqa: E402

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#7a7973"
GRID = "#e6e5e2"

# Validated categorical order. Slots are bound to methods once, here, so that a
# method keeps its colour in every figure it appears in.
SLOT_COLOURS: tuple[str, ...] = (
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#4a3aa7",  # violet
    "#e34948",  # red
    "#eda100",  # yellow
)

METHOD_SLOTS: dict[str, int] = {
    "dqn": 0,
    "greedy_objective_aware": 1,
    "supervised_rf": 2,
    "greedy_communication_aware": 3,
    "random": 4,
    "tabular_q": 5,
    "greedy_fastest_device": 3,
    "round_robin": 4,
}

# Exact methods are references rather than competitors, so they are drawn in ink.
REFERENCE_METHODS = frozenset({"dp_lower_bound", "dp_relaxed", "dp_exact", "exhaustive"})

MARKERS: tuple[str, ...] = ("o", "s", "^", "D", "v", "P")


def colour_for(method: str) -> str:
    """Return the fixed colour of a method."""
    if method in REFERENCE_METHODS:
        return TEXT_SECONDARY
    return SLOT_COLOURS[METHOD_SLOTS.get(method, 0) % len(SLOT_COLOURS)]


def marker_for(method: str) -> str:
    """Return the fixed marker shape of a method, the second identity channel."""
    if method in REFERENCE_METHODS:
        return "*"
    return MARKERS[METHOD_SLOTS.get(method, 0) % len(MARKERS)]


def apply_style() -> None:
    """Install the shared Matplotlib style."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "medium",
            "axes.titlepad": 12,
            "axes.labelsize": 10,
            "axes.labelcolor": TEXT_SECONDARY,
            "axes.edgecolor": GRID,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "grid.linestyle": "-",
            "text.color": TEXT_PRIMARY,
            "xtick.color": TEXT_SECONDARY,
            "ytick.color": TEXT_SECONDARY,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "lines.linewidth": 2.0,
            "lines.markersize": 6,
        }
    )


def _finish(axes, *, title: str, subtitle: str | None, x_grid: bool, y_grid: bool) -> None:
    """Apply chrome common to every figure.

    The title and subtitle are drawn as text above the axes rather than through
    ``set_title`` so that the subtitle can carry its own, quieter style without
    colliding with the title.
    """
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)
    axes.xaxis.grid(x_grid)
    axes.yaxis.grid(y_grid)

    title_y = 1.10 if subtitle else 1.04
    axes.text(
        0.0,
        title_y,
        title,
        transform=axes.transAxes,
        fontsize=12.5,
        fontweight="medium",
        color=TEXT_PRIMARY,
        va="bottom",
    )
    if subtitle:
        axes.text(
            0.0,
            1.035,
            subtitle,
            transform=axes.transAxes,
            fontsize=9,
            color=TEXT_MUTED,
            va="bottom",
        )


def save(figure, name: str) -> Path:
    """Write a figure to ``results/figures/<name>.png`` and close it."""
    path = figure_path(name)
    figure.savefig(path)
    plt.close(figure)
    return path


def bars_by_method(
    summary: pd.DataFrame,
    metric: str,
    name: str,
    *,
    title: str,
    subtitle: str | None = None,
    value_format: str = "{:,.3f}",
    reference_method: str | None = None,
    log_scale: bool = False,
) -> Path:
    """Horizontal bar chart of one metric per method, with confidence intervals.

    Args:
        summary: Output of :func:`~src.utils.metrics.summarise_by_method`.
        metric: Metric stem, e.g. ``"objective"``.
        name: File stem for the figure.
        title: Chart title.
        subtitle: Optional line of context under the title.
        value_format: Format string for the value printed at each bar's end.
        reference_method: Method drawn in reference ink, e.g. the optimum.
        log_scale: Whether the value axis is logarithmic, used for runtimes that
            span several orders of magnitude.

    Returns:
        The path of the written figure.
    """
    apply_style()
    column = f"{metric}_mean"
    frame = summary[summary[column].notna()].copy()
    order = sort_methods(frame["method"])
    frame = frame.set_index("method").loc[order].reset_index()
    # Highest bar at the top reads more naturally on a horizontal chart.
    frame = frame.iloc[::-1].reset_index(drop=True)

    positions = range(len(frame))
    values = frame[column].to_numpy()
    low = frame.get(f"{metric}_ci_low", pd.Series(values)).to_numpy()
    high = frame.get(f"{metric}_ci_high", pd.Series(values)).to_numpy()
    errors = [values - low, high - values]

    colours = [
        TEXT_SECONDARY
        if (reference_method is not None and method == reference_method)
        or method in REFERENCE_METHODS
        else SLOT_COLOURS[0]
        for method in frame["method"]
    ]

    height = max(2.6, 0.44 * len(frame) + 1.5)
    figure, axes = plt.subplots(figsize=(7.8, height))
    axes.barh(
        list(positions),
        values,
        height=0.55,
        color=colours,
        xerr=errors,
        error_kw={"ecolor": TEXT_MUTED, "elinewidth": 1.0, "capsize": 3},
    )
    axes.set_yticks(list(positions))
    axes.set_yticklabels([label_for(method) for method in frame["method"]])
    axes.set_xlabel(METRIC_LABELS.get(metric, metric))
    if log_scale:
        axes.set_xscale("log")

    span = float(max(high)) if len(high) else 1.0
    for position, value, upper in zip(positions, values, high, strict=True):
        offset = upper * 1.06 if log_scale else upper + span * 0.02
        axes.text(
            offset,
            position,
            value_format.format(value),
            va="center",
            fontsize=9,
            color=TEXT_SECONDARY,
        )
    axes.set_xlim(right=(span * 3.0 if log_scale else span * 1.18))

    _finish(axes, title=title, subtitle=subtitle, x_grid=True, y_grid=False)
    return save(figure, name)


def dots_by_method(
    summary: pd.DataFrame,
    metric: str,
    name: str,
    *,
    title: str,
    subtitle: str | None = None,
    value_format: str = "{:,.1f}",
    log_scale: bool = True,
) -> Path:
    """Dot plot of one metric per method, for quantities spanning many decades.

    A bar chart on a logarithmic axis is misleading: bar *length* no longer
    encodes the value, and the apparent zero point is wherever the axis happens
    to start. A dot plot encodes the value as position, which stays truthful
    under a log transform -- the right form for runtimes that range from
    microseconds to minutes.
    """
    apply_style()
    column = f"{metric}_mean"
    frame = summary[summary[column].notna()].copy()
    order = sort_methods(frame["method"])
    frame = frame.set_index("method").loc[order].reset_index().iloc[::-1].reset_index(drop=True)

    positions = list(range(len(frame)))
    values = frame[column].to_numpy()
    low = frame.get(f"{metric}_ci_low", pd.Series(values)).to_numpy()
    high = frame.get(f"{metric}_ci_high", pd.Series(values)).to_numpy()

    height = max(2.6, 0.44 * len(frame) + 1.5)
    figure, axes = plt.subplots(figsize=(7.8, height))
    for position, value, lower, upper, method in zip(
        positions, values, low, high, frame["method"], strict=True
    ):
        colour = TEXT_SECONDARY if method in REFERENCE_METHODS else SLOT_COLOURS[0]
        axes.plot([lower, upper], [position, position], color=TEXT_MUTED, linewidth=1.2, zorder=1)
        axes.plot(
            [value],
            [position],
            marker="o",
            markersize=9,
            color=colour,
            markeredgecolor=SURFACE,
            markeredgewidth=1.5,
            zorder=2,
        )
        axes.text(
            upper * 1.25 if log_scale else upper + values.max() * 0.03,
            position,
            value_format.format(value),
            va="center",
            fontsize=9,
            color=TEXT_SECONDARY,
        )

    axes.set_yticks(positions)
    axes.set_yticklabels([label_for(method) for method in frame["method"]])
    axes.set_xlabel(METRIC_LABELS.get(metric, metric))
    if log_scale:
        axes.set_xscale("log")
        axes.set_xlim(min(low) * 0.5, max(high) * 6.0)
    else:
        axes.set_xlim(0, max(high) * 1.2)
    axes.set_ylim(-0.7, len(frame) - 0.3)

    _finish(axes, title=title, subtitle=subtitle, x_grid=True, y_grid=False)
    return save(figure, name)


def lines_by_x(
    summary: pd.DataFrame,
    x_column: str,
    metric: str,
    name: str,
    *,
    title: str,
    subtitle: str | None = None,
    x_label: str | None = None,
    methods: Sequence[str] | None = None,
    log_y: bool = False,
) -> Path:
    """Line chart of one metric against a numeric axis, one line per method.

    Every series gets its own marker shape as well as its own colour, and the
    final point of each line is labelled directly, so the chart stays readable
    without relying on colour alone.
    """
    apply_style()
    column = f"{metric}_mean"
    frame = summary[summary[column].notna()].copy()
    chosen = list(methods) if methods is not None else sort_methods(frame["method"])

    figure, axes = plt.subplots(figsize=(7.6, 4.6))
    right_edge = frame[x_column].max()
    for method in chosen:
        series = frame[frame["method"] == method].sort_values(x_column)
        if series.empty:
            continue
        axes.plot(
            series[x_column],
            series[column],
            color=colour_for(method),
            marker=marker_for(method),
            markeredgecolor=SURFACE,
            markeredgewidth=1.5,
            label=label_for(method),
        )
        last = series.iloc[-1]
        axes.annotate(
            label_for(method),
            xy=(last[x_column], last[column]),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=8.5,
            color=colour_for(method),
        )

    axes.set_xlabel(x_label if x_label is not None else x_column.replace("_", " "))
    axes.set_ylabel(METRIC_LABELS.get(metric, metric))
    if log_y:
        axes.set_yscale("log")
    axes.set_xlim(right=right_edge * 1.24)
    axes.legend(loc="upper left", ncols=2)
    _finish(axes, title=title, subtitle=subtitle, x_grid=False, y_grid=True)
    return save(figure, name)


def grouped_bars(
    summary: pd.DataFrame,
    group_column: str,
    metric: str,
    name: str,
    *,
    title: str,
    subtitle: str | None = None,
    x_label: str | None = None,
    methods: Sequence[str] | None = None,
    group_order: Sequence | None = None,
    value_format: str = "{:,.2f}",
) -> Path:
    """Grouped bar chart: one group per condition, one bar per method.

    Used for the dynamic-condition experiments, where the question is how each
    method's cost changes as the environment changes.
    """
    apply_style()
    column = f"{metric}_mean"
    frame = summary[summary[column].notna()].copy()
    chosen = list(methods) if methods is not None else sort_methods(frame["method"])
    groups = list(group_order) if group_order is not None else list(dict.fromkeys(frame[group_column]))

    figure, axes = plt.subplots(figsize=(8.4, 4.8))
    total = max(len(chosen), 1)
    # A 2px surface gap between adjacent bars rather than a border around them.
    slot_width = 0.82 / total
    bar_width = slot_width * 0.88

    for index, method in enumerate(chosen):
        series = frame[frame["method"] == method].set_index(group_column)
        values, lows, highs, positions = [], [], [], []
        for group_index, group in enumerate(groups):
            if group not in series.index:
                continue
            row = series.loc[group]
            values.append(float(row[column]))
            lows.append(float(row.get(f"{metric}_ci_low", row[column])))
            highs.append(float(row.get(f"{metric}_ci_high", row[column])))
            positions.append(group_index - 0.41 + slot_width * (index + 0.5))
        if not values:
            continue
        errors = [
            [value - low for value, low in zip(values, lows, strict=True)],
            [high - value for value, high in zip(values, highs, strict=True)],
        ]
        bars = axes.bar(
            positions,
            values,
            width=bar_width,
            color=colour_for(method),
            label=label_for(method),
            yerr=errors,
            error_kw={"ecolor": TEXT_MUTED, "elinewidth": 1.0, "capsize": 2.5},
        )
        for bar, value in zip(bars, values, strict=True):
            axes.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                value_format.format(value),
                ha="center",
                va="bottom",
                fontsize=7.5,
                color=TEXT_SECONDARY,
                rotation=90 if total > 3 else 0,
            )

    axes.set_xticks(range(len(groups)))
    axes.set_xticklabels([str(group).replace("_", " ") for group in groups])
    axes.set_xlabel(x_label if x_label is not None else group_column.replace("_", " "))
    axes.set_ylabel(METRIC_LABELS.get(metric, metric))
    axes.margins(y=0.18)
    axes.legend(loc="upper left", ncols=min(len(chosen), 3))
    _finish(axes, title=title, subtitle=subtitle, x_grid=False, y_grid=True)
    return save(figure, name)


__all__ = [
    "GRID",
    "METHOD_SLOTS",
    "SLOT_COLOURS",
    "SURFACE",
    "apply_style",
    "bars_by_method",
    "colour_for",
    "dots_by_method",
    "grouped_bars",
    "lines_by_x",
    "marker_for",
    "save",
]
