"""Statistics for comparing placement methods.

Every method is evaluated on the *same* scenario seeds, so comparisons here are
**paired**: the right question is "how much better is A than B on the same
problem", not "are the two populations different". Paired analysis removes the
scenario-to-scenario variance, which in this study is far larger than the
difference between methods -- an unpaired comparison would drown any real effect
in noise.

Confidence intervals are reported for every headline number, and the paired
Wilcoxon signed-rank test is used rather than a t-test because objective ratios
across scenarios are noticeably right-skewed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class Interval:
    """A point estimate with a confidence interval."""

    estimate: float
    low: float
    high: float
    n: int

    @property
    def half_width(self) -> float:
        """Half the width of the interval, for symmetric error bars."""
        return (self.high - self.low) / 2.0

    def format(self, decimals: int = 3) -> str:
        """Render as ``estimate [low, high]``."""
        return (
            f"{self.estimate:.{decimals}f} "
            f"[{self.low:.{decimals}f}, {self.high:.{decimals}f}]"
        )


@dataclass(frozen=True)
class PairedComparison:
    """The result of comparing two methods over the same scenarios.

    Attributes:
        name_a: Label of the first method.
        name_b: Label of the second method.
        difference: Mean of ``a - b`` with its confidence interval. Negative
            means A costs less, which is better for every metric here.
        relative_difference_pct: Mean of ``(a - b) / b`` as a percentage.
        win_rate: Fraction of scenarios in which A is strictly cheaper than B.
        p_value: Two-sided paired Wilcoxon signed-rank p-value.
        n: Number of paired scenarios.
    """

    name_a: str
    name_b: str
    difference: Interval
    relative_difference_pct: float
    win_rate: float
    p_value: float
    n: int

    def summary(self) -> str:
        """One-line human-readable verdict."""
        direction = "better" if self.difference.estimate < 0 else "worse"
        return (
            f"{self.name_a} vs {self.name_b}: {abs(self.relative_difference_pct):.2f}% "
            f"{direction} on average, wins {self.win_rate:.0%} of {self.n} scenarios, "
            f"p={self.p_value:.2e}"
        )


def mean_interval(values, confidence: float = 0.95) -> Interval:
    """Mean with a Student-t confidence interval.

    Args:
        values: Sample values.
        confidence: Coverage of the interval.

    Returns:
        An :class:`Interval`. A single observation yields a degenerate interval
        rather than an error, so callers need not special-case tiny samples.
    """
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return Interval(float("nan"), float("nan"), float("nan"), 0)
    mean = float(array.mean())
    if array.size == 1:
        return Interval(mean, mean, mean, 1)
    standard_error = float(stats.sem(array))
    if standard_error == 0.0:
        return Interval(mean, mean, mean, int(array.size))
    half = standard_error * stats.t.ppf((1 + confidence) / 2.0, array.size - 1)
    return Interval(mean, mean - half, mean + half, int(array.size))


def bootstrap_interval(
    values,
    confidence: float = 0.95,
    resamples: int = 10_000,
    seed: int = 0,
) -> Interval:
    """Percentile bootstrap confidence interval for the mean.

    Used where the sampling distribution is visibly skewed, such as optimality
    gaps, which are bounded below by zero but have a long right tail.
    """
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return Interval(float("nan"), float("nan"), float("nan"), 0)
    if array.size == 1:
        value = float(array[0])
        return Interval(value, value, value, 1)
    rng = np.random.default_rng(seed)
    draws = rng.choice(array, size=(resamples, array.size), replace=True).mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    return Interval(
        estimate=float(array.mean()),
        low=float(np.quantile(draws, tail)),
        high=float(np.quantile(draws, 1.0 - tail)),
        n=int(array.size),
    )


def paired_comparison(
    values_a,
    values_b,
    name_a: str = "A",
    name_b: str = "B",
    confidence: float = 0.95,
) -> PairedComparison:
    """Compare two methods measured on the same scenarios, in the same order.

    Args:
        values_a: Metric for method A, one entry per scenario.
        values_b: Metric for method B, aligned with ``values_a``.
        name_a: Label for method A.
        name_b: Label for method B.
        confidence: Coverage of the reported interval.

    Returns:
        A :class:`PairedComparison`.

    Raises:
        ValueError: If the two sequences have different lengths, which would mean
            the comparison is not actually paired.
    """
    first = np.asarray(list(values_a), dtype=np.float64)
    second = np.asarray(list(values_b), dtype=np.float64)
    if first.shape != second.shape:
        raise ValueError(
            f"paired comparison needs aligned samples, got {first.shape} and {second.shape}"
        )

    differences = first - second
    with np.errstate(divide="ignore", invalid="ignore"):
        relative = np.where(second != 0, differences / second, np.nan)

    if np.allclose(differences, 0.0):
        p_value = 1.0
    else:
        p_value = float(stats.wilcoxon(first, second, zero_method="zsplit").pvalue)

    return PairedComparison(
        name_a=name_a,
        name_b=name_b,
        difference=mean_interval(differences, confidence),
        relative_difference_pct=float(np.nanmean(relative) * 100.0),
        win_rate=float(np.mean(differences < 0.0)),
        p_value=p_value,
        n=int(first.size),
    )


__all__ = [
    "Interval",
    "PairedComparison",
    "bootstrap_interval",
    "mean_interval",
    "paired_comparison",
]
