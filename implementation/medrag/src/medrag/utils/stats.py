from __future__ import annotations

import math
from statistics import NormalDist
from typing import Sequence



def wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    n = int(n)
    k = int(k)
    if n <= 0:
        return 0.0, 0.0
    z = float(NormalDist().inv_cdf(1.0 - float(alpha) / 2.0))
    phat = float(k) / float(n)
    denom = 1.0 + (z * z) / float(n)
    center = (phat + (z * z) / (2.0 * float(n))) / denom
    radius = (
        z
        * math.sqrt((phat * (1.0 - phat) / float(n)) + (z * z) / (4.0 * float(n) * float(n)))
        / denom
    )
    lo = max(0.0, float(center - radius))
    hi = min(1.0, float(center + radius))
    return lo, hi


def _binomial_two_sided_p(min_count: int, total: int) -> float:
    if total <= 0:
        return 1.0
    tail = 0.0
    for i in range(0, int(min_count) + 1):
        tail += math.comb(int(total), int(i)) * (0.5 ** int(total))
    return float(min(1.0, 2.0 * tail))


def mcnemar(b: int, c: int, exact: bool = True) -> dict[str, float | int | str]:
    b = int(b)
    c = int(c)
    total = int(b + c)
    if total <= 0:
        return {
            "b": b,
            "c": c,
            "n_discordant": total,
            "statistic": 0.0,
            "p_value": 1.0,
            "method": "degenerate",
        }
    if bool(exact):
        p_value = _binomial_two_sided_p(min(b, c), total)
        statistic = float((abs(b - c) - 1.0) ** 2 / float(total)) if total > 0 else 0.0
        return {
            "b": b,
            "c": c,
            "n_discordant": total,
            "statistic": statistic,
            "p_value": float(p_value),
            "method": "exact_binomial",
        }
    statistic = float((abs(b - c) - 1.0) ** 2 / float(total))
    p_value = float(1.0 - math.erf(math.sqrt(max(statistic, 0.0) / 2.0)))
    return {
        "b": b,
        "c": c,
        "n_discordant": total,
        "statistic": statistic,
        "p_value": p_value,
        "method": "chi2_cc",
    }


def mcnemar_from_paired(baseline_correct: Sequence[bool], current_correct: Sequence[bool], exact: bool = True) -> dict[str, float | int | str]:
    if len(baseline_correct) != len(current_correct):
        raise ValueError("baseline_correct and current_correct must have the same length")
    b = 0
    c = 0
    for base_ok, cur_ok in zip(baseline_correct, current_correct):
        if bool(base_ok) and not bool(cur_ok):
            b += 1
        elif (not bool(base_ok)) and bool(cur_ok):
            c += 1
    return mcnemar(b=b, c=c, exact=exact)


def paired_bootstrap_ci(
    baseline: Sequence[float],
    current: Sequence[float],
    alpha: float = 0.05,
    n_boot: int = 10000,
    seed: int = 1234,
) -> tuple[float, float]:
    import numpy as np
    if len(baseline) != len(current):
        raise ValueError("baseline and current must have the same length")
    n = len(baseline)
    if n == 0:
        return 0.0, 0.0
    base = np.asarray([float(x) for x in baseline], dtype=np.float64)
    cur = np.asarray([float(x) for x in current], dtype=np.float64)
    diff = cur - base
    rng = np.random.default_rng(int(seed))
    samples = np.empty(int(n_boot), dtype=np.float64)
    for i in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        samples[i] = float(np.mean(diff[idx]))
    lo = float(np.quantile(samples, float(alpha) / 2.0))
    hi = float(np.quantile(samples, 1.0 - float(alpha) / 2.0))
    return lo, hi


def _self_test() -> None:
    lo, hi = wilson_ci(20, 52)
    assert abs(lo - 0.262) < 0.01, (lo, hi)
    assert abs(hi - 0.516) < 0.01, (lo, hi)
    mc = mcnemar(10, 2)
    assert mc["n_discordant"] == 12


_self_test()
