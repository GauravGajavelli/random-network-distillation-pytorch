"""Shared utilities for parsing TensorBoard event files in evaluation scripts."""
from typing import List, Tuple

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_scalar(log_dir: str, tag: str) -> Tuple[List[int], List[float]]:
    """Return (steps, values) for the given scalar tag in log_dir.

    Returns empty lists if the tag is absent so callers can degrade gracefully
    when a metric wasn't logged in a given run.
    """
    ea = EventAccumulator(log_dir, size_guidance={"scalars": 0})
    ea.Reload()
    if tag not in ea.Tags().get("scalars", []):
        return [], []
    events = ea.Scalars(tag)
    return [e.step for e in events], [e.value for e in events]


def tail_mean(values: List[float], n: int = 100) -> float:
    """Mean of the last n values; falls back to mean of all if fewer than n."""
    if not values:
        return float("nan")
    tail = values[-n:]
    return sum(tail) / len(tail)


def available_tags(log_dir: str) -> List[str]:
    """List all scalar tags present in a TensorBoard log directory."""
    ea = EventAccumulator(log_dir, size_guidance={"scalars": 0})
    ea.Reload()
    return sorted(ea.Tags().get("scalars", []))
