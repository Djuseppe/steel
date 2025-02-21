import numpy as np

from src.config import settings


def asymmetric_cost_metric(
        y_pred, t_threshold: int | float = settings.t_threshold,
        cost_below: int = settings.cost_below,
        cost_above: int = settings.cost_above,
        weights: np.ndarray | None = None
) -> np.floating:
    """Calculates the weighted asymmetric cost metric."""
    cost = np.where(y_pred < t_threshold, cost_below, cost_above * np.floor((y_pred - t_threshold) / cost_above))
    if weights is None:
        return np.mean(cost)
    return np.mean(cost * weights)
