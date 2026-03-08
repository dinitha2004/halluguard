import math
from statistics import mean, pstdev


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def compute_sep_score(hidden_state):
    """
    Temporary SEP-style token score.
    Input: one token hidden-state vector (list of floats)
    Output: score between 0 and 1

    This is a lightweight prototype score based on:
    - average absolute activation
    - hidden-state standard deviation

    Later, this can be replaced by a trained probe.
    """
    if not hidden_state:
        return 0.0

    abs_vals = [abs(x) for x in hidden_state]

    avg_abs = mean(abs_vals)
    std_val = pstdev(hidden_state) if len(hidden_state) > 1 else 0.0

    raw_score = (0.6 * avg_abs) + (0.4 * std_val)

    # normalize to 0 to 1 range using sigmoid
    score = sigmoid(raw_score - 1.5)

    return round(score, 4)