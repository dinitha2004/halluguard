def clamp(value, min_value=0.0, max_value=1.0):
    if value < min_value:
        return min_value
    if value > max_value:
        return max_value
    return value


def compute_final_token_risk(sep_score, hallushift_score):
    """
    Prototype final token risk score.

    Combines:
    - SEP score
    - HalluShift score

    Output: value between 0 and 1
    """
    sep_score = 0.0 if sep_score is None else sep_score
    hallushift_score = 0.0 if hallushift_score is None else hallushift_score

    final_score = (0.6 * sep_score) + (0.4 * hallushift_score)
    return round(clamp(final_score), 4)


def get_risk_label(final_score):
    if final_score >= 0.75:
        return "HIGH"
    if final_score >= 0.45:
        return "MEDIUM"
    return "LOW"