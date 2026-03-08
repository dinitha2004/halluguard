def clamp(value, min_value=0.0, max_value=1.0):
    if value < min_value:
        return min_value
    if value > max_value:
        return max_value
    return value


def compute_final_token_risk(sep_score, hallushift_score):
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


def _is_special_token(token_text: str) -> bool:
    return token_text.startswith("<|") and token_text.endswith("|>")


def _finalize_span(span_items):
    span_text = "".join(item["token_text"] for item in span_items).strip()
    start_step = span_items[0]["step"]
    end_step = span_items[-1]["step"]

    risks = [item["final_risk_score"] for item in span_items]
    avg_risk = round(sum(risks) / len(risks), 4)
    max_risk = round(max(risks), 4)

    if any(item["risk_label"] == "HIGH" for item in span_items):
        span_label = "HIGH"
    else:
        span_label = "MEDIUM"

    return {
        "span_text": span_text,
        "start_step": start_step,
        "end_step": end_step,
        "avg_risk": avg_risk,
        "max_risk": max_risk,
        "span_label": span_label,
        "token_count": len(span_items),
    }


def build_risky_spans(token_data, min_label="MEDIUM"):
    label_rank = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
    min_rank = label_rank.get(min_label, 1)

    spans = []
    current_span = []

    for item in token_data:
        token_text = item["token_text"]
        risk_label = item.get("risk_label", "LOW")

        if _is_special_token(token_text):
            if current_span:
                spans.append(_finalize_span(current_span))
                current_span = []
            continue

        if label_rank.get(risk_label, 0) >= min_rank:
            current_span.append(item)
        else:
            if current_span:
                spans.append(_finalize_span(current_span))
                current_span = []

    if current_span:
        spans.append(_finalize_span(current_span))

    return spans