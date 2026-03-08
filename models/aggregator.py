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

def compute_overall_hallucination_score(token_data, spans):
    usable_tokens = [
        item for item in token_data
        if not _is_special_token(item["token_text"])
    ]

    if not usable_tokens:
        return 0.0

    token_risks = [item["final_risk_score"] for item in usable_tokens]
    avg_token_risk = sum(token_risks) / len(token_risks)
    max_token_risk = max(token_risks)

    span_avg_risk = 0.0
    span_max_risk = 0.0

    if spans:
        span_avg_risk = sum(span["avg_risk"] for span in spans) / len(spans)
        span_max_risk = max(span["max_risk"] for span in spans)

    has_high_token = any(item["risk_label"] == "HIGH" for item in usable_tokens)
    has_high_span = any(span["span_label"] == "HIGH" for span in spans)
    has_medium_span = any(span["span_label"] == "MEDIUM" for span in spans)

    base_token_score = (avg_token_risk + max_token_risk) / 2

    if has_high_token or has_high_span:
        overall_score = (0.65 * base_token_score) + (0.35 * span_max_risk)
    elif has_medium_span:
        overall_score = (0.35 * base_token_score) + (0.15 * span_avg_risk)
    else:
        overall_score = 0.20 * base_token_score

    return round(clamp(overall_score), 4)


def get_overall_label(overall_score):
    if overall_score >= 0.75:
        return "HIGH"
    if overall_score >= 0.45:
        return "MEDIUM"
    return "LOW"