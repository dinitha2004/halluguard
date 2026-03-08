import html


def is_special_token(token_text: str) -> bool:
    return token_text.startswith("<|") and token_text.endswith("|>")


def token_to_html(token_text: str) -> str:
    escaped = html.escape(token_text)
    escaped = escaped.replace(" ", "&nbsp;")
    escaped = escaped.replace("\n", "<br>")
    return escaped


def get_token_style(risk_label: str) -> str:
    if risk_label == "HIGH":
        return "background-color:#ffb3b3; color:#7a0000; padding:2px 0; border-radius:3px;"
    return ""


def build_highlighted_response(token_data):
    html_parts = []

    for item in token_data:
        token_text = item["token_text"]

        if is_special_token(token_text):
            continue

        risk_label = item.get("risk_label", "LOW")
        final_risk_score = item.get("final_risk_score", 0.0)

        token_html = token_to_html(token_text)
        token_style = get_token_style(risk_label)

        if risk_label == "HIGH":
            html_parts.append(
                f'<span title="Risk: {final_risk_score} | Label: {risk_label}" style="{token_style}">{token_html}</span>'
            )
        else:
            html_parts.append(token_html)

    combined_html = "".join(html_parts).strip()

    if not combined_html:
        combined_html = "No response generated."

    return f"""
    <div style="line-height:1.8; font-size:16px; padding:10px; border:1px solid #ddd; border-radius:8px;">
        {combined_html}
    </div>
    """