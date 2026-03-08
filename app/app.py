from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import gradio as gr
from db.database import init_db, save_run
from models.generator import generate_with_hidden_states
from models.eat import build_highlighted_response
from models.aggregator import build_risky_spans

init_db()


def run_demo(prompt):
    prompt = prompt.strip()

    if not prompt:
        return (
            "Please enter a prompt.",
            "No database record saved.",
            "No token preview.",
            "<div>Please enter a prompt.</div>",
            "No span preview."
        )

    try:
        response, token_data = generate_with_hidden_states(prompt)
        highlighted_response = build_highlighted_response(token_data)
        spans = build_risky_spans(token_data, min_label="MEDIUM")

        run_id = save_run(
            prompt=prompt,
            response_text=response,
            overall_score=None,
            token_scores=[],
            spans=[]
        )

        preview_lines = []
        for item in token_data[:10]:
            preview_lines.append(
                f"Step {item['step']} | Token: {repr(item['token_text'])} | Hidden size: {len(item['hidden_state'])} | SEP: {item['sep_score']} | Shift: {item['hallushift_score']} | Risk: {item['final_risk_score']} | Label: {item['risk_label']}"
            )

        token_preview = "\n".join(preview_lines) if preview_lines else "No token data generated."

        span_lines = []
        for i, span in enumerate(spans, start=1):
            span_lines.append(
                f"Span {i} | Text: {repr(span['span_text'])} | Steps: {span['start_step']}-{span['end_step']} | Avg risk: {span['avg_risk']} | Max risk: {span['max_risk']} | Label: {span['span_label']}"
            )

        span_preview = "\n".join(span_lines) if span_lines else "No candidate spans detected."
        status = f"Saved successfully to SQLite. Run ID: {run_id}"

        return response, status, token_preview, highlighted_response, span_preview

    except Exception as e:
        return (
            f"Error: {str(e)}",
            "Database save skipped because generation failed.",
            "No token preview.",
            f"<div>Error: {str(e)}</div>",
            "No span preview."
        )


with gr.Blocks(title="HalluGuard Prototype") as demo:
    gr.Markdown("# HalluGuard Prototype")
    gr.Markdown("Llama 3.2 + SQLite + Gradio + Token Highlighting + Span Aggregation")

    prompt_box = gr.Textbox(
        label="Enter your prompt",
        lines=5,
        placeholder="Ask a factual question here..."
    )

    run_button = gr.Button("Generate")

    answer_box = gr.Textbox(
        label="Model Response",
        lines=5
    )

    status_box = gr.Textbox(
        label="System Status",
        lines=2
    )

    token_box = gr.Textbox(
        label="TBG Token Preview",
        lines=12
    )

    gr.Markdown("### Highlighted Risk View")
    highlight_box = gr.HTML()

    span_box = gr.Textbox(
        label="Span Aggregator Preview",
        lines=8
    )

    run_button.click(
        fn=run_demo,
        inputs=prompt_box,
        outputs=[answer_box, status_box, token_box, highlight_box, span_box]
    )

if __name__ == "__main__":
    demo.launch()