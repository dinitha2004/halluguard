from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import gradio as gr
from db.database import init_db, save_run, get_recent_runs, get_run_details
from models.generator import generate_with_hidden_states
from models.eat import build_highlighted_response
from models.aggregator import (
    build_risky_spans,
    compute_overall_hallucination_score,
    get_overall_label,
)

init_db()


def run_demo(prompt):
    prompt = prompt.strip()

    if not prompt:
        return (
            "Please enter a prompt.",
            "No database record saved.",
            "No overall score.",
            "No token preview.",
            "<div>Please enter a prompt.</div>",
            "No span preview."
        )

    try:
        response, token_data = generate_with_hidden_states(prompt)
        highlighted_response = build_highlighted_response(token_data)
        spans = build_risky_spans(token_data, min_label="MEDIUM")
        overall_score = compute_overall_hallucination_score(token_data, spans)
        overall_label = get_overall_label(overall_score)

        run_id = save_run(
            prompt=prompt,
            response_text=response,
            overall_score=overall_score,
            token_scores=token_data,
            spans=spans
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
        overall_preview = f"Overall score: {overall_score} | Overall label: {overall_label}"

        return response, status, overall_preview, token_preview, highlighted_response, span_preview

    except Exception as e:
        return (
            f"Error: {str(e)}",
            "Database save skipped because generation failed.",
            "No overall score.",
            "No token preview.",
            f"<div>Error: {str(e)}</div>",
            "No span preview."
        )
    
def load_recent_runs():
    runs = get_recent_runs(limit=10)

    if not runs:
        return "No saved runs yet."

    lines = []
    for item in runs:
        prompt_preview = item["prompt"].replace("\n", " ")
        if len(prompt_preview) > 60:
            prompt_preview = prompt_preview[:60] + "..."

        lines.append(
            f"Run {item['id']} | Score: {item['overall_score']} | Time: {item['created_at']} | Prompt: {prompt_preview}"
        )

    return "\n".join(lines)


def inspect_saved_run(run_id_text):
    run_id_text = str(run_id_text).strip()

    if not run_id_text:
        return "Please enter a run ID."

    if not run_id_text.isdigit():
        return "Run ID must be a number."

    run_data = get_run_details(int(run_id_text))

    if not run_data:
        return f"No run found for ID {run_id_text}."

    run = run_data["run"]
    tokens = run_data["tokens"]
    spans = run_data["spans"]

    overall_score = run["overall_score"]
    overall_label = get_overall_label(overall_score) if overall_score is not None else "UNKNOWN"

    lines = [
        f"Run ID: {run['id']}",
        f"Created at: {run['created_at']}",
        f"Overall score: {overall_score}",
        f"Overall label: {overall_label}",
        f"Prompt: {run['prompt']}",
        f"Response: {run['response_text']}",
        "",
        "TOKENS:",
    ]

    for item in tokens[:10]:
        lines.append(
            f"Step {item['token_index']} | Token: {repr(item['token_text'])} | Final: {item['final_score']} | Label: {item['risk_label']}"
        )

    if len(tokens) > 10:
        lines.append(f"... {len(tokens) - 10} more tokens")

    lines.append("")
    lines.append("SPANS:")

    if spans:
        for span in spans:
            lines.append(
                f"Text: {repr(span['span_text'])} | Steps: {span['start_token_index']}-{span['end_token_index']} | Avg: {span['avg_score']} | Max: {span['max_score']} | Label: {span['span_label']}"
            )
    else:
        lines.append("No spans saved.")

    return "\n".join(lines)


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

    overall_box = gr.Textbox(
    label="Overall Hallucination Score",
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

    gr.Markdown("### Run History / Inspection")

    history_box = gr.Textbox(
        label="Recent Saved Runs",
        lines=8,
        value=load_recent_runs()
    )

    refresh_history_button = gr.Button("Refresh Run History")

    run_id_box = gr.Textbox(
        label="Enter Run ID to Inspect",
        lines=1,
        placeholder="Example: 1"
    )

    inspect_button = gr.Button("Inspect Run")

    run_details_box = gr.Textbox(
        label="Run Details",
        lines=16
    )

    run_button.click(
        fn=run_demo,
        inputs=prompt_box,
        outputs=[answer_box, status_box, overall_box, token_box, highlight_box, span_box]
    )

    refresh_history_button.click(
        fn=load_recent_runs,
        inputs=[],
        outputs=history_box
    )

    inspect_button.click(
        fn=inspect_saved_run,
        inputs=run_id_box,
        outputs=run_details_box
    )

if __name__ == "__main__":
    demo.launch()