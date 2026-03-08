from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import gradio as gr
from db.database import init_db, save_run
from models.generator import generate_with_hidden_states

init_db()


def run_demo(prompt):
    prompt = prompt.strip()

    if not prompt:
        return "Please enter a prompt.", "No database record saved.", "No token preview."

    try:
        response, token_data = generate_with_hidden_states(prompt)

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
                f"Step {item['step']} | Token: {repr(item['token_text'])} | Hidden size: {len(item['hidden_state'])} | SEP: {item['sep_score']}"
            )

        token_preview = "\n".join(preview_lines) if preview_lines else "No token data generated."
        status = f"Saved successfully to SQLite. Run ID: {run_id}"

        return response, status, token_preview

    except Exception as e:
        return (
            f"Error: {str(e)}",
            "Database save skipped because generation failed.",
            "No token preview."
        )


with gr.Blocks(title="HalluGuard Prototype") as demo:
    gr.Markdown("# HalluGuard Prototype")
    gr.Markdown("Llama 3.2 + SQLite + Gradio + TBG Preview")

    prompt_box = gr.Textbox(
        label="Enter your prompt",
        lines=5,
        placeholder="Ask a factual question here..."
    )

    run_button = gr.Button("Generate")

    answer_box = gr.Textbox(
        label="Model Response",
        lines=8
    )

    status_box = gr.Textbox(
        label="System Status",
        lines=2
    )

    token_box = gr.Textbox(
        label="TBG Token Preview",
        lines=12
    )

    run_button.click(
        fn=run_demo,
        inputs=prompt_box,
        outputs=[answer_box, status_box, token_box]
    )

if __name__ == "__main__":
    demo.launch()