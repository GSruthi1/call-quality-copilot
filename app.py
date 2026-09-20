"""Call Quality Copilot - Gradio UI."""
import os
import threading
from pathlib import Path

import gradio as gr

from main import DIMENSIONS, REVIEW_HIGH, REVIEW_LOW, analyze
from main import _get_vectorstore as get_vectorstore

SAMPLES_DIR = Path(__file__).parent / "samples"

SEVERITY_ICON = {"critical": "🔴", "major": "🟠", "minor": "🟡"}

BANNER_CSS = """
.flag-banner {background:#fff4e5; border:2px solid #f59e0b; color:#7a4300; border-radius:10px;
              padding:14px 18px; font-size:1.05rem; margin:4px 0;}
.flag-banner, .flag-banner * {color:#7a4300 !important;}
.flag-banner b {font-size:1.15rem;}
"""


def _sample_choices() -> list[tuple[str, str]]:
    """(label, filename) pairs, e.g. ('01 - Billing dispute', '01_billing_dispute.txt')."""
    choices = []
    for path in sorted(SAMPLES_DIR.glob("*.txt")):
        number, _, name = path.stem.partition("_")
        choices.append((f"{number} - {name.replace('_', ' ').capitalize()}", path.name))
    return choices


def load_sample(filename: str | None) -> str:
    if not filename:
        return ""
    return (SAMPLES_DIR / filename).read_text(encoding="utf-8")


def _bar(score: int) -> str:
    return "●" * score + "○" * (5 - score)


def _flag_banner(scores: dict) -> str:
    reasons = [f"{dim.replace('_', ' ')} is very low ({value})" for dim, value in scores.items() if value < REVIEW_LOW]
    average = sum(scores.values()) / len(scores)
    if average > REVIEW_HIGH:
        reasons.append(f"the average score is exceptionally high ({average:.1f})")
    return (
        '<div class="flag-banner"><b>⚠️ Flagged for human review</b><br>'
        f"An outlier score needs a person to confirm it: {'; '.join(reasons)}.</div>"
    )


def _findings_markdown(findings: list[dict]) -> str:
    if not findings:
        return "✅ No policy violations found."
    lines = []
    for f in findings:
        where = f"turn {f['turn']}" if f["turn"] is not None else "whole call"
        quote = f" - “{f['agent_quote']}”" if f["agent_quote"] and f["agent_quote"] != "N/A" else ""
        lines.append(f"- {SEVERITY_ICON.get(f['severity'], '⚪')} **{f['policy']}** ({where}){quote}  \n  {f['issue']}")
    return "\n".join(lines)


def run_analysis(transcript: str):
    if not transcript or not transcript.strip():
        raise gr.Error("Paste a transcript or pick a sample first.")
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise gr.Error("ANTHROPIC_API_KEY is not set. Add it as an environment variable (or a Space secret) and restart.")
    try:
        result = analyze(transcript)
    except ValueError as exc:  # unparseable transcript
        raise gr.Error(str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - surface API/network errors in the UI
        raise gr.Error(f"Analysis failed: {exc}") from exc

    scores = result["scores"]
    rows = [[dim.replace("_", " ").title(), scores[dim], _bar(scores[dim])] for dim in DIMENSIONS]
    rows.append(["Average", round(sum(scores.values()) / len(scores), 1), ""])
    banner = _flag_banner(scores) if result["needs_review"] else ""
    coaching = "\n".join(f"{i}. {tip}" for i, tip in enumerate(result["coaching"], 1))
    return banner, rows, coaching, _findings_markdown(result["rag_findings"])


with gr.Blocks(title="Call Quality Copilot") as demo:
    gr.Markdown(
        "# 📞 Call Quality Copilot\n"
        "Scores a Nextel Communications support call on five dimensions, checks it against the policy "
        "knowledge base, and suggests coaching. Outlier scores are flagged for human review."
    )
    with gr.Row():
        sample = gr.Dropdown(choices=_sample_choices(), value=None, label="Sample transcript", scale=3)
        analyze_btn = gr.Button("Analyze", variant="primary", scale=1)
    transcript = gr.Textbox(
        label="Transcript",
        lines=14,
        placeholder="Paste a transcript here. Each line starts with 'Agent:' or 'Customer:'.",
    )

    banner = gr.HTML()
    scores_table = gr.Dataframe(
        headers=["Dimension", "Score (1-5)", ""],
        datatype=["str", "number", "str"],
        interactive=False,
        label="Scores",
    )
    gr.Markdown("### Coaching")
    coaching = gr.Markdown()
    with gr.Accordion("Policy findings (from the knowledge base)", open=False):
        findings = gr.Markdown()

    sample.change(load_sample, inputs=sample, outputs=transcript)
    analyze_btn.click(run_analysis, inputs=transcript, outputs=[banner, scores_table, coaching, findings])

def _warm_up() -> None:
    """Build the knowledge-base index in the background so the first Analyze is not slow."""
    try:
        get_vectorstore()
    except Exception:  # noqa: BLE001 - the real error surfaces on the first Analyze
        pass


if __name__ == "__main__":
    threading.Thread(target=_warm_up, daemon=True).start()
    port = os.getenv("PORT")  # set by hosting platforms such as Railway
    demo.launch(
        server_name="0.0.0.0" if port else None,
        server_port=int(port) if port else None,
        css=BANNER_CSS,
    )
