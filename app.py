"""Call Quality Copilot - Gradio UI."""
import os
import threading
from pathlib import Path

import anthropic
import gradio as gr

from main import DIMENSIONS, REVIEW_HIGH, REVIEW_LOW, is_waiting_for_review, pending_review, resume_analysis, start_analysis
from main import _get_vectorstore as get_vectorstore

SAMPLES_DIR = Path(__file__).parent / "samples"

SEVERITY_ICON = {"critical": "🔴", "major": "🟠", "minor": "🟡"}

BANNER_CSS = """
.flag-banner {background:#fff4e5; border:2px solid #f59e0b; border-radius:10px; padding:14px 18px; font-size:1.05rem; margin:4px 0;}
.flag-banner, .flag-banner * {color:#7a4300 !important;}
.review-done {background:#ecfdf3; border:2px solid #16a34a; border-radius:10px; padding:14px 18px; font-size:1.05rem; margin:4px 0;}
.review-done, .review-done * {color:#14532d !important;}
.flag-banner b, .review-done b {font-size:1.15rem;}
"""

WAITING_TEXT = "*Coaching is generated after the review is submitted.*"


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


def _label(dim: str) -> str:
    return dim.replace("_", " ").title()


def _bar(score: int) -> str:
    return "●" * score + "○" * (5 - score)


def _score_rows(scores: dict, original: dict | None = None) -> list[list]:
    """Table rows; when a reviewer changed a score, the model's original value is shown next to it."""
    rows = []
    for dim in DIMENSIONS:
        bar = _bar(scores[dim])
        if original and original[dim] != scores[dim]:
            bar += f"   (model said {original[dim]})"
        rows.append([_label(dim), scores[dim], bar])
    rows.append(["Average", round(sum(scores.values()) / len(scores), 1), ""])
    return rows


def _flag_reasons(scores: dict) -> list[str]:
    reasons = [f"{dim.replace('_', ' ')} is very low ({value})" for dim, value in scores.items() if value < REVIEW_LOW]
    average = sum(scores.values()) / len(scores)
    if average > REVIEW_HIGH:
        reasons.append(f"the average score is exceptionally high ({average:.1f})")
    return reasons


def _flag_banner(scores: dict) -> str:
    return (
        '<div class="flag-banner"><b>⚠️ Flagged for human review</b><br>'
        f"An outlier score needs a person to confirm it: {'; '.join(_flag_reasons(scores))}.<br>"
        "The pipeline is paused. Confirm or correct the scores below to generate coaching.</div>"
    )


def _review_banner(review: dict) -> str:
    changed = [
        f"{dim.replace('_', ' ')} {review['original_scores'][dim]} → {review['final_scores'][dim]}"
        for dim in DIMENSIONS if review["original_scores"][dim] != review["final_scores"][dim]
    ]
    outcome = f"scores edited ({'; '.join(changed)})" if changed else "scores approved as they were"
    note = f"<br>Reviewer note: {review['note']}" if review["note"] else ""
    was = "; ".join(_flag_reasons(review["original_scores"]))
    return f'<div class="review-done"><b>✅ Reviewed by a human: {outcome}</b><br>Flagged because: {was}.{note}</div>'


def _findings_markdown(findings: list[dict]) -> str:
    if not findings:
        return "✅ No policy violations found."
    lines = []
    for f in findings:
        where = f"turn {f['turn']}" if f["turn"] is not None else "whole call"
        quote = f" - “{f['agent_quote']}”" if f["agent_quote"] and f["agent_quote"] != "N/A" else ""
        lines.append(f"- {SEVERITY_ICON.get(f['severity'], '⚪')} **{f['policy']}** ({where}){quote}  \n  {f['issue']}")
    return "\n".join(lines)


def _guarded(fn, *args):
    """Run a pipeline call and turn API failures into short messages for the UI."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise gr.Error("ANTHROPIC_API_KEY is not set. Add it as an environment variable and restart.")
    try:
        return fn(*args)
    except ValueError as exc:  # unparseable transcript or invalid reviewer scores
        raise gr.Error(str(exc)) from exc
    except anthropic.AuthenticationError as exc:
        raise gr.Error("Anthropic rejected the API key. Check ANTHROPIC_API_KEY.") from exc
    except anthropic.BadRequestError as exc:
        if "credit balance" in str(exc).lower():
            raise gr.Error("The Anthropic account is out of credits. Add credits in the Anthropic console (Plans & Billing), then try again.") from exc
        raise gr.Error(f"Analysis failed: {exc}") from exc
    except anthropic.RateLimitError as exc:
        raise gr.Error("Anthropic is rate-limiting requests. Wait a minute and try again.") from exc
    except Exception as exc:  # noqa: BLE001 - surface other API/network errors in the UI
        raise gr.Error(f"Analysis failed: {exc}") from exc


def run_analysis(transcript: str):
    """Analyze button. Returns: banner, table, coaching, findings, findings-open, review panel, 5 sliders, note, thread id."""
    if not transcript or not transcript.strip():
        raise gr.Error("Paste a transcript or pick a sample first.")
    thread_id, result = _guarded(start_analysis, transcript)
    payload = pending_review(result)
    findings = _findings_markdown(result["rag_findings"])
    if payload is None:  # not flagged: finished
        coaching = "\n".join(f"{i}. {tip}" for i, tip in enumerate(result["coaching"], 1))
        return ("", _score_rows(result["scores"]), coaching, findings, gr.Accordion(open=False),
                gr.Group(visible=False), *[result["scores"][d] for d in DIMENSIONS], "", None)
    scores = payload["scores"]  # flagged: paused at human_review
    return (_flag_banner(scores), _score_rows(scores), WAITING_TEXT, findings, gr.Accordion(open=True),
            gr.Group(visible=True), *[scores[d] for d in DIMENSIONS], "", thread_id)


def submit_review(thread_id: str | None, *values):
    """Submit-review button: resume the paused run with the reviewer's scores and note."""
    *slider_values, note = values
    if not thread_id or not is_waiting_for_review(thread_id):
        raise gr.Error("There is no pending review. Click Analyze to start again.")
    scores = {dim: int(v) for dim, v in zip(DIMENSIONS, slider_values)}
    result = _guarded(resume_analysis, thread_id, {"scores": scores, "note": note, "reviewer": "reviewer"})
    review = {**result["review"], "final_scores": result["scores"]}
    coaching = "\n".join(f"{i}. {tip}" for i, tip in enumerate(result["coaching"], 1))
    return (_review_banner(review), _score_rows(result["scores"], review["original_scores"]), coaching,
            gr.skip(), gr.skip(), gr.Group(visible=False), None)


with gr.Blocks(title="Call Quality Copilot") as demo:
    gr.Markdown(
        "# 📞 Call Quality Copilot\n"
        "Scores a Nextel Communications support call on five dimensions, checks it against the policy "
        "knowledge base, and suggests coaching. Outlier scores pause the pipeline until a person reviews them."
    )
    with gr.Row():
        sample = gr.Dropdown(choices=_sample_choices(), value=None, label="Sample transcript", scale=3)
        analyze_btn = gr.Button("Analyze", variant="primary", scale=1)
    transcript = gr.Textbox(
        label="Transcript",
        lines=14,
        placeholder="Paste a transcript here. Each line starts with 'Agent:' or 'Customer:'.",
    )
    thread = gr.State(None)

    banner = gr.HTML()
    scores_table = gr.Dataframe(
        headers=["Dimension", "Score (1-5)", ""],
        datatype=["str", "number", "str"],
        interactive=False,
        label="Scores",
    )
    with gr.Group(visible=False) as review_panel:
        gr.Markdown("### Reviewer decision\nConfirm the scores, or correct any you disagree with, then submit.")
        with gr.Row():
            sliders = [gr.Slider(1, 5, value=3, step=1, label=_label(dim)) for dim in DIMENSIONS]
        review_note = gr.Textbox(label="Reviewer note (optional)", lines=2)
        submit_btn = gr.Button("Submit review and generate coaching", variant="primary")
    gr.Markdown("### Coaching")
    coaching = gr.Markdown()
    with gr.Accordion("Policy findings (from the knowledge base)", open=False) as findings_box:
        findings = gr.Markdown()

    sample.change(load_sample, inputs=sample, outputs=transcript)
    analyze_btn.click(
        run_analysis, inputs=transcript,
        outputs=[banner, scores_table, coaching, findings, findings_box, review_panel, *sliders, review_note, thread],
    )
    submit_btn.click(
        submit_review, inputs=[thread, *sliders, review_note],
        outputs=[banner, scores_table, coaching, findings, findings_box, review_panel, thread],
    )


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
