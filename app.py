"""Call Quality Copilot - Gradio UI."""
import html
import os
import threading
from pathlib import Path

import anthropic
import gradio as gr

from main import DIMENSIONS, REVIEW_HIGH, REVIEW_LOW, is_waiting_for_review, pending_review, resume_analysis, start_analysis
from main import _get_vectorstore as get_vectorstore

SAMPLES_DIR = Path(__file__).parent / "samples"

THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.slate,
    secondary_hue=gr.themes.colors.slate,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
    radius_size=gr.themes.sizes.radius_sm,
).set(
    body_background_fill="#f5f6f8",
    body_background_fill_dark="#0f1115",
    block_shadow="none",
    block_border_width="1px",
    button_primary_background_fill="#1f2937",
    button_primary_background_fill_hover="#111827",
    button_primary_border_color="#1f2937",
    button_primary_text_color="#ffffff",
)

APP_CSS = """
.gradio-container {max-width: 1400px !important;}
.html-container {padding: 0 !important;}
footer {display: none !important;}

.app-header {padding: 6px 2px 2px;}
.app-header h1 {font-size: 1.5rem; font-weight: 650; letter-spacing: -0.015em; margin: 0 0 4px;}
.app-header p {margin: 0; max-width: 60rem; font-size: .95rem; color: var(--body-text-color-subdued);}
.app-header .meta {margin-top: 6px; font-size: .74rem; letter-spacing: .07em; text-transform: uppercase; color: var(--body-text-color-subdued);}

.panel {border: 1px solid var(--border-color-primary); border-radius: 8px; padding: 14px 16px; background: var(--background-fill-primary);}
.panel-title {font-size: .72rem; font-weight: 650; letter-spacing: .08em; text-transform: uppercase; color: var(--body-text-color-subdued); margin-bottom: 8px;}
.panel-hint {font-size: .9rem; color: var(--body-text-color-subdued); margin: -2px 0 6px;}
.empty {border: 1px dashed var(--border-color-primary); border-radius: 8px; padding: 56px 20px; text-align: center; color: var(--body-text-color-subdued);}

.notice {border: 1px solid; border-left-width: 4px; border-radius: 8px; padding: 12px 16px;}
.notice-label {font-size: .72rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; margin-bottom: 3px;}
.notice.review {background: #fff7e6; border-color: #efc574; border-left-color: #d97706;}
.notice.review, .notice.review * {color: #6b3f00 !important;}
.notice.reviewed {background: #eefaf3; border-color: #a9d8bd; border-left-color: #15803d;}
.notice.reviewed, .notice.reviewed * {color: #14532d !important;}

.panel table.scores, .panel table.scores tr, .panel table.scores td {border: 0 !important; background: transparent !important;}
.panel table.scores {width: 100%; border-collapse: collapse;}
.panel table.scores td {padding: 8px 4px !important; border-top: 1px solid var(--border-color-primary) !important; vertical-align: middle;}
.panel table.scores tr:first-child td {border-top: 0 !important;}
.review-panel {border: 1px solid var(--border-color-primary); border-radius: 8px; padding: 14px 16px !important; background: var(--background-fill-primary); gap: 10px;}
table.scores .dim {width: 36%;}
table.scores .num {width: 20%; font-weight: 650; font-variant-numeric: tabular-nums;}
table.scores .meter {width: 44%;}
table.scores tr.avg td {font-weight: 650;}
.track {height: 8px; border-radius: 4px; background: var(--background-fill-secondary); overflow: hidden;}
.fill {height: 100%; border-radius: 4px;}
.fill.low {background: #b91c1c;} .fill.mid {background: #b45309;} .fill.high {background: #15803d;}
.was {margin-left: 8px; font-size: .78rem; font-weight: 400; color: var(--body-text-color-subdued);}

.tag {display: inline-block; padding: 1px 8px; border-radius: 999px; font-size: .68rem; font-weight: 700; letter-spacing: .05em; text-transform: uppercase;}
.tag.critical {background: #fde8e8; color: #9b1c1c;}
.tag.major {background: #fdf0dc; color: #92400e;}
.tag.minor {background: #eef1f5; color: #475569;}
.coach ol {margin: 0; padding-left: 1.25rem;}
.coach li {margin: 0 0 8px; line-height: 1.5;}
.coach li:last-child {margin-bottom: 0;}
.findings summary {cursor: pointer; list-style: none; font-size: .72rem; font-weight: 650; letter-spacing: .08em; text-transform: uppercase; color: var(--body-text-color-subdued);}
.findings summary::-webkit-details-marker {display: none;}
.findings summary::before {content: "▸"; display: inline-block; width: 1.1em;}
.findings[open] summary::before {content: "▾";}
.findings[open] summary {margin-bottom: 8px;}
.findings .count {margin-left: 6px; padding: 0 7px; border-radius: 999px; background: var(--background-fill-secondary); letter-spacing: 0;}
.findings > .issue {margin-top: 6px;}
.finding {padding: 10px 0; border-top: 1px solid var(--border-color-primary);}
.finding:first-child {border-top: 0; padding-top: 2px;}
.finding .policy {margin-left: 6px; font-weight: 600;}
.finding .where {margin-left: 6px; font-size: .85rem; color: var(--body-text-color-subdued);}
.finding .quote {margin-top: 4px; font-style: italic;}
.finding .issue {margin-top: 2px; color: var(--body-text-color-subdued);}
"""

HEADER_HTML = """
<div class="app-header">
  <h1>Call Quality Copilot</h1>
  <p>Scores a support call on five dimensions, audits it against company policy, and drafts coaching for the agent.
  Outlier scores pause the pipeline until a person reviews them.</p>
  <div class="meta">Demo data: fictional company &ldquo;Nextel Communications&rdquo;</div>
</div>
"""

EMPTY_HTML = '<div class="empty">Choose a sample call or paste a transcript, then click Analyze.</div>'
COACHING_WAITING = ('<div class="panel coach"><div class="panel-title">Coaching</div>'
                    '<div class="issue">Generated after the review is submitted.</div></div>')


def _esc(value) -> str:
    return html.escape(str(value), quote=True)


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


def _score_table(scores: dict, original: dict | None = None) -> str:
    """Score table with a meter per dimension; a reviewer's change shows the model's original value."""
    rows = []
    for dim in DIMENSIONS:
        value = scores[dim]
        level = "low" if value <= 2 else "mid" if value == 3 else "high"
        was = f'<span class="was">was {original[dim]}</span>' if original and original[dim] != value else ""
        rows.append(
            f'<tr><td class="dim">{_label(dim)}</td><td class="num">{value}{was}</td>'
            f'<td class="meter"><div class="track"><div class="fill {level}" style="width:{value * 20}%"></div></div></td></tr>'
        )
    average = sum(scores.values()) / len(scores)
    rows.append(f'<tr class="avg"><td class="dim">Average</td><td class="num">{average:.1f}</td><td></td></tr>')
    return f'<div class="panel"><div class="panel-title">Scores</div><table class="scores">{"".join(rows)}</table></div>'


def _flag_reasons(scores: dict) -> list[str]:
    reasons = [f"{dim.replace('_', ' ')} is very low ({value})" for dim, value in scores.items() if value < REVIEW_LOW]
    average = sum(scores.values()) / len(scores)
    if average > REVIEW_HIGH:
        reasons.append(f"the average score is exceptionally high ({average:.1f})")
    return reasons


def _flag_banner(scores: dict) -> str:
    return (
        '<div class="notice review flag-banner"><div class="notice-label">Review required</div>'
        f"Outlier scores need a person to confirm them: {_esc('; '.join(_flag_reasons(scores)))}. "
        "The pipeline is paused until the scores are confirmed or corrected.</div>"
    )


def _review_banner(review: dict) -> str:
    changed = [
        f"{dim.replace('_', ' ')} {review['original_scores'][dim]} → {review['final_scores'][dim]}"
        for dim in DIMENSIONS if review["original_scores"][dim] != review["final_scores"][dim]
    ]
    outcome = f"Scores edited: {'; '.join(changed)}." if changed else "Scores approved as they were."
    note = f"<br>Reviewer note: {_esc(review['note'])}" if review["note"] else ""
    was = _esc("; ".join(_flag_reasons(review["original_scores"])))
    return (
        '<div class="notice reviewed review-done"><div class="notice-label">Reviewed by a person</div>'
        f"{_esc(outcome)} Flagged because: {was}.{note}</div>"
    )


def _findings_html(findings: list[dict], expanded: bool) -> str:
    """Policy findings as a collapsible panel (a native <details> element, styled in APP_CSS)."""
    if not findings:
        return '<div class="panel findings"><div class="panel-title" style="margin:0">Policy findings</div><div class="issue">No policy violations found.</div></div>'
    items = []
    for f in findings:
        severity = f["severity"] if f["severity"] in ("critical", "major", "minor") else "minor"
        where = f"turn {f['turn']}" if f["turn"] is not None else "whole call"
        has_quote = f["agent_quote"] and f["agent_quote"] != "N/A"
        quote = f'<div class="quote">“{_esc(f["agent_quote"])}”</div>' if has_quote else ""
        items.append(
            f'<div class="finding"><span class="tag {severity}">{severity}</span>'
            f'<span class="policy">{_esc(f["policy"])}</span><span class="where">{where}</span>'
            f'{quote}<div class="issue">{_esc(f["issue"])}</div></div>'
        )
    opened = " open" if expanded else ""
    return (f'<details class="panel findings"{opened}><summary>Policy findings <span class="count">{len(findings)}</span></summary>'
            f'{"".join(items)}</details>')


def _coaching_html(tips: list[str]) -> str:
    items = "".join(f"<li>{_esc(tip)}</li>" for tip in tips)
    return f'<div class="panel coach"><div class="panel-title">Coaching</div><ol>{items}</ol></div>'


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
    """Analyze button. Returns: banner, scores, coaching, findings, review panel, 5 sliders, note, thread id."""
    if not transcript or not transcript.strip():
        raise gr.Error("Paste a transcript or pick a sample first.")
    thread_id, result = _guarded(start_analysis, transcript)
    payload = pending_review(result)
    if payload is None:  # not flagged: finished
        return ("", _score_table(result["scores"]), _coaching_html(result["coaching"]),
                _findings_html(result["rag_findings"], expanded=False), gr.Column(visible=False),
                *[result["scores"][d] for d in DIMENSIONS], "", None)
    scores = payload["scores"]  # flagged: paused at human_review, findings open for the reviewer
    return (_flag_banner(scores), _score_table(scores), COACHING_WAITING,
            _findings_html(result["rag_findings"], expanded=True), gr.Column(visible=True),
            *[scores[d] for d in DIMENSIONS], "", thread_id)


def submit_review(thread_id: str | None, *values):
    """Submit-review button: resume the paused run with the reviewer's scores and note."""
    *slider_values, note = values
    if not thread_id or not is_waiting_for_review(thread_id):
        raise gr.Error("There is no pending review. Click Analyze to start again.")
    scores = {dim: int(v) for dim, v in zip(DIMENSIONS, slider_values)}
    result = _guarded(resume_analysis, thread_id, {"scores": scores, "note": note, "reviewer": "reviewer"})
    review = {**result["review"], "final_scores": result["scores"]}
    return (_review_banner(review), _score_table(result["scores"], review["original_scores"]),
            _coaching_html(result["coaching"]), gr.skip(), gr.Column(visible=False), None)


with gr.Blocks(title="Call Quality Copilot") as demo:
    gr.HTML(HEADER_HTML)
    thread = gr.State(None)
    with gr.Row(equal_height=False):
        with gr.Column(scale=5):
            sample = gr.Dropdown(choices=_sample_choices(), value=None, label="Sample call")
            transcript = gr.Textbox(
                label="Transcript",
                lines=24,
                max_lines=24,
                placeholder="Paste a transcript here. Each line starts with 'Agent:' or 'Customer:'.",
            )
            analyze_btn = gr.Button("Analyze", variant="primary")
        with gr.Column(scale=6):
            banner = gr.HTML()
            scores_html = gr.HTML(EMPTY_HTML)
            with gr.Column(visible=False, elem_classes="review-panel") as review_panel:
                gr.HTML('<div class="panel-title">Reviewer decision</div>'
                        '<div class="panel-hint">Confirm the scores, or correct any you disagree with.</div>')
                with gr.Row():
                    sliders = [gr.Slider(1, 5, value=3, step=1, label=_label(dim)) for dim in DIMENSIONS[:3]]
                with gr.Row():
                    sliders += [gr.Slider(1, 5, value=3, step=1, label=_label(dim)) for dim in DIMENSIONS[3:]]
                review_note = gr.Textbox(label="Reviewer note (optional)", lines=2)
                submit_btn = gr.Button("Submit review and generate coaching", variant="primary")
            coaching = gr.HTML()
            findings = gr.HTML()

    sample.change(load_sample, inputs=sample, outputs=transcript)
    analyze_btn.click(
        run_analysis, inputs=transcript,
        outputs=[banner, scores_html, coaching, findings, review_panel, *sliders, review_note, thread],
    )
    submit_btn.click(
        submit_review, inputs=[thread, *sliders, review_note],
        outputs=[banner, scores_html, coaching, findings, review_panel, thread],
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
        theme=THEME,
        css=APP_CSS,
    )
