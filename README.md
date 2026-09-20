# Call Quality Copilot

Built as a technical demonstration of LangGraph multi-agent pipelines with RAG and human-in-the-loop routing.

![Demo: analyzing a support call](docs/demo.gif)

Paste a support-call transcript (or pick one of 10 samples) and the copilot scores the agent on five dimensions,
checks what the agent said against a policy knowledge base, writes coaching notes, and flags outlier results for
a person to review. All company data is fictional ("Nextel Communications").

## Architecture

```mermaid
flowchart LR
    T[Transcript] --> P["parse_transcript<br/>turns + key moments"]
    P --> R["rag_check<br/>ChromaDB search,<br/>then policy audit"]
    KB[("kb/ policy docs<br/>local embeddings")] --> R
    R --> S["score_agent<br/>5 scores (Pydantic)<br/>+ needs_review flag"]
    S -->|"flagged"| H["human_review<br/>graph pauses until a<br/>reviewer confirms or edits"]
    S -->|"not flagged"| C
    H --> C["generate_coaching<br/>grounded in final scores,<br/>findings and policy"]
    C --> UI["Gradio UI<br/>scores, coaching,<br/>review banner"]
```

The pipeline runs in a LangGraph `StateGraph` over one typed state. Four core nodes always run; a fifth,
`human_review`, runs only for flagged calls:

```python
class State(TypedDict):
    transcript: str
    parsed: dict          # speaker turns, key moments
    rag_findings: list    # policy violations found
    scores: dict          # 5 dimensions, 1-5 each
    coaching: list        # 2-3 specific recommendations
    needs_review: bool    # HITL flag
    review: dict          # the human reviewer's decision (set by human_review)
```

| Node | What it does |
|---|---|
| `parse_transcript` | Splits `Agent:` / `Customer:` turns and extracts key moments (emotion, refund or billing talk, escalation requests, recording disclosure, verification, promises, dead air) plus stats such as repeated sentences and repeated PIN requests. Plain Python, no LLM. |
| `rag_check` | Embeds the key moments and agent turns, searches a ChromaDB index of `kb/`, then asks Claude which of the retrieved policies the agent broke. Output: findings with quote, turn number and severity. |
| `score_agent` | One structured-output call into a Pydantic model: `empathy`, `accuracy`, `resolution`, `efficiency`, `script_adherence`, each an integer 1-5 with a rubric in its `Field` description. Sets `needs_review`. |
| `human_review` | Runs only for flagged calls. Uses LangGraph's `interrupt` to pause the graph; a reviewer confirms or edits the scores, and the run resumes. The state keeps what the model said, what the human decided, and their note. |
| `generate_coaching` | 2-3 specific recommendations grounded in the final scores, the findings, the retrieved policy text and the reviewer's note. |

### Human-in-the-loop

`score_agent` sets `needs_review` when **any score is below 2** (a severe failure) or the **average is above 4.5**
(a call that looks too perfect). A conditional edge then routes on that flag:

- **Not flagged:** straight to `generate_coaching`.
- **Flagged:** to `human_review`, which calls LangGraph's `interrupt`. The graph checkpoints and stops. The UI shows the
  model's scores, the policy findings and a reviewer panel (a 1-5 slider per dimension and a note). On submit the run
  resumes, the final scores replace the model's, and coaching is generated from them. From the terminal,
  `python main.py <file>` prompts for the review instead.

Reviewer scores are validated (whole numbers 1-5, all five dimensions) before the run resumes, and an invalid edit
leaves the run paused.

## What this demonstrates

- A typed LangGraph pipeline where each node has one job and the LLM is only used where judgment is needed.
- RAG that ends in a decision, not a summary: retrieval feeds a policy audit, whose findings feed scoring and coaching.
- Structured outputs validated with Pydantic, with a retry, and errors that say what went wrong (out of credits,
  bad key, unparseable transcript).
- A real human-in-the-loop step: a conditional edge, a checkpointed pause, and a resume with the reviewer's edits, not just a warning flag.
- A human-review rule chosen from measured behaviour rather than assumed (see below).
- Cost awareness: the default model was picked by comparing three Claude models on the same calls.

## Where it applies

The pipeline shape is the reusable part: parse the conversation, audit it against a knowledge base, score it on a
rubric, coach, and send outliers to a person. Six ways it maps to other domains:

| Domain | How the same pipeline maps | What would change |
|---|---|---|
| **Contact-centre quality management** | Score calls consistently, tie coaching to exact turns, let supervisors review only outliers. | Real policy documents; speech-to-text with speaker labels for audio. |
| **IT service management** | A ticket thread is the transcript. Runbooks and SLA rules are the knowledge base. Findings and scores drive escalation decisions. | Rubric (for example SLA adherence, correct resolution, escalation) and parser patterns. |
| **Sales-call effectiveness** | Score discovery, objection handling and next-step commitment against a sales playbook. | Playbook as the knowledge base and a new rubric in the `AgentScores` model. |
| **Healthcare care-coordination audits** | Audit outreach calls against care protocols and compliance rules. | Protocol documents. This demo does no PHI handling: real calls would need de-identification and a suitable agreement with the LLM provider. |
| **Human-in-the-loop agent workflows** | A rule decides which outputs a person must confirm before they are trusted. | The rule and the pause are already here; a production version would add a durable checkpointer and reviewer sign-in. |
| **Client-specific deployments** | The knowledge base, model and API endpoint are configuration. | The five scoring dimensions and the rubric text live in one Pydantic model in `main.py`, so a new client means new documents plus a small code edit. |

## Results on the 10 sample calls

`kb/` holds five policy documents; `samples/` holds ten transcripts written against them. Score order is
empathy / accuracy / resolution / efficiency / script adherence. These came from one live run on Claude Opus 5.

| Sample | Scenario | Result |
|---|---|---|
| 01 | Duplicate charge handled flawlessly | 5/5/5/4/4, avg 4.6: **flagged** (too perfect) |
| 02 | Router troubleshooting | 4/4/4/4/4 |
| 03 | Plan upgrade | 4/5/4/5/4 |
| 04 | Device return inside the 14-day window | 4/5/4/5/4 |
| 05 | Late-fee call, no recording disclosure or closing | script adherence 1: **flagged** |
| 06 | Number transfer, repeated PIN request and 15s dead air | efficiency 2 |
| 07 | Tablet return, agent quotes a 2-3 day refund (policy: 5-7) | accuracy 2 |
| 08 | Outage call, rude agent, refuses a supervisor | 1/1/1/2/1: **flagged** |
| 09 | Third-party caller: no verification, $200 credit, false guarantee, reads a full card number | accuracy 1: **flagged** |
| 10 | Cancellation and FCC threat, agent never escalates | resolution 1: **flagged** |

Samples 01, 08 and 09 were written to trigger review, and they do. 05 and 10 flag too, because Claude gave a 1 on
one dimension.

### Model choice

Compared on samples 02, 07, 08 and 09 (same review flags on every model):

| Model | Cost per analysis | Difference |
|---|---|---|
| Claude Haiku 4.5 (default) | about $0.013 | Scores efficiency more leniently (sample 08: 4 instead of 2) |
| Claude Sonnet 5 | about $0.039 | Closest to Opus |
| Claude Opus 5 | not measured; list price is 5x Haiku per token | Reference scores above |

Costs are computed from measured token counts. Set `ANTHROPIC_MODEL` to change the model.

## Design decisions

- **Local embeddings.** Anthropic has no embeddings API, so the knowledge base is embedded with Chroma's built-in
  MiniLM model. That means one vendor and one API key, at the price of memory: the app peaks around 650 MB when it
  builds the index (measured), which rules out 512 MB free hosting tiers.
- **Review rule.** The first version flagged any single score above 4.5. Because scores are whole numbers, that
  flagged 7 of 10 samples. Applying the high side to the average gives 5 of 10, and every flag is explainable.
- **Scale calibration.** The first live run flagged all 10 calls because the model handed out 5s freely. The rubric
  now defines 4 as "meets the standard", requires a quoted behaviour for any 5, and reserves 1 for severe failures.
- **Coaching grounded in policy.** An early version sometimes advised against a mandatory step (identity
  verification). The coach now sees the same retrieved policy text as the auditor.
- **Pinned API URL.** The client ignores an ambient `ANTHROPIC_BASE_URL`, which silently redirected calls to a local
  server on one dev machine. Override with `CQC_ANTHROPIC_BASE_URL`.

## Limitations

- Ten synthetic calls and a fictional company. Scores were not compared with human QA reviewers.
- Scores come from an LLM and can move by a point between runs and models, especially on borderline calls.
- Paused reviews live in server memory (an in-memory checkpointer). A restart loses them, and there is no reviewer
  sign-in. A production version would swap the in-memory checkpointer for PostgreSQL or Redis, which LangGraph
  supports natively. It would also need reviewer authentication.
- The pipeline is a single sequence with one branch, not several cooperating agents.
- Input is typed transcripts. Audio would need a speech-to-text step with speaker labels first.
- No automated test suite is shipped; verification used stubbed and live runs kept outside the repo.

## Run locally

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
python app.py                                   # Gradio UI at http://127.0.0.1:7860
python main.py samples/08_service_outage.txt    # or run it in a terminal; a flagged call prompts for a review
```

The Chroma index is built on first use in `.chroma/` and rebuilt automatically when `kb/` changes. The embedding
model (about 80 MB) downloads once on first run.

## Project structure

```
main.py            LangGraph pipeline (state, four nodes, scoring and coaching schemas)
app.py             Gradio UI
kb/                five policy documents for the fictional company
samples/           ten sample transcripts
docs/demo.gif      the demo above
Dockerfile         container image
requirements.txt
```
