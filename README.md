---
title: Call Quality Copilot
emoji: 📞
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.28.0
python_version: "3.11"
app_file: app.py
pinned: false
short_description: LangGraph + RAG call QA with human-in-the-loop routing
---

# Call Quality Copilot

Built as a technical demonstration of LangGraph multi-agent pipelines with RAG and human-in-the-loop routing.

Paste a support-call transcript (or pick one of the 10 samples) and the copilot scores the agent on five
dimensions, checks what the agent said against a policy knowledge base, writes coaching notes, and flags
outlier results for a human to review. All company data is fictional ("Nextel Communications").

## Pipeline

```
START -> parse_transcript -> rag_check -> score_agent -> generate_coaching -> END
```

| Node | What it does |
|---|---|
| `parse_transcript` | Splits the text into `Agent:` / `Customer:` turns and extracts key moments (emotion, refund/billing, escalation requests, recording disclosure, verification, promises, dead air) plus stats such as repeated sentences and repeated PIN requests. Plain Python, no LLM. |
| `rag_check` | Embeds the key moments and agent turns, runs a similarity search over the ChromaDB index of `kb/`, then asks the LLM which of the retrieved policies the agent broke. Output: a list of findings with quote, turn, severity. |
| `score_agent` | One structured-output call into a Pydantic model with five integer scores (1-5), each with a `Field` description. Sets `needs_review`. |
| `generate_coaching` | 2-3 specific recommendations grounded in the scores and findings. |

### Scoring dimensions

`empathy`, `accuracy` (judged from the RAG findings), `resolution`, `efficiency`, `script_adherence`.

### Human-in-the-loop rule

`needs_review` is set when **any score is below 2**, or the **average of the five scores is above 4.5**. Both
ends are outliers a human should confirm before the result is trusted or used in coaching: a severe failure on
any dimension, or a call that looks too perfect.

The original idea was "any score `> 4.5`". Scores are integers, so that flags every call with a single 5. In a
live run it flagged 7 of 10 samples, so the high side is applied to the average (in `needs_human_review()`).

LangGraph edges can only route, not write state, so the rule is applied at the end of `score_agent`
(`needs_human_review()` in `main.py`) rather than in a separate edge.

## Run locally

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
python app.py                              # Gradio UI at http://127.0.0.1:7860
python main.py samples/01_billing_dispute.txt   # or run the pipeline from the CLI
```

Optional: `ANTHROPIC_MODEL` (default `claude-haiku-4-5`, about $0.013 per analysis). `claude-sonnet-5` (about
$0.04) or `claude-opus-5` are more discriminating but cost 3-10x more.
Scoring and coaching use Claude with Pydantic structured outputs. Anthropic has no embeddings API, so the
knowledge base is embedded locally with Chroma's built-in MiniLM model (downloaded once on first run, no key
needed). The Chroma index is built on first use in `.chroma/` and rebuilt automatically when `kb/` changes.

## Sample data

`kb/` holds five policy documents (`refund_policy`, `greeting_script`, `escalation_rules`, `product_faq`,
`compliance_rules`). `samples/` holds ten transcripts written against them:

| Sample | Scenario | Result (Claude Opus 5, one live run) |
|---|---|---|
| 01 | Duplicate charge handled flawlessly | 5/5/5/4/4, avg 4.6: **flagged** (too perfect) |
| 02 | Router troubleshooting | 4/4/4/4/4 |
| 03 | Plan upgrade | 4/5/4/5/4 |
| 04 | Device return inside the 14-day window | 4/5/4/5/4 |
| 05 | Late-fee call, no recording disclosure or closing | script adherence 1: **flagged** |
| 06 | Number transfer with a repeated PIN request and 15s dead air | efficiency 2 |
| 07 | Tablet return, agent quotes 2-3 day refund timing (policy: 5-7) | accuracy 2 |
| 08 | Outage call, rude agent, refuses supervisor | 1/1/1/2/1: **flagged** |
| 09 | Third-party caller: no verification, $200 credit, false guarantee, reads full card number | accuracy 1: **flagged** |
| 10 | Cancellation and FCC threat, agent never escalates | resolution 1: **flagged** |

Score order: empathy / accuracy / resolution / efficiency / script adherence. Cheaper models score slightly
differently: on samples 02, 07, 08 and 09, Sonnet 5 and Haiku 4.5 reproduced the same review flags as Opus 5, but
Haiku scores efficiency more leniently (sample 08: 4 instead of 2).

Samples 01, 08 and 09 were written to trigger review, and they do. 05 and 10 flag too, because Claude gave a 1 on
one dimension. Scores come from the LLM and can vary between runs and models, especially for borderline calls.

## Deploy

**Railway** (used for the live demo): `Dockerfile` and `railway.json` are included. Create a Railway project
from this folder (`railway init`, then `railway up`), set the service variables `ANTHROPIC_API_KEY` and
`PORT=8000`, and generate a domain. The app binds to `$PORT` automatically.

**Hugging Face Spaces**: the YAML block at the top of this README is the Space configuration (Gradio SDK,
Python 3.11). Push the folder to a Gradio Space and add `ANTHROPIC_API_KEY` under
**Settings -> Variables and secrets**. Note that Hugging Face currently requires a PRO subscription to host
Gradio Spaces on its free CPU tier.

Either way, every visitor's analysis is billed to that API key, so keep the deployment private or set a spend
limit in the Anthropic console.
