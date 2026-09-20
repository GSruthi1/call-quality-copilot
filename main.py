"""Call Quality Copilot - LangGraph pipeline.

parse_transcript -> rag_check -> score_agent -> generate_coaching

Run from the CLI:  python main.py samples/01_billing_dispute.txt
"""
import hashlib
import json
import os
import re
import sys
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
from typing import TypedDict

import anthropic
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field, ValidationError

BASE_DIR = Path(__file__).parent
KB_DIR = BASE_DIR / "kb"
CHROMA_DIR = BASE_DIR / ".chroma"

LLM_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-opus-5")
# Anthropic has no embeddings API, so the knowledge base is embedded locally with Chroma's
# built-in ONNX all-MiniLM-L6-v2 model (downloaded once on first use, no API key needed).
EMBEDDING_MODEL = "chroma-default-all-MiniLM-L6-v2"

# HITL thresholds. A score below REVIEW_LOW is a severe failure a human should confirm; an overall
# average above REVIEW_HIGH is a consistently "perfect" call, which is also worth a second look.
REVIEW_LOW = 2
REVIEW_HIGH = 4.5

DIMENSIONS = ("empathy", "accuracy", "resolution", "efficiency", "script_adherence")


class State(TypedDict):
    transcript: str
    parsed: dict          # speaker turns, key moments
    rag_findings: list    # policy violations found
    scores: dict          # 5 dimensions, 1-5 each
    coaching: list        # 2-3 specific recommendations
    needs_review: bool    # HITL flag


# --------------------------------------------------------------------------- #
# Structured-output schemas
# --------------------------------------------------------------------------- #

class PolicyFinding(BaseModel):
    """One place where the agent broke, or failed to follow, a Nextel policy."""

    policy: str = Field(description="Policy document and section that was broken, e.g. 'refund_policy.txt > Device return and refund window'.")
    turn: int | None = Field(description="Index of the agent turn where it happened, or null if it is an omission across the whole call.")
    agent_quote: str = Field(description="Short verbatim quote from the agent, or 'N/A' for an omission.")
    issue: str = Field(description="One sentence saying what the agent did or failed to do, and what the policy requires.")
    severity: str = Field(description="One of: 'minor', 'major', 'critical'. Critical = compliance/privacy breach or a promise Nextel cannot keep.")


class PolicyFindings(BaseModel):
    findings: list[PolicyFinding] = Field(description="Genuine violations only. Empty list if the agent complied with every retrieved policy.")


class AgentScores(BaseModel):
    """Quality scores for the agent on one call. Every score is an integer from 1 (very poor) to 5 (exceptional)."""

    reasoning: str = Field(description="2-4 sentences of evidence from the transcript that justify the scores. Write this first. For every dimension you will score 5, quote the specific agent behaviour that went beyond the standard; if you cannot quote one, score it 4.")
    empathy: int = Field(ge=1, le=5, description="Did the agent acknowledge the customer's emotion in their own words before moving to a fix, and stay polite? 1 = rude, mocking or dismissive, 2 = no acknowledgement of a clearly upset customer, 3 = generic or late acknowledgement, 4 = timely, sincere acknowledgement (meets the standard), 5 = exceptional: personalised, well-timed and reassuring throughout.")
    accuracy: int = Field(ge=1, le=5, description="Based on the policy findings, was the information the agent gave correct? 1 = critical violation such as a privacy breach, false guarantee or unauthorized promise, 2 = a major wrong statement, 3 = minor slips only, 4 = everything stated was correct (meets the standard), 5 = exceptional: correct and proactively volunteered the details the customer needed (fees, timelines, case numbers).")
    resolution: int = Field(ge=1, le=5, description="Was the issue resolved, or escalated properly per the escalation rules? 1 = unresolved and a required escalation was refused or ignored, 2 = unresolved or a required escalation was skipped, 3 = partly resolved, 4 = resolved or correctly escalated (meets the standard), 5 = exceptional: resolved with clear next steps, verification of the outcome and a written reference.")
    efficiency: int = Field(ge=1, le=5, description="Unnecessary repetition, repeated requests for information, dead air, or off-script chatter? 1 = call derailed by waste, 2 = significant waste (repeated questions, unexplained dead air, off-topic talk), 3 = some repetition or delay, 4 = focused with no material waste (meets the standard), 5 = exceptional: unusually tight and well-paced.")
    script_adherence: int = Field(ge=1, le=5, description="Did the agent give the required opening (Nextel greeting, name, recording disclosure, open question) and closing (recap, 'anything else?', sign-off)? 1 = no recognisable opening or closing at all, 2 = several required elements missing, 3 = one or two elements missing, 4 = every required element present (meets the standard), 5 = exceptional: every element delivered naturally and personalised.")


class CoachingPlan(BaseModel):
    recommendations: list[str] = Field(min_length=2, max_length=3, description="2-3 specific, actionable coaching recommendations for the agent, most important first.")


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #

class LocalEmbeddings(Embeddings):
    """LangChain adapter around Chroma's local default embedding function."""

    def __init__(self) -> None:
        self._fn = DefaultEmbeddingFunction()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self._fn(texts)]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


@lru_cache(maxsize=1)
def _get_client() -> anthropic.Anthropic:
    return anthropic.Anthropic()  # reads ANTHROPIC_API_KEY


@lru_cache(maxsize=1)
def _get_embeddings() -> LocalEmbeddings:
    return LocalEmbeddings()


def _structured(schema: type[BaseModel], system: str, user: str) -> BaseModel:
    """Call Claude for a Pydantic-validated result; retry once if the output fails validation."""
    for attempt in range(2):
        try:
            response = _get_client().messages.parse(
                model=LLM_MODEL,
                max_tokens=16000,
                system=system,
                messages=[{"role": "user", "content": user}],
                output_format=schema,
            )
        except ValidationError:
            if attempt == 1:
                raise
            continue
        if response.stop_reason == "refusal":
            raise RuntimeError("The model declined to analyze this transcript.")
        if response.parsed_output is not None:
            return response.parsed_output
        if attempt == 1:
            raise RuntimeError(f"No structured output returned (stop_reason={response.stop_reason}).")


def _numbered_transcript(parsed: dict) -> str:
    return "\n".join(f"[{t['index']}] {t['speaker']}: {t['text']}" for t in parsed["turns"])


def needs_human_review(scores: dict) -> bool:
    """HITL rule: flag if any score is below 2, or the average of the scores is above 4.5.

    Scores are integers, so applying "> 4.5" to each score would flag every call that earns a single 5.
    """
    return any(v < REVIEW_LOW for v in scores.values()) or sum(scores.values()) / len(scores) > REVIEW_HIGH


# --------------------------------------------------------------------------- #
# Node 1: parse_transcript
# --------------------------------------------------------------------------- #

_TURN_RE = re.compile(r"^\s*(agent|customer)\s*:\s*(.*)$", re.IGNORECASE)
_EVENT_RE = re.compile(r"^\s*\[(?P<event>[^\]]+)\]\s*$")
_TIMED_RE = re.compile(r"\[\s*(?P<kind>silence|pause|hold)[^\]\d]*(?P<n>\d+)\s*(?P<unit>seconds?|secs?|s|minutes?|mins?|m)\b[^\]]*\]", re.IGNORECASE)
_CARD_RE = re.compile(r"\b\d{4}[ -]?\d{4}[ -]?\d{4}[ -]?\d{4}\b")

# (moment type, speaker it applies to, pattern)
_MOMENT_PATTERNS = [
    ("customer_emotion", "Customer", r"frustrat|annoy|upset|angry|furious|ridiculous|unacceptable|disappoint|worried|fed up|awful|not happy|third time|three times|real problem"),
    ("escalation_request", "Customer", r"supervisor|manager|escalat|cancellation department|someone else"),
    ("cancellation_or_legal", "Customer", r"cancel|switch(?:ing)? (?:to|carriers)\b|\bfcc\b|lawyer|attorney|lawsuit|complaint"),
    ("refund_or_billing", "Customer", r"refund|money back|\bcredit\b|return (?:it|the|my)|charged twice|double charge|late fee|dispute|charged"),
    ("recording_disclosure", "Agent", r"may be recorded|recorded for"),
    ("verification", "Agent", r"\bpin\b|last four|last 4|verify"),
    ("closing_check", "Agent", r"anything else"),
    ("promise_or_guarantee", "Agent", r"guarantee|promise|you have my word|for sure"),
    ("credit_or_refund_offer", "Agent", r"\$\s?\d[\d,]*(?:\.\d\d)?[^.?!]*(?:credit|refund)|(?:credit|refund)[^.?!]*\$\s?\d[\d,]*"),
    ("competitor_mention", "Agent", r"other guys|competitor|verizon|at&t|t-mobile"),
    ("sensitive_data", "Either", r"card number|social security|full ssn"),
]


def _seconds(n: str, unit: str) -> int:
    return int(n) * 60 if unit.lower().startswith("m") else int(n)


def _repeated_agent_sentences(agent_texts: list[str]) -> list[str]:
    """Agent sentences that (nearly) repeat an earlier agent sentence: a hint for the efficiency score."""
    sentences = []
    for text in agent_texts:
        for s in re.split(r"(?<=[.!?])\s+", text):
            if len(s.split()) >= 6:
                sentences.append(s.strip())
    repeats = []
    for i, later in enumerate(sentences):
        if any(SequenceMatcher(None, earlier.lower(), later.lower()).ratio() >= 0.7 for earlier in sentences[:i]):
            repeats.append(later)
    return repeats


def parse_transcript(state: State) -> dict:
    """Split the transcript into Agent/Customer turns and pull out the key moments."""
    turns: list[dict] = []
    events: list[str] = []
    for line in state["transcript"].splitlines():
        if not line.strip():
            continue
        turn = _TURN_RE.match(line)
        if turn:
            turns.append({"index": len(turns), "speaker": turn.group(1).capitalize(), "text": turn.group(2).strip()})
        elif (event := _EVENT_RE.match(line)):
            events.append(event.group("event"))
        elif turns:  # continuation of the previous speaker's turn
            turns[-1]["text"] += " " + line.strip()
        # text before the first speaker label is ignored

    if not turns:
        raise ValueError("No 'Agent:' or 'Customer:' turns found. Each line of the transcript must start with 'Agent:' or 'Customer:'.")

    moments: list[dict] = []

    def add(turn: dict, kind: str, **extra):
        moments.append({"turn": turn["index"], "speaker": turn["speaker"], "type": kind, "text": turn["text"][:240], **extra})

    agent_turns = [t for t in turns if t["speaker"] == "Agent"]
    if agent_turns:
        add(agent_turns[0], "opening")
        add(agent_turns[-1], "closing")

    dead_air = 0
    for turn in turns:
        lowered = turn["text"].lower()
        for kind, speaker, pattern in _MOMENT_PATTERNS:
            if speaker in (turn["speaker"], "Either") and re.search(pattern, lowered):
                add(turn, kind)
        if _CARD_RE.search(turn["text"]) and not any(m["turn"] == turn["index"] and m["type"] == "sensitive_data" for m in moments):
            add(turn, "sensitive_data")
        if turn["speaker"] == "Agent":
            for m in _TIMED_RE.finditer(turn["text"]):
                secs = _seconds(m.group("n"), m.group("unit"))
                if m.group("kind").lower() != "hold" and secs > 10:
                    dead_air += 1
                    add(turn, "dead_air", seconds=secs)

    moments.sort(key=lambda m: m["turn"])
    agent_texts = [t["text"] for t in agent_turns]
    parsed = {
        "turns": turns,
        "key_moments": moments,
        "events": events,
        "stats": {
            "agent_turns": len(agent_turns),
            "customer_turns": len(turns) - len(agent_turns),
            "agent_words": sum(len(t.split()) for t in agent_texts),
            "customer_words": sum(len(t["text"].split()) for t in turns if t["speaker"] == "Customer"),
            "dead_air_events": dead_air,
            "agent_pin_requests": sum(1 for t in agent_texts if re.search(r"\bpin\b", t, re.IGNORECASE) and "?" in t),
            "repeated_agent_sentences": _repeated_agent_sentences(agent_texts),
        },
    }
    return {"parsed": parsed}


# --------------------------------------------------------------------------- #
# Node 2: rag_check  (ChromaDB similarity search over kb/, then an LLM policy check)
# --------------------------------------------------------------------------- #

def _load_kb_chunks() -> list[Document]:
    """One chunk per '## ' section of each policy file, prefixed with the document title."""
    chunks = []
    for path in sorted(KB_DIR.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        title, _, rest = text.partition("\n")
        for section in re.split(r"^## ", rest, flags=re.MULTILINE)[1:]:
            heading, _, body = section.partition("\n")
            chunks.append(Document(
                page_content=f"{title.strip()}\n{heading.strip()}\n{body.strip()}",
                metadata={"source": path.name, "section": heading.strip()},
            ))
    if not chunks:
        raise RuntimeError(f"No policy documents found in {KB_DIR}")
    return chunks


@lru_cache(maxsize=1)
def _get_vectorstore() -> Chroma:
    """Local persistent Chroma index. The collection name embeds a hash of the KB, so edits re-index."""
    chunks = _load_kb_chunks()
    digest = hashlib.sha1((EMBEDDING_MODEL + "".join(c.page_content for c in chunks)).encode()).hexdigest()[:12]
    store = Chroma(
        collection_name=f"kb_{digest}",
        embedding_function=_get_embeddings(),
        persist_directory=str(CHROMA_DIR),
    )
    if not store.get(limit=1)["ids"]:
        store.add_documents(chunks, ids=[f"{c.metadata['source']}::{c.metadata['section']}" for c in chunks])
    return store


_BASELINE_QUERIES = [
    "required call opening greeting and recording disclosure",
    "required call closing sign-off",
    "identity verification before discussing account information",
    "when the agent must escalate or transfer to a supervisor",
    "agent authority limits for refunds and credits",
    "false guarantees, promises and misleading statements",
]
MAX_QUERIES = 18
CHUNKS_PER_QUERY = 3
MAX_CHUNKS = 18


def _build_queries(parsed: dict) -> list[str]:
    queries = list(_BASELINE_QUERIES)
    seen = set()
    candidates = [m["text"] for m in parsed["key_moments"] if m["type"] not in ("opening", "closing")]
    candidates += sorted((t["text"] for t in parsed["turns"] if t["speaker"] == "Agent"), key=len, reverse=True)
    for text in candidates:
        key = text.lower()
        if key not in seen and len(text.split()) >= 6:
            seen.add(key)
            queries.append(text)
    return queries[:MAX_QUERIES]


def retrieve_policies(parsed: dict) -> list[Document]:
    """Similarity search over the KB using every key moment/agent turn; best-matching unique chunks."""
    store = _get_vectorstore()
    queries = _build_queries(parsed)
    vectors = _get_embeddings().embed_documents(queries)  # one batched local embedding call
    best: dict[str, tuple[float, Document]] = {}
    for vector in vectors:
        for doc, distance in store.similarity_search_by_vector_with_relevance_scores(vector, k=CHUNKS_PER_QUERY):
            key = f"{doc.metadata['source']}::{doc.metadata['section']}"
            if key not in best or distance < best[key][0]:
                best[key] = (distance, doc)
    ranked = sorted(best.values(), key=lambda pair: pair[0])
    return [doc for _, doc in ranked[:MAX_CHUNKS]]


_RAG_SYSTEM = """You are a compliance auditor for Nextel Communications, a telecom company.
You are given (1) policy excerpts retrieved from the knowledge base and (2) a call transcript with numbered turns.
List every place the AGENT violated, or failed to follow, one of the retrieved policies. Rules:
- Only report violations that are clearly supported by both a policy excerpt and the transcript. Do not invent policies or facts.
- Include omissions (e.g. missing recording disclosure, missing identity verification, no closing) with turn=null.
- Judge only the agent. Customer statements are never violations.
- Severity: critical = privacy/compliance breach, false guarantee, or promise beyond authority; major = wrong policy information or a refused/skipped required escalation; minor = script or etiquette slip.
- If the agent complied with everything, return an empty list."""


def rag_check(state: State) -> dict:
    """Retrieve relevant policies from ChromaDB and check the agent's behaviour against them."""
    parsed = state["parsed"]
    docs = retrieve_policies(parsed)
    policies = "\n\n---\n\n".join(f"[{d.metadata['source']} > {d.metadata['section']}]\n{d.page_content}" for d in docs)
    stats = parsed["stats"]
    user = (
        f"POLICY EXCERPTS:\n{policies}\n\n"
        f"CALL TRANSCRIPT (turn numbers in brackets):\n{_numbered_transcript(parsed)}\n\n"
        f"Parser hints: dead-air events >10s: {stats['dead_air_events']}; "
        f"agent PIN requests: {stats['agent_pin_requests']}."
    )
    result = _structured(PolicyFindings, _RAG_SYSTEM, user)
    return {"rag_findings": [f.model_dump() for f in result.findings]}


# --------------------------------------------------------------------------- #
# Node 3: score_agent
# --------------------------------------------------------------------------- #

_SCORE_SYSTEM = """You are a senior call-quality analyst for Nextel Communications scoring ONE agent on ONE call.
Score the five dimensions independently, as integers 1-5. Calibration:
- 4 means the agent fully MET the standard on that dimension. That is what a good agent normally earns, so a competent call with no problems scores 4s.
- 5 is reserved for training-exemplar performance that goes clearly beyond the standard. Expect a 5 on at most about one call in ten; do not give 5 just because nothing went wrong.
- 1 is reserved for severe failures (rude or abusive conduct, privacy or compliance breach, false guarantees, refusing a required escalation). A single wrong fact or a missed script line is a 2 or 3, not a 1.
- 3 is acceptable with clear gaps.
- Use the policy findings as hard evidence: a critical finding caps accuracy at 1-2; a major finding at 2-3; no findings and correct statements means 4-5.
- The required opening has 4 parts (Nextel greeting, agent name, recording disclosure, open question). The required closing has 3 (recap, 'anything else?', sign-off).
- Judge what the agent actually said. Do not reward good intentions or penalize the customer's behaviour."""


def score_agent(state: State) -> dict:
    """Structured scoring on five dimensions, then apply the human-review (HITL) rule."""
    parsed = state["parsed"]
    findings = state["rag_findings"]
    findings_text = "\n".join(
        f"- [{f['severity']}] {f['policy']} (turn {f['turn']}): {f['issue']}" for f in findings
    ) or "None - no policy violations were found."
    user = (
        f"TRANSCRIPT:\n{_numbered_transcript(parsed)}\n\n"
        f"POLICY FINDINGS:\n{findings_text}\n\n"
        f"PARSER STATS: {json.dumps(parsed['stats'])}"
    )
    result = _structured(AgentScores, _SCORE_SYSTEM, user)
    scores = {dim: getattr(result, dim) for dim in DIMENSIONS}
    # LangGraph edges can only route, not write state, so the HITL rule is applied here,
    # right after scoring and before coaching.
    return {"scores": scores, "needs_review": needs_human_review(scores)}


# --------------------------------------------------------------------------- #
# Node 4: generate_coaching
# --------------------------------------------------------------------------- #

_COACH_SYSTEM = """You are a supportive contact-centre coach for Nextel Communications.
Write 2-3 coaching recommendations for the agent, most important first. Each one must:
- target a specific moment in the call (quote or paraphrase what the agent said),
- say what to do instead, with example wording the agent can use,
- be at most 3 sentences.
Prioritise the lowest scores and any policy findings. If the call was excellent, say what to keep doing and how to make it repeatable.
If the call is flagged for human review, the first recommendation should tell the agent's supervisor what to review."""


def generate_coaching(state: State) -> dict:
    """Coaching recommendations grounded in the scores and RAG findings."""
    findings_text = "\n".join(f"- [{f['severity']}] {f['issue']} (turn {f['turn']})" for f in state["rag_findings"]) or "None."
    user = (
        f"TRANSCRIPT:\n{_numbered_transcript(state['parsed'])}\n\n"
        f"SCORES (1-5): {json.dumps(state['scores'])}\n"
        f"POLICY FINDINGS:\n{findings_text}\n"
        f"FLAGGED FOR HUMAN REVIEW: {state['needs_review']}"
    )
    result = _structured(CoachingPlan, _COACH_SYSTEM, user)
    return {"coaching": result.recommendations[:3]}


# --------------------------------------------------------------------------- #
# Graph
# --------------------------------------------------------------------------- #

def build_graph():
    graph = StateGraph(State)
    graph.add_node("parse_transcript", parse_transcript)
    graph.add_node("rag_check", rag_check)
    graph.add_node("score_agent", score_agent)
    graph.add_node("generate_coaching", generate_coaching)
    graph.add_edge(START, "parse_transcript")
    graph.add_edge("parse_transcript", "rag_check")
    graph.add_edge("rag_check", "score_agent")
    graph.add_edge("score_agent", "generate_coaching")
    graph.add_edge("generate_coaching", END)
    return graph.compile()


pipeline = build_graph()


def analyze(transcript: str) -> State:
    """Run the full pipeline on one transcript and return the final state."""
    return pipeline.invoke({
        "transcript": transcript,
        "parsed": {},
        "rag_findings": [],
        "scores": {},
        "coaching": [],
        "needs_review": False,
    })


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python main.py <transcript.txt>")
    final = analyze(Path(sys.argv[1]).read_text(encoding="utf-8"))
    print(json.dumps({k: v for k, v in final.items() if k not in ("transcript", "parsed")}, indent=2))
