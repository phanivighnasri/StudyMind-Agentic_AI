"""
Orchestrator Agent
------------------
Central routing layer. All queries pass through route_query() first.

Two-tier design:
  1. Fast regex path  — handles clear-cut cases instantly (no LLM cost)
  2. LLM classifier   — handles ambiguous / rephrased / vague queries

route_query() returns a routing dict used by main.py to dispatch agents.
"""

import re
import os
import json
from difflib import get_close_matches
from langchain_core.messages import SystemMessage, HumanMessage

# ── Intent routing ──────────────────────────────────────────────────────────

# ORDER MATTERS — first match wins.
# evaluate and reminder must come before quiz/plan to avoid false matches
# e.g. "evaluate my quiz results" must not trigger quiz intent.
INTENT_PATTERNS = {
    "evaluate": r"\b(evaluat|performance|how am i doing|my results?|my score|analysis|analyse|analyz)\b",
    "reminder": r"\b(remind(er)?s?|alert|notify|notification|set (a )?timer|break time|study timer)\b",
    "plan":     (
        r"\b(study plan|create plan|make plan|build plan|schedule|timetable|roadmap|"
        r"plan my study|make a plan|plan for|plan on|day plan|"
        r"\d+[\s\-]day (plan|study|schedule)|i have \d+ days?)\b"
    ),
    "quiz":     (
        r"\b(quiz|test me|test myself|mcq|generate questions?|practice test|"
        r"practice questions?|make (a )?quiz|give me (a )?quiz|start (a )?quiz|"
        r"assess me|create (a )?quiz|take (a )?quiz|question me)\b"
    ),
    # Summary intent — "summarize", "give me key points", "overview of X"
    "summary":  (
        r"\b(summar(ize|ise|y)|key points?|important points?|overview|"
        r"brief(ly)?|highlights?|main points?|give me (a )?(gist|outline))\b"
    ),
    # Topics intent — "list topics", "what's in the pdf", "what can I study"
    "topics":   (
        r"\b(list topics?|what topics?|topics? (in|from|of|covered)|"
        r"what('s| is) (in|covered|inside)|what did (i|you) upload|"
        r"contents? of|show topics?|extract topics?|"
        r"what can i (study|learn)|what is (in|inside) (the|this|my) (pdf|document|file|material)|"
        r"key concepts|main concepts)\b"
    ),
    # PDF info intent — "how many pdfs", "what files did I upload"
    "pdf_info": (
        r"\b(how many (pdf|file|document|material)s?|list (pdf|file|document)s?|"
        r"what (pdf|file|document)s? (have i|did i|are)|uploaded (pdf|file|document)s?|"
        r"my (pdf|file|document|upload)s?|show (pdf|file|document|upload)s?|"
        r"which (pdf|file|document)s?|number of (pdf|file|document)s?)\b"
    ),
}

# ── Acronym / alias expansion ────────────────────────────────────────────────

# Maps common abbreviations to their full forms so TF-IDF retrieval finds them.
TOPIC_ALIASES: dict[str, str] = {
    "scm":   "source code management",
    "ai":    "artificial intelligence",
    "ml":    "machine learning",
    "dl":    "deep learning",
    "nlp":   "natural language processing",
    "cv":    "computer vision",
    "oop":   "object oriented programming",
    "os":    "operating system",
    "dbms":  "database management system",
    "db":    "database",
    "cn":    "computer networks",
    "ds":    "data structures",
    "dsa":   "data structures and algorithms",
    "algo":  "algorithms",
    "se":    "software engineering",
    "api":   "application programming interface",
    "ui":    "user interface",
    "ux":    "user experience",
    "git":   "git version control",
    "vcs":   "version control system",
}


def expand_topic(topic: str) -> str:
    """
    Expand acronyms in a topic string to improve TF-IDF retrieval.
    e.g. "SCM basics" → "source code management basics"
    """
    words = topic.lower().split()
    return " ".join(TOPIC_ALIASES.get(w, w) for w in words)

EMOTIONAL_KEYWORDS = [
    "feel", "feeling", "felt", "sad", "low", "depressed", "anxious",
    "stress", "stressed", "tired", "exhausted", "overwhelmed", "nervous",
    "scared", "hopeless", "frustrated", "angry", "upset", "worried",
    "crying", "cry", "unhappy", "demotivated", "burnt out", "burnout",
    "not motivated", "i give up", "struggling",
]

# Greetings and small-talk that require no document context
_GREETING_RE = re.compile(
    r"^\s*(hi|hello|hey|good\s+(morning|afternoon|evening|night)|bye|goodbye|"
    r"see\s+you|thanks|thank\s+you|cheers|ok|okay|great|cool|awesome|got\s+it)\b",
    re.IGNORECASE,
)
_GENERAL_CHAT_RE = re.compile(
    r"\b(who are you|what can you do|what are you|how are you|your name|"
    r"what is your purpose|can you help|tell me about yourself)\b",
    re.IGNORECASE,
)

AGENT_LABELS = {
    "quiz":      "Content Agent",
    "summary":   "Content Agent",
    "evaluate":  "Evaluation Agent",
    "plan":      "Planner Agent",
    "reminder":  "Reminder Agent",
    "emotional": "Orchestrator Agent",
    "chat":      "Orchestrator Agent",
    "topics":    "Content Agent",
    "pdf_info":  "Orchestrator Agent",
    "qa":        "Content Agent",
}


def detect_intent(user_message: str) -> str:
    """Return one of: quiz, evaluate, plan, reminder, emotional, chat, qa"""
    msg = user_message.lower().strip()

    # Emotional check has highest priority
    if any(kw in msg for kw in EMOTIONAL_KEYWORDS):
        return "emotional"

    # Greetings / small-talk — respond directly without document lookup
    if _GREETING_RE.search(msg) or _GENERAL_CHAT_RE.search(msg):
        return "chat"

    for intent, pattern in INTENT_PATTERNS.items():
        if re.search(pattern, msg, re.IGNORECASE):
            return intent

    return "qa"


def get_agent_label(intent: str) -> str:
    return AGENT_LABELS.get(intent, "Content Agent")


# ── System prompt builder ────────────────────────────────────────────────────

def build_system_prompt(uploaded_files: list) -> str:
    """
    Build the global StudyMind AI system context injected into every LLM call.
    Gives the model PDF awareness and consistent behavior rules.
    """
    if uploaded_files:
        pdf_lines = "\n".join(f"  - {f['filename']}" for f in uploaded_files)
        pdf_context = f"PDFs currently in memory:\n{pdf_lines}"
    else:
        pdf_context = "No PDFs uploaded yet."

    return (
        "You are StudyMind AI — an intelligent, professional study assistant.\n\n"
        f"{pdf_context}\n\n"
        "Behavior rules:\n"
        "- Answer directly and confidently.\n"
        "- NEVER start with 'Based on the context', 'According to the passage', "
        "or 'According to the retrieved content'.\n"
        "- If a topic is not in the provided material, say: "
        "'This is not clearly covered in this document, but here's a general explanation:'\n"
        "- For casual conversation or greetings: respond naturally, no document lookup.\n"
        "- Always produce structured responses: use headings and bullet points.\n"
        "- Never mix content from different PDFs."
    )


def _llm_invoke(llm, system_prompt: str, human_prompt: str):
    """Invoke ChatGroq with proper SystemMessage / HumanMessage separation."""
    return llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt),
    ])


# ── Canned responses ─────────────────────────────────────────────────────────

def get_emotional_response(llm=None, user_message: str = "", system_prompt: str = "") -> str:
    """LLM-generated emotional support. Falls back to canned message if LLM unavailable."""
    if llm and user_message:
        try:
            system = system_prompt or build_system_prompt([])
            human = (
                "The student is sharing their emotional state. "
                "Respond with genuine empathy and understanding. "
                "Acknowledge exactly what they said, then offer 2–3 concrete tips "
                "(e.g. take a break, breathe, hydrate). "
                "End with a warm offer to continue studying together. "
                "Keep it natural and under 100 words.\n\n"
                f"Student: {user_message}\nCoach:"
            )
            res = _llm_invoke(llm, system, human)
            return res.content.strip()
        except Exception:
            pass

    return (
        "I can sense you might be feeling overwhelmed right now — and that is completely okay.\n\n"
        "A few gentle suggestions:\n"
        "- Take a short 5–10 minute break to reset\n"
        "- Try some slow deep breaths or a quick walk\n"
        "- Stay hydrated and have something to eat\n\n"
        "You are doing better than you think. Whenever you feel ready, I'm right here!"
    )


def get_pdf_info_response(uploaded_files: list) -> str:
    """Answer system-state questions about uploaded PDFs — no LLM needed."""
    if not uploaded_files:
        return (
            "No PDFs have been uploaded yet.\n\n"
            "Use the sidebar to upload a PDF and I'll index it for you."
        )
    count = len(uploaded_files)
    lines = [f"**{count} PDF{'s' if count > 1 else ''} currently in memory:**\n"]
    for i, f in enumerate(uploaded_files, 1):
        ts = f.get("uploaded_at", "")[:10]
        chunks = f.get("chunks", "?")
        lines.append(f"{i}. `{f['filename']}` — {chunks} chunks — added {ts}")
    lines.append("\nYou can ask questions, generate quizzes, or create study plans from any of these.")
    return "\n".join(lines)


def get_chat_response(llm, user_message: str, system_prompt: str = "") -> str:
    """Handle greetings and general conversation — no document context needed."""
    system = system_prompt or build_system_prompt([])
    human = (
        "Respond naturally to the student's message.\n"
        "- For greetings (hi, hello, bye, thanks): reply warmly and briefly.\n"
        "- For capability questions: explain you can answer questions from uploaded PDFs, "
        "generate quizzes, build study plans, set reminders, and evaluate performance.\n"
        "- Keep the response short (2–4 sentences max) and conversational.\n\n"
        f"Student: {user_message}\nAssistant:"
    )
    res = _llm_invoke(llm, system, human)
    return res.content.strip()


# ── Content DB helpers ───────────────────────────────────────────────────────

def content_db_exists(db_path: str) -> bool:
    """True only when vectorstore files are present (upload was successful)."""
    return os.path.exists(os.path.join(db_path, "vectors.pkl"))


# ── Central routing engine ────────────────────────────────────────────────────

# Maps LLM classifier output → internal intent names used by main.py
_LLM_INTENT_MAP = {
    "study_plan":       "plan",
    "topic_extraction": "topics",
    "general":          "chat",
    "emotion":          "emotional",
    "pdf_query":        "qa",
    "quiz":             "quiz",
    "summary":          "summary",
    "reminder":         "reminder",
    "evaluate":         "evaluate",
    "pdf_info":         "pdf_info",
}

# Intents that NEVER need RAG
_NO_RAG_INTENTS = {"chat", "emotional", "pdf_info", "evaluate", "reminder"}


def _match_source_filter(raw: str, uploaded_files: list) -> str | None:
    """
    Match LLM-extracted filename fragment against actual stored filenames.

    Three-pass strategy (most → least strict):
      1. Case-insensitive substring match (handles stripped quotes, partial names)
      2. Fuzzy stem match via difflib (handles typos like "jenkin" → "jenkins")
      3. Give up → return None (caller will search all PDFs)
    """
    if not raw or not uploaded_files:
        return None

    # Strip surrounding quotes the LLM sometimes adds
    raw_clean = raw.lower().strip().strip("\"'")

    # Pass 1 — substring containment
    for f in uploaded_files:
        name = f["filename"].lower()
        if raw_clean in name or name in raw_clean:
            return f["filename"]

    # Pass 2 — fuzzy match on filename stems (no extension)
    stems = [os.path.splitext(f["filename"].lower())[0] for f in uploaded_files]
    raw_stem = os.path.splitext(raw_clean)[0]
    matches = get_close_matches(raw_stem, stems, n=1, cutoff=0.6)
    if matches:
        idx = stems.index(matches[0])
        return uploaded_files[idx]["filename"]

    return None


def route_query(llm, user_message: str, uploaded_files: list, content_ready: bool) -> dict:
    """
    Central routing decision maker. Returns:
    {
        "intent":       internal intent string,
        "topic":        extracted topic (str, may be ""),
        "source_filter": filename to filter retrieval (str | None),
        "use_rag":      bool,
        "agent_label":  display name for active-agent banner,
    }

    Two-tier design:
      1. Fast deterministic path — emotional, greetings, pdf_info (no LLM cost)
      2. LLM classifier — everything else (handles typos, synonyms, vague phrasing)
      3. Regex fallback — if LLM call fails
    """
    msg = user_message.lower().strip()

    # ── Tier 1: fast deterministic cases ─────────────────────────────────────
    if any(kw in msg for kw in EMOTIONAL_KEYWORDS):
        return _build_route("emotional", use_rag=False)

    if _GREETING_RE.search(msg) or _GENERAL_CHAT_RE.search(msg):
        return _build_route("chat", use_rag=False)

    if re.search(INTENT_PATTERNS["pdf_info"], msg, re.IGNORECASE):
        return _build_route("pdf_info", use_rag=False)

    # ── Tier 2: LLM intent classifier ────────────────────────────────────────
    file_names = [f["filename"] for f in uploaded_files] if uploaded_files else []

    classifier_prompt = (
        "You are an intent classifier for an AI study assistant. "
        "Classify the user message and extract routing information.\n\n"
        f"Files the user has uploaded: {file_names if file_names else ['none']}\n\n"
        "Intent options:\n"
        "  study_plan       — wants a study plan, schedule, or timetable\n"
        "  quiz             — wants a quiz, test, MCQs, or practice questions\n"
        "  summary          — wants a summary, key points, overview, or highlights\n"
        "  topic_extraction — wants to know what topics/concepts are in a document\n"
        "  reminder         — wants reminders, timers, or break alerts\n"
        "  evaluate         — wants to see their quiz score, performance, or results\n"
        "  pdf_query        — has a factual question about their study material\n"
        "  general          — casual conversation unrelated to study tasks\n\n"
        "Extract:\n"
        "  topic: specific subject mentioned (empty string if none given)\n"
        "  source_filter: filename if user clearly identifies a specific file "
        "(null if user means all files or says 'the pdf'/'uploaded material')\n\n"
        "Rules:\n"
        "- Spelling mistakes are common — infer intent from context\n"
        "- 'test me', 'make questions', 'assess me' → quiz\n"
        "- 'give key points', 'what are the main ideas' → summary\n"
        "- 'uploaded pdf/document/material' without a specific name → source_filter: null\n\n"
        "Respond with ONLY valid JSON — no explanation, no markdown:\n"
        '{"intent": "...", "topic": "...", "source_filter": null}\n\n'
        f'User message: "{user_message}"\n\nJSON:'
    )

    try:
        res = llm.invoke(classifier_prompt)
        match = re.search(r'\{[^{}]*\}', res.content)
        if match:
            data = json.loads(match.group())
            raw_intent = data.get("intent", "pdf_query")
            intent = _LLM_INTENT_MAP.get(raw_intent, "qa")
            topic = str(data.get("topic") or "").strip()
            source_filter = _match_source_filter(data.get("source_filter"), uploaded_files)
            return _build_route(intent, topic=topic, source_filter=source_filter)
    except Exception:
        pass

    # ── Tier 3: regex fallback (if LLM fails) ────────────────────────────────
    for intent_key, pattern in INTENT_PATTERNS.items():
        if re.search(pattern, msg, re.IGNORECASE):
            return _build_route(intent_key)

    return _build_route("qa")


def _build_route(intent: str, topic: str = "", source_filter=None, use_rag: bool = None) -> dict:
    """Build a routing dict with computed use_rag if not explicitly provided."""
    if use_rag is None:
        use_rag = intent not in _NO_RAG_INTENTS
    return {
        "intent":        intent,
        "topic":         topic,
        "source_filter": source_filter,
        "use_rag":       use_rag,
        "agent_label":   AGENT_LABELS.get(intent, "Content Agent"),
    }


# ── General LLM fallback (no document context) ───────────────────────────────

def answer_general(llm, query: str, history: list = None, system_prompt: str = "") -> str:
    """
    Answer without RAG context — graceful fallback when topic not in documents.
    Never says 'out of syllabus'; always tries to help.
    """
    history_block = ""
    if history:
        turns = [
            ("Student" if m["role"] == "user" else "Assistant") + ": " + m["content"]
            for m in history[-8:]
        ]
        history_block = "Conversation so far:\n" + "\n".join(turns) + "\n\n"

    system = system_prompt or build_system_prompt([])
    human = (
        "The student's question may not be covered in their uploaded material.\n"
        "Answer as a general knowledge question. Be educational, clear, and structured.\n"
        "If relevant, begin with: "
        "'This is not clearly covered in this document, but here's a general explanation:'\n"
        "Use bullet points or numbered steps where helpful.\n\n"
        f"{history_block}"
        f"Question: {query}\n\nAnswer:"
    )
    res = _llm_invoke(llm, system, human)
    return res.content.strip()
