"""
Content Agent
-------------
Single source of knowledge. Responsible for:
  - Q&A from uploaded content (with "out of syllabus" detection)
  - Quiz generation from content
  - Persisting and loading quiz results from DB
"""

import json
import re
import os
from datetime import datetime
from langchain_core.messages import SystemMessage, HumanMessage

QUIZ_RESULTS_PATH = "./db/quiz_results.json"


# ── Context retrieval ─────────────────────────────────────────────────────────

def get_context(
    query: str,
    db_path: str,
    retrieve_content,
    source_filter: str = None,
) -> tuple:
    """
    Returns (context_text: str, max_similarity_score: float).
    source_filter: if set, restricts retrieval to that specific uploaded file.
    """
    result = retrieve_content(query, db_path, top_k=5, source_filter=source_filter)

    if isinstance(result, tuple):
        text, score = result
    else:
        text, score = str(result), 1.0  # legacy fallback

    return text, score


# ── Q&A ───────────────────────────────────────────────────────────────────────

def answer_question(
    llm,
    context: str,
    query: str,
    history: list = None,
    system_prompt: str = "",
) -> str:
    """
    Answer a question from document context.
    system_prompt: injected as SystemMessage so model has full PDF + behavior context.
    history: last N conversation turns for continuity.
    """
    if not context.strip():
        return "I could not find any relevant content. Please upload your study material first."

    # Build conversation history block — last 5 exchanges (10 messages)
    history_block = ""
    if history:
        turns = [
            ("Student" if m["role"] == "user" else "Assistant") + ": " + m["content"]
            for m in history[-10:]
        ]
        if turns:
            history_block = "Recent conversation:\n" + "\n".join(turns) + "\n\n"

    human = (
        "Answer the student's question directly and clearly.\n"
        "Rules:\n"
        "- Give structured answers with bullet points or numbered steps.\n"
        "- Do NOT start with 'Based on the context' or 'According to the passage'.\n"
        "- Answer confidently from the study material provided.\n\n"
        f"{history_block}"
        f"Study material:\n{context}\n\n"
        f"Question: {query}\n\nAnswer:"
    )

    if system_prompt:
        res = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=human)])
    else:
        res = llm.invoke(human)
    return res.content.strip()


# ── Topic extraction ──────────────────────────────────────────────────────────

def extract_topics(llm, context: str, system_prompt: str = "") -> str:
    """
    Scan retrieved content and return a structured topic list.
    system_prompt: injected as SystemMessage for PDF-aware context.
    """
    if not context.strip():
        return "No content available. Please upload a PDF first."

    human = (
        "Analyze the content below and extract all major topics and key concepts "
        "actually present in the text. Do NOT add anything not in the content.\n\n"
        "Format your response exactly as:\n\n"
        "📚 **Topics Covered**\n\n"
        "For each main topic:\n"
        "**Topic Name**\n"
        "↳ one-line explanation of what it covers\n\n"
        "Then add:\n"
        "**Suggested Study Order**\n"
        "List the topics in the recommended order to study them.\n\n"
        f"Content:\n{context}\n\nTopic List:"
    )

    if system_prompt:
        res = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=human)])
    else:
        res = llm.invoke(human)
    return res.content.strip()


# ── Summarization ─────────────────────────────────────────────────────────────

def summarize_content(llm, context: str, topic: str = "", system_prompt: str = "") -> str:
    """
    Generate a structured, exam-ready summary.
    system_prompt: injected as SystemMessage for PDF-aware context.
    """
    if not context.strip():
        return "No content available to summarize. Please upload your study material first."

    topic_line = f"Topic focus: {topic}\n\n" if topic else ""
    human = (
        "Create a structured, exam-ready summary of the content below.\n"
        "Use ONLY the provided content — do not add outside information.\n\n"
        f"{topic_line}"
        "Format your response EXACTLY as follows:\n\n"
        "📘 **What This Document Is About**\n"
        "(2–3 sentences giving a clear overview)\n\n"
        "🧠 **Key Points**\n"
        "- (bullet each important concept, one idea per bullet)\n\n"
        "📌 **Short Summary**\n"
        "(3–4 sentences capturing the most important ideas)\n\n"
        f"Content:\n{context}\n\nStructured Summary:"
    )

    if system_prompt:
        res = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=human)])
    else:
        res = llm.invoke(human)
    return res.content.strip()


# ── Quiz generation ───────────────────────────────────────────────────────────

def generate_quiz(llm, context: str, topic: str):
    """Returns parsed quiz dict or None on failure."""
    if not context.strip():
        return None

    prompt = (
        f"You are a quiz generator. Create exactly 5 multiple-choice questions based ONLY on the context below.\n"
        f"Topic: {topic}\n\n"
        f"Rules:\n"
        f"- Each question must be answerable from the context — no outside knowledge.\n"
        f"- Make distractors plausible but clearly wrong.\n"
        f"- Vary difficulty: mix recall, understanding, and application questions.\n"
        f"- Return ONLY valid JSON with no text before or after the JSON block.\n\n"
        f"Context:\n{context}\n\n"
        f"Required JSON format:\n"
        f"{{\n"
        f'  "mcqs": [\n'
        f"    {{\n"
        f'      "question": "Question text here?",\n'
        f'      "options": [\n'
        f'        {{"key": "A", "text": "Option A text"}},\n'
        f'        {{"key": "B", "text": "Option B text"}},\n'
        f'        {{"key": "C", "text": "Option C text"}},\n'
        f'        {{"key": "D", "text": "Option D text"}}\n'
        f"      ],\n"
        f'      "answer": "A",\n'
        f'      "topic": "{topic}"\n'
        f"    }}\n"
        f"  ]\n"
        f"}}\n\n"
        f"JSON:"
    )

    res = llm.invoke(prompt)

    try:
        match = re.search(r'\{[\s\S]*\}', res.content)
        if not match:
            return None
        return json.loads(match.group())
    except Exception as e:
        print("Quiz parsing error:", e)
        return None


# ── Quiz result persistence ───────────────────────────────────────────────────

def save_quiz_result(score: int, total: int, weak_topics: list):
    """Append a quiz result to the JSON DB file."""
    os.makedirs(os.path.dirname(QUIZ_RESULTS_PATH), exist_ok=True)

    results = _load_results_raw()
    results.append({
        "timestamp": datetime.now().isoformat(),
        "score": score,
        "total": total,
        "percentage": round((score / total) * 100, 1) if total > 0 else 0.0,
        "weak_topics": weak_topics,
    })

    with open(QUIZ_RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)


def load_latest_quiz_result() -> dict | None:
    """Return the most recent quiz result dict, or None if none exist."""
    results = _load_results_raw()
    return results[-1] if results else None


def load_all_quiz_results() -> list:
    """Return all stored quiz results (oldest first)."""
    return _load_results_raw()


def _load_results_raw() -> list:
    if not os.path.exists(QUIZ_RESULTS_PATH):
        return []
    try:
        with open(QUIZ_RESULTS_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return []
