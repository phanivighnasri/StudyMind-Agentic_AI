"""
Planner Agent
-------------
Generates day-wise study plans with deadline awareness.
RULE: Will not run without uploaded content (enforced by caller).
"""

import re


# ── Deadline extraction ───────────────────────────────────────────────────────

# Maps English number words → integers
_WORD_TO_INT = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "fourteen": 14, "fifteen": 15,
    "twenty": 20, "thirty": 30,
}


def _to_int(word: str) -> int | None:
    """Convert a digit string or English word to int, or return None."""
    try:
        return int(word)
    except (ValueError, TypeError):
        return _WORD_TO_INT.get(str(word).lower())


def extract_days(user_query: str) -> int:
    """
    Parse time constraints from natural language.

    Recognises:
      "tomorrow"            → 1
      "today" / "tonight"   → 1
      "in 2 days"           → 2
      "3 days"              → 3
      "this weekend"        → 2
      "next week" / "in a week" → 7
      "in 2 weeks"          → 14
      "in a month"          → 30

    Returns number of days (int, always >= 1). Defaults to 7 if nothing found.
    """
    q = user_query.lower()

    # "tomorrow" or "by tomorrow"
    if re.search(r"\btomorrow\b", q):
        return 1

    # "today" or "tonight"
    if re.search(r"\btoday\b|\btonight\b", q):
        return 1

    # "this weekend" → 2 days of intensive study
    if re.search(r"\bthis\s+weekend\b", q):
        return 2

    # "in X days" / "within X days" / "X days" / "X-day"
    m = re.search(
        r"\b(?:in|within)\s+(\w+)\s+days?\b"   # "in 3 days"
        r"|\b(\w+)\s*[-\s]day\b"                 # "3-day" / "3 day"
        r"|\b(\w+)\s+days?\s+(?:left|remaining|to go)\b",  # "3 days left"
        q,
    )
    if m:
        word = next(g for g in m.groups() if g is not None)
        days = _to_int(word)
        if days:
            return max(1, days)

    # "in X weeks" / "X weeks"
    m = re.search(r"\b(?:in\s+)?(\w+)\s+weeks?\b", q)
    if m:
        weeks = _to_int(m.group(1))
        if weeks:
            return max(1, weeks * 7)

    # "next week" / "in a week" / "in one week"
    if re.search(r"\bnext\s+week\b|\bin\s+(?:a|one|1)\s+week\b", q):
        return 7

    # "in a month"
    if re.search(r"\bin\s+(?:a|one|1)\s+month\b", q):
        return 30

    return 7   # default


def _intensity_label(days: int) -> str:
    if days <= 2:
        return "INTENSIVE"
    elif days <= 5:
        return "MODERATE"
    return "DETAILED"


def _extract_specific_topic(user_query: str) -> str:
    """
    Strip plan-command keywords and day-count patterns from the query
    to isolate the specific topic the student wants to study.
    Returns empty string if no specific topic was given.
    """
    q = user_query.lower()
    # Remove plan-command phrases
    for kw in sorted([
        "generate study plan", "create study plan", "make study plan",
        "build study plan", "study plan on", "study plan for",
        "study plan about", "study plan", "create plan", "make plan",
        "build plan", "plan on", "plan for", "plan about", "plan",
    ], key=len, reverse=True):  # longest first to avoid partial matches
        q = q.replace(kw, "")
    # Remove day/time patterns
    q = re.sub(r"\b(for\s+)?\d+\s*[-\s]?\s*days?\b", "", q)
    q = re.sub(r"\b(in|within)\s+\d+\s+days?\b", "", q)
    q = re.sub(r"\b(next\s+week|this\s+week|tomorrow|tonight|today)\b", "", q)
    return q.strip(" ,-.")


# ── Plan generation ───────────────────────────────────────────────────────────

def generate_study_plan(
    llm,
    weak_topics: list,
    context: str,
    user_query: str = "",
) -> str:
    """
    Generate a structured day-wise study plan.

    Parameters
    ----------
    llm         : LLM instance
    weak_topics : list of topic strings (from quiz results)
    context     : retrieved content from vectorstore
    user_query  : the original user message — used to extract deadline
    """
    days = extract_days(user_query)
    intensity = _intensity_label(days)

    # Detect if the user gave a specific topic or wants a general plan
    specific_topic = _extract_specific_topic(user_query)
    if specific_topic:
        topic_instruction = f"Focus on this topic: **{specific_topic}**"
    else:
        topic_instruction = (
            "No specific topic was given — infer all study topics directly "
            "from the content provided below. Do NOT make up topics."
        )

    weak_str = (
        ", ".join(weak_topics)
        if weak_topics
        else "none identified — cover all topics evenly"
    )

    if days == 1:
        day_desc = "1 day only — INTENSIVE mode (today/tomorrow)"
    elif days <= 2:
        day_desc = f"{days} days — INTENSIVE mode (focus only on critical topics)"
    elif days <= 5:
        day_desc = f"{days} days — MODERATE mode (balanced coverage with revision)"
    else:
        day_desc = f"{days} days — DETAILED mode (thorough coverage and revision)"

    prompt = (
        f"You are a study planner. Create a structured, realistic day-wise study plan "
        f"based ONLY on the content provided below.\n\n"
        f"TIME CONSTRAINT: The student has {day_desc}.\n"
        f"TOPIC FOCUS: {topic_instruction}\n\n"
        f"TOPIC DISTRIBUTION RULES (MANDATORY — violation = wrong answer):\n"
        f"1. First, identify ALL unique topics/sections available in the content.\n"
        f"2. Assign each topic to EXACTLY ONE day — never repeat a topic across days.\n"
        f"3. If a topic is large, split it into sub-sections within its assigned day.\n"
        f"4. If days > topics available, use extra days for revision, practice, or mock tests.\n"
        f"5. NEVER write the same heading or concept on two different days.\n\n"
        f"DAY COUNT RULES (MANDATORY):\n"
        f"- Generate EXACTLY {days} days — not more, not less.\n"
        f"- Stop at Day {days}.\n\n"
        f"Output format MUST be:\n"
        f"Day 1:\n"
        f"Day 2:\n"
        f"...\n"
        f"Day {days}:\n\n"
        f"Weak topics needing extra attention (prioritise these): {weak_str}\n\n"
        f"Content to study from:\n{context}\n\n"
        f"Format requirements:\n"
        f"- Heading: '📅 Study Plan ({days}-Day {intensity})'\n"
        f"- For each day: list topics with time slots (e.g., 9:00 AM – 10:00 AM: Topic X)\n"
        f"- Include 10-minute breaks between study blocks\n"
        f"- For INTENSIVE plans: cover only the highest-priority topics\n"
        f"- End with a single motivational sentence\n\n"
        f"Study Plan:"
    )

    res = llm.invoke(prompt)
    return res.content.strip()
