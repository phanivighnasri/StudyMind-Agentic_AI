"""
Reminder Agent
--------------
Two modes:

1. generate_reminders(llm, study_plan)
   → Full day-wise schedule from a study plan.
   → Returns {"type": "reminder_schedule", "display_text": str, "reminders": [...]}

2. extract_quick_reminder(user_message)
   → Single break/study timer from a casual request like "remind me in 25 minutes".
   → Returns {"type": "reminder", "duration": <minutes>, "message": str}
      so the frontend can start a countdown immediately.
"""

import json
import re


_DURATION_RE = re.compile(
    r"(\d+)\s*(hour|hr|minute|min|second|sec)s?",
    re.IGNORECASE,
)


def extract_quick_reminder(user_message: str) -> dict | None:
    """
    Parse a casual reminder request and return a structured action dict.

    Examples:
      "remind me in 25 minutes" → {"type": "reminder", "duration": 25, "message": "Break over!"}
      "set a 1 hour timer"      → {"type": "reminder", "duration": 60, "message": "Time's up!"}

    Returns None if no duration could be parsed.
    """
    total_minutes = 0
    for match in _DURATION_RE.finditer(user_message):
        value = int(match.group(1))
        unit  = match.group(2).lower()
        if unit.startswith("hour") or unit.startswith("hr"):
            total_minutes += value * 60
        elif unit.startswith("min"):
            total_minutes += value
        elif unit.startswith("sec"):
            total_minutes += max(1, value // 60)

    if total_minutes == 0:
        return None

    # Choose a contextual message
    msg = "Break over! Time to get back to studying. 📚"
    low = user_message.lower()
    if any(w in low for w in ("study", "work", "focus")):
        msg = "Study session starting now! Let's go. 🚀"
    elif any(w in low for w in ("break", "rest", "relax", "nap")):
        msg = "Break's over — back to it! 📖"

    return {"type": "reminder", "duration": total_minutes, "message": msg}


def _parse_reminders_json(text: str) -> list:
    """Extract a JSON array from LLM output. Returns [] on failure."""
    match = re.search(r'\[[\s\S]*\]', text)
    if not match:
        return []
    try:
        return json.loads(match.group())
    except Exception:
        return []


def generate_reminders(llm, study_plan: str) -> dict:
    """
    Returns:
    {
        "type": "reminder_schedule",
        "display_text": str,          # human-readable, shown in chat
        "reminders": [                # machine-readable, for frontend timer
            {
                "day": 1,
                "time": "09:00",
                "duration_minutes": 60,
                "message": "Study Topic X",
                "type": "study" | "break" | "review"
            }, ...
        ]
    }
    """
    # Step 1: Generate readable day-by-day schedule
    text_res = llm.invoke(
        "You are a study reminder assistant. Convert the study plan below into "
        "a clear, actionable daily reminder schedule.\n\n"
        f"Study Plan:\n{study_plan}\n\n"
        "Generate:\n"
        "- A reminder for each study block with a specific time (e.g., 9:00 AM — Study Topic X for 1 hour)\n"
        "- A short motivational note at the start of each day\n"
        "- Break reminders (e.g., 10:00 AM — 10-minute break)\n"
        "- An end-of-day review reminder\n\n"
        "Format as a clean, readable day-by-day schedule.\n\n"
        "Reminder Schedule:"
    )
    display_text = text_res.content.strip()

    # Step 2: Convert to structured JSON for frontend timer integration
    json_res = llm.invoke(
        "Convert this study reminder schedule into a JSON array of reminder objects.\n"
        "Each object must have these exact keys:\n"
        "  day (int), time (HH:MM in 24-hour format), duration_minutes (int),\n"
        "  message (str), type ('study' | 'break' | 'review')\n"
        "Return ONLY a valid JSON array — no explanation, no markdown.\n\n"
        f"Schedule:\n{display_text}\n\n"
        "JSON:"
    )
    reminders = _parse_reminders_json(json_res.content)

    return {
        "type": "reminder_schedule",
        "display_text": display_text,
        "reminders": reminders,
    }
