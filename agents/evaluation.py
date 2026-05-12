"""
Evaluation Agent
----------------
Reads quiz results from DB and analyses student performance.

RULES:
  - Will not run without quiz results (enforced by caller).
  - NEVER generates a quiz directly — redirects user to Content Agent via CTA.
  - Score < 70%  → suggest quiz retake + study plan
  - Score >= 70% → suggest advanced topics or next quiz
"""


def analyze_performance(llm, quiz_result: dict) -> str:
    """
    Analyse quiz performance and return structured feedback with a CTA.

    The response always ends with a deterministic next-step block so the
    LLM cannot accidentally omit it.
    """
    score = quiz_result.get("score", 0)
    total = quiz_result.get("total", 0)
    pct   = quiz_result.get("percentage", 0)
    weak  = quiz_result.get("weak_topics", [])
    ts    = quiz_result.get("timestamp", "recent")[:10]

    weak_str = ", ".join(weak) if weak else "none — all topics answered correctly"

    # ── Score-based branching ──────────────────────────────────────────────────
    if pct < 50:
        performance_tier = "needs significant improvement"
        cta_instruction = (
            "The student scored below 50% — this is a critical situation. "
            "Instruct them clearly to: (1) re-read the uploaded material, "
            "(2) type 'study plan' for a focused revision plan, "
            "(3) type 'quiz' to retake the quiz after studying."
        )
        # Hard-coded CTA appended after LLM output (guaranteed to appear)
        next_step = (
            "\n\n---\n"
            "### Next Steps\n"
            "Your score is below 50% — don't be discouraged, this is fixable!\n\n"
            "- Type **`study plan`** → Planner Agent will build a focused revision schedule\n"
            "- Type **`quiz`** → Content Agent will generate a new quiz after you revise\n"
            "- Re-read the sections on: **{weak}**"
        ).format(weak=weak_str)

    elif pct < 70:
        performance_tier = "below the passing threshold"
        cta_instruction = (
            "The student scored between 50–70%. Encourage them and suggest: "
            "(1) type 'quiz' to attempt the quiz again on weak topics, "
            "(2) type 'study plan' for a moderate revision plan."
        )
        next_step = (
            "\n\n---\n"
            "### Next Steps\n"
            "You are close — a little more revision will push you over 70%!\n\n"
            "- Type **`quiz`** → Content Agent will test you again on your weak areas\n"
            "- Type **`study plan`** → Planner Agent will create a targeted revision plan\n"
            "- Focus revision on: **{weak}**"
        ).format(weak=weak_str)

    elif pct < 90:
        performance_tier = "good — above passing threshold"
        cta_instruction = (
            "The student scored between 70–90%. Praise them and suggest: "
            "(1) type 'quiz' to test themselves on a new or harder topic, "
            "(2) type 'study plan' to continue with advanced content."
        )
        next_step = (
            "\n\n---\n"
            "### Next Steps\n"
            "Solid performance! Keep the momentum going.\n\n"
            "- Type **`quiz`** → Content Agent will test you on the next topic\n"
            "- Type **`study plan`** → Planner Agent will map out your advanced topics\n"
            "- Consider reviewing: **{weak}** to reach 90%+"
        ).format(weak=weak_str if weak else "any remaining gaps")

    else:
        performance_tier = "excellent"
        cta_instruction = (
            "The student scored 90% or above — outstanding. Congratulate them and suggest: "
            "(1) type 'quiz' to explore a new, more advanced topic, "
            "(2) type 'study plan' for an advanced-level study schedule."
        )
        next_step = (
            "\n\n---\n"
            "### Next Steps\n"
            "Outstanding score! You have mastered this material.\n\n"
            "- Type **`quiz`** → Content Agent will challenge you with a new topic\n"
            "- Type **`study plan`** → Planner Agent will design an advanced study path\n"
            "- You are ready to move to the next chapter!"
        )

    # ── LLM prompt ────────────────────────────────────────────────────────────
    prompt = (
        f"You are an expert study coach. Analyse the quiz performance below and "
        f"provide structured, encouraging feedback.\n\n"
        f"Quiz Date  : {ts}\n"
        f"Score      : {score} / {total} ({pct}%) — {performance_tier}\n"
        f"Weak Topics: {weak_str}\n\n"
        f"Format your response with these exact headings:\n"
        f"### Overall Assessment\n"
        f"(1-2 sentences summarising the result)\n\n"
        f"### Strengths\n"
        f"(bullet points — what the student answered correctly)\n\n"
        f"### Areas to Improve\n"
        f"(for each weak topic, give one specific, actionable tip)\n\n"
        f"### Recommended Actions\n"
        f"{cta_instruction}\n\n"
        f"NOTE: Do NOT generate a quiz yourself. "
        f"Only tell the student to type 'quiz' or 'study plan' as commands.\n\n"
        f"Analysis:"
    )

    res = llm.invoke(prompt)

    # Append guaranteed CTA block — LLM output cannot accidentally omit it
    return res.content.strip() + next_step
