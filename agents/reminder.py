def generate_reminders(llm, study_plan):
    prompt = f"""
Convert this study plan into reminders.

Study Plan:
{study_plan}

Give:
- Daily reminders
- Time-based schedule
"""
    res = llm.invoke(prompt)
    return res.content
