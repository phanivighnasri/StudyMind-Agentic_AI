def generate_study_plan(llm, weak_topics, context):
    prompt = f"""
Create a study plan.

Weak topics:
{weak_topics}

Context:
{context}

Give:
- Day-wise plan
- Topics per day
- Time allocation
"""
    res = llm.invoke(prompt)
    return res.content
