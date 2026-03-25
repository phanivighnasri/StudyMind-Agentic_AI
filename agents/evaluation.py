def analyze_performance(llm, weak_topics):
    prompt = f"""
Analyze performance.

Weak topics:
{weak_topics}

Give:
- Strengths
- Weaknesses
- Improvement tips
"""
    res = llm.invoke(prompt)
    return res.content
