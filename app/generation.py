from langchain_openai import ChatOpenAI
from config import GROK_API_KEY, BASE_URL

def generate_answer(context, query):

    llm = ChatOpenAI(
        api_key=GROK_API_KEY,
        base_url=BASE_URL,
        model="grok-beta",
        temperature=0
    )

    prompt = f"""
    You are an academic assistant.

    Strict Rules:
    - Answer ONLY using provided context.
    - If answer is not found, say:
      "Not found in provided material."
    - Do NOT use outside knowledge.

    Context:
    {context}

    Question:
    {query}
    """

    response = llm.invoke(prompt)
    return response.content
