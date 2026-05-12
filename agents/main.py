import os
import json
import re
import traceback
import streamlit as st
from dotenv import load_dotenv

from app.ingestion import build_vectorstore
from app.retrieval import retrieve_content
from langchain_openai import ChatOpenAI

from agents.content_agent import get_context, answer_question, generate_quiz
from agents.planner import generate_study_plan
from agents.reminder import generate_reminders
from agents.evaluation import analyze_performance

# -----------------------------
# ENV + PROXY
# -----------------------------
load_dotenv()

os.environ["http_proxy"] = "http://mgntguest1:rgurgu@staffnet.rgukt.ac.in:3128/"
os.environ["https_proxy"] = "http://mgntguest1:rgurgu@staffnet.rgukt.ac.in:3128/"

# -----------------------------
# CONFIG
# -----------------------------
CONTENT_DB_PATH = "db/content_db"
TEMP_DIR = "temp"

llm = ChatOpenAI(
    api_key=os.getenv("GROK_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
    model="llama-3.1-8b-instant",
    temperature=0.3,
    timeout=120,
    max_retries=3
)

# -----------------------------
# SESSION STATE
# -----------------------------
defaults = {
    "quiz": None,
    "quiz_generated": False,
    "force_quiz": False,
    "quiz_topic": "",
    "weak_topics": [],
    "study_plan": None,
    "quiz_result": None
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# -----------------------------
# UI
# -----------------------------
st.title("🤖 Orchestrator Agent")

# -----------------------------
# FILE UPLOAD (FIXED)
# -----------------------------
file = st.file_uploader("Upload Study Material", type="pdf")

if file:
    os.makedirs(TEMP_DIR, exist_ok=True)
    os.makedirs(CONTENT_DB_PATH, exist_ok=True)

    file_path = os.path.join(TEMP_DIR, file.name)

    with open(file_path, "wb") as f:
        f.write(file.read())

    build_vectorstore(file_path, CONTENT_DB_PATH)

    st.success("📘 Vector DB updated successfully")

# -----------------------------
# INPUT
# -----------------------------
un_clean_query = st.text_input("Ask something")
query = re.sub(r'[^\w\s]', '', un_clean_query.lower())
# -----------------------------
# FORCE QUIZ MODE
# -----------------------------
if st.session_state.force_quiz:

    st.write("📘 Quiz Agent Activated")

    if not os.path.exists(CONTENT_DB_PATH):
        st.warning("Upload study material first")

    else:
        context = get_context(
            st.session_state.quiz_topic,
            CONTENT_DB_PATH,
            retrieve_content
        )

        quiz = generate_quiz(llm, context, st.session_state.quiz_topic)

        if quiz:
            st.session_state.quiz = quiz
            st.session_state.quiz_generated = True
        else:
            st.error("Quiz generation failed")

    st.session_state.force_quiz = False

# -----------------------------
# ORCHESTRATOR
# -----------------------------
if query:

    q = query.lower().strip()

    # Emotional response
    if any(word in q for word in ["sad", "low", "tired", "depressed"]):
        st.write("🙂 Stay positive!")

    # -------------------------
    # EVALUATION
    # -------------------------
    elif "evaluate" in q:

        if not st.session_state.quiz_result:
            st.warning("Take quiz first")
        else:
            result = analyze_performance(llm, st.session_state.quiz_result)
            st.write(result)

            if st.button("Improve Quiz"):
                st.session_state.quiz = None
                st.session_state.quiz_generated = False
                st.session_state.force_quiz = True

                st.session_state.quiz_topic = (
                    " ".join(st.session_state.weak_topics)
                    if st.session_state.weak_topics
                    else "general"
                )

                st.rerun()

    # -------------------------
    # STUDY PLAN
    # -------------------------
    elif "plan" in q:

        if not os.path.exists(CONTENT_DB_PATH):
            st.warning("Upload data first")

        else:
            context = get_context(query, CONTENT_DB_PATH, retrieve_content)

           plan = generate_study_plan(
    llm,
    st.session_state.weak_topics,
    context,
    user_query=query   # 🔥 THIS IS THE FIX
)

            st.session_state.study_plan = plan
            st.write(plan)

    # -------------------------
    # REMINDER
    # -------------------------
    elif "reminder" in q:

        if not st.session_state.study_plan:
            st.warning("Create plan first")
        else:
            rem = generate_reminders(llm, st.session_state.study_plan)
            st.write(rem)

    # -------------------------
    # QUIZ MODE
    # -------------------------
    elif "quiz" in q:

        if not os.path.exists(CONTENT_DB_PATH):
            st.warning("Upload data first")

        elif not st.session_state.quiz_generated:

            topic = q.replace("quiz on", "").replace("generate quiz on", "").strip()
            st.session_state.quiz_topic = topic if topic else "general"

            context = get_context(
                st.session_state.quiz_topic,
                CONTENT_DB_PATH,
                retrieve_content
            )

            quiz = generate_quiz(llm, context, st.session_state.quiz_topic)

            if quiz:
                st.session_state.quiz = quiz
                st.session_state.quiz_generated = True
            else:
                st.warning("Quiz generation failed")

    # -------------------------
    # NORMAL QA (FIXED)
    # -------------------------
    else:

        if not os.path.exists(CONTENT_DB_PATH):
            st.warning("Upload data first")
        else:
            context = get_context(query, CONTENT_DB_PATH, retrieve_content)

            if not context:
                st.warning("No relevant content found in uploaded document.")
            else:
                answer = answer_question(llm, context, query)
                st.write(answer)

# -----------------------------
# QUIZ UI
# -----------------------------
if st.session_state.quiz:

    st.subheader(f"📝 Quiz on: {st.session_state.quiz_topic}")

    score = 0
    answers = []

    for i, q in enumerate(st.session_state.quiz["mcqs"]):

        st.write(f"Q{i+1}: {q['question']}")

        texts = [opt["text"] for opt in q["options"]]
        keys = [opt["key"] for opt in q["options"]]

        ans = st.radio(
            "Choose one:",
            ["Select"] + texts,
            index=0,
            key=f"q{i}"
        )

        answers.append(ans)

    if st.button("Submit Quiz"):

        st.session_state.weak_topics = []

        for i, q in enumerate(st.session_state.quiz["mcqs"]):

            texts = [opt["text"] for opt in q["options"]]

            if answers[i] == "Select":
                continue

            selected_key = keys[texts.index(answers[i])]

            if selected_key == q["answer"]:
                score += 1
            else:
            	clean_topic = re.sub(r'[^\w\s]', '', q["topic"]).lower()
                st.session_state.weak_topics.append(clean_topic)

        st.session_state.quiz_result = {
            "score": score,
            "total": len(st.session_state.quiz["mcqs"]),
            "weak_topics": st.session_state.weak_topics
        }

        st.success(f"🎯 Score: {score}/{len(st.session_state.quiz['mcqs'])}")

    if st.button("New Quiz"):
        st.session_state.quiz = None
        st.session_state.quiz_generated = False
        st.session_state.force_quiz = True
        st.session_state.quiz_topic = "general"
        st.rerun()
