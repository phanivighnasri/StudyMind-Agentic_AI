from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import os, json, traceback, re
from app.ingestion import build_vectorstore
from app.retrieval import retrieve_content
from langchain_openai import ChatOpenAI

# 🔗 IMPORT AGENTS
from agents.planner import generate_study_plan
from agents.reminder import generate_reminders
from agents.evaluation import analyze_performance

# -----------------------------
# CONFIG
# -----------------------------
CONTENT_DB_PATH = "db/content_db"
SYLLABUS_DB_PATH = "db/syllabus_db"

os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""

llm = ChatOpenAI(
    api_key=os.getenv("GROK_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
    model="openai/gpt-oss-120b",
    temperature=0.3,
    timeout=60
)

# -----------------------------
# SESSION
# -----------------------------
for key in ["history","quiz_data","weak_topics","study_plan"]:
    if key not in st.session_state:
        st.session_state[key] = [] if key!="quiz_data" else None

# -----------------------------
# CORE FUNCTIONS
# -----------------------------
def get_context(query):
    syllabus = retrieve_content(query, SYLLABUS_DB_PATH, top_k=2) if os.path.exists(SYLLABUS_DB_PATH) else ""
    content = retrieve_content(query, CONTENT_DB_PATH, top_k=3)
    return (syllabus + "\n\n" + content)[:3000]

def safe_json(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return json.loads(match.group()) if match else None

# -----------------------------
# UI
# -----------------------------
st.title("📘 Multi-Agent Learning System")

# Upload
st.subheader("Upload PDFs")

syllabus = st.file_uploader("Syllabus", type="pdf")
content = st.file_uploader("Study Material", type="pdf")

if syllabus and not os.path.exists(SYLLABUS_DB_PATH):
    os.makedirs("temp", exist_ok=True)
    path = os.path.join("temp", syllabus.name)
    open(path,"wb").write(syllabus.read())
    build_vectorstore(path, SYLLABUS_DB_PATH)
    st.success("Syllabus stored ✅")

if content and not os.path.exists(CONTENT_DB_PATH):
    os.makedirs("temp", exist_ok=True)
    path = os.path.join("temp", content.name)
    open(path,"wb").write(content.read())
    build_vectorstore(path, CONTENT_DB_PATH)
    st.success("Content stored ✅")

# -----------------------------
# INPUT
# -----------------------------
query = st.text_input("Ask something")

col1,col2,col3,col4,col5 = st.columns(5)

quiz_btn = col1.button("Quiz")
summary_btn = col2.button("Summary")
plan_btn = col3.button("Study Plan")
eval_btn = col4.button("Evaluation")
reminder_btn = col5.button("Reminder")

# -----------------------------
# PROCESS
# -----------------------------
if query:
    try:
        if not os.path.exists(CONTENT_DB_PATH):
            st.warning("Upload content first")
        else:
            context = get_context(query)

            # QA
            if not any([quiz_btn,summary_btn,plan_btn,eval_btn,reminder_btn]):
                res = llm.invoke(f"Answer ONLY from context:\n{context}\nQ:{query}")
                st.write(res.content)
                st.session_state.history.append({"q":query,"a":res.content})

            # SUMMARY
            if summary_btn:
                res = llm.invoke(f"Summarize:\n{context}")
                st.write(res.content)

            # QUIZ
            if quiz_btn:
                res = llm.invoke(f"""
Return JSON quiz.

Context:
{context}

Format:
{{"mcqs":[{{"question":"","options":[{{"key":"A","text":""}}],"answer":"A","topic":""}}]}}
""")
                st.session_state.quiz_data = safe_json(res.content)

            # STUDY PLAN
            if plan_btn:
                plan = generate_study_plan(llm, st.session_state.weak_topics, context)
                st.session_state.study_plan = plan
                st.write("📅 Study Plan")
                st.write(plan)

            # EVALUATION
            if eval_btn:
                result = analyze_performance(llm, st.session_state.weak_topics)
                st.write("📊 Evaluation")
                st.write(result)

            # REMINDER
            if reminder_btn and st.session_state.study_plan:
                rem = generate_reminders(llm, st.session_state.study_plan)
                st.write("⏰ Reminders")
                st.write(rem)

    except:
        st.code(traceback.format_exc())

# -----------------------------
# QUIZ UI
# -----------------------------
if st.session_state.quiz_data:
    st.subheader("📝 Quiz")

    score = 0
    answers = []

    for i,q in enumerate(st.session_state.quiz_data["mcqs"]):
        st.write(q["question"])
        texts=[o["text"] for o in q["options"]]
        keys=[o["key"] for o in q["options"]]

        ans=st.radio("Choose",["Select"]+texts,key=i)
        answers.append(ans)

    if st.button("Submit"):
        st.session_state.weak_topics=[]

        for i,q in enumerate(st.session_state.quiz_data["mcqs"]):
            texts=[o["text"] for o in q["options"]]
            keys=[o["key"] for o in q["options"]]

            if answers[i]=="Select":
                continue

            sel=keys[texts.index(answers[i])]

            if sel==q["answer"]:
                score+=1
                st.success("Correct")
            else:
                st.error("Wrong")
                st.session_state.weak_topics.append(q["topic"])

        st.success(f"Score {score}/{len(answers)}")

# -----------------------------
# HISTORY
# -----------------------------
st.subheader("History")
for h in st.session_state.history:
    st.write(h["q"],"→",h["a"])

with open("history.json","w") as f:
    json.dump(st.session_state.history,f)
