"""
Multi-Agent Study Assistant — Single Streamlit UI
--------------------------------------------------
All user interactions happen here.
The Orchestrator Agent decides which specialist agent handles each request.

Agent routing:
  quiz      → Content Agent   (generate quiz, store results)
  evaluate  → Evaluation Agent (read results from DB)
  plan      → Planner Agent   (requires uploaded content)
  reminder  → Reminder Agent  (requires a study plan)
  emotional → Orchestrator    (generic supportive response)
  default   → Content Agent   (Q&A from uploaded material)
"""

import os
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from langchain_groq import ChatGroq

from app.ingestion import build_vectorstore, get_uploaded_files
from app.retrieval import retrieve_content
from db.database import (
    init_db, upsert_pdf, get_all_pdfs,
    save_study_plan, get_latest_study_plan,
    save_reminder, save_chat_message,
)

from agents.orchestrator import (
    detect_intent,
    get_agent_label,
    get_emotional_response,
    get_chat_response,
    get_pdf_info_response,
    route_query,
    answer_general,
    build_system_prompt,
    content_db_exists,
    expand_topic,
)
from agents.content_agent import (
    get_context,
    answer_question,
    summarize_content,
    extract_topics,
    generate_quiz,
    save_quiz_result,
    load_latest_quiz_result,
    load_all_quiz_results,
)
from agents.planner import generate_study_plan
from agents.reminder import generate_reminders, extract_quick_reminder
from agents.evaluation import analyze_performance

# ── Env ───────────────────────────────────────────────────────────────────────
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
CONTENT_DB_PATH = "./db/content_db"

# ── Database — initialise on every startup (idempotent) ───────────────────────
init_db()
OUT_OF_SYLLABUS_THRESHOLD = 0.05
TEMP_DIR = "temp"

# ── LLM ───────────────────────────────────────────────────────────────────────
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.3,
    groq_api_key=GROQ_API_KEY,
    request_timeout=120,
    max_retries=3,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Multi-Agent Study Assistant",
    page_icon="🤖",
    layout="wide",
)

# ── Session state defaults ────────────────────────────────────────────────────
_DEFAULTS = {
    "chat_history":    [],       # list of {role, content, agent}
    "active_agent":    None,     # label shown in banner
    "quiz":            None,     # current quiz dict
    "quiz_generated":  False,
    "quiz_topic":      "",
    "quiz_attempt":    0,        # incremented per new quiz to reset radio keys
    "weak_topics":     [],       # from latest quiz result
    "study_plan":      None,
    "quiz_result":     None,     # latest result (session cache)
    "content_ready":   False,    # True once vectorstore exists
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# Sync content_ready with disk state (survives page refresh)
if not st.session_state.content_ready:
    st.session_state.content_ready = content_db_exists(CONTENT_DB_PATH)

# Sync quiz_result from DB if session lost (page refresh)
if st.session_state.quiz_result is None:
    st.session_state.quiz_result = load_latest_quiz_result()

# Restore study plan from SQLite so reminders still work after page refresh
if st.session_state.study_plan is None:
    st.session_state.study_plan = get_latest_study_plan()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📚 Study Assistant")
    st.caption("Multi-Agent AI Platform")
    st.markdown("---")

    st.subheader("Content Agent — Upload Material")
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"], label_visibility="collapsed")

    if uploaded_file is not None:
        # Capture name immediately — never access uploaded_file.name inside except
        pdf_name = uploaded_file.name
        with st.spinner(f"Indexing '{pdf_name}'..."):
            try:
                save_dir = os.path.join(os.path.dirname(__file__), "data")
                os.makedirs(save_dir, exist_ok=True)
                file_path = os.path.join(save_dir, pdf_name)

                # Write bytes explicitly — avoids partial-write on large files
                file_bytes = uploaded_file.read()
                if not file_bytes:
                    st.error("Uploaded file is empty. Please upload a valid PDF.")
                    st.stop()

                with open(file_path, "wb") as fh:
                    fh.write(file_bytes)

                build_vectorstore(file_path, CONTENT_DB_PATH)
                st.session_state.content_ready = True
                st.session_state.active_agent = "Content Agent"

                # Persist PDF record to SQLite
                _meta = get_uploaded_files(CONTENT_DB_PATH)
                _chunks = next((f["chunks"] for f in _meta if f["filename"] == pdf_name), 0)
                upsert_pdf(pdf_name, _chunks)

                st.success(f"✅ '{pdf_name}' indexed successfully!")

            except Exception as exc:
                st.error(f"Upload failed: {exc}")
                st.stop()  # Prevents any further execution after failure

    st.markdown("---")
    uploaded_files = get_uploaded_files(CONTENT_DB_PATH)
    if uploaded_files:
        st.success(f"📗 {len(uploaded_files)} PDF(s) in memory")
        for f in uploaded_files:
            st.caption(f"📄 {f['filename']} ({f.get('chunks', '?')} chunks)")
    else:
        st.warning("📕 No content uploaded yet.\nUpload a PDF to begin.")

    # Quick stats — quiz history
    all_results = load_all_quiz_results()
    if all_results:
        st.markdown("---")
        st.subheader("📊 Quiz History")
        for r in all_results[-5:][::-1]:   # newest first, max 5
            ts = r.get("timestamp", "")[:10]
            pct = r.get("percentage", 0)
            st.write(f"• {ts} — {r['score']}/{r['total']} ({pct}%)")

    st.markdown("---")
    st.caption(
        "**You can say things like:**\n"
        "- 'test me on git branching'\n"
        "- 'summarize the SCM pdf'\n"
        "- 'make a 3-day study plan'\n"
        "- 'how am I performing?'\n"
        "- 'list topics in my material'\n"
        "- Any question — it just works"
    )

# ── Main area ─────────────────────────────────────────────────────────────────
st.title("🤖 Multi-Agent Study Assistant")
st.caption("Orchestrator Agent is always active — routing your requests to the right specialist.")

# Active agent banner
if st.session_state.active_agent:
    st.info(f"⚡ **{st.session_state.active_agent} Activated**")

st.markdown("---")

# ── Render chat history ───────────────────────────────────────────────────────
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and msg.get("agent"):
            st.caption(f"— {msg['agent']}")
        st.write(msg["content"])

# ── Quiz UI ───────────────────────────────────────────────────────────────────
if st.session_state.quiz and st.session_state.quiz_generated:
    st.markdown("---")
    st.subheader(f"📝 Quiz: {st.session_state.quiz_topic}")
    st.info("⚡ **Content Agent Activated** — Quiz Mode")
    st.caption("Select one answer per question. No answer is pre-selected — you must choose manually.")

    mcqs = st.session_state.quiz.get("mcqs", [])
    attempt = st.session_state.quiz_attempt   # used in widget keys to avoid stale state

    user_answers = {}   # {question_index: chosen_key or None}

    for i, q in enumerate(mcqs):
        st.write(f"**Q{i + 1}: {q['question']}**")
        option_labels = [f"{opt['key']}. {opt['text']}" for opt in q["options"]]

        selected_label = st.radio(
            "Select your answer:",
            options=option_labels,
            index=None,                          # ← NO default selection
            key=f"quiz_a{attempt}_q{i}",
        )

        # Extract key letter (e.g. "A" from "A. Option text")
        user_answers[i] = selected_label.split(".")[0].strip() if selected_label else None

    col1, col2 = st.columns([1, 1])

    with col1:
        if st.button("✅ Submit Quiz", key=f"submit_{attempt}"):
            score = 0
            weak = []
            skipped = sum(1 for k in user_answers.values() if k is None)

            for i, q in enumerate(mcqs):
                chosen = user_answers.get(i)
                if chosen is None:
                    continue
                if chosen == q["answer"]:
                    score += 1
                else:
                    topic = q.get("topic", st.session_state.quiz_topic)
                    weak.append(topic)

            total = len(mcqs)

            if skipped > 0:
                st.warning(f"You skipped {skipped} question(s) — they are counted as wrong.")

            pct = round((score / total) * 100, 1) if total > 0 else 0.0
            result = {
                "score": score,
                "total": total,
                "percentage": pct,
                "weak_topics": list(set(weak)),
            }

            # Persist to session AND disk DB
            st.session_state.quiz_result = result
            st.session_state.weak_topics = result["weak_topics"]
            save_quiz_result(score, total, result["weak_topics"])

            # Build result message
            result_msg = (
                f"Quiz completed!\n\n"
                f"**Score: {score}/{total} ({pct}%)**\n\n"
                + (f"Weak areas: {', '.join(set(weak))}" if weak else "Excellent — no weak areas!")
            )
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": result_msg,
                "agent": "Content Agent",
            })

            # Clear quiz
            st.session_state.quiz = None
            st.session_state.quiz_generated = False
            st.rerun()

    with col2:
        if st.button("🔄 Cancel Quiz", key=f"cancel_{attempt}"):
            st.session_state.quiz = None
            st.session_state.quiz_generated = False
            st.rerun()

    st.markdown("---")

# ── Chat input ────────────────────────────────────────────────────────────────
_placeholder = (
    "Complete the quiz above before asking new questions."
    if st.session_state.quiz_generated
    else "Ask me anything about your study material…"
)

def _render_countdown_timer(message: str, duration_minutes: int) -> None:
    """
    Inject a JS countdown timer + browser notification into the Streamlit page.
    The timer runs client-side so it does NOT block the Python thread.
    Browser notifications fire when the countdown hits zero.
    """
    seconds = duration_minutes * 60
    safe_msg = message.replace("'", "\\'").replace('"', '\\"')
    components.html(
        f"""
        <div id="sm-timer" style="
            padding:14px 18px; background:#eef6ff; border-left:4px solid #3b82f6;
            border-radius:8px; font-family:sans-serif; font-size:1.05em; margin-top:8px;">
            ⏱️ Loading timer…
        </div>
        <script>
        (function() {{
            var remaining = {seconds};
            var el = document.getElementById('sm-timer');

            // Ask for notification permission once
            if ('Notification' in window && Notification.permission === 'default') {{
                Notification.requestPermission();
            }}

            function fmt(s) {{
                var m = Math.floor(s / 60), sec = s % 60;
                return m + ':' + (sec < 10 ? '0' : '') + sec;
            }}

            function tick() {{
                if (remaining <= 0) {{
                    el.innerHTML = '✅ <strong>Timer done!</strong> — {safe_msg}';
                    el.style.background = '#d1fae5';
                    el.style.borderColor = '#10b981';
                    if ('Notification' in window && Notification.permission === 'granted') {{
                        new Notification('StudyMind AI ⏰', {{
                            body: '{safe_msg}',
                            tag: 'studymind-reminder'
                        }});
                    }}
                    return;
                }}
                el.innerHTML = '⏱️ <strong>' + fmt(remaining) + '</strong> — {safe_msg}';
                remaining--;
                setTimeout(tick, 1000);
            }}
            tick();
        }})();
        </script>
        """,
        height=70,
    )


user_input = st.chat_input(_placeholder)

# Block new messages while quiz is active
if user_input and st.session_state.quiz_generated:
    st.warning("Please finish or cancel the quiz above before sending a new message.")
    user_input = None

if user_input:
    user_input = user_input.strip()

    # Add user message to history + persist to SQLite
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    save_chat_message("user", user_input)

    # ── Central routing — LLM-based intent + topic + source extraction ────────
    _uploaded = get_uploaded_files(CONTENT_DB_PATH)
    routing = route_query(llm, user_input, _uploaded, st.session_state.content_ready)
    intent        = routing["intent"]
    agent_label   = routing["agent_label"]
    routing_topic = routing.get("topic", "")
    source_filter = routing.get("source_filter")
    st.session_state.active_agent = agent_label
    response = ""

    # Build system prompt once — injected into every LLM call this turn
    sys_prompt = build_system_prompt(_uploaded)

    # ── Multi-PDF clarification guard ─────────────────────────────────────────
    # For summary / topics: if >1 PDF and user didn't specify which one → ask
    if intent in {"summary", "topics"} and len(_uploaded) > 1 and source_filter is None:
        file_list = "\n".join(f"- `{f['filename']}`" for f in _uploaded)
        response = (
            f"You have **{len(_uploaded)} PDFs** in memory. "
            f"Which one would you like me to use?\n\n{file_list}\n\n"
            "Just mention the filename and I'll proceed right away."
        )

    # ── EMOTIONAL ────────────────────────────────────────────────────────────
    elif intent == "emotional":
        response = get_emotional_response(llm, user_input, system_prompt=sys_prompt)

    # ── CHAT (greetings / small-talk) ─────────────────────────────────────────
    elif intent == "chat":
        response = get_chat_response(llm, user_input, system_prompt=sys_prompt)

    # ── SUMMARY ───────────────────────────────────────────────────────────────
    elif intent == "summary":
        if not st.session_state.content_ready:
            response = answer_general(llm, user_input, st.session_state.chat_history[:-1],
                                      system_prompt=sys_prompt)
        else:
            with st.spinner("Content Agent: summarizing your material…"):
                topic = expand_topic(routing_topic) if routing_topic else user_input
                ctx, sim_score = get_context(topic, CONTENT_DB_PATH, retrieve_content,
                                             source_filter=source_filter)
                if sim_score < OUT_OF_SYLLABUS_THRESHOLD:
                    response = answer_general(llm, user_input, st.session_state.chat_history[:-1],
                                              system_prompt=sys_prompt)
                else:
                    response = summarize_content(llm, ctx, topic=topic,
                                                 system_prompt=sys_prompt)

    # ── PDF INFO ──────────────────────────────────────────────────────────────
    elif intent == "pdf_info":
        response = get_pdf_info_response(get_uploaded_files(CONTENT_DB_PATH))

    # ── TOPICS ────────────────────────────────────────────────────────────────
    elif intent == "topics":
        if not st.session_state.content_ready:
            response = (
                "No PDFs uploaded yet. Please upload a PDF using the sidebar "
                "and I'll extract all topics from it."
            )
        else:
            with st.spinner("Content Agent: scanning topics in your material…"):
                ctx, _ = get_context(
                    "topics concepts overview introduction headings",
                    CONTENT_DB_PATH, retrieve_content,
                    source_filter=source_filter,
                )
                response = extract_topics(llm, ctx, system_prompt=sys_prompt)

    # ── QUIZ ─────────────────────────────────────────────────────────────────
    elif intent == "quiz":
        if not st.session_state.content_ready:
            response = (
                "Content Agent needs study material before generating a quiz. "
                "Please upload a PDF using the sidebar first."
            )
        else:
            # LLM already extracted topic — expand acronyms, fall back to "general"
            topic = expand_topic(routing_topic.strip()) if routing_topic else "general"

            st.session_state.quiz_topic = topic
            st.session_state.quiz_attempt += 1   # reset radio widget keys

            with st.spinner("Content Agent: generating quiz…"):
                ctx, sim_score = get_context(topic, CONTENT_DB_PATH, retrieve_content,
                                             source_filter=source_filter)

                if sim_score < OUT_OF_SYLLABUS_THRESHOLD:
                    response = answer_general(llm, user_input, st.session_state.chat_history[:-1],
                                              system_prompt=sys_prompt)
                else:
                    quiz = generate_quiz(llm, ctx, topic)
                    if quiz and quiz.get("mcqs"):
                        st.session_state.quiz = quiz
                        st.session_state.quiz_generated = True
                        response = f"Quiz ready on '{topic}'! Answer the questions below."
                    else:
                        response = (
                            "Quiz generation failed. Try a more specific topic "
                            "or rephrase your request."
                        )

    # ── EVALUATE ─────────────────────────────────────────────────────────────
    elif intent == "evaluate":
        if not st.session_state.content_ready:
            response = (
                "Evaluation Agent requires study content to be uploaded first. "
                "Please upload your PDF using the sidebar."
            )
        else:
            # Load from session cache first, then DB — NEVER generates a new quiz
            quiz_result = st.session_state.quiz_result or load_latest_quiz_result()

            if not quiz_result:
                response = (
                    "Evaluation Agent found no quiz results. "
                    "Please take a quiz first — type 'quiz' to start one."
                )
            else:
                with st.spinner("Evaluation Agent: analysing performance…"):
                    response = analyze_performance(llm, quiz_result)

    # ── PLAN ─────────────────────────────────────────────────────────────────
    elif intent == "plan":
        if not st.session_state.content_ready:
            response = (
                "Planner Agent requires study content. "
                "Please upload your PDF material first."
            )
        else:
            # Pull weak topics from session or DB
            quiz_result = st.session_state.quiz_result or load_latest_quiz_result()
            weak = (
                st.session_state.weak_topics
                or (quiz_result.get("weak_topics", []) if quiz_result else [])
            )

            with st.spinner("Planner Agent: building study plan…"):
                # LLM already extracted specific topic; fall back to broad retrieval if empty
                retrieval_query = (expand_topic(routing_topic) if routing_topic
                                   else "overview introduction topics summary")
                ctx, _ = get_context(retrieval_query, CONTENT_DB_PATH, retrieve_content,
                                     source_filter=source_filter)
                plan = generate_study_plan(llm, weak, ctx, user_query=user_input)
                st.session_state.study_plan = plan
                save_study_plan(plan)         # persist to SQLite
                response = plan

    # ── REMINDER ─────────────────────────────────────────────────────────────
    elif intent == "reminder":
        # ── Quick timer: "remind me in 25 minutes" ────────────────────────────
        quick = extract_quick_reminder(user_input)
        if quick:
            dur  = quick["duration"]
            msg  = quick["message"]
            save_reminder(msg, dur)           # persist to SQLite
            response = (
                f"⏰ **Timer set for {dur} minute{'s' if dur != 1 else ''}!**\n\n"
                f"I'll alert you when it's done. The countdown has started below."
            )
            # Timer renders immediately (client-side JS, non-blocking)
            _render_countdown_timer(msg, dur)

        # ── Full schedule from study plan ─────────────────────────────────────
        elif not st.session_state.study_plan:
            response = (
                "Reminder Agent needs a study plan first. "
                "Type **'create a study plan'** and I'll generate one, "
                "then I can set up your reminders."
            )
        else:
            with st.spinner("Reminder Agent: creating schedule…"):
                result = generate_reminders(llm, st.session_state.study_plan)
                response = result["display_text"]

                # Persist each reminder to SQLite
                for r in result.get("reminders", []):
                    save_reminder(r.get("message", ""), r.get("duration_minutes", 0))

                # Render reminder cards + timers for break blocks
                if result.get("reminders"):
                    st.markdown("### ⏰ Reminder Schedule")
                    for r in result["reminders"]:
                        icon = {"study": "📚", "break": "☕", "review": "🔁"}.get(
                            r.get("type", "study"), "📌"
                        )
                        dur = r.get("duration_minutes", 0)
                        st.info(
                            f"{icon} **Day {r.get('day','')} — {r.get('time','')}**  \n"
                            f"{r.get('message','')}  \n"
                            f"Duration: {dur} min"
                        )
                        # Attach a live countdown to break blocks only
                        if r.get("type") == "break" and dur > 0:
                            _render_countdown_timer(r.get("message", "Break time!"), dur)

    # ── QA (default) ─────────────────────────────────────────────────────────
    else:
        history = st.session_state.chat_history[:-1]  # exclude current turn
        if not st.session_state.content_ready:
            response = answer_general(llm, user_input, history, system_prompt=sys_prompt)
        else:
            with st.spinner("Content Agent: searching your material…"):
                expanded_query = expand_topic(routing_topic or user_input)
                ctx, sim_score = get_context(expanded_query, CONTENT_DB_PATH, retrieve_content,
                                             source_filter=source_filter)

                if sim_score < OUT_OF_SYLLABUS_THRESHOLD:
                    response = answer_general(llm, user_input, history, system_prompt=sys_prompt)
                else:
                    response = answer_question(llm, ctx, user_input, history=history,
                                               system_prompt=sys_prompt)

    # ── Store assistant message + persist + rerun ─────────────────────────────
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": response,
        "agent": agent_label,
    })
    save_chat_message("assistant", response, agent_label)
    st.rerun()
