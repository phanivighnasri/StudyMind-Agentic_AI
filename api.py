"""
Flask Backend — StudyMind AI
-----------------------------
Serves the HTML frontend and exposes the Orchestrator via REST API.

Endpoints
---------
POST /login          — authenticate user (DB-backed)
POST /signup         — register user (DB-backed)
GET  /status         — check whether content DB is ready
GET  /pdfs           — list uploaded PDFs for current user
POST /upload         — upload & index a PDF
POST /chat           — send message → orchestrator → response + agent label
POST /quiz/submit    — submit quiz answers → score + store result to DB
GET  /events         — get calendar events for current user
POST /events         — create a calendar event
DELETE /events/<id>  — remove a calendar event
GET  /               — serve index5.html
"""

import os
import re
import uuid
from datetime import datetime, timedelta

from flask import Flask, request, jsonify, send_file, session as flask_session
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv
from langchain_groq import ChatGroq

from agents.orchestrator import (
    route_query, get_agent_label,
    get_emotional_response, get_chat_response, get_pdf_info_response,
    content_db_exists, expand_topic, build_system_prompt, answer_general,
)
from agents.content_agent import (
    get_context, answer_question, generate_quiz, summarize_content,
    extract_topics, save_quiz_result, load_latest_quiz_result,
)
from agents.planner import generate_study_plan
from agents.reminder import generate_reminders, extract_quick_reminder
from agents.evaluation import analyze_performance
from app.retrieval import retrieve_content
from app.ingestion import build_vectorstore, get_uploaded_files
from db.database import (
    init_db,
    create_user, get_user_by_email,
    upsert_pdf, get_all_pdfs,
    save_study_plan, get_latest_study_plan,
    save_reminder,
    save_chat_message,
    save_event, get_events, delete_event,
)

# ── Config ────────────────────────────────────────────────────────────────────
load_dotenv()
init_db()

GROQ_API_KEY      = os.getenv("GROQ_API_KEY", "")
CONTENT_DB_PATH   = "./db/content_db"
OUT_OF_SYLLABUS_T = 0.05
TEMP_DIR          = "temp"

# ── LLM ───────────────────────────────────────────────────────────────────────
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.3,
    groq_api_key=GROQ_API_KEY,
    request_timeout=120,
    max_retries=3,
)

# ── App ───────────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET", "studymind-secret-key-change-in-prod")
CORS(app, supports_credentials=True)

# ── Server-side session store ─────────────────────────────────────────────────
_sessions: dict[str, dict] = {}


def _get_sess() -> dict:
    sid = flask_session.get("sid")
    if not sid or sid not in _sessions:
        sid = str(uuid.uuid4())
        flask_session["sid"] = sid
        _sessions[sid] = {
            "user_id":      0,
            "quiz_result":  None,
            "study_plan":   None,
            "weak_topics":  [],
            "current_quiz": None,
            "quiz_topic":   "",
        }
    return _sessions[sid]


def _current_user_id() -> int:
    """Return DB user id from session, or 0 for anonymous."""
    return _get_sess().get("user_id", 0)


# ── Plan → calendar events parser ─────────────────────────────────────────────

_TIME_SLOT_RE = re.compile(
    r"(\d{1,2}):(\d{2})\s*(AM|PM)\s*[–\-]+\s*\d{1,2}:\d{2}\s*(?:AM|PM)[:\s]*(.+)",
    re.IGNORECASE,
)
_DAY_RE = re.compile(r"Day\s+(\d+)", re.IGNORECASE)


def _parse_plan_events(plan_text: str) -> list[dict]:
    """
    Extract time-slotted tasks from a study plan and return calendar event dicts.
    Each event: {title, start_dt (ISO str), type, color}
    """
    events = []
    today = datetime.now().date()
    current_day = 1

    for line in plan_text.splitlines():
        line = line.strip()
        day_m = _DAY_RE.search(line)
        if day_m:
            current_day = int(day_m.group(1))
            continue

        slot_m = _TIME_SLOT_RE.search(line)
        if not slot_m:
            continue

        hour   = int(slot_m.group(1))
        minute = int(slot_m.group(2))
        ampm   = slot_m.group(3).upper()
        topic  = slot_m.group(4).strip().rstrip(".")

        if ampm == "PM" and hour != 12:
            hour += 12
        elif ampm == "AM" and hour == 12:
            hour = 0

        event_date = today + timedelta(days=current_day - 1)
        start_dt   = datetime(event_date.year, event_date.month, event_date.day,
                              hour, minute).isoformat()

        is_break = bool(re.search(r"\bbreak\b", topic, re.IGNORECASE))
        events.append({
            "title":    topic,
            "start_dt": start_dt,
            "type":     "break" if is_break else "study",
            "color":    "#9e9e9e" if is_break else "#7c5cff",
        })

    return events


# ── Auth ──────────────────────────────────────────────────────────────────────

@app.route("/signup", methods=["POST"])
def signup():
    data = request.json or {}
    name, email, password = data.get("name"), data.get("email"), data.get("password")
    if not all([name, email, password]):
        return jsonify({"success": False, "message": "All fields required"}), 400
    if get_user_by_email(email):
        return jsonify({"success": False, "message": "Email already registered"}), 400
    try:
        create_user(name, email, generate_password_hash(password))
        return jsonify({"success": True, "message": "Account created. Please sign in."})
    except Exception as exc:
        return jsonify({"success": False, "message": str(exc)}), 500


@app.route("/login", methods=["POST"])
def login():
    data = request.json or {}
    email, password = data.get("email"), data.get("password")
    if not all([email, password]):
        return jsonify({"success": False, "message": "Email and password required"}), 400
    user = get_user_by_email(email)
    if user and check_password_hash(user["password_hash"], password):
        flask_session["user"] = email
        sess = _get_sess()
        sess["user_id"] = user["id"]
        # Restore latest study plan from DB
        plan = get_latest_study_plan()
        if plan:
            sess["study_plan"] = plan
        return jsonify({"success": True, "message": "Login successful", "name": user["name"]})
    return jsonify({"success": False, "message": "Invalid email or password"}), 401


# ── Content DB status ─────────────────────────────────────────────────────────

@app.route("/status", methods=["GET"])
def status():
    return jsonify({"content_ready": content_db_exists(CONTENT_DB_PATH)})


# ── PDFs ──────────────────────────────────────────────────────────────────────

@app.route("/pdfs", methods=["GET"])
def list_pdfs():
    """Return all uploaded PDFs from both DB and vectorstore metadata."""
    try:
        pdfs = get_all_pdfs()
    except Exception:
        pdfs = []
    return jsonify({"pdfs": pdfs})


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"success": False, "message": "No file provided"}), 400
    f = request.files["file"]
    if not f.filename.lower().endswith(".pdf"):
        return jsonify({"success": False, "message": "Only PDF files are supported"}), 400

    os.makedirs(TEMP_DIR, exist_ok=True)
    filename  = f.filename
    save_path = os.path.join(TEMP_DIR, filename)
    f.save(save_path)

    try:
        chunks = build_vectorstore(save_path, CONTENT_DB_PATH)
        upsert_pdf(filename, chunks or 0)
        return jsonify({
            "success":  True,
            "message":  f"'{filename}' indexed successfully.",
            "filename": filename,
        })
    except Exception as exc:
        return jsonify({"success": False, "message": str(exc)}), 500


# ── Calendar events ───────────────────────────────────────────────────────────

@app.route("/events", methods=["GET"])
def list_events():
    uid = _current_user_id()
    return jsonify({"events": get_events(uid)})


@app.route("/events", methods=["POST"])
def create_event():
    data    = request.json or {}
    uid     = _current_user_id()
    title   = data.get("title", "Event")
    start   = data.get("start_dt", datetime.now().isoformat())
    etype   = data.get("type", "study")
    color   = data.get("color", "#7c5cff")
    eid     = save_event(uid, title, start, etype, color)
    return jsonify({"success": True, "id": eid})


@app.route("/events/<int:event_id>", methods=["DELETE"])
def remove_event(event_id: int):
    delete_event(event_id)
    return jsonify({"success": True})


# ── Main chat endpoint (Orchestrator) ─────────────────────────────────────────

@app.route("/chat", methods=["POST"])
def chat():
    """
    Body: { "message": "..." }
    Response: { "response": <str|dict>, "agent": "...", "quiz": <dict|null>, "events": [...] }
    """
    data    = request.json or {}
    message = data.get("message", "").strip()
    if not message:
        return jsonify({"error": "Empty message"}), 400

    sess    = _get_sess()
    uid     = sess.get("user_id", 0)

    # Build uploaded files list (from DB + vectorstore metadata for max accuracy)
    uploaded_files = get_all_pdfs()
    content_ready  = content_db_exists(CONTENT_DB_PATH)
    sys_prompt     = build_system_prompt(uploaded_files)

    # Two-tier intent routing (LLM classifier + regex fallback)
    route      = route_query(llm, message, uploaded_files, content_ready)
    intent     = route["intent"]
    topic      = route["topic"] or message
    src_filter = route["source_filter"]
    agent_label = route["agent_label"]

    response     = ""
    quiz_payload = None
    cal_events   = []   # calendar events to send back to frontend

    # ── EMOTIONAL ────────────────────────────────────────────────────────────
    if intent == "emotional":
        response = get_emotional_response(llm, message, sys_prompt)

    # ── CHAT (greetings, capability questions) ────────────────────────────────
    elif intent == "chat":
        response = get_chat_response(llm, message, sys_prompt)

    # ── PDF INFO ──────────────────────────────────────────────────────────────
    elif intent == "pdf_info":
        response = get_pdf_info_response(uploaded_files)

    # ── EVALUATE ─────────────────────────────────────────────────────────────
    elif intent == "evaluate":
        quiz_result = sess.get("quiz_result") or load_latest_quiz_result()
        if not quiz_result:
            response = (
                "No quiz results found yet. "
                "Take a quiz first by typing 'quiz on <topic>'."
            )
        else:
            response = analyze_performance(llm, quiz_result)

    # ── REMINDER ─────────────────────────────────────────────────────────────
    elif intent == "reminder":
        # Try quick reminder first ("remind me in 25 minutes")
        quick = extract_quick_reminder(message)
        if quick:
            response = quick
            # Add calendar event at trigger time
            trigger_dt = (datetime.now() + timedelta(minutes=quick["duration"])).isoformat()
            save_event(uid, f"⏰ {quick['message']}", trigger_dt, "reminder", "#9c27b0")
            cal_events.append({
                "title":    f"⏰ {quick['message']}",
                "start_dt": trigger_dt,
                "type":     "reminder",
                "color":    "#9c27b0",
            })
        elif not sess.get("study_plan"):
            response = (
                "Reminder Agent needs a study plan first. "
                "Type 'study plan' to generate one."
            )
        else:
            response = generate_reminders(llm, sess["study_plan"])
            # Create calendar events from the schedule
            if isinstance(response, dict) and response.get("reminders"):
                today = datetime.now().date()
                for rem in response["reminders"]:
                    try:
                        day   = int(rem.get("day", 1))
                        hhmm  = rem.get("time", "09:00")
                        h, m  = map(int, hhmm.split(":"))
                        dt    = datetime(today.year, today.month, today.day, h, m)
                        dt   += timedelta(days=day - 1)
                        color = "#9c27b0" if rem.get("type") == "break" else "#f57c00"
                        ev = {
                            "title":    rem.get("message", "Study"),
                            "start_dt": dt.isoformat(),
                            "type":     rem.get("type", "study"),
                            "color":    color,
                        }
                        save_event(uid, ev["title"], ev["start_dt"], ev["type"], ev["color"])
                        cal_events.append(ev)
                    except Exception:
                        continue

    # ── PLAN ─────────────────────────────────────────────────────────────────
    elif intent == "plan":
        if not content_ready:
            response = "Planner Agent requires study material. Upload a PDF first."
        else:
            quiz_result = sess.get("quiz_result") or load_latest_quiz_result()
            weak = sess.get("weak_topics") or (
                quiz_result.get("weak_topics", []) if quiz_result else []
            )
            retrieval_q = expand_topic(topic)
            ctx, _ = get_context(retrieval_q, CONTENT_DB_PATH, retrieve_content,
                                  source_filter=src_filter)
            plan = generate_study_plan(llm, weak, ctx, user_query=message)
            sess["study_plan"] = plan
            save_study_plan(plan)
            response = plan

            # Parse plan into calendar events
            cal_events = _parse_plan_events(plan)
            for ev in cal_events:
                save_event(uid, ev["title"], ev["start_dt"], ev["type"], ev["color"])

    # ── QUIZ ─────────────────────────────────────────────────────────────────
    elif intent == "quiz":
        if not content_ready:
            response = "Content Agent needs study material. Upload a PDF first."
        else:
            quiz_topic = expand_topic(topic) if topic else "general"
            ctx, sim = get_context(quiz_topic, CONTENT_DB_PATH, retrieve_content,
                                    source_filter=src_filter)
            if sim < OUT_OF_SYLLABUS_T:
                response = answer_general(llm, message, system_prompt=sys_prompt)
            else:
                quiz = generate_quiz(llm, ctx, quiz_topic)
                if quiz and quiz.get("mcqs"):
                    sess["current_quiz"] = quiz
                    sess["quiz_topic"]   = quiz_topic
                    quiz_payload = quiz
                    response = f"Quiz ready on '{quiz_topic}'! Answer below."
                else:
                    response = "Quiz generation failed. Try a more specific topic."

    # ── SUMMARY ──────────────────────────────────────────────────────────────
    elif intent == "summary":
        if not content_ready:
            response = "Content Agent needs study material. Upload a PDF first."
        else:
            summary_topic = expand_topic(topic) if topic else message
            ctx, sim = get_context(summary_topic, CONTENT_DB_PATH, retrieve_content,
                                    source_filter=src_filter)
            if sim < OUT_OF_SYLLABUS_T:
                response = answer_general(llm, message, system_prompt=sys_prompt)
            else:
                response = summarize_content(llm, ctx, topic=summary_topic,
                                              system_prompt=sys_prompt)

    # ── TOPICS ───────────────────────────────────────────────────────────────
    elif intent == "topics":
        if not content_ready:
            response = "Content Agent needs study material. Upload a PDF first."
        else:
            ctx, sim = get_context(message, CONTENT_DB_PATH, retrieve_content,
                                    source_filter=src_filter)
            if sim < OUT_OF_SYLLABUS_T:
                response = "Could not find enough content. Make sure a PDF is uploaded."
            else:
                response = extract_topics(llm, ctx, system_prompt=sys_prompt)

    # ── QA (default RAG) ─────────────────────────────────────────────────────
    else:
        if not content_ready:
            response = (
                "Please upload a PDF first so I can answer questions from your material."
            )
        else:
            qa_topic = expand_topic(topic) if topic else message
            ctx, sim = get_context(qa_topic, CONTENT_DB_PATH, retrieve_content,
                                    source_filter=src_filter)
            if sim < OUT_OF_SYLLABUS_T:
                # Graceful fallback — never say "out of syllabus"
                response = answer_general(llm, message, system_prompt=sys_prompt)
            else:
                response = answer_question(llm, ctx, message, system_prompt=sys_prompt)

    # Persist chat turn
    save_chat_message("user",      message,  "")
    save_chat_message("assistant", response if isinstance(response, str) else
                      (response.get("display_text", "") if isinstance(response, dict) else ""),
                      agent_label)

    return jsonify({
        "response": response,
        "agent":    agent_label,
        "quiz":     quiz_payload,
        "events":   cal_events,
    })


# ── Quiz submission ───────────────────────────────────────────────────────────

@app.route("/quiz/submit", methods=["POST"])
def quiz_submit():
    data    = request.json or {}
    answers = data.get("answers", {})

    sess = _get_sess()
    quiz = sess.get("current_quiz")
    if not quiz:
        return jsonify({"error": "No active quiz. Generate one first."}), 400

    mcqs  = quiz.get("mcqs", [])
    score = 0
    weak  = []

    for i, q in enumerate(mcqs):
        chosen = answers.get(str(i))
        if chosen == q["answer"]:
            score += 1
        else:
            weak.append(q.get("topic", sess.get("quiz_topic", "general")))

    total  = len(mcqs)
    pct    = round((score / total) * 100, 1) if total > 0 else 0.0
    result = {"score": score, "total": total, "percentage": pct,
              "weak_topics": list(set(weak))}

    sess["quiz_result"]  = result
    sess["weak_topics"]  = result["weak_topics"]
    sess["current_quiz"] = None
    save_quiz_result(score, total, result["weak_topics"])

    return jsonify(result)


# ── Serve frontend ────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_file("index5.html")


if __name__ == "__main__":
    app.run(port=5000, debug=True)
