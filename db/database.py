"""
StudyMind AI — SQLite Persistent Storage
-----------------------------------------
Single-file database for all persistent data.
Tables: pdfs, study_plans, reminders, chats

No ORM — plain sqlite3 for zero extra dependencies.
Call init_db() once on startup; all other functions are safe to call anytime.
"""

import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "studymind.db")


# ── Connection ────────────────────────────────────────────────────────────────

def _conn() -> sqlite3.Connection:
    """Open a connection with row_factory for dict-like access."""
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


# ── Schema ────────────────────────────────────────────────────────────────────

def init_db() -> None:
    """Create all tables if they don't exist. Safe to call on every startup."""
    with _conn() as con:
        con.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                name          TEXT    NOT NULL,
                email         TEXT    NOT NULL UNIQUE,
                password_hash TEXT    NOT NULL,
                created_at    TEXT    NOT NULL
            );

            CREATE TABLE IF NOT EXISTS pdfs (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id      INTEGER NOT NULL DEFAULT 0,
                filename     TEXT    NOT NULL UNIQUE,
                chunks       INTEGER NOT NULL DEFAULT 0,
                upload_time  TEXT    NOT NULL
            );

            CREATE TABLE IF NOT EXISTS study_plans (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id     INTEGER NOT NULL DEFAULT 0,
                content     TEXT NOT NULL,
                created_at  TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS reminders (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id          INTEGER NOT NULL DEFAULT 0,
                message          TEXT    NOT NULL,
                duration_minutes INTEGER NOT NULL DEFAULT 0,
                status           TEXT    NOT NULL DEFAULT 'pending',
                created_at       TEXT    NOT NULL
            );

            CREATE TABLE IF NOT EXISTS chats (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id    INTEGER NOT NULL DEFAULT 0,
                role       TEXT NOT NULL,
                content    TEXT NOT NULL,
                agent      TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS events (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id    INTEGER NOT NULL DEFAULT 0,
                title      TEXT    NOT NULL,
                start_dt   TEXT    NOT NULL,
                type       TEXT    NOT NULL DEFAULT 'study',
                color      TEXT    NOT NULL DEFAULT '#7c5cff',
                created_at TEXT    NOT NULL
            );
        """)


# ── PDFs ──────────────────────────────────────────────────────────────────────

def upsert_pdf(filename: str, chunks: int) -> None:
    """Insert or update a PDF record (deduplicates by filename)."""
    with _conn() as con:
        con.execute(
            """INSERT INTO pdfs (filename, chunks, upload_time)
               VALUES (?, ?, ?)
               ON CONFLICT(filename) DO UPDATE SET
                   chunks = excluded.chunks,
                   upload_time = excluded.upload_time""",
            (filename, chunks, datetime.now().isoformat()),
        )


def get_all_pdfs() -> list[dict]:
    """Return all uploaded PDF records ordered by upload time."""
    with _conn() as con:
        rows = con.execute(
            "SELECT filename, chunks, upload_time FROM pdfs ORDER BY upload_time ASC"
        ).fetchall()
    return [{"filename": r["filename"], "chunks": r["chunks"],
             "uploaded_at": r["upload_time"]} for r in rows]


# ── Study Plans ───────────────────────────────────────────────────────────────

def save_study_plan(content: str) -> None:
    with _conn() as con:
        con.execute(
            "INSERT INTO study_plans (content, created_at) VALUES (?, ?)",
            (content, datetime.now().isoformat()),
        )


def get_latest_study_plan() -> str | None:
    with _conn() as con:
        row = con.execute(
            "SELECT content FROM study_plans ORDER BY id DESC LIMIT 1"
        ).fetchone()
    return row["content"] if row else None


# ── Reminders ─────────────────────────────────────────────────────────────────

def save_reminder(message: str, duration_minutes: int) -> int:
    """Persist a reminder and return its row id."""
    with _conn() as con:
        cur = con.execute(
            """INSERT INTO reminders (message, duration_minutes, status, created_at)
               VALUES (?, ?, 'pending', ?)""",
            (message, duration_minutes, datetime.now().isoformat()),
        )
        return cur.lastrowid


def complete_reminder(reminder_id: int) -> None:
    with _conn() as con:
        con.execute(
            "UPDATE reminders SET status = 'done' WHERE id = ?",
            (reminder_id,),
        )


def get_pending_reminders() -> list[dict]:
    with _conn() as con:
        rows = con.execute(
            "SELECT id, message, duration_minutes FROM reminders WHERE status = 'pending'"
        ).fetchall()
    return [{"id": r["id"], "message": r["message"],
             "duration_minutes": r["duration_minutes"]} for r in rows]


# ── Chats ─────────────────────────────────────────────────────────────────────

def save_chat_message(role: str, content: str, agent: str = "") -> None:
    """Persist a single chat turn (role: 'user' | 'assistant')."""
    with _conn() as con:
        con.execute(
            "INSERT INTO chats (role, content, agent, created_at) VALUES (?, ?, ?, ?)",
            (role, content, agent, datetime.now().isoformat()),
        )


def get_recent_chats(limit: int = 20) -> list[dict]:
    """Return the most recent N chat messages (oldest first)."""
    with _conn() as con:
        rows = con.execute(
            "SELECT role, content, agent FROM chats ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [{"role": r["role"], "content": r["content"], "agent": r["agent"]}
            for r in reversed(rows)]


# ── Users ─────────────────────────────────────────────────────────────────────

def create_user(name: str, email: str, password_hash: str) -> int:
    """Insert a new user and return their id. Raises IntegrityError on duplicate email."""
    with _conn() as con:
        cur = con.execute(
            "INSERT INTO users (name, email, password_hash, created_at) VALUES (?, ?, ?, ?)",
            (name, email, password_hash, datetime.now().isoformat()),
        )
        return cur.lastrowid


def get_user_by_email(email: str) -> dict | None:
    """Return {id, name, email, password_hash} or None if not found."""
    with _conn() as con:
        row = con.execute(
            "SELECT id, name, email, password_hash FROM users WHERE email = ?",
            (email,),
        ).fetchone()
    return dict(row) if row else None


# ── Events (Calendar) ─────────────────────────────────────────────────────────

def save_event(user_id: int, title: str, start_dt: str,
               event_type: str = "study", color: str = "#7c5cff") -> int:
    """Persist a calendar event and return its id."""
    with _conn() as con:
        cur = con.execute(
            """INSERT INTO events (user_id, title, start_dt, type, color, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (user_id, title, start_dt, event_type, color, datetime.now().isoformat()),
        )
        return cur.lastrowid


def get_events(user_id: int) -> list[dict]:
    """Return all calendar events for a user."""
    with _conn() as con:
        rows = con.execute(
            "SELECT id, title, start_dt, type, color FROM events WHERE user_id = ? ORDER BY start_dt ASC",
            (user_id,),
        ).fetchall()
    return [{"id": r["id"], "title": r["title"], "start_dt": r["start_dt"],
             "type": r["type"], "color": r["color"]} for r in rows]


def delete_event(event_id: int) -> None:
    with _conn() as con:
        con.execute("DELETE FROM events WHERE id = ?", (event_id,))
