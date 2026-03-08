import sqlite3
from pathlib import Path
from datetime import datetime

DB_PATH = Path(__file__).resolve().parent / "halluguard.db"


def get_connection():
    return sqlite3.connect(DB_PATH)


def init_db():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prompt TEXT NOT NULL,
            response_text TEXT NOT NULL,
            overall_score REAL,
            created_at TEXT NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS token_scores (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            token_index INTEGER NOT NULL,
            token_text TEXT NOT NULL,
            sep_score REAL,
            shift_score REAL,
            final_score REAL,
            FOREIGN KEY (run_id) REFERENCES runs(id)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS spans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            span_text TEXT NOT NULL,
            start_char INTEGER,
            end_char INTEGER,
            span_score REAL,
            FOREIGN KEY (run_id) REFERENCES runs(id)
        )
    """)

    conn.commit()
    conn.close()


def save_run(prompt, response_text, overall_score=None, token_scores=None, spans=None):
    token_scores = token_scores or []
    spans = spans or []

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO runs (prompt, response_text, overall_score, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (prompt, response_text, overall_score, datetime.now().isoformat())
    )

    run_id = cursor.lastrowid

    for item in token_scores:
        cursor.execute(
            """
            INSERT INTO token_scores (
                run_id, token_index, token_text, sep_score, shift_score, final_score
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                item.get("token_index", 0),
                item.get("token_text", ""),
                item.get("sep_score"),
                item.get("shift_score"),
                item.get("final_score"),
            )
        )

    for item in spans:
        cursor.execute(
            """
            INSERT INTO spans (
                run_id, span_text, start_char, end_char, span_score
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                run_id,
                item.get("span_text", ""),
                item.get("start_char"),
                item.get("end_char"),
                item.get("span_score"),
            )
        )

    conn.commit()
    conn.close()

    return run_id