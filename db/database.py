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
            risk_label TEXT,
            FOREIGN KEY (run_id) REFERENCES runs(id)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS spans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            span_text TEXT NOT NULL,
            start_token_index INTEGER NOT NULL,
            end_token_index INTEGER NOT NULL,
            avg_score REAL,
            max_score REAL,
            span_label TEXT,
            FOREIGN KEY (run_id) REFERENCES runs(id)
        )
    """)

    conn.commit()
    conn.close()


def save_run(prompt, response_text, overall_score, token_scores, spans):
    conn = get_connection()
    cursor = conn.cursor()

    created_at = datetime.utcnow().isoformat()

    cursor.execute("""
        INSERT INTO runs (prompt, response_text, overall_score, created_at)
        VALUES (?, ?, ?, ?)
    """, (prompt, response_text, overall_score, created_at))

    run_id = cursor.lastrowid

    for item in token_scores:
        cursor.execute("""
            INSERT INTO token_scores (
                run_id, token_index, token_text, sep_score, shift_score, final_score, risk_label
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            run_id,
            item["step"],
            item["token_text"],
            item.get("sep_score"),
            item.get("hallushift_score"),
            item.get("final_risk_score"),
            item.get("risk_label"),
        ))

    for span in spans:
        cursor.execute("""
            INSERT INTO spans (
                run_id, span_text, start_token_index, end_token_index, avg_score, max_score, span_label
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            run_id,
            span["span_text"],
            span["start_step"],
            span["end_step"],
            span.get("avg_risk"),
            span.get("max_risk"),
            span.get("span_label"),
        ))

    conn.commit()
    conn.close()
    return run_id

def get_recent_runs(limit=10):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, prompt, response_text, overall_score, created_at
        FROM runs
        ORDER BY id DESC
        LIMIT ?
    """, (limit,))

    rows = cursor.fetchall()
    conn.close()

    results = []
    for row in rows:
        results.append({
            "id": row[0],
            "prompt": row[1],
            "response_text": row[2],
            "overall_score": row[3],
            "created_at": row[4],
        })
    return results


def get_run_details(run_id):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, prompt, response_text, overall_score, created_at
        FROM runs
        WHERE id = ?
    """, (run_id,))
    run_row = cursor.fetchone()

    if not run_row:
        conn.close()
        return None

    cursor.execute("""
        SELECT token_index, token_text, sep_score, shift_score, final_score, risk_label
        FROM token_scores
        WHERE run_id = ?
        ORDER BY token_index ASC
    """, (run_id,))
    token_rows = cursor.fetchall()

    cursor.execute("""
        SELECT span_text, start_token_index, end_token_index, avg_score, max_score, span_label
        FROM spans
        WHERE run_id = ?
        ORDER BY start_token_index ASC
    """, (run_id,))
    span_rows = cursor.fetchall()

    conn.close()

    return {
        "run": {
            "id": run_row[0],
            "prompt": run_row[1],
            "response_text": run_row[2],
            "overall_score": run_row[3],
            "created_at": run_row[4],
        },
        "tokens": [
            {
                "token_index": row[0],
                "token_text": row[1],
                "sep_score": row[2],
                "shift_score": row[3],
                "final_score": row[4],
                "risk_label": row[5],
            }
            for row in token_rows
        ],
        "spans": [
            {
                "span_text": row[0],
                "start_token_index": row[1],
                "end_token_index": row[2],
                "avg_score": row[3],
                "max_score": row[4],
                "span_label": row[5],
            }
            for row in span_rows
        ]
    }