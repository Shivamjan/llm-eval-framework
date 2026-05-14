"""
Logger - stores eval runs in SQLite.
Nothing fancy. Just works.
"""

import sqlite3
import json
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "logs", "eval_logs.db")


def get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    return conn


def init_db():
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS eval_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            query TEXT,
            retrieved_docs TEXT,
            llm_response TEXT,
            prompt_label TEXT,
            relevance_score REAL,
            faithfulness_score REAL,
            hallucination_flag INTEGER,
            hallucination_reason TEXT,
            extra_notes TEXT
        )
    """)
    conn.commit()
    conn.close()


def log_eval(
    query: str,
    retrieved_docs: list[str],
    llm_response: str,
    relevance_score: float,
    faithfulness_score: float,
    hallucination_flag: bool,
    hallucination_reason: str = "",
    prompt_label: str = "default",
    extra_notes: str = ""
):
    init_db()
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        INSERT INTO eval_runs (
            timestamp, query, retrieved_docs, llm_response, prompt_label,
            relevance_score, faithfulness_score, hallucination_flag,
            hallucination_reason, extra_notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().isoformat(),
        query,
        json.dumps(retrieved_docs),
        llm_response,
        prompt_label,
        round(relevance_score, 4),
        round(faithfulness_score, 4),
        int(hallucination_flag),
        hallucination_reason,
        extra_notes
    ))
    conn.commit()
    conn.close()


def get_all_logs() -> list[dict]:
    init_db()
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM eval_runs ORDER BY timestamp DESC")
    rows = c.fetchall()
    conn.close()
    result = []
    for row in rows:
        d = dict(row)
        d["retrieved_docs"] = json.loads(d["retrieved_docs"])
        d["hallucination_flag"] = bool(d["hallucination_flag"])
        result.append(d)
    return result


def get_logs_by_label(label: str) -> list[dict]:
    init_db()
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute(
        "SELECT * FROM eval_runs WHERE prompt_label = ? ORDER BY timestamp DESC",
        (label,)
    )
    rows = c.fetchall()
    conn.close()
    result = []
    for row in rows:
        d = dict(row)
        d["retrieved_docs"] = json.loads(d["retrieved_docs"])
        d["hallucination_flag"] = bool(d["hallucination_flag"])
        result.append(d)
    return result
