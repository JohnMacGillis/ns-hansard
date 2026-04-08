"""
Build SQLite database from scraped Hansard transcripts.
Creates tables for members, sitting days, and speeches with FTS5 search.
"""

import json
import sqlite3
import os
import re

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DB_PATH = os.path.join(DATA_DIR, "hansard.db")


def build():
    print("Building Hansard database...")

    # Load scraped data
    with open(os.path.join(DATA_DIR, "transcripts.json")) as f:
        transcripts = json.load(f)

    # Remove old DB
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # --- Schema ---
    c.executescript("""
        CREATE TABLE members (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            slug TEXT UNIQUE NOT NULL,
            party TEXT,
            constituency TEXT,
            is_honourable INTEGER DEFAULT 0
        );

        CREATE TABLE sitting_days (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT UNIQUE NOT NULL,
            slug TEXT NOT NULL,
            url TEXT,
            segment_count INTEGER DEFAULT 0
        );

        CREATE TABLE speeches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            member_id INTEGER,
            sitting_id INTEGER,
            section TEXT,
            timestamp TEXT,
            text TEXT NOT NULL,
            word_count INTEGER DEFAULT 0,
            FOREIGN KEY (member_id) REFERENCES members(id),
            FOREIGN KEY (sitting_id) REFERENCES sitting_days(id)
        );

        CREATE INDEX idx_speeches_member ON speeches(member_id);
        CREATE INDEX idx_speeches_sitting ON speeches(sitting_id);
        CREATE INDEX idx_speeches_section ON speeches(section);
        CREATE INDEX idx_sitting_date ON sitting_days(date);
    """)

    # --- Insert sitting days ---
    sitting_map = {}  # date -> id
    for t in transcripts:
        if t["segment_count"] == 0:
            continue
        if t["date"] in sitting_map:
            # Duplicate date (e.g., amended transcript) — skip the duplicate
            continue
        c.execute(
            "INSERT INTO sitting_days (date, slug, url, segment_count) VALUES (?, ?, ?, ?)",
            (t["date"], t["slug"], t["url"], t["segment_count"])
        )
        sitting_map[t["date"]] = c.lastrowid

    print(f"  Inserted {len(sitting_map)} sitting days")

    # --- Insert members + speeches ---
    member_map = {}  # slug -> id

    # Determine if honourable from name prefix
    def is_hon(name):
        return 1 if name.startswith("HON.") or name.startswith("THE ") else 0

    def clean_name(name):
        """Normalize display name."""
        # Remove HON. prefix for storage, keep it as flag
        n = name.strip()
        n = re.sub(r'^HON\.\s*', '', n)
        # Title case
        parts = n.split()
        titled = []
        for p in parts:
            if p in ("OF", "THE", "AND", "DE", "LA"):
                titled.append(p.lower())
            elif len(p) <= 3 and p == p.upper():
                titled.append(p)  # Keep acronyms
            else:
                titled.append(p.capitalize())
        return " ".join(titled)

    speech_count = 0

    for t in transcripts:
        if t["date"] not in sitting_map:
            continue
        sitting_id = sitting_map[t["date"]]

        for seg in t["segments"]:
            slug = seg.get("speaker_slug", "")
            if not slug:
                continue

            # Insert or get member
            if slug not in member_map:
                name = clean_name(seg["speaker"])
                hon = is_hon(seg["speaker"])
                try:
                    c.execute(
                        "INSERT INTO members (name, slug, is_honourable) VALUES (?, ?, ?)",
                        (name, slug, hon)
                    )
                    member_map[slug] = c.lastrowid
                except sqlite3.IntegrityError:
                    c.execute("SELECT id FROM members WHERE slug = ?", (slug,))
                    member_map[slug] = c.fetchone()[0]

            member_id = member_map[slug]
            text = seg.get("text", "").strip()
            if not text:
                continue

            word_count = len(text.split())
            c.execute(
                "INSERT INTO speeches (member_id, sitting_id, section, timestamp, text, word_count) VALUES (?, ?, ?, ?, ?, ?)",
                (member_id, sitting_id, seg.get("section"), seg.get("timestamp"), text, word_count)
            )
            speech_count += 1

    print(f"  Inserted {len(member_map)} members")
    print(f"  Inserted {speech_count} speeches")

    # --- Build FTS5 index ---
    c.executescript("""
        CREATE VIRTUAL TABLE speeches_fts USING fts5(
            text,
            content='speeches',
            content_rowid='id'
        );
        INSERT INTO speeches_fts(rowid, text) SELECT id, text FROM speeches;
    """)
    print("  Built FTS5 search index")

    # --- Compute member stats view ---
    c.executescript("""
        CREATE VIEW member_stats AS
        SELECT
            m.id,
            m.name,
            m.slug,
            m.party,
            m.constituency,
            m.is_honourable,
            COUNT(s.id) as speech_count,
            COALESCE(SUM(s.word_count), 0) as total_words,
            COUNT(DISTINCT s.sitting_id) as days_active,
            ROUND(COALESCE(AVG(s.word_count), 0), 0) as avg_words_per_speech
        FROM members m
        LEFT JOIN speeches s ON s.member_id = m.id
        GROUP BY m.id
        ORDER BY total_words DESC;
    """)
    print("  Created member_stats view")

    conn.commit()

    # --- Summary ---
    c.execute("SELECT COUNT(*) FROM speeches")
    total = c.fetchone()[0]
    c.execute("SELECT SUM(word_count) FROM speeches")
    words = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM members")
    members = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM sitting_days")
    days = c.fetchone()[0]

    conn.close()

    print(f"\n  Database: {DB_PATH}")
    print(f"  Size: {os.path.getsize(DB_PATH) / 1024 / 1024:.1f} MB")
    print(f"  {days} sitting days, {members} members, {total} speeches, {words:,} words")


if __name__ == "__main__":
    build()
