"""
Score Hansard speeches on quality using Claude.
Rates substance, responsiveness, and flags absurd moments.
"""

import sqlite3
import os
import json
import time

import anthropic

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DB_PATH = os.path.join(DATA_DIR, "hansard.db")
PROGRESS_FILE = os.path.join(DATA_DIR, "score_progress.json")

SCORE_PROMPT = """You are analyzing speeches from the Nova Scotia Legislature (Hansard).

For each speech, score it on quality. Return a JSON array with one object per speech:

{
  "substance": 0-10,
  "responsiveness": 0-10,
  "absurd": true/false,
  "absurd_reason": "brief explanation if absurd, null otherwise",
  "highlight_type": "best" | "worst" | null,
  "highlight_quote": "the most notable 1-2 sentence excerpt, or null"
}

Scoring guide:

**substance** (0-10):
- 0-2: Empty filler, "I agree with my colleague", heckling, repetitive talking points with no specifics
- 3-4: Generic position statement with no data, examples, or policy detail
- 5-6: Makes a clear point with some specifics
- 7-8: Substantive argument with data, examples, or detailed policy reasoning
- 9-10: Exceptional — original analysis, compelling evidence, or a genuinely important point

**responsiveness** (0-10):
- Only score this for speeches in ORAL QUESTIONS sections (Question Period)
- 0-2: Complete dodge, talks about something else entirely, reads a prepared non-answer
- 3-4: Acknowledges the topic but doesn't answer the actual question
- 5-6: Partially answers
- 7-8: Directly answers with specifics
- 9-10: Answers fully and adds useful context
- For non-Question-Period speeches, set to -1

**absurd** — flag as true ONLY when the speaker:
- Says something factually incorrect that they should know better
- Contradicts something they said earlier in the same session
- Makes a claim so disconnected from reality it's notable
- Uses an analogy or comparison that is genuinely bizarre
- Do NOT flag normal partisan disagreement, rhetorical exaggeration, or opinions you disagree with

**highlight_type** — "best" if this is an unusually good speech worth featuring, "worst" if notably bad, null for average

**highlight_quote** — if highlight_type is set, extract the most notable 1-2 sentences

Be tough but fair. Most speeches should score 3-6 on substance. Reserve 8+ for genuinely good speeches. Reserve absurd flags for things that would make a reasonable person do a double-take.

Speeches to score:
"""

client = anthropic.Anthropic()


def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"scored_ids": []}


def save_progress(progress):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f)


def score_batch(speeches):
    items = []
    for i, s in enumerate(speeches):
        text = s["text"][:800]
        section = s["section"] or "Unknown"
        items.append(f"[{i}] Speaker: {s['speaker_name']}\nSection: {section}\nSpeech: {text}")

    prompt = SCORE_PROMPT + "\n\n" + "\n\n".join(items)

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}]
        )
        text = response.content[0].text
        start = text.find("[")
        end = text.rfind("]") + 1
        if start >= 0 and end > start:
            return json.loads(text[start:end])
    except Exception as e:
        print(f"    Error: {e}")
    return None


def run():
    print("=" * 60)
    print("Scoring Hansard speeches with Claude")
    print("=" * 60)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    # Add columns
    for col in ["substance_score", "responsiveness_score", "is_absurd", "absurd_reason",
                "highlight_type", "highlight_quote"]:
        try:
            col_type = "INTEGER" if col in ("substance_score", "responsiveness_score", "is_absurd") else "TEXT"
            c.execute(f"ALTER TABLE speeches ADD COLUMN {col} {col_type}")
        except:
            pass
    conn.commit()

    progress = load_progress()
    scored = set(progress["scored_ids"])
    print(f"  Already scored: {len(scored)}")

    # Get speeches worth scoring (50+ words, real speeches only)
    c.execute("""
        SELECT s.id, s.section, s.text, s.word_count, m.name as speaker_name
        FROM speeches s
        JOIN members m ON s.member_id = m.id
        WHERE s.word_count >= 50
        AND s.speech_type = 'speech'
        ORDER BY s.id
    """)
    all_speeches = [dict(row) for row in c.fetchall()]
    to_score = [s for s in all_speeches if s["id"] not in scored]

    print(f"  Speeches to score: {len(to_score)} (of {len(all_speeches)} eligible)")

    if not to_score:
        print("  Nothing to score!")
        conn.close()
        return

    BATCH_SIZE = 10  # Smaller batches for quality scoring
    total_batches = (len(to_score) + BATCH_SIZE - 1) // BATCH_SIZE
    scored_count = 0
    absurd_count = 0
    highlight_count = 0

    for i in range(0, len(to_score), BATCH_SIZE):
        batch = to_score[i:i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1

        print(f"\n  Batch {batch_num}/{total_batches} ({len(batch)} speeches)...", end=" ", flush=True)

        results = score_batch(batch)

        if results and len(results) == len(batch):
            for s, r in zip(batch, results):
                sub = r.get("substance", 5)
                resp = r.get("responsiveness", -1)
                absurd = 1 if r.get("absurd", False) else 0
                absurd_reason = r.get("absurd_reason")
                hl_type = r.get("highlight_type")
                hl_quote = r.get("highlight_quote")

                c.execute("""UPDATE speeches SET
                    substance_score = ?, responsiveness_score = ?,
                    is_absurd = ?, absurd_reason = ?,
                    highlight_type = ?, highlight_quote = ?
                    WHERE id = ?""",
                    (sub, resp, absurd, absurd_reason, hl_type, hl_quote, s["id"]))
                scored.add(s["id"])
                scored_count += 1
                if absurd:
                    absurd_count += 1
                if hl_type:
                    highlight_count += 1

            conn.commit()
            progress["scored_ids"] = list(scored)
            save_progress(progress)
            print(f"OK ({scored_count} total, {absurd_count} absurd, {highlight_count} highlights)")
        else:
            print(f"MISMATCH — skipping")

        time.sleep(0.5)

    conn.close()

    print(f"\n{'=' * 60}")
    print(f"DONE")
    print(f"  Scored: {scored_count}")
    print(f"  Absurd moments: {absurd_count}")
    print(f"  Highlights: {highlight_count}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    run()
