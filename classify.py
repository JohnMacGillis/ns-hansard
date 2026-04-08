"""
Classify Hansard speeches by topic using Claude.
Groups speeches into batches for efficiency, assigns topic tags and one-line summaries.
"""

import sqlite3
import os
import json
import time

try:
    import anthropic
except ImportError:
    print("Installing anthropic...")
    os.system("pip3 install anthropic")
    import anthropic

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DB_PATH = os.path.join(DATA_DIR, "hansard.db")
PROGRESS_FILE = os.path.join(DATA_DIR, "classify_progress.json")

TOPICS = [
    "Healthcare", "Education", "Housing", "Budget & Finance", "Justice & Public Safety",
    "Environment & Climate", "Economy & Jobs", "Infrastructure & Transportation",
    "Social Services", "Indigenous Affairs", "Municipal Affairs",
    "Agriculture & Natural Resources", "Technology & Innovation",
    "Labour & Workers", "Seniors", "Children & Youth", "Immigration",
    "Arts & Culture", "Procedural", "Tributes & Recognition", "Other"
]

CLASSIFY_PROMPT = """You are classifying speeches from the Nova Scotia Legislature (Hansard).

For each speech, provide:
1. **topic**: The single most relevant topic from this list: """ + ", ".join(TOPICS) + """
2. **summary**: A one-line summary (max 15 words) of what the speaker said.

Rules:
- "Tributes & Recognition" = member statements recognizing people, events, milestones
- "Procedural" = points of order, motions to adjourn, procedural exchanges
- Short interjections like "Hear, hear!" or one-word responses = "Procedural" with summary "Interjection"
- Pick the most specific topic that fits

Respond with a JSON array matching the input order. Example:
[{"topic": "Healthcare", "summary": "Calls for more ER doctors in rural hospitals"}, ...]

Speeches to classify:
"""

client = anthropic.Anthropic()


def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"classified_ids": []}


def save_progress(progress):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f)


def classify_batch(speeches):
    """Classify a batch of speeches using Claude."""
    # Format speeches for the prompt
    items = []
    for i, s in enumerate(speeches):
        text = s["text"][:500]  # Truncate to save tokens
        section = s["section"] or "Unknown"
        items.append(f"[{i}] Section: {section}\nSpeech: {text}")

    prompt = CLASSIFY_PROMPT + "\n\n" + "\n\n".join(items)

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}]
        )
        text = response.content[0].text

        # Extract JSON from response
        # Find the JSON array
        start = text.find("[")
        end = text.rfind("]") + 1
        if start >= 0 and end > start:
            results = json.loads(text[start:end])
            return results
    except Exception as e:
        print(f"    Error: {e}")

    return None


def run():
    print("=" * 60)
    print("Classifying Hansard speeches with Claude")
    print("=" * 60)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    # Add columns if not exist
    try:
        c.execute("ALTER TABLE speeches ADD COLUMN topic TEXT")
    except:
        pass
    try:
        c.execute("ALTER TABLE speeches ADD COLUMN summary TEXT")
    except:
        pass
    conn.commit()

    # Load progress
    progress = load_progress()
    classified = set(progress["classified_ids"])
    print(f"  Already classified: {len(classified)}")

    # Get unclassified speeches
    c.execute("""
        SELECT id, section, text, word_count FROM speeches
        WHERE id NOT IN ({})
        ORDER BY id
    """.format(",".join(str(i) for i in classified) if classified else "0"))
    all_speeches = [dict(row) for row in c.fetchall()]

    # Filter out very short interjections — classify them directly
    short_speeches = [s for s in all_speeches if s["word_count"] <= 5]
    regular_speeches = [s for s in all_speeches if s["word_count"] > 5]

    print(f"  Short interjections (auto-tagged): {len(short_speeches)}")
    print(f"  Regular speeches to classify: {len(regular_speeches)}")

    # Auto-tag short interjections
    for s in short_speeches:
        c.execute("UPDATE speeches SET topic = 'Procedural', summary = 'Interjection' WHERE id = ?", (s["id"],))
        classified.add(s["id"])

    conn.commit()
    progress["classified_ids"] = list(classified)
    save_progress(progress)
    print(f"  Auto-tagged {len(short_speeches)} interjections")

    # Batch classify regular speeches
    BATCH_SIZE = 20
    total_batches = (len(regular_speeches) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"\n  Processing {len(regular_speeches)} speeches in {total_batches} batches...")

    classified_count = 0
    errors = 0

    for i in range(0, len(regular_speeches), BATCH_SIZE):
        batch = regular_speeches[i:i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1

        # Skip already classified
        batch = [s for s in batch if s["id"] not in classified]
        if not batch:
            continue

        print(f"\n  Batch {batch_num}/{total_batches} ({len(batch)} speeches)...", end=" ", flush=True)

        results = classify_batch(batch)

        if results and len(results) == len(batch):
            for s, r in zip(batch, results):
                topic = r.get("topic", "Other")
                summary = r.get("summary", "")
                # Validate topic
                if topic not in TOPICS:
                    topic = "Other"
                c.execute("UPDATE speeches SET topic = ?, summary = ? WHERE id = ?",
                          (topic, summary, s["id"]))
                classified.add(s["id"])
                classified_count += 1

            conn.commit()
            progress["classified_ids"] = list(classified)
            save_progress(progress)
            print(f"OK ({classified_count} total)")
        else:
            errors += 1
            expected = len(batch)
            got = len(results) if results else 0
            print(f"MISMATCH (expected {expected}, got {got})")
            # Try one-by-one for this batch
            for s in batch:
                r = classify_batch([s])
                if r and len(r) == 1:
                    topic = r[0].get("topic", "Other")
                    summary = r[0].get("summary", "")
                    if topic not in TOPICS:
                        topic = "Other"
                    c.execute("UPDATE speeches SET topic = ?, summary = ? WHERE id = ?",
                              (topic, summary, s["id"]))
                    classified.add(s["id"])
                    classified_count += 1
            conn.commit()
            progress["classified_ids"] = list(classified)
            save_progress(progress)

        # Rate limit: ~1 request/sec
        time.sleep(0.5)

    conn.close()

    print(f"\n{'=' * 60}")
    print(f"DONE")
    print(f"  Classified: {classified_count}")
    print(f"  Errors: {errors}")
    print(f"  Total tagged: {len(classified)}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    run()
