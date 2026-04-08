"""
Embed Hansard speeches using Voyage AI for semantic search.
Stores embeddings as numpy arrays in SQLite for fast cosine similarity.
"""

import sqlite3
import os
import json
import time
import struct

try:
    import numpy as np
except ImportError:
    os.system("pip3 install numpy")
    import numpy as np

try:
    import voyageai
except ImportError:
    os.system("pip3 install voyageai")
    import voyageai

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DB_PATH = os.path.join(DATA_DIR, "hansard.db")
PROGRESS_FILE = os.path.join(DATA_DIR, "embed_progress.json")

# Voyage AI model — voyage-3 for general text (voyage-law-2 is legal-specific but this is parliamentary)
MODEL = "voyage-3"
BATCH_SIZE = 32  # Voyage supports up to 128, but keep it reasonable
EMBEDDING_DIM = 1024


def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"embedded_ids": []}


def save_progress(progress):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f)


def embedding_to_blob(embedding):
    """Convert float list to bytes for SQLite storage."""
    return struct.pack(f'{len(embedding)}f', *embedding)


def blob_to_embedding(blob):
    """Convert bytes back to numpy array."""
    n = len(blob) // 4
    return np.array(struct.unpack(f'{n}f', blob), dtype=np.float32)


def run():
    print("=" * 60)
    print("Embedding Hansard speeches with Voyage AI")
    print("=" * 60)

    vo = voyageai.Client()

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    # Create embeddings table
    c.executescript("""
        CREATE TABLE IF NOT EXISTS speech_embeddings (
            speech_id INTEGER PRIMARY KEY,
            embedding BLOB NOT NULL,
            tokens_used INTEGER DEFAULT 0,
            FOREIGN KEY (speech_id) REFERENCES speeches(id)
        );
    """)
    conn.commit()

    # Load progress
    progress = load_progress()
    embedded = set(progress["embedded_ids"])
    print(f"  Already embedded: {len(embedded)}")

    # Get speeches to embed (skip very short ones < 10 words)
    c.execute("""
        SELECT id, text, section, word_count FROM speeches
        WHERE word_count >= 10
        ORDER BY id
    """)
    all_speeches = [dict(row) for row in c.fetchall()]

    # Filter already embedded
    to_embed = [s for s in all_speeches if s["id"] not in embedded]
    print(f"  Speeches to embed: {len(to_embed)} (of {len(all_speeches)} eligible)")

    if not to_embed:
        print("  Nothing to embed!")
        conn.close()
        return

    total_batches = (len(to_embed) + BATCH_SIZE - 1) // BATCH_SIZE
    embedded_count = 0
    total_tokens = 0

    for i in range(0, len(to_embed), BATCH_SIZE):
        batch = to_embed[i:i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1

        # Prepare texts — truncate to ~1000 chars to manage token costs
        texts = []
        for s in batch:
            section = s["section"] or ""
            text = s["text"][:1000]
            # Prepend section for context
            if section:
                texts.append(f"{section}: {text}")
            else:
                texts.append(text)

        print(f"  Batch {batch_num}/{total_batches} ({len(batch)} speeches)...", end=" ", flush=True)

        try:
            result = vo.embed(texts, model=MODEL, input_type="document")
            embeddings = result.embeddings
            tokens = result.total_tokens

            for s, emb in zip(batch, embeddings):
                blob = embedding_to_blob(emb)
                c.execute(
                    "INSERT OR REPLACE INTO speech_embeddings (speech_id, embedding, tokens_used) VALUES (?, ?, ?)",
                    (s["id"], blob, tokens // len(batch))
                )
                embedded.add(s["id"])
                embedded_count += 1

            total_tokens += tokens
            conn.commit()
            progress["embedded_ids"] = list(embedded)
            save_progress(progress)

            print(f"OK ({tokens} tokens, {embedded_count} total)")

        except Exception as e:
            print(f"ERROR: {e}")
            time.sleep(5)  # Back off on errors

        # Rate limit
        time.sleep(0.3)

    conn.close()

    print(f"\n{'=' * 60}")
    print(f"DONE")
    print(f"  Embedded: {embedded_count} speeches")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    run()
