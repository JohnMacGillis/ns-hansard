"""
Incremental update: fetch sittings published since the last run and add them
to the existing database.

Unlike build_db.py — which drops the database and rebuilds it from scratch —
this only inserts sitting days that aren't in the database yet. Classification,
quality scores and embeddings on existing speeches are preserved, because those
cost real API spend to produce and re-deriving them is pure waste.

New speeches land with topic / summary / score columns NULL, so the existing
classify.py, score.py and embed.py pick them up on their next run without any
special-casing.

    python3 update.py                 # fetch and insert anything new
    python3 update.py --dry-run       # report what would change, touch nothing
    python3 update.py --session assembly-65-session-2

A sitting page that exists but parses to zero speeches (the House has scheduled
the day but not published the transcript) is reported and skipped, not inserted.
"""

import argparse
import os
import sqlite3
import sys

import scraper
from build_db import DB_PATH, clean_name, is_hon


def ensure_fts_row(c, speech_id, text):
    """Keep the external-content FTS5 index in sync.

    speeches_fts is content='speeches' with no triggers, so every insert into
    speeches needs a matching insert here or the row is invisible to search.
    """
    c.execute(
        "INSERT INTO speeches_fts(rowid, text) VALUES (?, ?)",
        (speech_id, text),
    )


def existing_state(conn):
    """Dates already stored, and the member slug -> id map."""
    c = conn.cursor()
    dates = {row[0] for row in c.execute("SELECT date FROM sitting_days")}
    members = {row[0]: row[1] for row in c.execute("SELECT slug, id FROM members")}
    return dates, members


def insert_sitting(c, transcript, member_map):
    """Insert one sitting day plus its speeches. Returns (speeches, new_members)."""
    c.execute(
        "INSERT INTO sitting_days (date, slug, url, segment_count) VALUES (?, ?, ?, ?)",
        (
            transcript["date"],
            transcript["slug"],
            transcript["url"],
            transcript["segment_count"],
        ),
    )
    sitting_id = c.lastrowid

    speeches = 0
    new_members = []

    for seg in transcript["segments"]:
        slug = seg.get("speaker_slug", "")
        if not slug:
            continue

        text = seg.get("text", "").strip()
        if not text:
            continue

        if slug not in member_map:
            try:
                c.execute(
                    "INSERT INTO members (name, slug, is_honourable) VALUES (?, ?, ?)",
                    (clean_name(seg["speaker"]), slug, is_hon(seg["speaker"])),
                )
                member_map[slug] = c.lastrowid
                new_members.append(clean_name(seg["speaker"]))
            except sqlite3.IntegrityError:
                c.execute("SELECT id FROM members WHERE slug = ?", (slug,))
                member_map[slug] = c.fetchone()[0]

        c.execute(
            "INSERT INTO speeches (member_id, sitting_id, section, timestamp, text, word_count) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                member_map[slug],
                sitting_id,
                seg.get("section"),
                seg.get("timestamp"),
                text,
                len(text.split()),
            ),
        )
        ensure_fts_row(c, c.lastrowid, text)
        speeches += 1

    return speeches, new_members


def pending_counts(conn):
    """How much of the corpus still needs each enrichment pass."""
    c = conn.cursor()
    columns = {row[1] for row in c.execute("PRAGMA table_info(speeches)")}
    total = c.execute("SELECT COUNT(*) FROM speeches").fetchone()[0]

    def missing(col):
        if col not in columns:
            return total
        return c.execute(
            f"SELECT COUNT(*) FROM speeches WHERE {col} IS NULL"
        ).fetchone()[0]

    has_embed_table = c.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='speech_embeddings'"
    ).fetchone()[0]
    if has_embed_table:
        unembedded = c.execute(
            "SELECT COUNT(*) FROM speeches WHERE word_count >= 10 AND id NOT IN "
            "(SELECT speech_id FROM speech_embeddings)"
        ).fetchone()[0]
    else:
        unembedded = c.execute(
            "SELECT COUNT(*) FROM speeches WHERE word_count >= 10"
        ).fetchone()[0]

    return {
        "total": total,
        "unclassified": missing("topic"),
        "unscored": missing("substance_score"),
        "unembedded": unembedded,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would change without writing")
    ap.add_argument("--session", default=None,
                    help="session slug, e.g. assembly-65-session-2")
    args = ap.parse_args()

    if not os.path.exists(DB_PATH):
        sys.exit(
            f"No database at {DB_PATH}.\n"
            "This is the incremental updater — it needs an existing database.\n"
            "For a first build run: python3 scraper.py && python3 build_db.py"
        )

    session = args.session or scraper.SESSION
    print("=" * 60)
    print(f"NS Hansard incremental update — {session}")
    print("=" * 60)

    conn = sqlite3.connect(DB_PATH)
    have_dates, member_map = existing_state(conn)
    print(f"\n  Database holds {len(have_dates)} sitting days, {len(member_map)} members")

    print("\n[1/3] Checking the legislature site for new sittings...")
    live = scraper.get_sitting_dates(session=session)
    new = [d for d in live if d["date"] not in have_dates]

    print(f"\n  Live sitting pages: {len(live)}")
    print(f"  Already stored:     {len(live) - len(new)}")
    print(f"  New:                {len(new)}")

    if not new:
        print("\n  Nothing to do — the database is current.")
        conn.close()
        return

    for d in new:
        print(f"     {d['date']}  {d['slug']}")

    if args.dry_run:
        print("\n  --dry-run: stopping before any fetch or write.")
        conn.close()
        return

    print(f"\n[2/3] Fetching and parsing {len(new)} transcript(s)...")
    parsed = []
    for i, d in enumerate(new, 1):
        print(f"\n  [{i}/{len(new)}] {d['date']} ({d['slug']})")
        raw_file = os.path.join(scraper.RAW_DIR, f"{d['slug']}.html")
        if os.path.exists(raw_file):
            print("    Using cached HTML")
            html = open(raw_file, encoding="utf8").read()
        else:
            html = scraper.fetch_url(d["url"])
            if not html:
                print("    SKIPPED (fetch failed)")
                continue
            with open(raw_file, "w", encoding="utf8") as f:
                f.write(html)

        segments = scraper.parse_transcript(html, d["date"])
        print(f"    Parsed {len(segments)} speech segments")
        if not segments:
            print("    SKIPPED (no transcript published yet)")
            continue

        parsed.append({
            "date": d["date"],
            "slug": d["slug"],
            "url": d["url"],
            "segment_count": len(segments),
            "segments": segments,
        })

    if not parsed:
        print("\n  No transcripts with content — nothing inserted.")
        conn.close()
        return

    print(f"\n[3/3] Inserting {len(parsed)} sitting(s)...")
    c = conn.cursor()
    total_speeches = 0
    all_new_members = []
    try:
        for t in parsed:
            speeches, new_members = insert_sitting(c, t, member_map)
            total_speeches += speeches
            all_new_members.extend(new_members)
            print(f"    {t['date']}: {speeches} speeches")
        conn.commit()
    except Exception:
        conn.rollback()
        conn.close()
        raise

    print(f"\n  Inserted {len(parsed)} sitting days, {total_speeches} speeches")
    if all_new_members:
        print(f"  New members: {', '.join(sorted(set(all_new_members)))}")

    pending = pending_counts(conn)
    conn.close()

    print(f"\n  Corpus now {pending['total']:,} speeches. Still to enrich:")
    print(f"    unclassified: {pending['unclassified']:,}  (python3 classify.py)")
    print(f"    unscored:     {pending['unscored']:,}  (python3 score.py)")
    print(f"    unembedded:   {pending['unembedded']:,}  (python3 embed.py)")


if __name__ == "__main__":
    main()
