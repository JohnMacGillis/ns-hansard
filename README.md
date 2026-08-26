---
title: NS Hansard Search
emoji: 🏛️
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
short_description: Searchable record of the Nova Scotia House of Assembly
---

# NS Hansard Search

A searchable, classified record of everything said in the Nova Scotia House of
Assembly — built from the legislature's own published Hansard transcripts.

Nova Scotia publishes Hansard as one HTML page per sitting day. That is a public
record in the sense that it exists, but not in the sense that anyone can use it:
there is no way to ask what a given MLA has said about housing, or which topics
a session actually spent its time on. This turns those pages into a database you
can query, and puts a search interface in front of it.

**Scope:** 65th General Assembly, Session 1 (December 2024 – present).

---

## Pipeline

Six stages, each independently re-runnable. Every stage is idempotent and
resumable — you can stop any of them at any point and re-run without redoing
completed work or paying twice for it.

| Stage | Script | What it does |
|---|---|---|
| Scrape | `scraper.py` | Walks the session's paginated listing, fetches each sitting page, parses it into structured speaker segments |
| Build | `build_db.py` | Builds SQLite from scratch — members, sitting days, speeches, FTS5 index, `member_stats` view |
| Update | `update.py` | **Incremental.** Fetches only sittings not yet stored and inserts them, preserving existing enrichment |
| Classify | `classify.py` | Assigns each speech a topic (21-item taxonomy) and a one-line summary |
| Score | `score.py` | Rates substance and responsiveness, flags notable and absurd moments |
| Embed | `embed.py` | Voyage AI embeddings stored as BLOBs for cosine-similarity semantic search |

`server.py` serves the search interface over the resulting database.

### Parsing

Hansard has no structured export — the speaker attribution is a profile link and
the section headings are bold text, so `scraper.py` reconstructs the structure
from the markup: speaker links become `member_id`, `**ALL CAPS**` lines become
sections, bracketed lines become timestamps or stage directions. Raw HTML is
cached under `data/raw_html/` so a re-parse never re-fetches.

### Design decisions

**Be a good guest.** The scraper identifies itself
(`NSHansardScraper/1.0 (civic research project)`), sleeps 1.5s between requests,
and caches every page it fetches so a re-run costs the legislature nothing.

**Never pay twice.** `classify.py`, `score.py` and `embed.py` each keep an
on-disk progress file of completed IDs. A crashed or cancelled run resumes where
it stopped. This matters: the corpus is large enough that a full pass is a real
API bill, and a pipeline that restarts from zero on failure is one you stop
running.

**Updates are incremental, not rebuilds.** `build_db.py` drops and rebuilds the
database, which discards every topic, score and embedding on it. `update.py`
exists so the routine case — a few new sitting days — doesn't destroy work that
cost money to produce. New speeches land with their enrichment columns `NULL`,
so the classify/score/embed passes pick them up with no special-casing.

**Model choice is a runtime decision.** The enrichment scripts default to
`claude-opus-5` and honour `NS_HANSARD_MODEL`, so a bulk backfill can be run on
a cheaper model without editing code:

```bash
NS_HANSARD_MODEL=claude-haiku-4-5 python3 classify.py
```

---

## Running it

```bash
pip install -r requirements-pipeline.txt
export ANTHROPIC_API_KEY=...    # classify.py, score.py
export VOYAGE_API_KEY=...       # embed.py
```

First build:

```bash
python3 scraper.py && python3 build_db.py
```

Routine update — check for new sittings, insert them, then enrich:

```bash
python3 update.py --dry-run    # what would change, no writes
python3 update.py
python3 classify.py && python3 score.py && python3 embed.py
```

`update.py` finishes by reporting how many speeches are still unclassified,
unscored and unembedded, so the enrichment backlog is always visible.

When the House opens a new session, point the scraper at it — nothing else
changes:

```bash
NS_HANSARD_SESSION=assembly-65-session-2 python3 update.py
```

Serving locally (`server.py` reads `PORT`, defaults to 8080):

```bash
python3 server.py
```

Production runs the same entry point via the `Procfile`. `requirements.txt` is
deliberately minimal — it holds only what the web process needs, which is why
the pipeline dependencies live in `requirements-pipeline.txt`.

---

## Schema

```
members          id, name, slug, party, constituency, is_honourable
sitting_days     id, date (UNIQUE), slug, url, segment_count
speeches         id, member_id, sitting_id, section, timestamp, text, word_count,
                 topic, summary,                        -- classify.py
                 substance_score, responsiveness_score,  -- score.py
                 is_absurd, absurd_reason, highlight_type, highlight_quote
speech_embeddings  speech_id, embedding (BLOB), tokens_used   -- embed.py
speeches_fts     FTS5, external content over speeches.text
member_stats     VIEW — per-member speech count, total words, days active
```

`speeches_fts` is an external-content FTS5 table with no triggers, so any code
inserting into `speeches` must insert the matching row itself. `update.py` does;
if you add another writer, it has to as well.

Sitting days are unique **by date**, not by slug — the legislature occasionally
publishes a corrected transcript under a suffixed slug (`house_26mar25-0`), and
storing both would double-count that day.

---

## Current state, and what to trust

The corpus is complete: every published sitting of the session is scraped and
searchable via full-text search.

The **enrichment passes are partial** — as of August 2026 roughly 11% of
speeches carry a topic, about half carry quality scores, and embeddings have not
been generated at all. Two consequences worth being explicit about:

- **Topic counts are not findings.** The topic distribution reflects which
  speeches happen to have been classified, not what the House discussed. Don't
  read "49 speeches on healthcare" as a fact about the legislature.
- **Semantic search falls back to keyword search** until `embed.py` has run.

Run `python3 update.py` to see current coverage.

### On the scores

`score.py` asks a language model to rate substance and responsiveness and to
flag absurd moments. These are model judgments, not measurements, and they were
manually reviewed — an early pass produced several hundred false-positive
"absurd" flags that were reviewed and removed by hand. Treat the scores as a
way to find interesting speeches, not as an objective ranking of MLAs. Speaking
frequently is also not the same as governing well: backbenchers and cabinet
ministers have structurally different speaking opportunities.

---

## Known issues

- Member display names are normalized with naive title-casing, so hyphenated
  and Mc/Mac names render wrong (`Smith-mccrossin` for Smith-McCrossin).
  Fixing it requires a one-off migration of the stored `members.name` values.
- `classify.py` and `score.py` parse the model's JSON out of prose with a
  bracket search. Structured outputs (`output_config.format`) would make that
  guaranteed rather than best-effort.
- Only one session at a time; there is no cross-session view.

---

## Data source

Transcripts are the property of the Nova Scotia House of Assembly and are
reproduced here from <https://nslegislature.ca> for research and public
accountability purposes.
