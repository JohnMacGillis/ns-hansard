"""
NS Hansard Search — FastAPI server.
MLA accountability tool for the 65th Assembly of Nova Scotia.
"""

import sqlite3
import os
import json
import struct
import re
import urllib.request
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import voyageai
    HAS_VOYAGE = bool(os.environ.get("VOYAGE_API_KEY"))
except ImportError:
    HAS_VOYAGE = False

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DB_PATH = os.path.join(DATA_DIR, "hansard.db")
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
RAW_HTML_DIR = os.path.join(DATA_DIR, "raw_html")
PORT = int(os.environ.get("PORT", os.environ.get("Port", 8080)))

# Global embedding matrix (loaded at startup if available)
EMBEDDING_MATRIX = None
EMBEDDING_IDS = None
EMBEDDING_DIM = 1024
VO_CLIENT = None


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def dict_rows(cursor):
    return [dict(row) for row in cursor.fetchall()]


def grade_mla(member, total_days, max_words, max_speeches):
    """
    Grade an MLA based on their legislative record.
    Factors: participation rate, volume, substance (avg speech length),
    and breadth (days active relative to total).
    """
    # Participation: what % of sitting days did they show up?
    participation = member["days_active"] / total_days if total_days else 0

    # Volume: words spoken relative to the most active MLA
    volume = member["total_words"] / max_words if max_words else 0

    # Substance: average words per speech (longer = more substantive, up to a point)
    avg = member["avg_words_per_speech"] or 0
    # 200+ words per speech is solid, <50 is mostly interjections
    substance = min(avg / 250, 1.0)

    # Composite score (0-100)
    score = (participation * 35) + (volume * 30) + (substance * 35)
    score = round(score * 100)

    if score >= 70:
        letter = "A"
        label = "Active contributor"
    elif score >= 50:
        letter = "B"
        label = "Regular participant"
    elif score >= 35:
        letter = "C"
        label = "Below average"
    elif score >= 20:
        letter = "D"
        label = "Minimal contribution"
    else:
        letter = "F"
        label = "Not showing up"

    return {
        "letter": letter,
        "score": score,
        "label": label,
        "participation_pct": round(participation * 100),
        "volume_pct": round(volume * 100),
        "substance_pct": round(substance * 100)
    }


def load_embeddings():
    """Preload all speech embeddings into a numpy matrix for fast search."""
    global EMBEDDING_MATRIX, EMBEDDING_IDS, VO_CLIENT
    if not HAS_NUMPY:
        print("  NumPy not available — semantic search disabled")
        return

    conn = get_db()
    c = conn.cursor()
    try:
        c.execute("SELECT speech_id, embedding FROM speech_embeddings ORDER BY speech_id")
        rows = c.fetchall()
    except:
        print("  No embeddings table — semantic search disabled")
        conn.close()
        return

    if not rows:
        print("  No embeddings found — semantic search disabled")
        conn.close()
        return

    ids = []
    vectors = []
    for row in rows:
        ids.append(row["speech_id"])
        n = len(row["embedding"]) // 4
        vec = np.array(struct.unpack(f'{n}f', row["embedding"]), dtype=np.float32)
        vectors.append(vec)

    EMBEDDING_MATRIX = np.array(vectors)
    # Normalize for cosine similarity
    norms = np.linalg.norm(EMBEDDING_MATRIX, axis=1, keepdims=True)
    norms[norms == 0] = 1
    EMBEDDING_MATRIX = EMBEDDING_MATRIX / norms
    EMBEDDING_IDS = ids

    if HAS_VOYAGE:
        VO_CLIENT = voyageai.Client()

    print(f"  Loaded {len(ids)} embeddings ({EMBEDDING_MATRIX.shape})")
    conn.close()


def semantic_search(query, top_k=50, threshold=0.3):
    """Search speeches by semantic similarity."""
    if EMBEDDING_MATRIX is None or VO_CLIENT is None:
        return []

    # Embed the query
    result = VO_CLIENT.embed([query], model="voyage-3", input_type="query")
    query_vec = np.array(result.embeddings[0], dtype=np.float32)
    query_vec = query_vec / np.linalg.norm(query_vec)

    # Cosine similarity via matrix multiplication
    similarities = EMBEDDING_MATRIX @ query_vec

    # Get top results above threshold
    top_indices = np.argsort(similarities)[::-1][:top_k]
    results = []
    for idx in top_indices:
        sim = float(similarities[idx])
        if sim < threshold:
            break
        results.append((EMBEDDING_IDS[idx], sim))

    return results


class HansardHandler(SimpleHTTPRequestHandler):

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)

        # API routes
        if path == "/api/lookup":
            return self.handle_postal_lookup(params)
        elif path == "/api/members":
            return self.handle_members(params)
        elif path.startswith("/api/members/"):
            slug = path.split("/api/members/")[1].strip("/")
            return self.handle_member_profile(slug, params)
        elif path == "/api/search":
            return self.handle_search(params)
        elif path.startswith("/api/sitting/"):
            date = path.split("/api/sitting/")[1].strip("/")
            return self.handle_sitting(date)
        elif path.startswith("/api/transcript/"):
            slug = path.split("/api/transcript/")[1].strip("/")
            return self.handle_raw_transcript(slug)
        elif path == "/api/stats":
            return self.handle_stats()

        # Serve static files
        if path == "/" or path == "":
            path = "/index.html"

        # Try static directory
        file_path = os.path.join(STATIC_DIR, path.lstrip("/"))
        if os.path.isfile(file_path):
            self.path = path
            self.directory = STATIC_DIR
            return SimpleHTTPRequestHandler.do_GET(self)

        # 404
        self.send_json({"error": "Not found"}, 404)

    def send_json(self, data, code=200):
        body = json.dumps(data, default=str).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def handle_postal_lookup(self, params):
        """Look up MLA by postal code using Represent API."""
        pc = params.get("pc", [""])[0].strip().replace(" ", "").upper()
        if not pc or len(pc) < 3:
            return self.send_json({"error": "Enter a valid postal code"}, 400)

        try:
            url = f"https://represent.opennorth.ca/postcodes/{pc}/?sets=nova-scotia-electoral-districts-2019"
            req = urllib.request.Request(url, headers={"User-Agent": "NSHansard/1.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            boundaries = data.get("boundaries_centroid", []) or data.get("boundaries_concordance", [])
            if not boundaries:
                return self.send_json({"error": "No constituency found for this postal code"}, 404)

            district_name = boundaries[0].get("name", "")

            # Try to match to our members
            conn = get_db()
            c = conn.cursor()
            c.execute("SELECT * FROM member_stats WHERE constituency LIKE ?", (f"%{district_name}%",))
            member = c.fetchone()

            # If no match by constituency, try to find by representatives data
            if not member:
                # Get reps from the API too
                reps = data.get("representatives_centroid", [])
                for rep in reps:
                    if "nova-scotia" in rep.get("elected_office", "").lower() or "MLA" in rep.get("elected_office", ""):
                        rep_name = rep.get("name", "")
                        # Try matching by name
                        c.execute("SELECT * FROM member_stats WHERE name LIKE ?", (f"%{rep_name.split()[-1]}%",))
                        member = c.fetchone()
                        if member:
                            break

            conn.close()

            result = {
                "postal_code": pc,
                "constituency": district_name,
                "member": dict(member) if member else None
            }
            self.send_json(result)

        except Exception as e:
            self.send_json({"error": f"Lookup failed: {str(e)}"}, 500)

    def handle_stats(self):
        conn = get_db()
        c = conn.cursor()
        c.execute("SELECT COUNT(*) as n FROM members")
        members = c.fetchone()["n"]
        c.execute("SELECT COUNT(*) as n FROM sitting_days")
        days = c.fetchone()["n"]
        c.execute("SELECT COUNT(*) as n FROM speeches")
        speeches = c.fetchone()["n"]
        c.execute("SELECT SUM(word_count) as n FROM speeches")
        words = c.fetchone()["n"]
        conn.close()
        self.send_json({
            "members": members,
            "sitting_days": days,
            "speeches": speeches,
            "total_words": words
        })

    def handle_members(self, params):
        conn = get_db()
        c = conn.cursor()
        sort = params.get("sort", ["total_words"])[0]
        valid_sorts = {"total_words", "speech_count", "days_active", "name", "avg_words_per_speech", "grade"}
        if sort not in valid_sorts:
            sort = "total_words"

        c.execute("SELECT * FROM member_stats WHERE slug != 'speaker'")
        members = dict_rows(c)

        # Get total sitting days for participation rate
        c.execute("SELECT COUNT(*) as n FROM sitting_days")
        total_days = c.fetchone()["n"]

        conn.close()

        # Grade each member
        if members:
            max_words = max(m["total_words"] for m in members)
            max_speeches = max(m["speech_count"] for m in members)

            for m in members:
                m["grade"] = grade_mla(m, total_days, max_words, max_speeches)
                m["participation_rate"] = round(m["days_active"] / total_days * 100) if total_days else 0

        if sort == "grade":
            grade_order = {"A": 0, "B": 1, "C": 2, "D": 3, "F": 4}
            members.sort(key=lambda m: (grade_order.get(m["grade"]["letter"], 5), -m["total_words"]))
        else:
            order = "ASC" if sort == "name" else "DESC"
            reverse = sort != "name"
            members.sort(key=lambda m: m.get(sort, 0) if sort != "name" else m.get("name", ""), reverse=reverse)

        self.send_json(members)

    def handle_member_profile(self, slug, params):
        conn = get_db()
        c = conn.cursor()

        # Member info
        c.execute("SELECT * FROM member_stats WHERE slug = ?", (slug,))
        member = c.fetchone()
        if not member:
            conn.close()
            return self.send_json({"error": "Member not found"}, 404)
        member = dict(member)

        # Compute grade
        c.execute("SELECT COUNT(*) as n FROM sitting_days")
        total_days = c.fetchone()["n"]
        c.execute("SELECT MAX(total_words) as mw, MAX(speech_count) as ms FROM member_stats WHERE slug != 'speaker'")
        maxes = c.fetchone()
        member["grade"] = grade_mla(member, total_days, maxes["mw"] or 1, maxes["ms"] or 1)
        member["participation_rate"] = round(member["days_active"] / total_days * 100) if total_days else 0

        # Topic breakdown
        c.execute("""
            SELECT topic, COUNT(*) as count, SUM(word_count) as words
            FROM speeches WHERE member_id = ? AND topic IS NOT NULL AND topic != 'Procedural'
            GROUP BY topic ORDER BY words DESC
        """, (member["id"],))
        topics = dict_rows(c)

        # Section breakdown
        c.execute("""
            SELECT section, COUNT(*) as count, SUM(word_count) as words
            FROM speeches WHERE member_id = ?
            GROUP BY section ORDER BY words DESC
        """, (member["id"],))
        sections = dict_rows(c)

        # Recent speeches (paginated)
        page = int(params.get("page", ["0"])[0])
        per_page = 50
        offset = page * per_page

        c.execute("""
            SELECT s.id, s.section, s.timestamp, s.text, s.word_count, s.topic, s.summary,
                   sd.date, sd.slug as sitting_slug
            FROM speeches s
            JOIN sitting_days sd ON s.sitting_id = sd.id
            WHERE s.member_id = ?
            ORDER BY sd.date DESC, s.id ASC
            LIMIT ? OFFSET ?
        """, (member["id"], per_page, offset))
        speeches = dict_rows(c)

        # Total speech count for pagination
        c.execute("SELECT COUNT(*) as n FROM speeches WHERE member_id = ?", (member["id"],))
        total = c.fetchone()["n"]

        conn.close()
        self.send_json({
            "member": member,
            "topics": topics,
            "sections": sections,
            "speeches": speeches,
            "total_speeches": total,
            "page": page,
            "per_page": per_page
        })

    def handle_search(self, params):
        q = params.get("q", [""])[0].strip()
        if not q:
            return self.send_json({"error": "Query required"}, 400)

        mode = params.get("mode", ["auto"])[0]
        conn = get_db()
        c = conn.cursor()

        results = []
        search_mode = "fts"

        # Try semantic search first if available
        if EMBEDDING_MATRIX is not None and VO_CLIENT is not None and mode != "fts":
            try:
                sem_results = semantic_search(q, top_k=50, threshold=0.3)
                if sem_results:
                    search_mode = "semantic"
                    ids = [r[0] for r in sem_results]
                    sims = {r[0]: r[1] for r in sem_results}
                    placeholders = ",".join("?" * len(ids))
                    c.execute(f"""
                        SELECT s.id, s.section, s.timestamp, s.text, s.word_count,
                               s.topic, s.summary,
                               m.name, m.slug as member_slug, m.is_honourable,
                               sd.date, sd.slug as sitting_slug
                        FROM speeches s
                        JOIN members m ON s.member_id = m.id
                        JOIN sitting_days sd ON s.sitting_id = sd.id
                        WHERE s.id IN ({placeholders})
                    """, ids)
                    rows = dict_rows(c)
                    # Add similarity scores and sort
                    for r in rows:
                        r["similarity"] = round(sims.get(r["id"], 0), 3)
                    results = sorted(rows, key=lambda r: r["similarity"], reverse=True)
            except Exception as e:
                print(f"  Semantic search error: {e}")

        # Fall back to FTS
        if not results:
            search_mode = "fts"
            try:
                c.execute("""
                    SELECT s.id, s.section, s.timestamp, s.text, s.word_count,
                           s.topic, s.summary,
                           m.name, m.slug as member_slug, m.is_honourable,
                           sd.date, sd.slug as sitting_slug,
                           rank
                    FROM speeches_fts fts
                    JOIN speeches s ON fts.rowid = s.id
                    JOIN members m ON s.member_id = m.id
                    JOIN sitting_days sd ON s.sitting_id = sd.id
                    WHERE speeches_fts MATCH ?
                    ORDER BY rank
                    LIMIT 50
                """, (q,))
                results = dict_rows(c)
            except Exception:
                c.execute("""
                    SELECT s.id, s.section, s.timestamp, s.text, s.word_count,
                           s.topic, s.summary,
                           m.name, m.slug as member_slug, m.is_honourable,
                           sd.date, sd.slug as sitting_slug
                    FROM speeches s
                    JOIN members m ON s.member_id = m.id
                    JOIN sitting_days sd ON s.sitting_id = sd.id
                    WHERE s.text LIKE ?
                    ORDER BY sd.date DESC
                    LIMIT 50
                """, (f"%{q}%",))
                results = dict_rows(c)

        conn.close()
        self.send_json({"query": q, "count": len(results), "results": results, "mode": search_mode})

    def handle_sitting(self, date):
        conn = get_db()
        c = conn.cursor()

        c.execute("SELECT * FROM sitting_days WHERE date = ?", (date,))
        sitting = c.fetchone()
        if not sitting:
            conn.close()
            return self.send_json({"error": "Sitting day not found"}, 404)
        sitting = dict(sitting)

        c.execute("""
            SELECT s.id, s.section, s.timestamp, s.text, s.word_count,
                   m.name, m.slug as member_slug, m.is_honourable
            FROM speeches s
            JOIN members m ON s.member_id = m.id
            WHERE s.sitting_id = ?
            ORDER BY s.id ASC
        """, (sitting["id"],))
        speeches = dict_rows(c)

        conn.close()
        self.send_json({
            "sitting": sitting,
            "speeches": speeches,
            "count": len(speeches)
        })

    def handle_raw_transcript(self, slug):
        """Serve the original Hansard HTML for full transcript view."""
        html_file = os.path.join(RAW_HTML_DIR, f"{slug}.html")
        if not os.path.exists(html_file):
            return self.send_json({"error": "Transcript not found"}, 404)

        with open(html_file, "r") as f:
            html = f.read()

        # Extract just the main content area
        # Send as JSON with HTML content
        self.send_json({"slug": slug, "html": html})

    def log_message(self, format, *args):
        if "/api/" in str(args[0]) if args else False:
            super().log_message(format, *args)


import re  # needed for search

if __name__ == "__main__":
    print(f"NS Hansard Search starting on port {PORT}")
    print(f"Database: {DB_PATH}")
    print(f"Static files: {STATIC_DIR}")

    # Quick stats
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM members")
    print(f"  Members: {c.fetchone()[0]}")
    c.execute("SELECT COUNT(*) FROM speeches")
    print(f"  Speeches: {c.fetchone()[0]}")
    c.execute("SELECT COUNT(*) FROM sitting_days")
    print(f"  Sitting days: {c.fetchone()[0]}")
    conn.close()

    # Load embeddings for semantic search
    load_embeddings()

    server = HTTPServer(("0.0.0.0", PORT), HansardHandler)
    print(f"\nListening at http://localhost:{PORT}")
    server.serve_forever()
