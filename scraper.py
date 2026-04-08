"""
Scraper for Nova Scotia Hansard transcripts (65th Assembly, Session 1).
Fetches listing pages to get all sitting dates, then fetches and parses
each transcript into structured speaker segments.
"""

import re
import json
import time
import os
import urllib.request
from html.parser import HTMLParser
from datetime import datetime

BASE_URL = "https://nslegislature.ca"
SESSION_URL = f"{BASE_URL}/legislative-business/hansard-debates/assembly-65-session-1"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
RAW_DIR = os.path.join(DATA_DIR, "raw_html")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RAW_DIR, exist_ok=True)


def fetch_url(url, delay=1.5):
    """Fetch URL content with polite delay."""
    print(f"  Fetching: {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "NSHansardScraper/1.0 (civic research project)"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            content = resp.read().decode("utf-8")
        time.sleep(delay)
        return content
    except Exception as e:
        print(f"  ERROR fetching {url}: {e}")
        return None


class TextExtractor(HTMLParser):
    """Extract text content from HTML, converting to a markdown-like format."""
    def __init__(self):
        super().__init__()
        self.result = []
        self.current_tag = None
        self.in_content = False
        self.skip_tags = {"script", "style", "nav", "header", "footer"}
        self.skip_depth = 0
        self.current_href = None
        self.current_title = None

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        if tag in self.skip_tags:
            self.skip_depth += 1
            return
        if self.skip_depth > 0:
            return
        self.current_tag = tag
        if tag == "a":
            self.current_href = attrs_dict.get("href", "")
            self.current_title = attrs_dict.get("title", "")
        elif tag == "strong" or tag == "b":
            self.result.append("**")
        elif tag == "br":
            self.result.append("\n")
        elif tag == "p":
            self.result.append("\n\n")

    def handle_endtag(self, tag):
        if tag in self.skip_tags:
            self.skip_depth -= 1
            return
        if self.skip_depth > 0:
            return
        if tag == "a":
            self.current_href = None
            self.current_title = None
        elif tag == "strong" or tag == "b":
            self.result.append("**")

    def handle_data(self, data):
        if self.skip_depth > 0:
            return
        text = data
        if self.current_href and self.current_title == "View Profile":
            # Format as markdown link for speaker identification
            self.result.append(f"[{text.strip()}]({self.current_href} \"{self.current_title}\")")
        else:
            self.result.append(text)

    def get_text(self):
        return "".join(self.result)


def get_sitting_dates():
    """Fetch all sitting dates from the session listing pages."""
    dates = []
    page = 0

    while True:
        url = f"{SESSION_URL}?page={page}" if page > 0 else SESSION_URL
        html = fetch_url(url)
        if not html:
            break

        # Extract date links: /assembly-65-session-1/house_YYmmmDD
        pattern = r'href="(/legislative-business/hansard-debates/assembly-65-session-1/house_[^"]+)"'
        links = re.findall(pattern, html)

        if not links:
            break

        for link in links:
            slug = link.split("/")[-1]  # house_26apr01
            if slug not in [d["slug"] for d in dates]:
                # Parse date from slug: house_YYmmmDD
                match = re.match(r"house_(\d{2})([a-z]{3})(\d{2})(-\d+)?$", slug)
                if match:
                    yy, mmm, dd, suffix = match.groups()
                    month_map = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
                                 "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12}
                    year = 2000 + int(yy)
                    month = month_map.get(mmm, 1)
                    day = int(dd)
                    try:
                        date_obj = datetime(year, month, day)
                        dates.append({
                            "slug": slug,
                            "path": link,
                            "date": date_obj.strftime("%Y-%m-%d"),
                            "url": BASE_URL + link
                        })
                    except ValueError:
                        print(f"  Skipping invalid date: {slug}")

        print(f"  Page {page}: found {len(links)} links ({len(dates)} total)")
        page += 1

    # Sort by date
    dates.sort(key=lambda d: d["date"])
    return dates


def parse_transcript(html_content, date_str):
    """Parse HTML transcript into structured speaker segments."""

    # Extract the main content area
    # The transcript is in the main article/content div
    extractor = TextExtractor()
    extractor.feed(html_content)
    text = extractor.get_text()

    # Split into lines and clean
    lines = text.split("\n")
    lines = [l.strip() for l in lines]
    lines = [l for l in lines if l]

    # Parse into segments
    segments = []
    current_speaker = None
    current_speaker_slug = None
    current_section = None
    current_text = []
    current_time = None

    # Speaker pattern: [NAME](/members/profiles/slug "View Profile")
    speaker_re = re.compile(r'\[([^\]]+)\]\(/members/(?:profiles/|speaker/)([^\s"]*)')

    # Section heading pattern: **SECTION NAME**
    section_re = re.compile(r'^\*\*([A-Z][A-Z\s\-:.,\'()&/0-9]+)\*\*$')

    # Timestamp pattern
    time_re = re.compile(r'^\[?\[?(\d{1,2}:\d{2}\s*[APap]\.?[Mm]\.?)\]?\]?$')

    # Stage direction pattern
    stage_re = re.compile(r'^\\\[.*\\\]$|^\[.*\]$')

    for line in lines:
        # Check for timestamp
        time_match = time_re.match(line.replace("\\[", "[").replace("\\]", "]"))
        if time_match:
            current_time = time_match.group(1).strip()
            continue

        # Check for section heading
        section_match = section_re.match(line)
        if section_match:
            # Save previous segment
            if current_speaker and current_text:
                segments.append({
                    "speaker": current_speaker,
                    "speaker_slug": current_speaker_slug,
                    "section": current_section,
                    "timestamp": current_time,
                    "text": " ".join(current_text).strip()
                })
                current_text = []

            current_section = section_match.group(1).strip()
            continue

        # Check for speaker
        speaker_match = speaker_re.search(line)
        if speaker_match:
            # Save previous segment
            if current_speaker and current_text:
                segments.append({
                    "speaker": current_speaker,
                    "speaker_slug": current_speaker_slug,
                    "section": current_section,
                    "timestamp": current_time,
                    "text": " ".join(current_text).strip()
                })
                current_text = []

            current_speaker = speaker_match.group(1).strip()
            current_speaker_slug = speaker_match.group(2).strip().rstrip("/")

            # Get the text after the speaker attribution
            # Remove the speaker link markup and the colon
            after_speaker = speaker_re.sub("", line).strip()
            after_speaker = re.sub(r'^[\s:]+', '', after_speaker)
            # Clean leftover markup artifacts
            after_speaker = re.sub(r'"View Profile"\)\s*', '', after_speaker)
            after_speaker = re.sub(r'[«»]', '', after_speaker).strip()
            if after_speaker:
                current_text.append(after_speaker)
            continue

        # Skip stage directions but preserve them in text
        if stage_re.match(line):
            if current_text is not None:
                current_text.append(f"[{line.strip('[]')}]")
            continue

        # Skip page references
        if re.match(r'^\[Page \d+\]$', line):
            continue

        # Regular text — append to current speech
        if current_speaker and line:
            # Clean any leftover markup
            cleaned = re.sub(r'"View Profile"\)\s*', '', line)
            cleaned = re.sub(r'[«»]', '', cleaned).strip()
            if cleaned:
                current_text.append(cleaned)

    # Save last segment
    if current_speaker and current_text:
        segments.append({
            "speaker": current_speaker,
            "speaker_slug": current_speaker_slug,
            "section": current_section,
            "timestamp": current_time,
            "text": " ".join(current_text).strip()
        })

    return segments


def scrape_all():
    """Main scraper: get dates, fetch transcripts, parse, save."""
    print("=" * 60)
    print("NS Hansard Scraper — 65th Assembly, Session 1")
    print("=" * 60)

    # Step 1: Get all sitting dates
    print("\n[1/3] Fetching sitting dates...")
    dates = get_sitting_dates()
    print(f"  Found {len(dates)} sitting dates")

    # Save dates index
    dates_file = os.path.join(DATA_DIR, "sitting_dates.json")
    with open(dates_file, "w") as f:
        json.dump(dates, f, indent=2)

    # Step 2: Fetch and parse each transcript
    print(f"\n[2/3] Fetching and parsing {len(dates)} transcripts...")
    all_transcripts = []
    members_seen = {}

    for i, date_info in enumerate(dates):
        slug = date_info["slug"]
        date_str = date_info["date"]
        print(f"\n  [{i+1}/{len(dates)}] {date_str} ({slug})")

        # Check for cached raw HTML
        raw_file = os.path.join(RAW_DIR, f"{slug}.html")
        if os.path.exists(raw_file):
            print(f"    Using cached HTML")
            with open(raw_file, "r") as f:
                html = f.read()
        else:
            html = fetch_url(date_info["url"])
            if not html:
                print(f"    SKIPPED (fetch failed)")
                continue
            # Cache it
            with open(raw_file, "w") as f:
                f.write(html)

        # Parse
        segments = parse_transcript(html, date_str)
        print(f"    Parsed {len(segments)} speech segments")

        # Track members
        for seg in segments:
            slug = seg.get("speaker_slug", "")
            if slug and slug != "speaker":
                if slug not in members_seen:
                    members_seen[slug] = {
                        "name": seg["speaker"],
                        "slug": slug,
                        "first_seen": date_str,
                        "speech_count": 0,
                        "total_words": 0
                    }
                members_seen[slug]["speech_count"] += 1
                members_seen[slug]["total_words"] += len(seg["text"].split())

        transcript = {
            "date": date_str,
            "slug": date_info["slug"],
            "url": date_info["url"],
            "segment_count": len(segments),
            "segments": segments
        }
        all_transcripts.append(transcript)

    # Step 3: Save results
    print(f"\n[3/3] Saving results...")

    # Save all transcripts
    transcripts_file = os.path.join(DATA_DIR, "transcripts.json")
    with open(transcripts_file, "w") as f:
        json.dump(all_transcripts, f, indent=2)
    print(f"  Saved {len(all_transcripts)} transcripts to {transcripts_file}")

    # Save members
    members_file = os.path.join(DATA_DIR, "members.json")
    members_list = sorted(members_seen.values(), key=lambda m: m["speech_count"], reverse=True)
    with open(members_file, "w") as f:
        json.dump(members_list, f, indent=2)
    print(f"  Found {len(members_list)} unique members")

    # Save raw HTML for full transcript view
    print(f"  Raw HTML cached in {RAW_DIR}")

    # Summary
    total_segments = sum(t["segment_count"] for t in all_transcripts)
    total_words = sum(m["total_words"] for m in members_list)
    print(f"\n{'=' * 60}")
    print(f"DONE")
    print(f"  Sitting days: {len(all_transcripts)}")
    print(f"  Speech segments: {total_segments}")
    print(f"  Unique members: {len(members_list)}")
    print(f"  Total words: {total_words:,}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    scrape_all()
