# Hugging Face Spaces (Docker SDK).
#
# The 19MB SQLite database is deliberately NOT in the Space repo — Hugging Face
# requires Git LFS for files over 10MB, and the repo's git history already
# carries several copies of the database. Instead the build fetches it from the
# GitHub repo, which stays the single source of truth for the data.
#
# To publish a data update: push to GitHub, then rebuild ("Factory rebuild" in
# the Space's Settings) so this layer re-fetches.

FROM python:3.13-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

# Spaces run the container as UID 1000.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PORT=7860 \
    PYTHONUNBUFFERED=1
WORKDIR /home/user/app

COPY --chown=user:user requirements.txt ./
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user:user . .

ARG DB_URL=https://github.com/JohnMacGillis/ns-hansard/raw/main/data/hansard.db
RUN mkdir -p data \
 && curl -fsSL "$DB_URL" -o data/hansard.db \
 && python3 -c "import sqlite3; n = sqlite3.connect('data/hansard.db').execute('SELECT COUNT(*) FROM speeches').fetchone()[0]; print(f'database ok: {n} speeches'); assert n > 0"

EXPOSE 7860
CMD ["python3", "server.py"]
