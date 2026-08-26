#!/usr/bin/env bash
#
# Deploy this repo to a Hugging Face Space.
#
#   ./deploy_hf.sh <hf-username>/<space-name>
#
# Pushes a single-commit snapshot of the tracked files, minus the SQLite
# database. Two reasons the history is not pushed as-is: Hugging Face rejects
# non-LFS files over 10MB, and this repo's history carries several copies of a
# 19MB database. The Dockerfile fetches the database from GitHub at build time.
#
# Requires a Hugging Face token with write access when git prompts for a
# password (Settings -> Access Tokens on huggingface.co).

set -euo pipefail

SPACE="${1:-}"
if [ -z "$SPACE" ]; then
    echo "usage: $0 <hf-username>/<space-name>" >&2
    echo "example: $0 JohnMacGillis/ns-hansard" >&2
    exit 1
fi

SRC="$(cd "$(dirname "$0")" && pwd)"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

cd "$SRC"

if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "warning: you have uncommitted changes; deploying the files as they are on disk." >&2
fi

echo "Staging tracked files (excluding data/hansard.db)..."
count=0
while IFS= read -r -d '' f; do
    [ "$f" = "data/hansard.db" ] && continue
    mkdir -p "$STAGE/$(dirname "$f")"
    cp "$f" "$STAGE/$f"
    count=$((count + 1))
done < <(git ls-files -z)
echo "  $count files"

if [ ! -f "$STAGE/Dockerfile" ]; then
    echo "error: Dockerfile is not tracked by git — commit it before deploying." >&2
    exit 1
fi

if ! head -1 "$STAGE/README.md" | grep -q '^---$'; then
    echo "error: README.md is missing the Hugging Face YAML frontmatter." >&2
    echo "       The Space needs it to know sdk: docker and app_port." >&2
    exit 1
fi

cd "$STAGE"
git init -q -b main
git add -A
git -c user.email=deploy@localhost -c user.name=deploy \
    commit -qm "Deploy NS Hansard Search"

echo "Pushing to https://huggingface.co/spaces/$SPACE"
git push -f "https://huggingface.co/spaces/$SPACE" main

echo
echo "Done. Watch the build at https://huggingface.co/spaces/$SPACE"
echo "Once it is running: https://$(echo "$SPACE" | tr '/' '-' | tr '[:upper:]' '[:lower:]').hf.space"
