#!/bin/bash
# Copy the Mac working copy to the Windows source of truth (E:\hackathon-project)
# and commit it there with git.
#   ./sync.sh "what changed"
# Needs the SSH master connection to be open:
#   ssh -fN -o ControlMaster=yes -o ControlPersist=4h -o ControlPath=~/.ssh/cm-%r@%h:%p <user>@<windows-host>
# Copies code, docs and small model outputs only. Raw data never leaves Windows.
set -euo pipefail
cd "$(dirname "$0")"
S=(ssh -o "ControlPath=$HOME/.ssh/cm-%r@%h:%p" -o BatchMode=yes -o LogLevel=ERROR <user>@<windows-host>)
MSG="${1:-sync from mac}"
MSG="${MSG//\"/\'}"

TMP=$(mktemp -t hacksync).tgz
trap 'rm -f "$TMP"' EXIT
COPYFILE_DISABLE=1 tar czf "$TMP" \
  --exclude .venv --exclude __pycache__ --exclude 'planner/cache' --exclude 'model/work' \
  --exclude .DS_Store --exclude '*.pyc' --exclude 'realtime/data' \
  README.md CLAUDE.md docs planner model .gitignore sync.sh $(for d in eval realtime; do [ -e "$d" ] && echo "$d"; done)

# Refuse to send more than 500 MB in one sync (guard against copying data by mistake).
SIZE=$(stat -f%z "$TMP")
if [ "$SIZE" -gt 524288000 ]; then echo "Refusing: archive is $SIZE bytes (>500 MB)"; exit 1; fi

"${S[@]}" "if not exist E:\\hackathon-project mkdir E:\\hackathon-project" >/dev/null
"${S[@]}" "tar xzf - -C E:/hackathon-project" < "$TMP"

"${S[@]}" "cd /d E:\\hackathon-project && (if not exist .git (git init -q -b main && git config user.name hackathon-team && git config user.email team@hackathon.local && git config core.autocrlf false)) && git add -A && git commit -q -m \"${MSG}\" -m \"Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\" & git log --oneline -3"
