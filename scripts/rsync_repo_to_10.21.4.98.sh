#!/usr/bin/env bash
# Sync the full AD_diagnosis repo to jiawen@10.21.4.98 (same path layout).
#
# Usage (run on this machine after SSH works):
#   bash /data/jiawen/AD_diagnosis/scripts/rsync_repo_to_10.21.4.98.sh
#
# One-time SSH setup if needed:
#   ssh-copy-id jiawen@10.21.4.98

set -euo pipefail

REMOTE="jiawen@10.21.4.98"
SRC="/data/jiawen/AD_diagnosis"
DEST="/data/jiawen/AD_diagnosis"

echo "Testing SSH to $REMOTE ..."
ssh -o BatchMode=yes "$REMOTE" 'echo SSH OK'

echo "Creating remote directory ..."
ssh "$REMOTE" "mkdir -p /data/jiawen"

echo "Syncing repo (~950M incl. .git) to ${REMOTE}:${DEST} ..."
rsync -avP \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude '.cursor/' \
  "$SRC/" "${REMOTE}:${DEST}/"

echo "Done. Remote path: ${REMOTE}:${DEST}"
