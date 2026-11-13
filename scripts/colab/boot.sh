#!/bin/bash
# Runs on the Colab runtime (push.py puts it in /content/ntl_boot). Installs the repo and the
# Kaggle credential, downloads the inputs, then runs the unattended full run, which restores the
# last backup from Kaggle first. Safe to start again at any time: a second copy exits at once.
VARIANT="${1:-mistral7b-full}"
B=/content/ntl_boot
exec >>"$B/boot.log" 2>&1
exec 9>"$B/lock"
flock -n 9 || { echo "already running"; exit 0; }
echo "=== boot $(date -u '+%F %T') UTC, variant $VARIANT ==="
mkdir -p /root/.kaggle
for f in kaggle.json access_token; do
  [ -f "$B/$f" ] && install -m 600 "$B/$f" "/root/.kaggle/$f"
done
if [ -f /root/.kaggle/access_token ]; then
  KAGGLE_API_TOKEN="$(tr -d '\r\n' </root/.kaggle/access_token)"
  export KAGGLE_API_TOKEN
fi
mkdir -p /content/ntl && tar -xzf "$B/repo.tgz" -C /content/ntl && cd /content/ntl || exit 1
pip install -q uv || exit 1
uv sync -q --group lora || exit 1
uv run python -m name_that_llm.remote_train fetch "$VARIANT" || exit 1
uv run python -m name_that_llm.remote_train unattended "$VARIANT"
echo "=== unattended exited with $? at $(date -u '+%F %T') UTC ==="
