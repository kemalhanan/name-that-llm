#!/bin/bash
# Runs on the Colab runtime: report the run, and start it if it is not running and not finished.
# The first line of output is NTL_STATUS=DONE | RUNNING | STARTED.
VARIANT="${1:-mistral7b-full}"
B=/content/ntl_boot
STATE="$(cat /content/ntl/data/remote/state.txt 2>/dev/null)"
if [[ "$STATE" == DONE_UPLOADED* ]]; then
  echo "NTL_STATUS=DONE"
elif ! flock -n "$B/lock" true 2>/dev/null; then
  echo "NTL_STATUS=RUNNING"
else
  setsid nohup bash "$B/boot.sh" "$VARIANT" </dev/null >/dev/null 2>&1 &
  echo "NTL_STATUS=STARTED"
fi
echo "state: ${STATE:-none yet}"
nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv,noheader 2>/dev/null
echo "--- boot.log ---"
grep -a -E "^\[unattended\]|^\[sync|^=== |restored|nothing to restore|cannot|exited" "$B/boot.log" 2>/dev/null | tail -n 5
echo "--- training log ---"
grep -a -E "^step |^fold |resuming|saved the|stage 1:|out-of-fold" "$B/boot.log" 2>/dev/null | tail -n 3
