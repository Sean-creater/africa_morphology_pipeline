#!/bin/bash
PYTHON=/home/sjs96_file_share/anaconda3/envs/Africa_env/bin/python3
SCRIPT=/home/sjs96_file_share/DL_workflow/K_Selection/imagery/generate_images.py
LOG=/home/sjs96_file_share/DL_workflow/K_Selection/logs/generate_images.log
TARGET=1160
RESTART_INTERVAL=600  # restart every 10 minutes

cd /home/sjs96_file_share/DL_workflow/K_Selection

while true; do
    COUNT=$(find outputs/images -name "*.png" 2>/dev/null | wc -l)
    echo "[$(date '+%H:%M:%S')] Images so far: $COUNT / $TARGET" >> "$LOG"
    if [ "$COUNT" -ge "$TARGET" ]; then
        echo "[$(date '+%H:%M:%S')] All $TARGET images done!" >> "$LOG"
        break
    fi
    $PYTHON "$SCRIPT" --project YOUR_GEE_PROJECT_ID --workers 4 >> "$LOG" 2>&1 &
    PY_PID=$!
    sleep $RESTART_INTERVAL
    kill $PY_PID 2>/dev/null
    wait $PY_PID 2>/dev/null
    echo "[$(date '+%H:%M:%S')] Restarting after ${RESTART_INTERVAL}s..." >> "$LOG"
    sleep 3
done
