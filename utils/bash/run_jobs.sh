#!/usr/bin/env bash
# Simple launcher for same training script, changing only config + log.

set -euo pipefail

# -------------------------
# Editable variables
# -------------------------
PYBIN="python"            # Python executable
TRAIN_SCRIPT="train.py" 
PARALLEL=false             # true = run all in parallel; false = sequential

# Jobs: "config_file|log_file|extra_args (optional)"
JOBS=(
  "soloba-v0.0.0.yaml|soloba-v0.0.0.log|"
  "quartznet-v1.0.0.yaml|quartznet-v1.0.0.log|"
  "soloni-v1.0.0.yaml|soloni-v1.0.0.log|"
)

# -------------------------
# Runner
# -------------------------
echo "Starting training jobs (parallel=$PARALLEL)..."
PIDS=()

for ENTRY in "${JOBS[@]}"; do
  IFS='|' read -r CFG LOG EXTRA <<< "$ENTRY"

  [[ -f "$CFG" ]] || echo "warn: config $CFG not found"

  echo "Launching: $TRAIN_SCRIPT --config $CFG  -> $LOG"
  nohup "$PYBIN" "$TRAIN_SCRIPT" --config "$CFG" > "$LOG" 2>&1 &

  PID=$!
  if "$PARALLEL"; then
    PIDS+=("$PID")
  else
    wait "$PID"
  fi
done

if "$PARALLEL"; then
  echo "Waiting for ${#PIDS[@]} job(s) to finish..."
  for pid in "${PIDS[@]}"; do
    wait "$pid"
  done
fi

echo "All training jobs completed!"
