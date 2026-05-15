#!/usr/bin/env bash
# Batch-size finder for the BassWave training pipeline on the 6700 XT 12 GB.
#
# Tries batch sizes 8, 4, 2, 1 in order. For each: runs ddsp_run with 3 train
# steps and a 240-second timeout. The first one that completes without OOM
# AND without any other crash wins.
#
# Usage:
#   bash find_batch_size.sh

set -uo pipefail   # pipefail: tee no longer hides ddsp_run's failure exit.

# Pulisci ambiente tossico (libcuda di vecchie installazioni cuda interferisce).
unset LD_LIBRARY_PATH

TEST_DIR=/tmp/bs_finder
GIN_FILE=papers/basswave/basswave_44k.gin

WINNER=""

for BS in 8 4 2 1; do
  LOG="/tmp/bs_finder_log_$BS.txt"
  echo
  echo "==================================================================="
  echo "Trying batch_size=$BS"
  echo "==================================================================="
  rm -rf "$TEST_DIR"
  mkdir -p "$TEST_DIR"

  # Run with strict step limit + checkpoint disabled.
  set +e          # don't abort on non-zero; we capture the rc explicitly.
  timeout 240 ddsp_run \
        --mode=train \
        --save_dir="$TEST_DIR" \
        --gin_file="$GIN_FILE" \
        --gin_param="batch_size=$BS" \
        --gin_param="train.num_steps=3" \
        --gin_param="train.steps_per_save=999999" \
        --gin_param="train.steps_per_summary=999999" \
        2>&1 | tee "$LOG"
  RC=${PIPESTATUS[0]}
  set -e

  # Failure inspection: scan the log first, then look at exit code.
  if grep -qi -E 'OOM|out of memory|ResourceExhausted' "$LOG"; then
    echo "BS=$BS: FAILED — OOM detected in log."
    continue
  fi
  if grep -qi -E 'Traceback \(most recent call last\)|^[A-Z][A-Za-z]*Error:' \
      "$LOG"; then
    # Print the last few lines of the traceback so the user can see WHY.
    echo "BS=$BS: FAILED — Python exception detected:"
    tail -n 3 "$LOG"
    # If this isn't an OOM-style error, all smaller batch sizes will fail
    # the same way. Stop early.
    echo
    echo "(This looks like a non-memory error — smaller batch sizes will"
    echo " hit the same problem. Aborting batch-size sweep.)"
    rm -rf "$TEST_DIR"
    exit 2
  fi
  if [ "$RC" -ne 0 ]; then
    echo "BS=$BS: FAILED — non-zero exit code ($RC), no traceback in log."
    continue
  fi

  WINNER=$BS
  echo "BS=$BS: SUCCEEDED (3 training steps completed cleanly)."
  break
done

rm -rf "$TEST_DIR"

echo
echo "==================================================================="
if [ -n "$WINNER" ]; then
  echo "Winning batch size: $WINNER"
  echo "Add to your training launch:  --gin_param=\"batch_size=$WINNER\""
else
  echo "All batch sizes failed. Inspect /tmp/bs_finder_log_*.txt for the"
  echo "actual error."
fi
echo "==================================================================="
