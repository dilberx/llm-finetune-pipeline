#!/usr/bin/env bash
# Train the first STOP_AT steps of the pilot with a checkpoint every 10 steps,
# then stop. See configs/finetune_early_ladder.yaml for why.
#
# Usage:  scripts/early_ladder.sh            (about 45 minutes on the RTX 3080)
#         STOP_AT=350 MIN_FREE=7000 scripts/early_ladder.sh
set -uo pipefail

CONFIG="${CONFIG:-configs/finetune_early_ladder.yaml}"
OUT="${OUT:-checkpoints/early_ladder}"
STOP_AT="${STOP_AT:-350}"
# The pilot peaked about 6.8 GB above what Windows held, and froze-style
# thrashing starts when less than that is free.
MIN_FREE="${MIN_FREE:-7000}"

cd "$(dirname "$0")/.."
PY=venv/bin/python

if compgen -G "$OUT/checkpoint-*" >/dev/null; then
    echo "$OUT already holds checkpoints; finetune.py would resume from them." >&2
    echo "Move the directory aside and rerun." >&2
    exit 1
fi
scripts/gpu_preflight.sh "$MIN_FREE" || exit 1

mkdir -p "$OUT"
$PY -m src.training.finetune --config "$CONFIG" > "$OUT/train.log" 2>&1 &
TRAINER=$!
echo "training started (pid $TRAINER), stopping after checkpoint-$STOP_AT"

# Watch the step counter, not the process: a thrashing run stays alive.
PREV=""; STALL=0
while kill -0 "$TRAINER" 2>/dev/null; do
    sleep 30
    if [ -s "$OUT/checkpoint-$STOP_AT/trainer_state.json" ]; then
        sleep 20   # let the adapter file finish writing
        kill "$TRAINER"; wait "$TRAINER" 2>/dev/null
        echo "DONE: stopped after checkpoint-$STOP_AT"; exit 0
    fi
    CUR=$(tr '\r' '\n' < "$OUT/train.log" | grep -aoE "[0-9]+/[0-9]+ \[" | tail -1 | cut -d/ -f1)
    if [ "$CUR" = "$PREV" ]; then STALL=$((STALL + 1)); else STALL=0; fi
    PREV=$CUR
    if [ "$STALL" -ge 20 ]; then
        kill "$TRAINER"; echo "STALLED at step ${CUR:-0} for 10 minutes; stopped" >&2; exit 1
    fi
done
echo "trainer exited before checkpoint-$STOP_AT; see $OUT/train.log" >&2
exit 1
