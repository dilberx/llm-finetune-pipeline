#!/usr/bin/env bash
# Train on a rented GPU box, then pack the adapters for download.
#
# Only the training runs remotely. Evaluation fits the local 10 GB card and
# costs nothing there, so the box can be released as soon as this finishes.
#
# On your machine, after renting a box (Linux, CUDA 12.x driver, >= 24 GB):
#
#   ssh <box> 'git clone -b eval-statistics-and-synthesis-fix \
#       https://github.com/dilberx/llm-finetune-pipeline.git'
#   rsync -avz data/curated/train.jsonl data/curated/eval.jsonl \
#       <box>:llm-finetune-pipeline/data/curated/
#   ssh <box> 'cd llm-finetune-pipeline && nohup scripts/cloud_train.sh > run.log 2>&1 &'
#
# Then, when run.log ends with DONE:
#
#   rsync -avz <box>:llm-finetune-pipeline/adapters.tar.gz .
#   tar -xzf adapters.tar.gz        # unpacks into checkpoints/
#
# data/curated is gitignored, which is why it is copied rather than cloned.
#
# Environment:
#   CONFIG=configs/finetune_cloud.yaml   which training config to use
#   MIN_FREE=20000                       preflight threshold in MiB
set -euo pipefail

CONFIG="${CONFIG:-configs/finetune_cloud.yaml}"
MIN_FREE="${MIN_FREE:-20000}"

cd "$(dirname "$0")/.."

for f in data/curated/train.jsonl data/curated/eval.jsonl; do
    [ -s "$f" ] || { echo "missing $f: copy data/curated/ from your machine first" >&2; exit 1; }
done

# finetune.py silently resumes from any checkpoint-* it finds. On a fresh box
# there are none; refuse rather than guess if there are.
if compgen -G "checkpoints/checkpoint-*" >/dev/null; then
    echo "checkpoints/ already holds checkpoint-* directories; finetune.py would resume" >&2
    echo "from them. Move them aside (e.g. checkpoints/_archive_<date>/) and rerun." >&2
    exit 1
fi

if [ ! -x venv/bin/python ]; then
    python3 -m venv venv
    venv/bin/pip install -q --upgrade pip
    venv/bin/pip install -q -r requirements-train.txt
fi

# Strict: this run's speed decides whether the full corpus is worth paying
# for, so it must not share the card with anything.
PREFLIGHT_STRICT=1 scripts/gpu_preflight.sh "$MIN_FREE"

start=$(date +%s)
venv/bin/python -m src.training.finetune --config "$CONFIG" 2>&1 | tee train.log
echo "training wall time: $(( $(date +%s) - start )) s" | tee -a train.log

# Adapters and trainer state only. Optimizer and scheduler state are what a
# resume needs, not an eval, and they are most of each checkpoint's size.
tar -czf adapters.tar.gz \
    --exclude='optimizer.pt' --exclude='scheduler.pt' --exclude='rng_state*.pth' \
    checkpoints train.log
ls -lh adapters.tar.gz
echo DONE
