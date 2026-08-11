#!/bin/bash
# =====================================================================
# One-click S4b evaluation launcher.
#
# Runs BOTH S4b evaluations for one checkpoint:
#   1) anti-shortcut eval : no-object hallucination rate + query-swap mask drift
#   2) formal eval        : RefCOCO/+/g val mask IoU + Ref-YT-VOS J&F
#
# Usage:
#   bash scripts/evo_seg/run_s4b_evals.sh <checkpoint.pt> [run_tag] [--smoke]
#
#   <checkpoint.pt>  a checkpoint produced by train_s4b.py
#                    (e.g. .../lisa_merged_v1/checkpoint_latest.pt)
#   [run_tag]        output tag; defaults to the checkpoint parent dir name
#   [--smoke]        run tiny 3-sample versions of both evals first, to verify
#                    the pipeline initializes (this machine has a recurring
#                    eval-init deadlock; smoke catches it before a long run)
#
# Outputs (outside the repo, under evo_artifacts):
#   .../results/s4b/eval/<run_tag>_shortcut_user/   (anti-shortcut)
#   .../results/s4b/eval/<run_tag>_formal_user/     (formal, per-shard + summary)
#
# OMP_NUM_THREADS is capped by default: on this shared 192-core node the
# default 192-thread OpenMP pool can deadlock during eval initialization
# (futex spin, no progress) when several processes start at once.
# Override with OMP_NUM_THREADS=... MKL_NUM_THREADS=... GPU_COUNT=...
# =====================================================================
set -euo pipefail

# ---------- fixed environment ----------
WORKTREE=/9950backfile/chenjiahui/EvoVILA-Seg-worktree
PY=/9950backfile/chenjiahui/.conda/envs/evovila/bin/python
CONFIG=${CONFIG:-configs/evo_seg/s4b_lisa_merged.yaml}
VILA_MODEL=${VILA_MODEL:-/9950backfile/chenjiahui/evo_artifacts/models/VILA1.5-3b}
SAM2_ROOT=${SAM2_ROOT:-/9950backfile/zhangyafei/sam2}
SAM2_CKPT=${SAM2_CKPT:-$SAM2_ROOT/checkpoints/sam2.1_hiera_tiny.pt}
MANIFEST_ROOT=/9950backfile/chenjiahui/evo_artifacts/datasets/s4b/manifests
IMG_MANIFEST=$MANIFEST_ROOT/image_merged_all
VID_MANIFEST=$MANIFEST_ROOT/video_ryvos
REFCOCO_MANIFEST=$MANIFEST_ROOT/image_refcoco
NO_OBJECT_MANIFEST=$REFCOCO_MANIFEST/val.no_object.jsonl
RESULTS_ROOT=/9950backfile/chenjiahui/evo_artifacts/results/s4b/eval
GPU_COUNT=${GPU_COUNT:-8}

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}

# ---------- arguments ----------
if [ $# -lt 1 ]; then
  echo "usage: $0 <checkpoint.pt> [run_tag] [--smoke]" >&2
  exit 2
fi
CKPT=$(readlink -f "$1")
if [ ! -f "$CKPT" ]; then
  echo "error: checkpoint not found: $CKPT" >&2
  exit 2
fi
TAG="${2:-$(basename "$(dirname "$CKPT")")}"
SMOKE=0
if [ "${3:-}" = "--smoke" ]; then
  SMOKE=1
fi

SHORTCUT_OUT=$RESULTS_ROOT/${TAG}_shortcut_user
FORMAL_OUT=$RESULTS_ROOT/${TAG}_formal_user
mkdir -p "$SHORTCUT_OUT" "$FORMAL_OUT"

echo "== checkpoint : $CKPT"
echo "== run tag    : $TAG"
echo "== mode       : $([ "$SMOKE" = 1 ] && echo SMOKE || echo FULL)"
echo "== shortcut -> $SHORTCUT_OUT"
echo "== formal   -> $FORMAL_OUT"
echo "== OMP_NUM_THREADS=$OMP_NUM_THREADS MKL_NUM_THREADS=$MKL_NUM_THREADS GPU_COUNT=$GPU_COUNT"

cd "$WORKTREE"

# ---------- smoke mode: 3-sample versions of both evals ----------
if [ "$SMOKE" = 1 ]; then
  echo
  echo ">>> [smoke 1/2] anti-shortcut (max-samples 3)"
  "$PY" scripts/evo_seg/eval_antishortcut.py \
    --config "$CONFIG" --checkpoint "$CKPT" --vila-model "$VILA_MODEL" \
    --sam2-source-root "$SAM2_ROOT" --sam2-checkpoint "$SAM2_CKPT" \
    --no-object-manifest "$NO_OBJECT_MANIFEST" --val-manifest-dir "$REFCOCO_MANIFEST" \
    --output "$SHORTCUT_OUT" --device cuda:0 --max-samples 3

  echo
  echo ">>> [smoke 2/2] formal eval (max-samples 3, image only)"
  "$PY" scripts/evo_seg/eval_s4b.py \
    --config "$CONFIG" --checkpoint "$CKPT" --vila-model "$VILA_MODEL" \
    --sam2-source-root "$SAM2_ROOT" --sam2-checkpoint "$SAM2_CKPT" \
    --manifest-dir "$IMG_MANIFEST" --video-manifest-dir "$VID_MANIFEST" \
    --output "$FORMAL_OUT" --device cuda:0 --max-samples 3 --skip-video

  echo
  echo "==> SMOKE OK: both evals initialize and run without hanging."
  echo "==> Launch the full runs with:"
  echo "    bash scripts/evo_seg/run_s4b_evals.sh '$CKPT' '$TAG'"
  exit 0
fi

# ---------- 1/3 anti-shortcut (single GPU) ----------
echo
echo ">>> [1/3] anti-shortcut eval (no-object + query-swap) on cuda:0"
"$PY" scripts/evo_seg/eval_antishortcut.py \
  --config "$CONFIG" --checkpoint "$CKPT" --vila-model "$VILA_MODEL" \
  --sam2-source-root "$SAM2_ROOT" --sam2-checkpoint "$SAM2_CKPT" \
  --no-object-manifest "$NO_OBJECT_MANIFEST" --val-manifest-dir "$REFCOCO_MANIFEST" \
  --output "$SHORTCUT_OUT" --device cuda:0
echo "==> anti-shortcut summary: $SHORTCUT_OUT/merged_summary.txt"
cat "$SHORTCUT_OUT/merged_summary.txt" 2>/dev/null || true

# ---------- 2/3 formal eval, GPU_COUNT shards ----------
echo
echo ">>> [2/3] formal eval (RefCOCO/+/g IoU + Ref-YT-VOS J&F) on $GPU_COUNT GPUs"
pids=()
for i in $(seq 0 $((GPU_COUNT - 1))); do
  CUDA_VISIBLE_DEVICES=$i "$PY" scripts/evo_seg/eval_s4b.py \
    --config "$CONFIG" --checkpoint "$CKPT" --vila-model "$VILA_MODEL" \
    --sam2-source-root "$SAM2_ROOT" --sam2-checkpoint "$SAM2_CKPT" \
    --manifest-dir "$IMG_MANIFEST" --video-manifest-dir "$VID_MANIFEST" \
    --output "$FORMAL_OUT" --device cuda:0 \
    --shard-index "$i" --shard-count "$GPU_COUNT" \
    > /tmp/${TAG}_formal_shard_$i.log 2>&1 &
  pids+=("$!")
done
echo "==> waiting for $GPU_COUNT shard(s) ... (logs: /tmp/${TAG}_formal_shard_*.log)"
fail=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    fail=1
  fi
done
if [ "$fail" = 1 ]; then
  echo "error: at least one formal shard failed; see /tmp/${TAG}_formal_shard_*.log" >&2
  exit 1
fi

# ---------- 3/3 merge ----------
echo
echo ">>> [3/3] merge formal shards"
"$PY" scripts/evo_seg/eval_s4b.py \
  --config "$CONFIG" --checkpoint "$CKPT" \
  --vila-model "$VILA_MODEL" --sam2-source-root "$SAM2_ROOT" --sam2-checkpoint "$SAM2_CKPT" \
  --output "$FORMAL_OUT" --merge-only | tee "$FORMAL_OUT/merged_summary.txt"

echo
echo "==> ALL DONE"
echo "   anti-shortcut: $SHORTCUT_OUT/merged_summary.txt"
echo "   formal       : $FORMAL_OUT/merged_summary.txt"
