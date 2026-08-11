#!/bin/bash
unset PYTHONPATH
# S4b v2 3B: SAM2.1-large + 强 LoRA + 视觉塔 LoRA + RefCOCO/+/g + Ref-YT-VOS + MeViS。
# 用法: bash scripts/evo_seg/run_s4b_lisa3b_v2.sh [--resume]
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
cd /9950backfile/chenjiahui/EvoVILA-Seg-worktree
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
NP=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
RESUME_FLAG=""
if [ "${1:-}" = "--resume" ]; then RESUME_FLAG="--resume"; echo "== resuming from checkpoint_latest.pt"; fi
echo "== GPUs: $CUDA_VISIBLE_DEVICES (nproc=$NP)"
/9950backfile/chenjiahui/.conda/envs/evovila/bin/torchrun --nproc_per_node="$NP" --master_port=29630 \
  scripts/evo_seg/train_s4b.py --config configs/evo_seg/s4b_lisa3b_v2.yaml \
  --vila-model /9950backfile/chenjiahui/evo_artifacts/models/VILA1.5-3b \
  --sam2-source-root /9950backfile/zhangyafei/sam2 \
  --sam2-checkpoint /9950backfile/zhangyafei/sam2/checkpoints/sam2.1_hiera_large.pt \
  --manifest-dir /9950backfile/chenjiahui/evo_artifacts/datasets/s4b/manifests/image_merged_all \
  --video-manifest-dir /9950backfile/chenjiahui/evo_artifacts/datasets/s4b/manifests/video_ryvos,/9950backfile/chenjiahui/evo_artifacts/datasets/s4b/manifests/video_mevis \
  --no-object-image-manifest /9950backfile/chenjiahui/evo_artifacts/datasets/s4b/manifests/image_merged_all/train.no_object.jsonl \
  --output /9950backfile/chenjiahui/evo_artifacts/results/s4b/lisa3b_v2_v1 \
  --retention-image-paths /9950backfile/chenjiahui/EvoVILA-Seg-worktree/demo_images/demo_img.png /9950backfile/chenjiahui/EvoVILA-Seg-worktree/demo_images/demo_img_1.png \
  --retention-video-dir /9950backfile/chenjiahui/evo_artifacts/datasets/s4b/retention_frames \
  --wandb-project lisa-evovila $RESUME_FLAG
