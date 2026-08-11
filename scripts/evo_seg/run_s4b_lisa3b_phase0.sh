#!/bin/bash
unset PYTHONPATH
# S4b Phase-0 3B, 单模型 DDP 多卡训练:
#   epoch-cycle 全量数据覆盖 + 强 LoRA + 可训练 SAM2 mask decoder + 梯度累积。
# 用法: bash scripts/evo_seg/run_s4b_lisa3b_phase0.sh
# 默认 GPU 4-7 (0-3 被 starVLA 占用)。想换卡:
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/evo_seg/run_s4b_lisa3b_phase0.sh
# 64 核机器上必须限制每进程线程数, 否则多进程线程超订会非常慢甚至卡死。
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
cd /9950backfile/chenjiahui/EvoVILA-Seg-worktree
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
NP=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
RESUME_FLAG=""
if [ "${1:-}" = "--resume" ]; then RESUME_FLAG="--resume"; echo "== resuming from checkpoint_latest.pt"; fi
echo "== GPUs: $CUDA_VISIBLE_DEVICES (nproc=$NP, OMP_NUM_THREADS=8)"
/9950backfile/chenjiahui/.conda/envs/evovila/bin/torchrun --nproc_per_node="$NP" --master_port=29619 \
  scripts/evo_seg/train_s4b.py --config configs/evo_seg/s4b_lisa3b_phase0.yaml \
  --vila-model /9950backfile/chenjiahui/evo_artifacts/models/VILA1.5-3b \
  --sam2-source-root /9950backfile/zhangyafei/sam2 \
  --sam2-checkpoint /9950backfile/zhangyafei/sam2/checkpoints/sam2.1_hiera_tiny.pt \
  --manifest-dir /9950backfile/chenjiahui/evo_artifacts/datasets/s4b/manifests/image_merged_all \
  --video-manifest-dir /9950backfile/chenjiahui/evo_artifacts/datasets/s4b/manifests/video_ryvos \
  --no-object-image-manifest /9950backfile/chenjiahui/evo_artifacts/datasets/s4b/manifests/image_merged_all/train.no_object.jsonl \
  --output /9950backfile/chenjiahui/evo_artifacts/results/s4b/lisa3b_phase0_v1 \
  --retention-image-paths /9950backfile/chenjiahui/EvoVILA-Seg-worktree/demo_images/demo_img.png /9950backfile/chenjiahui/EvoVILA-Seg-worktree/demo_images/demo_img_1.png \
  --retention-video-dir /9950backfile/chenjiahui/evo_artifacts/datasets/s4b/retention_frames \
  --wandb-project lisa-evovila $RESUME_FLAG
