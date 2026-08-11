#!/bin/bash
unset PYTHONPATH
# S4b LISA merged-data training on VILA1.5-7b with the SAM2 mask decoder trainable.
# Usage: bash scripts/evo_seg/run_s4b_lisa7b_merged.sh
cd /9950backfile/chenjiahui/EvoVILA-Seg-worktree
/9950backfile/chenjiahui/.conda/envs/evovila/bin/torchrun --nproc_per_node=8 --master_port=29618 \
  scripts/evo_seg/train_s4b.py --config configs/evo_seg/s4b_lisa7b_merged.yaml \
  --vila-model /9950backfile/chenjiahui/evo_artifacts/models/VILA1.5-7b \
  --sam2-source-root /9950backfile/zhangyafei/sam2 \
  --sam2-checkpoint /9950backfile/zhangyafei/sam2/checkpoints/sam2.1_hiera_tiny.pt \
  --manifest-dir /9950backfile/chenjiahui/evo_artifacts/datasets/s4b/manifests/image_merged_all \
  --video-manifest-dir /9950backfile/chenjiahui/evo_artifacts/datasets/s4b/manifests/video_ryvos \
  --no-object-image-manifest /9950backfile/chenjiahui/evo_artifacts/datasets/s4b/manifests/image_merged_all/train.no_object.jsonl \
  --output /9950backfile/chenjiahui/evo_artifacts/results/s4b/lisa7b_merged_v1 \
  --retention-image-paths /9950backfile/chenjiahui/EvoVILA-Seg-worktree/demo_images/demo_img.png /9950backfile/chenjiahui/EvoVILA-Seg-worktree/demo_images/demo_img_1.png \
  --retention-video-dir /9950backfile/chenjiahui/evo_artifacts/datasets/s4b/retention_frames \
  --wandb-project lisa-evovila
