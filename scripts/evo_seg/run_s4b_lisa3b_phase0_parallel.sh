#!/bin/bash
unset PYTHONPATH
# S4b Phase-0 3B "多卡分别起": 每张卡起 1 个独立训练进程 (无 NCCL, 互不影响),
# 产出 N 个独立模型, 跑完用 eval 挑最好的。等价于 4 个并行实验。
# 用法: bash scripts/evo_seg/run_s4b_lisa3b_phase0_parallel.sh
# 默认 GPU 4-7, 可覆盖:  CUDA_VISIBLE_DEVICES=4,5,6,7 bash .../run_s4b_lisa3b_phase0_parallel.sh
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
cd /9950backfile/chenjiahui/EvoVILA-Seg-worktree
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
IFS=',' read -r -a GPUS <<< "$CUDA_VISIBLE_DEVICES"
BASE_PORT=29700
BASE_SEED=100
idx=0
for gpu in "${GPUS[@]}"; do
  out="/9950backfile/chenjiahui/evo_artifacts/results/s4b/lisa3b_phase0_v1_gpu${idx}"
  port=$((BASE_PORT + idx))
  seed=$((BASE_SEED + idx))
  echo ">> GPU $gpu | output=$out | seed=$seed | port=$port"
  setsid nohup env CUDA_VISIBLE_DEVICES="$gpu" \
    /9950backfile/chenjiahui/.conda/envs/evovila/bin/torchrun --nproc_per_node=1 --master_port="$port" \
      scripts/evo_seg/train_s4b.py --config configs/evo_seg/s4b_lisa3b_phase0.yaml \
      --vila-model /9950backfile/chenjiahui/evo_artifacts/models/VILA1.5-3b \
      --sam2-source-root /9950backfile/zhangyafei/sam2 \
      --sam2-checkpoint /9950backfile/zhangyafei/sam2/checkpoints/sam2.1_hiera_tiny.pt \
      --manifest-dir /9950backfile/chenjiahui/evo_artifacts/datasets/s4b/manifests/image_merged_all \
      --video-manifest-dir /9950backfile/chenjiahui/evo_artifacts/datasets/s4b/manifests/video_ryvos \
      --no-object-image-manifest /9950backfile/chenjiahui/evo_artifacts/datasets/s4b/manifests/image_merged_all/train.no_object.jsonl \
      --output "$out" --seed "$seed" \
      --retention-image-paths /9950backfile/chenjiahui/EvoVILA-Seg-worktree/demo_images/demo_img.png /9950backfile/chenjiahui/EvoVILA-Seg-worktree/demo_images/demo_img_1.png \
      --retention-video-dir /9950backfile/chenjiahui/evo_artifacts/datasets/s4b/retention_frames \
      --wandb-project lisa-evovila \
      > "/tmp/lisa3b_phase0_gpu${idx}.log" 2>&1 < /dev/null &
  idx=$((idx + 1))
done
echo "== launched $idx independent process(es). logs: /tmp/lisa3b_phase0_gpu*.log"
