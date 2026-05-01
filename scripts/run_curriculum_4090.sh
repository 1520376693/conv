#!/usr/bin/env bash
set -u

cd /root/conv || exit 1

PIPELINE_LOG=/root/conv/train_pipeline_curriculum.log
echo "[$(date '+%F %T')] curriculum pipeline started" >> "${PIPELINE_LOG}"

run_stage() {
  local name="$1"
  local config="$2"
  local log="$3"
  echo "[$(date '+%F %T')] starting ${name}: ${config}" >> "${PIPELINE_LOG}"
  python scripts/train_separator.py --config "${config}" > "${log}" 2>&1
  local status=$?
  echo "[$(date '+%F %T')] finished ${name} status=${status}" >> "${PIPELINE_LOG}"
  return "${status}"
}

run_stage "stage1_2src" "configs/train_4090_stage1.yml" "train_separator_stage1.log" || exit $?
run_stage "stage2_2to3src" "configs/train_4090_stage2.yml" "train_separator_stage2.log" || exit $?
run_stage "stage3_2to4src" "configs/train_4090_stage3.yml" "train_separator_stage3.log" || exit $?

for kernel in 1 5; do
  echo "[$(date '+%F %T')] evaluating post_smooth_kernel=${kernel}" >> "${PIPELINE_LOG}"
  python scripts/evaluate_pipeline.py \
    --config configs/train_4090.yml \
    --separator_ckpt checkpoints/separator_4090/best.pt \
    --classifier_ckpt checkpoints/classifier_4090/best.pt \
    --out_dir "results/final_metrics_smooth${kernel}" \
    --max_batches 500 \
    --post_smooth_kernel "${kernel}" > "evaluate_pipeline_smooth${kernel}.log" 2>&1
  echo "[$(date '+%F %T')] evaluation smooth${kernel} status=$?" >> "${PIPELINE_LOG}"
done

echo "[$(date '+%F %T')] curriculum pipeline completed" >> "${PIPELINE_LOG}"
