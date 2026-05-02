#!/usr/bin/env bash
set -euo pipefail

cd /root/conv

TS="$(date +%Y%m%d_%H%M%S)"
BK="/hy-tmp/das_convtasnet_progress_${TS}"
mkdir -p "${BK}"

cat > "${BK}/README_backup.txt" <<'EOF'
DAS Conv-TasNet progress backup
Code pushed to GitHub main: 3a42072 Add separated-output classifier fine-tuning
Dataset is NOT included.

Recommended final artifacts:
- separator: checkpoints/separator_4090/best.pt
- separated-output classifier: checkpoints/classifier_sep_finetune/best.pt
- final evaluation: results/final_metrics_sepcls_finetune_smooth1

Key metrics:
- final separated classification accuracy: 0.822662
- final precision: 0.824308
- final recall: 0.822277
- final f1: 0.822345
- 2src accuracy: 0.922762
- 3src accuracy: 0.804557
- 4src accuracy: 0.725968
- stage1 final val SI-SNR: 1.9669 dB
- stage2 final val SI-SNR: 1.9095 dB
- stage3 final val SI-SNR: 1.4378 dB
EOF

cp -r configs src scripts "${BK}/"

mkdir -p \
  "${BK}/checkpoints/classifier_4090" \
  "${BK}/checkpoints/classifier_sep_finetune" \
  "${BK}/checkpoints/separator_4090" \
  "${BK}/checkpoints/separator_stage1_2src" \
  "${BK}/checkpoints/separator_stage2_2to3src" \
  "${BK}/results" \
  "${BK}/logs"

cp checkpoints/classifier_4090/best.pt "${BK}/checkpoints/classifier_4090/"
cp checkpoints/classifier_sep_finetune/best.pt "${BK}/checkpoints/classifier_sep_finetune/"
cp checkpoints/separator_4090/best.pt "${BK}/checkpoints/separator_4090/"
cp checkpoints/separator_stage1_2src/best.pt "${BK}/checkpoints/separator_stage1_2src/"
cp checkpoints/separator_stage2_2to3src/best.pt "${BK}/checkpoints/separator_stage2_2to3src/"

cp -r \
  results/final_metrics_smooth1 \
  results/final_metrics_smooth5 \
  results/final_metrics_polarity_smooth1 \
  results/final_metrics_sepcls_finetune_smooth1 \
  "${BK}/results/"

cp \
  train_classifier_4090.log \
  train_classifier_sep_finetune.log \
  train_pipeline_curriculum.log \
  train_separator_stage1.log \
  train_separator_stage2.log \
  train_separator_stage3.log \
  evaluate_pipeline_smooth1.log \
  evaluate_pipeline_smooth5.log \
  evaluate_pipeline_polarity_smooth1.log \
  evaluate_pipeline_sepcls_finetune_smooth1.log \
  "${BK}/logs/" 2>/dev/null || true

tar -czf "${BK}.tar.gz" -C /hy-tmp "$(basename "${BK}")"
du -sh "${BK}" "${BK}.tar.gz"
echo "BACKUP_DIR=${BK}"
echo "BACKUP_TAR=${BK}.tar.gz"
