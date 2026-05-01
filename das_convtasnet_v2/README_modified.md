# DAS Conv-TasNet 多声源分离与分类项目 v2

本版本针对 BJTUSensor DAS 数据集的 `.mat` 文件（常见形状 `[10000, 12]`）进行了重构，目标是快速提升“随机2/3源混合后的分离 + 分离后分类 + 定性/定量分析”的效果。

## 项目结构

```text
das_convtasnet_v2/
├── Conv_TasNet.py                 # 基础层、LayerNorm、TCN residual+skip block
├── DAS_Conv_TasNet.py             # 12通道DAS版Conv-TasNet Plus
├── DAS_classifier.py              # 残差2D-CNN分类器，兼容DASBaselineCNN类名
├── DAS_loss.py                    # 可变源数PIT-SI-SNR + 通道SI-SNR + 波形项 + 混合一致性
├── DASDataLoaders.py              # 单事件分类数据集、随机2/3源混合数据集、增强
├── das_preprocess.py              # .mat读取、差分、归一化、平滑、CNN输入转换
├── train_das_separator.py         # 分离模型训练
├── train_das_classifier.py        # 分类模型训练
├── eval_das_separator.py          # 分离定量指标：SI-SNR/SI-SNRi/SNR/SDR/MSE/MAE/PCC
├── eval_das_classification.py     # 分离后信号分类准确率、F1、混淆矩阵
├── separate_das.py                # 对真实/混合.mat文件分离并保存est_*.mat
├── plot_das_results.py            # 波形对比、热力图、mix与sum(est)对比
├── utils.py
├── options/
│   ├── train_das.yml
│   └── train_classifier.yml
└── requirements.txt
```

## 数据集目录要求

```text
/hy-tmp/DAS-dataset/
├── train/
│   ├── 01_background/
│   ├── 02_dig/
│   ├── 03_knock/
│   ├── 04_water/
│   ├── 05_shake/
│   └── 06_walk/
└── test/
    ├── 01_background/
    ├── 02_dig/
    ├── 03_knock/
    ├── 04_water/
    ├── 05_shake/
    └── 06_walk/
```

如果你的实际路径不是 `/hy-tmp/DAS-dataset`，修改：
- `options/train_das.yml` 中的 `datasets.root`
- `options/train_classifier.yml` 中的 `datasets.root`

## 训练顺序

### 1. 训练分类器

```bash
python train_das_classifier.py --opt options/train_classifier.yml
```

输出：
```text
checkpoints/classifier_plus/best.pt
checkpoints/classifier_plus/confusion_matrix_best.csv
```

### 2. 训练分离器

```bash
python train_das_separator.py --opt options/train_das.yml
```

输出：
```text
checkpoints/separator_plus/best.pt
```

### 3. 分离模型定量评估

```bash
python eval_das_separator.py \
  --opt options/train_das.yml \
  --checkpoint checkpoints/separator_plus/best.pt \
  --out_csv results/metrics/separator_metrics.csv \
  --max_batches 200
```

输出均值指标：
- `si_snr`
- `si_snri`
- `snr`
- `sdr`
- `mse`
- `mae`
- `pcc`

### 4. 分离后分类评估

```bash
python eval_das_classification.py \
  --sep_opt options/train_das.yml \
  --cls_opt options/train_classifier.yml \
  --sep_ckpt checkpoints/separator_plus/best.pt \
  --cls_ckpt checkpoints/classifier_plus/best.pt \
  --out_dir results/metrics \
  --max_batches 200 \
  --post_smooth_kernel 1
```

如果分离后分类准确率低于预期，可尝试：
```bash
--post_smooth_kernel 5
```
或：
```bash
--post_smooth_kernel 9
```

### 5. 对单个或文件夹中的 `.mat` 做分离

```bash
python separate_das.py \
  --opt options/train_das.yml \
  --checkpoint checkpoints/separator_plus/best.pt \
  --input /path/to/your/file_or_dir \
  --save_dir results/separated_mat \
  --cls_opt options/train_classifier.yml \
  --cls_ckpt checkpoints/classifier_plus/best.pt
```

### 6. 画定性分析图

```bash
python plot_das_results.py \
  --sample_dir results/separated_mat/某个样本名 \
  --save_dir results/figures/某个样本名
```

会输出：
- `mix_heatmap.png`
- `est_1_heatmap.png`
- `est_2_heatmap.png`
- `est_3_heatmap.png`
- `waveform_compare.png`
- `mix_vs_sum_est.png`

## 主要修改点

1. **分离网络改成真正的Conv-TasNet skip TCN结构**
   - 原版TCN只有残差相加，mask generator只能看到最后一层特征。
   - 新版每个dilation block输出skip，累加后生成mask，通常收敛更快、分离更稳定。

2. **背景类不再默认作为分离目标源**
   - `allow_background: false`
   - background改成mix中的加性噪声增强。
   - 这样模型主要学习 dig/knock/water/shake/walk 等真实扰动事件的分离，避免把背景强制分成一个源。

3. **随机混合增强更强**
   - 2/3源随机混合；
   - 源幅值随机；
   - 源时间平移；
   - 少量同类混合；
   - 背景噪声和白噪声；
   - 保持 `amp_range=0.01`。

4. **损失函数更适合12通道DAS**
   - global SI-SNR；
   - channel-wise SI-SNR，避免强通道支配；
   - waveform relative L1，稳定训练前期；
   - active outputs混合一致性；
   - unused outputs静音约束。

5. **分类器升级**
   - 从普通CNN换成残差2D-CNN；
   - 训练时加入噪声、平移、时间遮挡增强；
   - 加权采样缓解类别数量不完全均衡；
   - label smoothing增强泛化。

## 想实验4源分离

同时修改两个地方：

`options/train_das.yml`：
```yaml
datasets:
  max_sources: 4
net_conf:
  max_sources: 4
```

如果显存不足，把 `batch_size` 从8降到4。
