# DSPE-Net

论文 *DSPE-Net: From Satellite Imagery to ISW Prediction via Local Symbolic Regularization* 的代码实现。

DSPE-Net 从单幅观测递推预测海洋内孤立波。Conv-LSTM 生成下一步粗预测，PINN 使用从历史序列辨识的局部方程校正波场，再将校正结果送入下一步。

## 快速开始

```bash
git clone https://github.com/crrvst127/DSPE-Net.git
cd DSPE-Net
python -m pip install -r requirements.txt
```

将自己的二维 NPY 样本放入以下目录。文件名按时间自然排序，具体格式见 [数据说明](data/README.md)。

```text
data/
  development/   五折训练与验证序列
  evaluation/    独立评估序列，首个样本作为预测起点
  history/       用于辨识局部方程的历史观测
outputs/         模型、算子和预测结果
```

从仓库根目录运行完整流程：

```bash
python run_pipeline.py --scenario numerical
```

卫星数据使用 `--scenario satellite`。默认使用最近 5 个历史样本辨识，训练五折，使用第 1 折最终权重递推 10 步。结果保存在 `outputs/run/predictions/`，重复运行时用 `--output` 指定新目录。正式训练每折默认运行 5000 轮（数模）或 1000 轮（卫星）。可用以下命令先检查本地数据与流程：

```bash
python run_pipeline.py --epochs 1 --iterations 1 --steps 2 --output outputs/quickcheck
```

单独预测使用 [predict_recursive.py](predict_recursive.py)，已有权重和算子时可直接运行 `python predict_recursive.py --help` 查看参数。

## 致谢

Conv-LSTM 编解码器改编自 [jhhuang96/ConvLSTM-PyTorch](https://github.com/jhhuang96/ConvLSTM-PyTorch)。`SINDY/pysindy/` 保留了原实验使用的 [PySINDy](https://github.com/dynamicslab/pysindy) 源码；当前符号辨识入口使用本仓库的 STRidge 实现。
