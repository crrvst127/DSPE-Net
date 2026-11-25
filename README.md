# DSPE-Net

## 数据放置

| 目录 | 内容 |
|---|---|
| `data/development/` | 训练与验证序列，至少 10 个不同样本 |
| `data/evaluation/` | 独立评估序列，至少 2 个样本，首个样本为预测起点 |
| `data/history/` | 预测起点及之前的历史序列，至少 5 个样本 |

每个样本保存为一个二维 `.npy` 数组，形状为 `[高, 宽]`。所有样本使用相同网格，高和宽均为 4 的倍数。各目录内按时间顺序命名，例如 `001.npy`、`002.npy`。可选时间文件见[数据格式](data/README.md)。

## 致谢

Conv-LSTM 编解码器改编自 [ConvLSTM-PyTorch](https://github.com/jhhuang96/ConvLSTM-PyTorch)。`SINDY/pysindy/` 源自 [PySINDy](https://github.com/dynamicslab/pysindy)。
