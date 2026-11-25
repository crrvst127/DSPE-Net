# DSPE-Net

DSPE-Net 用于海洋内孤立波的多步预测。项目包含 Conv-LSTM 预测、局部方程辨识和 PINN 校正三个模块，可从一幅观测出发，生成后续各步的波场预测。

## 数据准备

将同一网格上的二维 `.npy` 样本按时间顺序放入以下目录。详见[输入格式](data/README.md)。

| 目录 | 内容 |
|---|---|
| `data/development/` | 训练与验证序列，至少 10 个不同样本 |
| `data/evaluation/` | 独立评估序列，至少 2 个样本，首个为预测起点 |
| `data/history/` | 预测起点及之前的历史观测，至少 5 个样本 |

## 运行

在仓库根目录安装依赖并运行：

```bash
python -m pip install -r requirements.txt
python run_pipeline.py
```

预测结果保存在 `outputs/run/predictions/`。已有权重和方程时，可单独使用 [predict_recursive.py](predict_recursive.py) 运行预测。

## 致谢

Conv-LSTM 编解码器改编自 [ConvLSTM-PyTorch](https://github.com/jhhuang96/ConvLSTM-PyTorch)。`SINDY/pysindy/` 源自 [PySINDy](https://github.com/dynamicslab/pysindy)。
