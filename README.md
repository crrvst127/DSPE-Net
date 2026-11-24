# DSPE-Net

论文 *DSPE-Net: From Satellite Imagery to ISW Prediction via Local Symbolic Regularization* 的代码实现。

DSPE-Net 从单幅观测递推预测海洋内孤立波。Conv-LSTM 生成下一步粗预测，PINN 使用从历史序列辨识的局部方程校正波场，再将校正结果送入下一步。

## 安装与运行

在仓库根目录安装依赖：

```bash
python -m pip install -r requirements.txt
```

输入为配准到同一网格的二维 NPY 数组。清单按时间顺序每行列出一个样本路径，时间文件按相同顺序列出采集时刻（分钟）。以下文件名为示例：

```bash
# 从历史序列辨识局部方程
python SINDY/prepare_sequence.py history.txt history.mat --times times.txt
python -m SINDY.identify_equations history.mat operators.json --scenario numerical

# 训练 Conv-LSTM，单独留出递推评估序列
python CLSTM/train_convlstm.py development.txt evaluation.txt models --scenario numerical

# 从一个观测起点递推预测，默认输出 10 步
python predict_recursive.py input.npy models/fold_1/final.pt operators.json predictions --scenario numerical --window 0 --initial-time 60
```

`--initial-time` 填输入观测在同一时间轴上的实际时刻，`--window` 选择截至该时刻已结束的历史窗口。卫星实验将场景设为 `satellite`；观测间隔不等时，用 `--intervals` 指定各步时长。

## 致谢

Conv-LSTM 编解码器改编自 [jhhuang96/ConvLSTM-PyTorch](https://github.com/jhhuang96/ConvLSTM-PyTorch)。`SINDY/pysindy/` 保留了原实验使用的 [PySINDy](https://github.com/dynamicslab/pysindy) 源码；当前符号辨识入口使用本仓库的 STRidge 实现。
