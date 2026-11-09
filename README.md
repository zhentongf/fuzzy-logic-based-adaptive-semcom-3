# 基于模糊逻辑的自适应车联网语义通信实验

该项目实现两种公平对比的传输方式：Deep-JSCC 语义传输（卷积编码器—信道—解码器）与原始 CIFAR-10 图像直接传输。两种方式均经过相同的距离路径损耗和 Nakagami-m 平坦衰落信道。

## 实验逻辑

SNR、距离与带宽首先分别线性归一化到 `[0, 1]`：SNR 对应 0–20 dB、距离对应 0–400 m、带宽对应 10–1000 bps。Mamdani 模糊控制器的主要规则是：高 SNR、近距离且大带宽选择直接传输；低 SNR、远距离且小带宽选择语义通信。控制器输出的质量分数小于 YAML 中的阈值时，选择语义通信。

距离损耗为 `G(d) = (d0 / max(d, d0))^alpha`，Nakagami 功率增益服从 `Gamma(m, 1/m)`，接收端为 `sqrt(G(d)g)x + n`。零距离按参考距离 `d0` 计算，避免非物理的无限增益。

Deep-JSCC 网络结构改编自 [chunbaobao/Deep-JSCC-PyTorch](https://github.com/chunbaobao/Deep-JSCC-PyTorch)：5 层 PReLU 卷积编码器、功率归一化、信道层和对称转置卷积解码器；本项目将其原 AWGN/Rayleigh 信道替换为统一的 Nakagami 信道。

## 运行

安装依赖后，先生成默认的 10,000 行均匀随机链路条件：

```powershell
python main.py --config config.example.yaml --generate-csv
```

然后训练并评估：

```powershell
python main.py --config config.example.yaml
```

训练始终使用完整 CIFAR-10 训练集，并按 `train.dynamic_snr_db` 每个 epoch 更换一次 SNR、循环使用；评估使用 CIFAR-10 测试集前 `eval_samples` 张图像。每一行条件 CSV 代表一个独立链路状态，并在该状态下传输全部 `eval_samples` 张图像，保存平均 MSE 与平均 PSNR。结果同时包含 `adaptive`、`all_direct`（全部直接传输）与 `all_semantic`（全部语义传输）三种 `transmission_mode`。首次运行时 `torchvision` 会下载 CIFAR-10。训练日志保存为带时间戳的 `outputs/train_log_*.csv`，评估结果保存为带时间戳的 `outputs/adaptive_evaluation_*.csv`。

绘图与训练/评估分离。将 `eval_plot.eval_results` 修改为目标评估 CSV 后，运行 `main_graph.py` 生成 PSNR 三联图。

在 YAML 中修改 `run.train`、`run.evaluate`、`train.dynamic_snr_db` 和 `data.eval_csv` 即可满足不同实验设置。若只评估已有模型，将 `run.train: false`，并使 `model.checkpoint` 指向现有模型即可。
