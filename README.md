# arneas — 基于线性自回归模型的图像压缩编码器

**ar-nearlossless-codec** · 逐像素自回归预测编码 · 近无损图像压缩 · 18 参数

一种极简参数（**三通道共 18 个参数**）的逐像素自回归近无损图像压缩方法：
**二维因果线性预测 → 均匀量化 → 算术熵编码**，完整思路与实验记录见 [PAPER.md](./PAPER.md)。

| 方法 | 码率 | 画质 | 相对 PNG |
|---|---|---|---|
| **本方法（Kodak 24 张镜像平均）** | **3.781 bpp** | **51.2 dB** | **−17.8%（24/24 全部低于 PNG）** |
| PNG（无损） | 4.602 bpp | 无损 | 基线 |

> Kodak 公开测试集镜像（500×500，24 张，可复现）：近无损（51 dB）画质下平均码率低于 PNG 17.8%（24/24）；全尺寸原始影像压缩比达 9.4 倍。

## 核心亮点

- 🪶 **极简**：单图最小二乘预测器，绿/红/蓝通道各 5/6/7 个参数（共 18 个），随图传输；
- 🧮 **可解析**：量化失真 E[ε²]=q²/12、画质上限 PSNRmax=58.92−20lg q 均可解析推导并实验验证；
- 📉 **可解释路线**：从一维窗口失败到二维窗口突破的完整思考过程，含全部失败实验记录（条件熵、符号合并、游程、色度下采样）；
- 🔬 **可复现**：Kodak 公开数据集 + 固定随机种子，一条命令跑完全部数字；
- ⚡ **快速**：numba 编译闭环，42 Mpx/s（较纯 Python 快约 1000 倍）。

## 快速开始

```bash
pip install numpy torch numba pillow

# Kodak 24 张完整基准（约 6 分钟，输出本仓库全部论文数字）
python3 experiments/kodak_bench.py

# 压缩单张图（q=2 近无损 / q=4 高画质）
cd graph_code
python3 main.py compress ../data/kodak/kodim01.png out.npz --K 16 --q 2 --steps 300 --window 2d
python3 main.py decompress out.npz out.png
```

## 为什么是 18 个参数？

本方法不训练"通用压缩器"，而是**对每一张图单独拟合**一个线性预测器：它回答的不是"一般图像该怎么猜"，而是"**这一张**图像该怎么猜"。详见 [PAPER.md §7.4](PAPER.md)。

## 结构

```
├── PAPER.md            # 完整论文（方法/思考过程/全部实验/负结果/定位）
├── LICENSE             # 双重许可（代码 CC BY-NC-SA；论文 CC BY-NC-ND）
├── NOTICE.md           # 使用与引用须知（防商用/专利/署名窃取）
├── graph_code/         # 核心代码：codec / entropy_codec / main / final_bench
├── experiments/        # 全部实验脚本（含负结果实验）
├── data/kodak/         # Kodak 公开测试集（24 张）
└── assets/             # 论文图表
```

## 许可

代码 **CC BY-NC-SA 4.0**（非商用）；论文 **CC BY-NC-ND 4.0**。详见 [LICENSE](./LICENSE) 与 [NOTICE](./NOTICE.md)。

---
*个人研究项目 · 由《动手学深度学习》(d2l) 启发、独立推演实现 · 2026*