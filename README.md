# The Floor-Control Circuit

**Locating and Steering Turn-Taking Decisions in Full-Duplex Speech Models**

在开放权重全双工语音模型内部，定位"话语权（floor）决策"的表征，因果验证之，将其做成免微调的推理时行为旋钮，并判定该决策的信息基础（语义 vs 声学）；同时回答该表征是否是跨架构（L1/L2）、跨语言的收敛解。

- 计划版本：v2.0（**冻结**，2026-07-17）· 投稿目标：ICLR 2027
- 当前阶段：W1+W2 合并冲刺（E0 基础设施 + MVE，Gate G0→G1），执行计划见 [`文档/02_w1+w2计划.md`](文档/02_w1+w2计划.md)

## 研究概要

四个核心问题：floor 决策的表征**在哪里**（E1 探针与几何）、**是否因果**（E2 patching/消融/剂量）、**能否操控**（E3 推理时 steering 与五旋钮范式对比）、**被什么信息驱动**（E4 语义 vs 声学 2×2 设计）；外加**收敛性**（E5 跨模型/跨架构/跨语言）。假设 H1–H5 与预注册判据见 [`文档/00_原始计划.md`](文档/00_原始计划.md) §1.2。

### 受试模型（5 个，横跨 L1/L2 两类架构与中英两语）

| 模型 | 架构类 | 角色 |
| --- | --- | --- |
| Moshi 7B | L2 多流 | 主受试，E1–E4 全量 |
| PersonaPlex-7B-v1 | L2 多流 | 同架构对照、复现层、方向移植（E5） |
| MiniCPM-o 4.5 | 端到端 omni | 中文原生受试（E1–E3 缩减网格）+ 跨架构收敛性 |
| Freeze-Omni | L1 隐状态预测器 | 正对照 + 层浮现曲线 + E3 模块化阈值旋钮基线 |
| dGSLM | L2 无文本双塔 | 仅探针级（E1 + E4-lite，文本支架必要性对照） |

### 数据与刺激

CANDOR（EN 主战场）· SmoothConv（ZH 专家金标）· DuplexConv（ZH 规模）· dualturn-switchboard-turn-taking（EN 12.5 Hz 帧级标签，G0 校准集）· 刺激库 S1–S5 由 Qwen3-TTS 合成（EN+ZH），噪声对照用 MUSAN。

## 仓库结构

```
文档/                 研究计划与执行文档（00 原始计划 / 01 本地路径 / 02 W1+W2 执行计划）
CLAUDE.md            AI 会话持久记忆与工作约定（权威）
AGENTS.md            智能体核心规则（精简版）
pyproject.toml       uv 项目配置（Python 3.12，共享信号处理工具链）
uv.lock              锁文件
```

代码目录（`src/floor_circuit/`、`runners/`、`configs/`、`scripts/`、`tests/`、`reports/`、`PREREG.md`）随 W1 脚手架落地，布局定义见 文档/02 §2。

## 环境与运行

- 实验一律在**本机 Windows** 执行：仓库位于 `C:\artificial_intelligence\repos\The_Floor_Control_Circuit`，模型与数据路径以 [`文档/01_本地路径.md`](文档/01_本地路径.md) 为准（2026-07-17 已核实）。
- 本仓库共享环境：`uv sync` 后用 `uv run` 调用（silero-vad、praat-parselmouth、pyloudnorm、环境内 ffmpeg 7.1、torch）。
- 五个受试模型各有独立 `.venv`，与本仓库环境隔离；跨环境只走 CLI 契约 + 磁盘 schema。
- 大型中间产物（激活缓存、刺激音频、生成音频）统一写入 `D:\data_storage\The_Floor_Control_Circuit`，**不入 Git**。
- Windows 注意事项（符号链接、`__main__` 守卫、ffmpeg 链路）见 文档/00 §11。

## 工作流

决策路径由 Gate 体系驱动（G0 事件管线校准 → G1 MVE 裁决 → G2 层/方向锁定 → G3/G3b 因果 → G4 语义敏感性 → G5 投稿裁决），完整定义见 文档/00 §7。预注册判据在 E1 全量前以 `PREREG.md` + git tag 冻结。

AI 协作约定（提交=commit+push、大文件不入库、中文写作、冻结参数变更流程等）见 [`CLAUDE.md`](CLAUDE.md)。
