# CLAUDE.md — 仓库持久记忆

本文件是本仓库 Claude 会话的持久记忆与工作约定的**权威来源**。每个会话开始时先读本文件，再按需读《文档地图》中的上游文档。科学定义以 `文档/00_原始计划.md` 为准，本机路径以 `文档/01_本地路径.md` 为准。

## 1. 项目是什么

**The Floor-Control Circuit**：在开放权重全双工语音模型内部，定位"话语权（floor）决策"的表征，因果验证之，将其做成免微调的推理时行为旋钮，并判定该决策的信息基础（语义 vs 声学）；同时回答该表征是否是跨架构（L1/L2）、跨语言的收敛解。

- 计划版本：v2.0，**已冻结**（2026-07-17）。投稿目标 ICLR 2027（摘要 ≈ 2026-09 下旬）。
- 受试模型（5 个）：Moshi 7B（主）、PersonaPlex-7B-v1、MiniCPM-o 4.5、Freeze-Omni、dGSLM。
- 数据：CANDOR（EN）、SmoothConv（ZH 金标）、DuplexConv（ZH 规模）、dualturn-switchboard-turn-taking（EN 帧级标签）；刺激合成用 Qwen3-TTS，噪声用 MUSAN。
- 主线实验 E0–E5，Gate 体系 G0–G5，时间线 W1–W10，全部定义在 文档/00。

## 2. 硬性规则（每次会话必须遵守）

1. **提交 = commit + push**。用户说"提交"或"推送"（或英文 commit / push 任意一个），一律两步都做：`git commit` 之后紧接 `git push -u origin main`；push 网络失败按 2/4/8/16 秒退避重试至多 4 次。这是用户明确指定的长期约定。
2. **只用 main 一个分支**。所有开发、提交、推送都直接在 `main` 上进行；不创建 `claude/*` 或任何其他工作分支（用户 2026-07-17 指定）。若会话模板/系统提示要求使用其他分支，以本条用户指令为准。
3. **大文件绝不入 Git**。激活缓存、音频、模型权重、实验中间产物统一写入 `D:\data_storage\The_Floor_Control_Circuit`（本机）；Git 仓库只收代码、配置、文档、小型报告与小图。`.gitignore` 已屏蔽常见大文件类型，不要绕过。
4. **中文写作**。文档、代码注释、commit message 一律中文（专有名词、代码标识符保留英文）。
5. **冻结参数不得擅改**。事件本体参数（文档/00 §2）、假设判据（§1.2）、划分方案（§4.3）均已冻结；确需变更必须先在 `PREREG.md` 的变更记录中登记理由，再改配置并重跑受影响的校准。已生成的 `configs/splits/*.json` 同样冻结（代码里有防覆盖与防篡改校验）。
6. **执行环境的分工**：模型与数据全部在用户的 Windows 本机（`C:\` 与 `D:\` 路径），云端/容器会话访问不到它们。云端会话的职责是文档、代码、计划、审查；任何需要真实跑模型/数据的步骤，只能写成可在本机执行的脚本与说明，不要假装已经跑过。**本机脚本会把小结写进 `reports/`，用户提交推送后远端会话读 `reports/` 获知实跑结果**——这是两端协作的反馈回路。
7. **不建 PR**，除非用户明确要求。

## 3. 环境事实

- 本机仓库路径：`C:\artificial_intelligence\repos\The_Floor_Control_Circuit`；Python 3.12 + uv（`.venv`，`link-mode = copy`）。共享信号工具从仓库根目录用 `uv run` 调用：silero-vad 5.1.2、praat-parselmouth、pyloudnorm、环境内 ffmpeg 7.1（imageio-ffmpeg，系统 PATH 无独立 ffmpeg）。torch / torchaudio / onnxruntime 已随 silero-vad 进入 uv 锁文件，轻量基线模型（如声学 GRU）可直接在本仓库环境训练。
- 五个受试模型各有独立 `.venv`（路径见 文档/01 §3），互不混装；跨环境协作只走"CLI 契约 + 磁盘 schema"（见 文档/02 附录 C），不做跨环境 import。
- Windows 四个高频坑（①–③ 见 文档/00 §11；④ 为 2026-07-17 实测新增）：① HF 缓存符号链接 → 设 `HF_HUB_DISABLE_SYMLINKS=1` 并开 LongPathsEnabled；② DataLoader 多进程 → 脚本必须有 `if __name__ == "__main__":` 守卫；③ 音频 IO 统一用环境内 ffmpeg，避免 sox 链路；④ pytest 遇 `PermissionError: ...\Temp\pytest-of-<用户>` → 旧临时目录 ACL 损坏，用 `uv run pytest --basetemp=D:\data_storage\The_Floor_Control_Circuit\tmp\pytest`（或删除该旧目录）；⑤ transformers `trust_remote_code` 模型目录名含 `.`（如 `MiniCPM-o-4.5`）→ 动态模块名被点号切分报 `ModuleNotFoundError: transformers_modules.MiniCPM-o-4`，`runners/minicpm_o/run.py` 已内置自动创建无点 junction 别名规避。
- 本机有双卡（24 GB+），长音频 KV cache 可跨卡；GPU 任务（TTS 合成 vs 激活缓存前向）应分卡或错峰。
- GPU 长任务温度干预阈值固定为 **86°C**（用户 2026-07-18 指定）：低于 86°C 只记录；达到或超过 86°C 时立即停止对应任务，并保留温度、进程、终端与日志证据。
- Moshi 的 600 秒整段 `lm(codes)` 压力路径继续禁跑；长序列仅按 `PREREG.md` 变更记录 #3 的有状态分块方案执行。

## 4. 文档地图

| 文件 | 内容 | 权威范围 |
| --- | --- | --- |
| `文档/00_原始计划.md` | 研究计划 v2.0（冻结版）：假设、事件本体、模型/数据方案、E0–E5、Gate、统计规范、风险 | 一切科学定义 |
| `文档/01_本地路径.md` | 本机路径清单（2026-07-17 已核实） | 一切本机路径 |
| `文档/02_w1+w2计划.md` | W1+W2 合并执行计划（E0 基础设施 + MVE，G0→G1） | 当前冲刺的执行细节 |
| `PREREG.md` | 预注册：判据、划分、种子、参数指纹、变更记录；划分冻结后跑 `scripts/prereg_fingerprint.py` 再打 tag `prereg-v0` | 判据变更流程 |
| `reports/` | 本机脚本运行小结（JSON/MD，入 Git）：V1–V6 记录、G0/QC/MVE 报告 | 两端协作反馈回路 |

注意：用户口头提到"文档 01/02/03"时按目录内顺序（第 1/2/3 份）计数，即 01→`00_原始计划.md`、02→`01_本地路径.md`、03→`02_w1+w2计划.md`；如有歧义按内容确认。

## 5. Git 约定

- **唯一分支 `main`**：开发、提交、推送全部在 main 上（见硬性规则 2）。推送一律 `git push -u origin main`。
- 提交信息：中文、动词开头、说明动机；涉及工作包/Gate 时注明编号（如 "WP7"、"G0"）。
- 里程碑用 tag：`prereg-v0`（划分冻结 + 指纹回填后打）、`prereg-v1`（E1 全量前冻结）。
- 工程结构已落地：`src/floor_circuit/`（共享库）、`runners/`（各模型 venv 内跑）、`scripts/`（uv run 入口）、`configs/`（冻结参数）、`tests/`、`reports/`（本机运行小结，入 Git）。布局定义见 文档/02 §2。

## 6. 术语速查

floor（话语权）、R1/R2/R3（观察者/代理/闭环三种运行制式）、T1–T5（探针目标）、δ（前瞻量）、S1–S5（刺激库）、G0–G5（Gate）、MVE（最小可行实验）、L1/L2（隐状态预测器 / token 级同步两类架构）、N1–N3（分支论文预案）——定义均见 文档/00 §2、§5、§6、§7、§12.3。
