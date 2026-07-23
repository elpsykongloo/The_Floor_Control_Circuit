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
8. **工具缓存统一外置**：测试与静态检查默认使用
   `pwsh -NoProfile -File scripts/run_checks.ps1 <all|pytest|ruff|mypy>`；pytest 临时数据固定写入
   `<data_root>\tmp\pytest`，pytest/ruff/mypy 缓存固定写入 `<data_root>\cache\tooling`。禁止在仓库根目录
   创建 `.pytest_tmp_*` 或把 `--basetemp` 指回仓库；历史缓存统一用
   `pwsh -NoProfile -File scripts/clean_repo_caches.ps1` 清理。

## 3. 环境事实

- 本机仓库路径：`C:\artificial_intelligence\repos\The_Floor_Control_Circuit`；Python 3.12 + uv（`.venv`，`link-mode = copy`）。共享信号工具从仓库根目录用 `uv run` 调用：silero-vad 5.1.2、praat-parselmouth、pyloudnorm、环境内 ffmpeg 7.1（imageio-ffmpeg，系统 PATH 无独立 ffmpeg）。torch / torchaudio / onnxruntime 已随 silero-vad 进入 uv 锁文件，轻量基线模型（如声学 GRU）可直接在本仓库环境训练。
- 五个受试模型各有独立 `.venv`（路径见 文档/01 §3），互不混装；跨环境协作只走"CLI 契约 + 磁盘 schema"（见 文档/02 附录 C），不做跨环境 import。
- 本地模型适配高频坑（①–③ 见 文档/00 §11；④ 起为本机实测新增）：① HF 缓存符号链接 → 设 `HF_HUB_DISABLE_SYMLINKS=1` 并开 LongPathsEnabled；② DataLoader 多进程 → 脚本必须有 `if __name__ == "__main__":` 守卫；③ 音频 IO 统一用环境内 ffmpeg，避免 sox 链路；④ pytest 遇 `PermissionError: ...\Temp\pytest-of-<用户>` → 旧临时目录 ACL 损坏，统一改走 `scripts/run_checks.ps1 pytest`，由脚本使用 D 盘固定临时根目录和进程锁；⑤ transformers `trust_remote_code` 模型目录名含 `.`（如 `MiniCPM-o-4.5`）→ 动态模块名被点号切分报 `ModuleNotFoundError: transformers_modules.MiniCPM-o-4`，`runners/minicpm_o/run.py` 已内置并核验无点 junction 别名；MiniCPM-o 4.5 上游纯读出仍会无条件初始化 TTS，runner 在 `generate_audio=false` 时临时跳过该无用初始化；⑥ PersonaPlex 的 Mimi 未暴露 `frame_size`，runner 仅在 `sample_rate/frame_rate` 可精确整除时推导帧长（本机为 24000/12.5=1920）。
- 本机有双卡（24 GB+），长音频 KV cache 可跨卡；GPU 任务（TTS 合成 vs 激活缓存前向）应分卡或错峰。
- E1 GPU 长任务由用户人工保障散热（PREREG #17，2026-07-22）；运行器不读取温度、不轮询 `nvidia-smi`，也不执行自动温控中止。
- Moshi 的 600 秒整段 `lm(codes)` 压力路径继续禁跑；长序列仅按 `PREREG.md` 变更记录 #3 的有状态分块方案执行。
- **上下文截断规程（PREREG #11）**：任何受试模型的 MVE/E1 探针分析窗必须先登记该模型的 context 上限并截断到规格内（Moshi：context=3000 步 → 可用标签步 0..2998；权威 `src/floor_circuit/mve/alignment.py`）。超上下文运行会出现 sink 淘汰尖峰与公共吸引态塌缩（2026-07-19 实测），其行不得进入任何判据。
- Moshi 正式 greedy 缓存按 `PREREG.md` #10 使用双卡各一个持久会话进程：会话级分片、双声道编码复用、显卡驻留缓冲与单步 CUDA Graph；历史有效 greedy 缓存由版本集合护栏续用。
- **G0 已全线退役（PREREG #16，2026-07-21 用户裁决）**：标记为初期探索的不成熟想法（判据建立在 Mimi 重建音频上，不具真值性质），一切修复/确认/质检/论文义务取消；`wp1_g0_*` 工具链封存不再运行，历史结果只留 PREREG 登记。配套登记事实：**CANDOR 无人工话轮标注**（transcribe_output.json = AWS ASR 原始输出；Audiophile/Cliffhanger/Backbiter 均为算法化话轮模型）——CANDOR 标签效度支柱 = T4 人工核验（#13）+ G1 优势佐证，Backbiter 仅为算法化同侪参考。
- **DuplexConv 完整发布物已闭合（PREREG #31，2026-07-23）**：固定上游提交 `0bb99da7ab7a2f6f86d6b23df92c9383e711d09a`；Windows 本机把上游 `Edu`、`edu`、`none_Edu` 分别映射为 `Edu_upper`、`edu_lower`、`none_Edu`，避免大小写路径冲突。全量库存为 189 个音频 TAR、3 个 `jsons.tar.gz`、93,709 对 WAV/JSON，总计 1,640,733,127,901 字节（1,528.05 GiB，含标注包）。数据根目录的 `LOCAL_LAYOUT.md` 与 `local_layout.json` 为本机权威布局；完整审计见 `reports/duplexconv_release_audit.md` 与 `reports/duplexconv_release_audit.json`。#16(b) 的下载等待条件已经解除；DuplexConv 训练清单仍须由专用 TAR 成员索引器另行生成、登记并冻结，现有 `wp2_freeze_splits.py` 不得直接用于该发布物。
- **E1 缓存 v2（PREREG #16(c)/#17）**：`wp_e1_cache_plan.py` 生成计划 v2（主计划 + 243/257 加权双分片，plan_id 内容寻址，音频/权重/配置全量摘要 + 240 s PCM 前缀指纹）；`runners/moshi/run_batch.py --plan <shard>` 自动走 v2 分支（全 32 层连续堆叠 [T,L,H] 分片、512 步双缓冲、下一会话音频预取、Mimi 0.08 秒单帧与 LM 单步各用 CUDA Graph 并逐段释放、资源遥测、断点续跑）；运行器不含温控路径。`wp_e1_cache_audit.py` 审计、`wp_e1_cache_parity.py` 与历史 MVE 前 3000 步逐位核验；zarr 摄取支持堆叠布局流式写入。**全量统一重跑，不复用 MVE 4 层缓存**；旧 `mve_r1_greedy/` 封存禁改。正式缓存已闭合（2026-07-22 审计 passed，1000/1000 路 789.5 GB）。
- **E1 探针网格（PREREG #18/#19/#20/#21，已实现待本机运行）**：`wp_e1_probe_grid.py` 五阶段 labels→parity→acoustic→run→finalize；10 规格 × 32 层 × 3 种子；G2 主目标预冻结 = T4；种子 = 会话级 90% 子抽样；多分类 macro-OVR AUC；torch-GPU 训练器有 sklearn 奇偶校验硬门。#19 按本机 31.75 GiB 内存改为单进程双卡按规格并行，每层训练 800 路与评估 200 路分阶段载入，正式命令用 `--num-shards 1 --devices cuda:0,cuda:1`；Windows 仓库 uv 环境锁定官方 CUDA 12.8 torch。#20 修正生产标签后缀为 `<sid>.labels.parquet`；本机 500 个 E1 会话实测缺 339 个标签，须按 `e1_probe/missing_label_sessions.txt` 调用独立的 `wp_e1_run_missing_events.py --session-list` 精确增算，保持权威事件脚本源码指纹不变。摄取用 `wp_e1_ingest.py`（勿用 wp5 批处理跑 E1 全量）。产物根 `<data_root>/e1_probe/`。
- **E1 格子落盘（PREREG #21）**：逐会话数组、键名、dtype 与原子替换语义不变，格子写入改为未压缩 NPZ；本机 D 盘 100 万行代表格实测写入由 0.352 s 降至 0.043 s，约增加 26% 文件体积。
- **E1 前置并行（PREREG #22/#24/#25/#26）**：acoustic 只读取冻结前 240 s，支持原子 NPY 与 `--acoustic-workers`；parity 可在训练前 20 会话的标签和 40 路 Zarr 就绪后提前运行。声学 1000/1000 已完成、GPU1 parity 已通过，标签四分片 339/339 已闭合，Zarr 摄取 1000/1000 passed。缺失标签逐通道载入完整波形以收紧峰值内存；摄取 3 worker 与标签 4 个互斥分片重叠。`--stage baselines --device cuda:1` 可提前生成 hazard 与声学 GRU 正式格，主网格按格子断点直接复用。T5 正式五分类须排除审计专用 `OVERLAP_UNRESOLVED=5`，hazard 仍保留该状态并按当前重叠处理。
- **E1 双卡动态调度（PREREG #27）**：正式 L0/L1 取证显示静态规格分卡会令 GPU1 每层提前约 80–98 s 清空队列；训练侧改为两个固定设备线程从共享的 `(spec, seed)` 优先队列动态取任务，评估侧按规格组动态取任务。逐格样本、C 路径、正式重训和原子断点不变；日志记录每任务设备/耗时及每层载入、拟合、评估分段耗时。
- **E1 层间预取（PREREG #28）**：冻结训练行映射的逐角色并集只占完整层 52.99%（约 9.70/18.31 GiB）；层缓存载入后仅保留原始行号与对应 fp16 值，装配时严格反查。双卡拟合期间只预取下一待运行层，按“双压紧训练层 + 6 GiB”做物理内存护栏，不满足即退回同步载入；当前训练缓存释放后才载入完整评估层。
- **E1 预取提交量护栏（PREREG #29）**：Windows 会把 CUDA 缓存分配计入提交量；每个 T5 单元返回 CPU 探针后立即释放所属设备的无引用 CUDA 缓存。后台预取若仍触发 `MemoryError`，同一进程必须自动改用完整层同步载入并永久关闭后续预取，保持 #27 动态调度继续运行，工程优化异常不得终止正式网格。
- **E1 正式收口与探索转向（PREREG #30/#33/#34）**：Moshi R1 正式 G2 = **fail**（T4@L29 优势 +0.0659 [+0.0581,+0.0743]；top-3 交集与方向余弦通过，唯一失败项 = PCA 有效秩 128 ≫ 16；#30 事后诊断过线位置 57–84、前 16 PC 仅解释 38–45% 方差）。用户裁决开启探索轨道（不必完全遵守预注册协议；G2=fail 原样保留、探索/正式分轨登记）。**E1-X 套件（#33，与 #32 几何解剖支线互补）**=`wp_e1x_suite.py` 七阶段（geometry→leadtime→decompose→trajectory→t2h→anatomy→finalize，`src/floor_circuit/e1x/`，产物根 `<data_root>/e1x/`，回传 `wp_e1x_summary.json`+`e1x_探索套件报告.md`）；**E2-lite（#34，= 文档/04 §4.2 方向注入升级案的操作化）**=方向级因果试点先于换位点重试：`wp_e2_lite_plan.py`（20 会话×13 条件=260 运行）→ `runners/moshi/run_steer.py`（moshi venv，R2 生成 + L29 注入 α·s_v·v̂，先 `--probe-api`/`--limit 2` 冒烟）→ `wp_e2_lite_analyze.py`（回传 `wp_e2_lite_summary.json`+`e2_lite_行为报告.md`）；α=0 基线顺带缓存 L28–31 激活（R2 观察数据）。causal_eval 25% 侧继续绝不读取。
- **E1 摄取单块单写（PREREG #23）**：stacked 布局先把 6 个源分片顺序拼接一次，再把每层 `(3000,4096)` Zarr 单块写一次；最终数组布局、压缩、dtype 与值不变，理论块写入量由约 4.72 GB/路降至 0.79 GB/路。单路 Windows 瞬时占用按 2/4/8/16 s 退避重试并只清理本路 `.partial`；回环改为保存层逐分片比较。旧摄取在 108/1000 因 WinError 5 退出后已从断点恢复。
- **E1 几何解剖支线（PREREG #32/#35/#36/#37/#38，探索性、已实现待本机运行）**：用户 2026-07-23 授权对 G2 有效秩失败做非裁决解剖。核心恒等式：T4 二分类探针的评估 AUC 0.8348 本身就是原始空间单方向 v\*=(w/σ) 的一维投影 AUC（正式 rank-1 PCA 仅 0.575），故有效秩 128 度量的是 v\* 与方差主轴的错向而非信息维数。`wp_e1_geometry_autopsy.py`（协议 v2，绑定 script/probe_gpu/grid/engine/alignment 源码 + 行域 + **全 768 fit 内容聚合 SHA** + 逐任务 fit 摘要）四阶段：directions（零激活方向族）→ spectrum（能量谱/**坐标 participation ratio（描述量）**/补空间重训/Mimi 主轴鉴定/转向包 `steering_L{29,30,31}.npz`，e1x-directions-v1 兼容、随机方向数=3、proj_std 与聚合 diffmeans 均用**去重训练并集**与 E1-X 同口径、可供 `wp_e2_lite_plan.py --directions-npz` 消费）→ trajectory（事件锁定轨迹，sup-t 同时带 + 连续 ≥3 步、查全部随机方向、缓存绑 steering SHA）→ finalize（八行判读矩阵，coordinate_distribution 行）。**核心纪律（#35/#36/#37/#38，均经用户反例 + 云端实证）：均值信号线性擦除秩恒为 1（LEACE）、方向剔除法（INLP r₁）受 Σ⁻¹d 旋转混淆、原生坐标特征折受跨坐标噪声抵消混淆、白化剔除塌缩为 mean-projection tautology、participation ratio 本身尺度非不变（重缩放两坐标即 1.0→0.0075）且经验 d̂ 受有限样本抬高（真支撑=1 的 PR/D 随 SNR 0.002–0.34）——五种冗余/分布性度量全部有混淆，"分布式 vs 局部化"在读出向量几何上不可辨识，一律降为描述量，交因果组件级证据（换位点 patching + E2 注入）裁决；换位点 patching 恢复无条件执行（几何不得取消，撤销 #32 起的条件执行降级）；秩-1 收窄为"单一 Fisher 判别轴"（不证唯一/无冗余）；top16 判读改嵌套 inner_val 部署 C\* 处配对掉幅（去评估集选择偏差，新增 top16_drop_stats）；均值投影自检/未收敛拟合拒绝出矩阵**。G2=fail 不变；判读矩阵（coordinate_distribution 出 descriptive_non_decisive）、E2 方向注入升级与换位点无条件执行见 文档/04 v0.5。

## 4. 文档地图

| 文件 | 内容 | 权威范围 |
| --- | --- | --- |
| `文档/00_原始计划.md` | 研究计划 v2.0（冻结版）：假设、事件本体、模型/数据方案、E0–E5、Gate、统计规范、风险 | 一切科学定义 |
| `文档/01_本地路径.md` | 本机路径清单（2026-07-17 已核实） | 一切本机路径 |
| `文档/02_w1+w2计划.md` | W1+W2 合并执行计划（E0 基础设施 + MVE，G0→G1；**已收口**：G1=full_e1） | W1+W2 历史执行记录 |
| `文档/03_w3+w4计划.md` | W3+W4 执行计划（E1 全量五模型 → G2；草案 v0.1 待批准冻结） | 当前冲刺的执行细节 |
| `文档/04_e1_几何解剖与探索路线.md` | E1 几何解剖独立分析与探索路线（PREREG #32 支线：再解读、实验 A–G、判读矩阵、E2 升级案） | 几何解剖支线的实验定义 |
| `PREREG.md` | 预注册：判据、划分、种子、参数指纹、变更记录；划分冻结后跑 `scripts/prereg_fingerprint.py` 再打 tag `prereg-v0` | 判据变更流程 |
| `reports/` | 本机脚本运行小结（JSON/MD，入 Git）：V1–V6 记录、G0/QC/MVE 报告 | 两端协作反馈回路 |

注意：用户口头提到"文档 01/02/03/04"时按目录内顺序计数，即 01→`00_原始计划.md`、02→`01_本地路径.md`、03→`02_w1+w2计划.md`、04→`03_w3+w4计划.md`；如有歧义按内容确认。

## 5. Git 约定

- **唯一分支 `main`**：开发、提交、推送全部在 main 上（见硬性规则 2）。推送一律 `git push -u origin main`。
- 提交信息：中文、动词开头、说明动机；涉及工作包/Gate 时注明编号（如 "WP7"、"G0"）。
- 里程碑用 tag：`prereg-v0`（划分冻结 + 指纹回填后打）、`prereg-v1`（E1 全量前冻结）。
- 工程结构已落地：`src/floor_circuit/`（共享库）、`runners/`（各模型 venv 内跑）、`scripts/`（uv run 入口）、`configs/`（冻结参数）、`tests/`、`reports/`（本机运行小结，入 Git）。布局定义见 文档/02 §2。

## 6. 术语速查

floor（话语权）、R1/R2/R3（观察者/代理/闭环三种运行制式）、T1–T5（探针目标）、δ（前瞻量）、S1–S5（刺激库）、G0–G5（Gate）、MVE（最小可行实验）、L1/L2（隐状态预测器 / token 级同步两类架构）、N1–N3（分支论文预案）——定义均见 文档/00 §2、§5、§6、§7、§12.3。
