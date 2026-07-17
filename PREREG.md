# PREREG · 预注册（v0）

状态：**v0 草案**（2026-07-17）。划分冻结（`scripts/wp2_freeze_splits.py`）并回填指纹
（`scripts/prereg_fingerprint.py`）后打 tag `prereg-v0`；E1 全量启动前升级冻结为 `prereg-v1`。
判据全文以 `文档/00_原始计划.md` §1.2 为权威，本文登记数值判据、划分、种子与变更流程；
探索性分析一律在报告中单列，不得事后并入确证性判据。

## 1. 假设与数值判据（摘自 00 §1.2，冻结）

- **H1 可解码性与前瞻性**：最优层线性探针 AUC − max(声学基线, 编解码/编码器基线, hazard 基线) ≥ **0.05**，
  会话级 cluster-bootstrap 95% CI 下界 > 0；在模型自身决策时钟 ≥ 3 步前瞻仍成立。
- **H2 因果性**：patching/方向注入翻转率 ≥ **50%** 且高出范数匹配随机方向 ≥ **30 pt**；剂量单调
  （Spearman ρ > 0.8）；内容方向对照无效（双重分离）。
- **H2b 整合器动力学**：漏积分拟合时间常数预测让位延迟分布（KS 检验不拒绝）。
- **H2c 通路中介**：top-k 用户流监测头中介效应比 ± CI 可量化；发言门控直接 logit 归因通路给出。
- **H3 可控性**：存在 α 使停顿处理 TOR ↓ ≥ **30%**、打断响应 TOR 不降、GPT 内容分 ↓ ≤ **5%**、
  UTMOS ↓ ≤ 0.15；SDT d′ 保持率不劣于 PAD/静音 logit 偏置。
- **H4 信息基础**：语义最小对差分显著（置换 p < 0.01），能量匹配对照不显著；2×2 交互项显著。
  全线不成立 → N3 反例论文路线。
- **H5 收敛性**：(a) 同架构方向可移植；(b) 跨架构 CKA/Procrustes 显著高于随机匹配；
  (c) 双语模型内跨语言探针迁移 AUC 降幅 ≤ 0.05。

## 2. Gate 判据（摘自 00 §7，冻结）

| Gate | 判据 | 过 → | 不过 → |
| --- | --- | --- | --- |
| G0 | 三层（变更记录 #1，已批准）：层1 官方算法逐帧全等（=1.000，固定 commit 2d0db21）；层2 VAD 一致性审计报告；层3 端到端 macro-F1 ≥ 门槛（官方 val 295 会话估计后冻结） | MVE | 修实现（≤1 周，参数不动） |
| G1 | 最优层 AUC 优势 ≥ +0.05 且 CI 下界 > 0（T1/T4 取较大者） | E1 全量 | +0.02~0.05 备胎 MVE；全部 <+0.02 → N1 |
| G2 | 跨 3 种子 top-3 层重叠 ≥ 2；有效秩 ≤ 16；方向余弦 ≥ 0.8 | E2 | 换位点重试 → N1' |
| G3 | H2 四判据 | E3 + G3b | N2 |
| G3b | top-8 头中介 ≥ 70% 且 Jaccard ≥ 0.6 | Tier-3 入正文 | Tier-3 后续工作 |
| G4 | H4 语义敏感 | 摘要 A | N3（摘要 B） |
| G5 | 完成度与日期 | ICLR 2027 | ICML 2027 |

## 3. 划分与种子（00 §4.3，生成即冻结）

- CANDOR：60/15/25（probe_train / probe_val / causal_eval），会话级；**causal_eval 侧在 E2 之前绝不读取**。
  已冻结（2026-07-17）：1,656 会话 → 994/248/414。
- SmoothConv：70/30（train / eval_sdt）；已冻结（2026-07-17）：2,503 会话 → 1,752/751。
  DuplexConv 仅训练侧（标注形态待定位，暂缓冻结）；DualTurn-SWB 沿官方 `splits.json`。
- **DualTurn 集合纪律（2026-07-17，变更记录 #1）**：G0 层3 门槛用官方 **val（295 会话）**估计；
  test 的前 20 个会话（按目录序）已在首轮联调中消耗，登记为**探索集**；
  其余 **118 个 test 会话锁定为一次性确认集**，门槛冻结前任何分析不得读取。
- 划分种子：**20260717**；刺激主种子：**20260717**（configs/stimuli.yaml master_seed）。
- MVE 抽样：probe_train 前 160 段 + probe_val 前 40 段（按冻结划分文件内的排序），每段前 10 min。
- R2 生成制式（v0 默认，E1 冻结前可经变更记录修订）：温度 **0.8**、种子 **0**；
  温度敏感性附 {0.6, 1.0}（00 §8）。
- 探针协议：线性 L2-logistic，C ∈ {0.01, 0.1, 1}（验证段选择），3 种子 {0,1,2}，负类下采样 5:1，
  会话级 cluster bootstrap 1,000 次；帧级重采样禁止。

## 4. 冻结参数指纹

由 `uv run python scripts/prereg_fingerprint.py` 自动回填（configs/*.yaml 与 configs/splits/*.json 的 sha256）。

<!-- FINGERPRINT:BEGIN -->

指纹生成时间：2026-07-17T12:54:19.350221+00:00

| 文件 | sha256 |
| --- | --- |
| configs\events.yaml | `9dc6441216cda4f2759201bc5c0395f241361202924be03d549b029e97255c51` |
| configs\grids.yaml | `41ccbaa7710aaf9b919f1ad1668f17cd0c14c50612e7687b9e055d9ababf1d12` |
| configs\paths.windows.yaml | `f4523178302f90b513740e220238890b605261ee4ca38b22ab5949a5aa581b0d` |
| configs\splits\candor.json | `0378ca258a5daebbf9bc5c966e552da7640a90499f2b3d9f0b21b465be93c02e` |
| configs\splits\smoothconv.json | `5e8612883f3eb671a6a58d51707cd0d5a3f23232725338e9d7832e13730991b7` |
| configs\stimuli.yaml | `dfdec98f07e5078cd326248c36d341cd9132ea8ef96250fb4e03fd6061942378` |
<!-- FINGERPRINT:END -->

## 5. 结果登记

| Gate | 日期 | 结果 | 证据 |
| --- | --- | --- | --- |
| G0·层1 | 2026-07-17 | 参考实现 138/138 会话逐帧全等（1,030,734 通道帧残差 0；与官方源码 2,000 组随机对拍一致）；库版待 --protocol-check 对拍确认 | reports/g0_reference_recompute.json |
| G0·层3 | — | 待定（门槛待 val 295 会话估计；首轮 20 会话 0.3424 为探索性诊断，不作裁决） | reports/g0_summary.json |
| G1 | — | 待定 | reports/mve_报告.md |

## 6. 变更记录

任何冻结参数/判据的变更必须先在此登记（日期、条目、理由、影响面、需重跑的校准），再改配置。

| 日期 | 状态 | 条目 | 理由 | 影响面 | 重跑 |
| --- | --- | --- | --- | --- | --- |
| 2026-07-17 | **已批准（修改版，用户裁决）** | **#1 G0 判据分层化（修改版）**：层1 协议正确性 = 固定官方提交 `2d0db21e767b953f5017c1cc697928b54161d645` 的 `relabel_context_aware.py`，要求逐帧全等（参考移植 `scripts/wp1_g0_reference_recompute.py` 已 138/138 全等；库版 `events/g0_official.py` 为第二实现，`--protocol-check` 持续对拍）；层2 VAD 一致性审计保留；层3 端到端门槛用官方 **val 295 会话**估计后冻结。集合纪律：test 前 20 会话（首轮联调已读）登记为探索集，其余 118 个锁定为一次性确认集。另：生产链 `process_session → detect_all`（事件本体，供 CANDOR/SmoothConv）仍需独立原始音频验证，不因 G0 重构而免检 | 用户 G0 首轮复核 + 层1 网格取证（64 组无解、结构性错误三处：无人恢复漏标 HOLD、BOT 最近活动语义、BC 首尾静音）；参考移植 138/138 全等确立语义权威 | G0 判据（00 §7）；`configs/events.yaml` g0 节；`events/g0_official.py` | 库版 `--protocol-check`（探索集 20 → val 295）→ val 三层全量定门槛 → 118 确认集一次性运行 |
| 2026-07-17 | **已批准（修改版，用户裁决）** | **#2 S1 拆臂（修改版）**：同文本原版/F0 拉平变换对的时长判据 ±5% 批准；complete/incomplete 字面前缀臂**不能单独作为 H4/G4 确证证据**。S1 拆为：**S1-A 确证臂**——跨条目配对 complete_i vs incomplete_j，总时长与语音活动时长均 ±5% 匹配（`stimuli/pairing.py` + `wp6_build_stimuli.py pair-a`）；**S1-B 探索臂**——保留真实前缀对，完整保存方向性时长比、韵律指标与人工抽听结果，仅作探索性分析 | V6 首轮质检 30% 通过率根因（前缀对时长判据结构性不可能）+ 用户裁决：前缀臂的时长/信息量混淆使其不足以单独确证语义敏感性 | S1 设计（00 §5）；`configs/stimuli.yaml`；E4/G4 证据结构 | 全量合成后跑 `pair-a` 报告配对成功率；S1-A 对数不足时扩充文本模板库 |
