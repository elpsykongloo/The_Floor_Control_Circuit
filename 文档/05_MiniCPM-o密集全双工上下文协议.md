# MiniCPM-o 密集全双工上下文宽度测量协议

## 1. 要回答的问题

本协议测量 MiniCPM-o 4.5 在严格密集全双工条件下，长前缀是否额外损害：

1. Qwen3 骨干选定层的隐藏状态；
2. 未干预的原生发言决策分布；
3. 贪心首 token 与完整单元生成序列；
4. KV 缓存的有限性和连续性。

模型配置给出的正式上限仍为 `max_position_embeddings=40960`。本协议可以给出
官方上限外的实证干净下界，但不会据此改写正式规格或项目冻结判据。

默认系统提示占 14 个位置，因此严格密集连续折算为
`(40960 - 14) / 32 = 1279.5625` 秒；按完整的一秒单元计，正式规格内只能完整
容纳 1279 个单元，第 1280 个单元会跨过配置上限。

## 2. 密集负载定义

官方一秒纯音频双工单元的位置组成是：

```text
<unit> 1 个位置
音频嵌入 10 个位置
最多 19 个模型正文/特殊 token
<|chunk_eos|> 1 个位置
</unit> 1 个位置
合计 32 个位置/秒
```

密集库通过官方 `streaming_prefill + streaming_generate` 状态机产生。生成时只屏蔽
`listen`、提前 `chunk_eos`、`chunk_tts_eos` 和 `turn_eos`；正文 token 仍由模型
按原始 logits 贪心选择。循环终点仍由官方代码写入 `chunk_eos` 和 `</unit>`。

该条件是位置消耗的饱和上界，代表用户持续输入、模型同时持续发言。模型输出音频
解码关闭，因为 TTS 位于 LLM token 决策之后，不写回 Qwen3 KV；模型自己的发言
token 仍完整进入上下文。

## 3. 三角对照

单看长流自身的隐藏状态变化，无法区分内容累积和位置退化；只拿长流与短后缀
比较，也会把合法的远端历史影响误写成退化。本协议在每个检查点为每条自然探针
运行三个条件：

- **完整长流**：保留从会话起点到当前时刻的全部 KV；
- **高位近期后缀**：重置 KV，只精确重放最近 64 个完整一秒单元，同时将这些
  动态 token 的 RoPE 位置整体平移，使当前探针的绝对位置与完整长流逐项相等；
- **低位近期后缀**：重置 KV，重放完全相同的最近 64 个单元，使用从零开始的
  默认位置；
- 三路近期音频嵌入、模型生成 token、当前探针音频、贪心解码参数和话轮状态
  逐项一致。

两组比较承担不同职责：

1. 高位近期后缀 vs 低位近期后缀：只改变绝对位置，用于上下文宽度主判据；
2. 完整长流 vs 高位近期后缀：绝对位置相同，只增加远端历史，用于登记内容累积
   和公共吸引态，不单独判定上下文失效。

高位近期后缀主比较检验局部绝对位置外推，不能代替“从数千秒前准确检索语义”
的远程记忆任务。完整长流保留远端 KV，可发现公共吸引态与运行行为分叉；远端
内容对当前决策的正常影响仍需借助带答案的任务探针另行判断。

每条独立音频运行都执行相同设计，正式联合结论至少使用三条 SHA-256 不同、来自
不同录音会话的输入；仅改变随机种子或使用同一会话的两个声道不计作三条独立
输入。

## 4. 检查点

默认目标秒数：

```text
64, 256, 512, 768, 1024, 1200, 1260, 1280, 1300, 1400,
1600, 1800, 1950, 2000, 2050, 2200, 2400, 2560, 2600
```

该网格覆盖：

- 早期配对基线；
- 官方 40960 位置附近；
- 2000 秒前后；
- `2C=81920` 附近。

每个检查点默认使用三个不同音频单元作为探针。报告同时记录目标秒数、真实输入
秒数、实际 KV 位置和观测位置速率，最终边界以实际位置为权威。检查点以严格
密集位置预算定位：

```text
目标位置 = 固定提示位置 + 目标秒数 × 32
```

自然探针可能选择倾听，从而少于 32 个位置；运行器会补充完整密集单元，直到实际
位置覆盖目标预算。因此“2000 秒已覆盖”要求实际位置达到
`固定提示位置 + 2000 × 32`，不会仅凭输入单元计数宣布覆盖。

## 5. 指标

高位近期后缀—低位近期后缀主比较计算：

1. 四个深度层的隐藏向量余弦；
2. 全词表中心化 logits 余弦；
3. 全词表 Jensen–Shannon 散度；
4. `{listen, speak, chunk_eos, turn_eos}` 概率的总变差；
5. 贪心首 token 一致率；
6. 完整生成 token 序列的归一化 Levenshtein 相似度；
7. 隐藏状态与 logits 有限性；
8. 长流/复位 KV 长度和真实位置速率。

完整长流—高位近期后缀另行计算同类指标，并在至少三条不同录音输入齐备时计算
隐藏范数比、跨输入平均方向余弦和未中心化第一公共方向占比。

## 6. 描述性绝对位置退化判据

第一检查点作为早期位置基线。某个检查点只有高位近期后缀相对低位近期后缀、且
相对早期基线发生实质恶化时才触发：

| 指标族 | 触发条件 |
| --- | --- |
| 表征 | 最差层隐藏余弦 ≤0.95 且下降 ≥0.03，或 logits 余弦 ≤0.95 且下降 ≥0.03 |
| 分布 | JS ≥0.05 且增加 ≥0.03，同时特殊决策总变差 ≥0.10 且增加 ≥0.05 |
| 行为 | 首 token 一致率 ≤0.80 且下降 ≥0.20，或序列相似度 ≤0.60 且下降 ≥0.20 |
| 历史结构 | 完整长流相对高位近期后缀出现 ≥2 倍隐藏范数偏离，或跨输入公共吸引态显著增加 |
| 数值 | 三路任一选定层或 logits 出现非有限值 |

确认退化要求：

- 绝对位置退化要求表征、分布、行为三个位置指标族中至少两个触发，并在下一个
  检查点持续；
- 历史行为分叉单独使用表征、分布、行为、历史结构四族中的至少两族及相同持续
  规则，不参与绝对位置退化裁决；
- 任意时点出现非有限值。

安全下界取首个确认退化检查点之前的检查点。未触发时，报告写成“实测干净下界”，
不能写成无限上下文。

## 7. 三层结论

报告必须分开给出：

1. **正式规格宽度**：`(40960 - 固定提示位置) / 32`；
2. **2000 秒三角对照结论**：2000 秒是否覆盖、绝对位置持续判据是否干净；
3. **官方上限外实证下界**：若 2000 或 2600 秒仍干净，只作为外推证据。

正式 floor 标签 AUC 不在本协议内。后续进入 E1 时，应在同样的长流/复位配对
位置上复算冻结探针分数。

## 8. 运行命令

### 8.1 冒烟

```powershell
$model = 'C:\artificial_intelligence\models\Full-Duplex\MiniCPM-o-4.5'
$repo = 'C:\artificial_intelligence\repos\The_Floor_Control_Circuit'
$out = 'D:\data_storage\The_Floor_Control_Circuit\context_stress\minicpm_o_dense\smoke'

uv run --no-sync --project $model python `
  "$repo\runners\minicpm_o\dense_context.py" `
  --model-root $model `
  --audio "$model\assets\HT_ref_audio.wav" `
  --out $out `
  --run-id smoke `
  --bank-units 4 `
  --checkpoint-seconds 8,16 `
  --max-seconds 16 `
  --probes-per-checkpoint 2 `
  --control-suffix-units 4 `
  --filler-group-units 2
```

### 8.2 正式单条运行

```powershell
$model = 'C:\artificial_intelligence\models\Full-Duplex\MiniCPM-o-4.5'
$repo = 'C:\artificial_intelligence\repos\The_Floor_Control_Circuit'
$audio = 'D:\data_storage\The_Floor_Control_Circuit\candor_extracted\67836c1d-1334-41a0-a33a-4f788e8b6fb3\audio_ch0.wav'
$out = 'D:\data_storage\The_Floor_Control_Circuit\context_stress\minicpm_o_dense\run1'

uv run --no-sync --project $model python `
  "$repo\runners\minicpm_o\dense_context.py" `
  --model-root $model `
  --audio $audio `
  --out $out `
  --run-id dense_run1 `
  --bank-units 32 `
  --seed 1
```

另外两条运行使用不同真实音频和不同种子。所有大数组留在 D 盘。

### 8.3 联合分析

```powershell
uv run --frozen python scripts\wp_minicpm_dense_context_analyze.py `
  --runs `
    D:\data_storage\The_Floor_Control_Circuit\context_stress\minicpm_o_dense\run0 `
    D:\data_storage\The_Floor_Control_Circuit\context_stress\minicpm_o_dense\run1 `
    D:\data_storage\The_Floor_Control_Circuit\context_stress\minicpm_o_dense\run2 `
  --out-json reports\minicpm_o_密集全双工上下文.json `
  --out-md reports\minicpm_o_密集全双工上下文.md
```

## 9. 运行护栏

- 密集逐 token 路径与批量重放必须满足隐藏和 logits 余弦均 ≥0.999，且 argmax
  一致；
- 滑窗固定关闭，避免把淘汰后的滚动历史误写成完整上下文；
- 温度达到 95°C 立即停止；
- 每个运行写入音频 SHA-256、生成 token 库、三路逐项位置校验、实际位置速率和
  trace SHA-256；
- 超过官方位置仍返回有限值时，只登记外推证据。
