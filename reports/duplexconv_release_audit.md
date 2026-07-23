# DuplexConv 完整发布物与本地布局审计报告

- 审计日期：2026-07-23
- 审计状态：**passed**
- 本机数据根目录：`D:\dataset\audio\Full_Duplex\qualialabsAI__DuplexConv`
- 上游仓库：[`qualialabsAI/DuplexConv`](https://huggingface.co/datasets/qualialabsAI/DuplexConv)
- 固定提交：[`0bb99da7ab7a2f6f86d6b23df92c9383e711d09a`](https://huggingface.co/datasets/qualialabsAI/DuplexConv/tree/0bb99da7ab7a2f6f86d6b23df92c9383e711d09a)
- 许可证：`CC-BY-NC-4.0`

## 1. 结论先行

1. 完整发布物已经闭合。固定提交包含 194 个文件，本机 194/194 文件均存在且字节数
   与远端一致；远端文件树在审计时共有 200 个条目，其中 6 个是目录记录。
2. 189 个音频 TAR 与 3 个 `jsons.tar.gz` 共形成 **93,709 对 WAV/JSON**；
   `audio_only=0`、`annotation_only=0`，三个类别均无跨分片重复会话。
3. 发布物占用 **1,640,733,127,901 字节（1,528.051801 GiB）**。其中音频 TAR
   为 1,527.836533 GiB，三个标注包为 0.215268 GiB。用户回传的三类
   322.06/132.53/1,073.24 GiB 属于音频口径；1,528.05 GiB 的总计包含标注包。
4. 93,709 个标注根对象合计 **2,000.214857 小时**，与数据卡声明的 2,000.21
   小时一致到两位小数；平均会话时长 76.841856 秒。全部对象采样率为 48 kHz，
   全部满足 `len(asr) == nTrack`。
5. Windows 本机固定使用 `Edu → Edu_upper`、`edu → edu_lower`、
   `none_Edu → none_Edu`。`upper/lower` 只描述上游目录首字母大小写，不表示教育层级。
6. 发布物完整性已达到生成训练清单的前置条件。训练清单本身仍未冻结；仓库现有
   `wp2_freeze_splits.py` 只发现散装 JSON，直接用于该 TAR 发布物会得到空集合，
   因此必须先实现专用成员索引器并单独登记。

## 2. 本轮产物

| 位置 | 产物 | 用途 |
| --- | --- | --- |
| 数据根目录 | `LOCAL_LAYOUT.md` | 面向本机操作者的固定布局、映射与解包规则 |
| 数据根目录 | `local_layout.json` | 机器可读的完整文件树映射、远端摘要、逐分片成员数和标注聚合 |
| 仓库 `reports/` | `duplexconv_release_audit.json` | 与数据根目录 JSON 逐字相同的可提交审计快照 |
| 仓库 `reports/` | `duplexconv_release_audit.md` | 本报告 |
| 仓库 `scripts/` | `wp2_audit_duplexconv_release.py` | 可重复执行的完整发布物审计器 |
| 仓库 `tests/` | `test_duplexconv_release_audit.py` | 路径映射、网络重试、TAR 安全成员和脱敏聚合测试 |

数据根目录新增的两个布局文件属于本机元数据，不计入上游 194 个发布物文件。

## 3. 固定提交与本地映射

### 3.1 发布物身份

| 字段 | 值 |
| --- | --- |
| 上游仓库 | `qualialabsAI/DuplexConv` |
| 固定提交 | `0bb99da7ab7a2f6f86d6b23df92c9383e711d09a` |
| 审计时远端 `main` | 与固定提交相同 |
| 远端最后修改时间 | 2026-06-12 04:47:34 UTC |
| 远端文件数 | 194 |
| 清单摘要 | `0169a70e7069e6160311953f0c735064423ff85a7f6413e0111ea711f04cab4c` |

清单摘要绑定逐文件上游路径、本机相对路径、字节数和远端 LFS SHA-256。后续上游
`main` 即使前移，本项目仍以本表固定提交为发布物身份。

### 3.2 Windows 路径消歧

上游同时存在只靠大小写区分的 `Edu` 与 `edu`。Windows 默认大小写不敏感，无法在同一
目录下可靠保留这两个名字，因此本机映射固定如下：

| 上游相对目录 | 本机相对目录 | 场景元数据 | 规则 |
| --- | --- | --- | --- |
| `Edu` | `Edu_upper` | `tutoring` | 保留上游大写路径语义 |
| `edu` | `edu_lower` | `tutoring` | 保留上游小写路径语义 |
| `none_Edu` | `none_Edu` | `social_chat` | 名称保持不变 |

索引时必须先应用本表映射，再进行大小写归一化或场景聚合。`Edu_upper` 与 `edu_lower`
完成会话唯一键核验后，可以按研究定义聚合到 `tutoring` 场景；原始存储路径和来源字段
仍须分别保留。

### 3.3 本地目录

```text
qualialabsAI__DuplexConv/
├── .gitattributes
├── README.md
├── LOCAL_LAYOUT.md
├── local_layout.json
├── Edu_upper/
│   ├── jsons.tar.gz
│   └── audios/Edu_0001.tar ... Edu_0045.tar
├── edu_lower/
│   ├── jsons.tar.gz
│   └── audios/edu_0001.tar ... edu_0019.tar
└── none_Edu/
    ├── jsons.tar.gz
    └── audios/none_Edu_0001.tar ... none_Edu_0125.tar
```

## 4. 全量库存与配对审计

| 本机类别 | 上游目录 | 音频 TAR | WAV | JSON | 成功配对 | 音频孤儿 | 标注孤儿 | 音频 GiB | 含标注 GiB | 标注小时 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `Edu_upper` | `Edu` | 45 | 22,050 | 22,050 | 22,050 | 0 | 0 | 322.061491 | 322.114490 | 495.917698 |
| `edu_lower` | `edu` | 19 | 9,437 | 9,437 | 9,437 | 0 | 0 | 132.531729 | 132.553775 | 204.127485 |
| `none_Edu` | `none_Edu` | 125 | 62,222 | 62,222 | 62,222 | 0 | 0 | 1,073.243313 | 1,073.383535 | 1,300.169675 |
| **合计** | — | **189** | **93,709** | **93,709** | **93,709** | **0** | **0** | **1,527.836533** | **1,528.051801** | **2,000.214857** |

字节级口径如下：

| 类别 | 音频字节 | 标注字节 | 合计字节 |
| --- | ---: | ---: | ---: |
| `Edu_upper` | 345,810,892,800 | 56,907,357 | 345,867,800,157 |
| `edu_lower` | 142,304,860,160 | 23,672,496 | 142,328,532,656 |
| `none_Edu` | 1,152,386,232,320 | 150,562,768 | 1,152,536,795,088 |
| **合计** | **1,640,501,985,280** | **231,142,621** | **1,640,733,127,901** |

配对键固定为“同一映射类别内，WAV 与 JSON 文件名去除扩展名后的完整会话标识”。
审计同时检查分片内重复、跨分片重复和两侧集合差异，三项均为零。

## 5. 标注结构与质量信号

### 5.1 根结构、轨道和采样率

全部 93,709 个 JSON 的根键集合一致：

```text
asr | fs | nTrack | timeLenInSec
```

`asr` 外层位置表示轨道，全部对象的外层长度都等于 `nTrack`；本轮发现
`asr_shape_errors=0`。轨道数分布如下：

| 类别 | 2 轨 | 3 轨 | 4 轨 | 合计 |
| --- | ---: | ---: | ---: | ---: |
| `Edu_upper` | 21,761 | 275 | 14 | 22,050 |
| `edu_lower` | 9,312 | 121 | 4 | 9,437 |
| `none_Edu` | 44,582 | 14,070 | 3,570 | 62,222 |
| **合计** | **75,655** | **14,466** | **3,588** | **93,709** |

全部会话的 `fs` 均为 48,000。会话时长边界：

| 类别 | 最短秒数 | 最长秒数 | 总小时 |
| --- | ---: | ---: | ---: |
| `Edu_upper` | 10.000 | 598.176 | 495.917698 |
| `edu_lower` | 10.000 | 500.096 | 204.127485 |
| `none_Edu` | 8.000 | 618.272 | 1,300.169675 |

### 5.2 句级状态

全量 1,749,177 条句级记录只观察到冻结的四种 `state` 值和缺字段：

| 原始状态 | 数量 | 归一化建议 |
| --- | ---: | --- |
| `<\|complete\|>` | 870,841 | `complete` |
| `<\|incomplete\|>` | 190,603 | `incomplete` |
| `<\|backchannel\|>` | 92,266 | `backchannel` |
| `<\|wait\|>` | 4,805 | `wait` |
| 缺字段 | 590,662 | `null`，不得推断为任一类别 |
| **合计** | **1,749,177** | — |

缺 `state` 的总体比例为 **33.767995%**，类别差异明显：

| 类别 | 句级记录 | 缺 `state` | 比例 |
| --- | ---: | ---: | ---: |
| `Edu_upper` | 397,192 | 76,409 | 19.24% |
| `edu_lower` | 162,622 | 31,916 | 19.63% |
| `none_Edu` | 1,189,363 | 482,337 | 40.55% |

因此，DuplexConv 适合承担弱标签大样本训练和稳健性放大；训练清单需要保存
`state_present` 与来源类别，涉及 `state` 的监督任务应报告类别分层覆盖率。

### 5.3 隐私删改占位

本轮发现 11,957 条 `sensitiveRedacted=true` 记录，占全部句级记录的 0.683579%：

| 类别 | 隐私删改占位 |
| --- | ---: |
| `Edu_upper` | 12 |
| `edu_lower` | 9 |
| `none_Edu` | 11,936 |
| **合计** | **11,957** |

这些记录应在训练清单中显式标记。涉及文本或状态监督时默认排除；涉及时长、容器结构等
无文本任务时，也应保留排除原因，避免和普通缺标签记录混淆。审计 JSON 不保存任何转写
正文。

## 6. 完整性证据与哈希边界

### 6.1 已完成的核验

- 固定提交 194/194 个文件逐一核对本机存在性和字节数。
- 189 个音频 TAR 全部扫描成员头；成员必须是普通相对文件，名称必须满足类别前缀与
  六位会话编号规则。
- 3 个标注包全部解压读取 JSON，并只输出聚合统计。
- 3 个标注包的本地 SHA-256 与远端 LFS SHA-256 逐项一致。
- `README.md`、`.gitattributes` 和 3 个标注包共 5 个文件计算了本地完整 SHA-256。
- 数据根目录 `local_layout.json` 与仓库审计 JSON 逐字一致。

### 6.2 关键摘要

| 对象 | SHA-256 |
| --- | --- |
| 发布物路径/字节/LFS 清单 | `0169a70e7069e6160311953f0c735064423ff85a7f6413e0111ea711f04cab4c` |
| 仓库 JSON / 数据根目录 JSON | `68ec519886fe4918c4a3fbe69fecb0a3d7f6b9bac9d4ce258dd5aff2ad278a6d` |
| 数据根目录 `LOCAL_LAYOUT.md` | `ee1a8bf1416af084f029ebbbf4501b63899aac1ddcee2442d172dc26272025a8` |
| `Edu_upper/jsons.tar.gz` | `1e6451ed46ce7ad03b3baf53efc240b7fdd9a752e6ca3cbac3a78fd8f2693240` |
| `edu_lower/jsons.tar.gz` | `406c6a10ad0620da363bc63cdd0fcaea45d367ed4feaa2d8ef85e403f97b9308` |
| `none_Edu/jsons.tar.gz` | `56e9bd338e60c0b404bffa01dfe09b5758ba75d6dee2efc1b97bb02dd42e1b16` |
| `README.md` | `a75a294a2211e86db64fe2c57f92b3a0f60f197f6c9db82640c457be60e7489c` |
| `.gitattributes` | `9e75dd981de037ec3769f24f790e126bc5a160b6871f510214e68dc70649aeeb` |

### 6.3 未执行的高成本核验

本轮没有顺序重读 1.5 TiB 音频内容计算本机完整 SHA-256。机器映射已经保存每个音频
TAR 的远端 LFS SHA-256，并完成本机字节数、TAR 可读性、成员名和成员集合核验。
这一边界能发现缺文件、截断、错误分片、容器损坏和配对缺口；同字节数的静默位翻转仍需
在机械盘空闲窗口执行完整内容哈希才能排除。

本轮也没有解码 WAV 负载或重新验证声道数、采样格式和音频可听性。后续训练清单可对每个
分片抽样读取 WAV 头，完整训练读入阶段再执行逐成员解码硬门。

## 7. 固定读取与解包规则

1. 原始 TAR 与 `jsons.tar.gz` 视为只读发布物，禁止原地修改、重打包或覆盖。
2. 默认流式访问。音频使用 `tarfile.open(path, mode="r:")`，标注使用
   `tarfile.open(path, mode="r:gz")`。
3. 成员必须是普通相对文件；绝对路径、`..`、符号链接、硬链接和重复目标路径全部硬失败。
4. 先按本报告映射解析类别，再用完整会话标识配对；不得先对 `Edu`/`edu` 做不区分
   大小写的路径归并。
5. 保留 WAV 多通道结构。混音、重采样、声道选择和截断均属于下游数据变换，必须另行
   登记并写入训练清单。
6. 确需物化时，写入
   `D:\data_storage\The_Floor_Control_Circuit\tmp\duplexconv_unpack\<revision>\<category>`；
   每次使用新的任务子目录，拒绝覆盖已有成员，完成后复核成员计数与配对集合。
7. Git 仓库只接收脚本、配置和小型报告；音频、解包结果、索引中间文件和训练缓存均不得
   进入 Git。

## 8. 对现有代码的影响

### 8.1 划分冻结脚本

`scripts/wp2_freeze_splits.py` 当前通过以下方式发现 DuplexConv 会话：

```python
dc_sessions = sorted(p.stem for p in dc_dir.rglob("*.json"))
```

发布物外层没有散装 JSON，所以该逻辑会得到空列表。#31 已明确禁止直接运行这一入口。
专用索引器至少应产出：

- 固定提交、映射类别、场景元数据；
- 音频 TAR 相对路径与 WAV 成员名；
- 标注包相对路径与 JSON 成员名；
- 会话标识、字节数、时长、轨道数；
- `state_present`、`sensitiveRedacted` 和其他质量标志；
- 原始发布物清单摘要与训练清单自身摘要。

### 8.2 统一解析器

当前 `src/floor_circuit/data/smoothconv.py` 面向平面
`segments/utterances/annotations/...` 列表。DuplexConv 的真实入口是
`asr[track][utterance]`，时间字段为 `startInMs/endInMs`，文本位于 `labels.txt`，
状态值带 `<|...|>` 包裹。需要独立的容器迭代层，再复用统一输出 schema；缺 `state`
必须保留为 `null`。

## 9. 历史局部快照的收口

原 `文档/07_DuplexConv发布物gz-md只读窥探.md` 基于下载中的 `Edu` 局部快照：
当时只有 43/45 个音频 TAR，因而出现 21,500 个 WAV 对 22,050 个 JSON、550 个
标注孤儿。完整下载补齐 `Edu_0044.tar` 与 `Edu_0045.tar` 后，`Edu_upper` 已达到
22,050 对、两侧孤儿均为零。

该历史文件已经从 `文档/` 移出并由本报告取代。其关于 `Edu` 根 schema、状态值和
嵌套结构的发现与本轮全量扫描一致；本报告进一步覆盖 `edu_lower`、`none_Edu`、
固定提交、三类路径映射和完整配对集合。

`reports/duplexconv_selfcheck.json` 同样保留为早期局部快照证据，不再承担当前库存
判断。当前机器可读权威为 `reports/duplexconv_release_audit.json`。

## 10. 冻结边界与使用限制

- PREREG #31 固定发布物提交、Windows 路径映射、库存数量和读取契约。
- 本轮没有选择、过滤或打乱训练会话，训练清单与训练顺序仍未冻结。
- 生成专用训练清单前，不得启动 DuplexConv 训练，也不得把当前完整性报告当作训练划分。
- 数据集采用 LLM 辅助标注；`none_Edu` 的缺 `state` 比例和隐私删改占位显著高于两个
  辅导目录。后续应把来源类别作为质量分层，并以 SmoothConv 人工标注数据做敏感性参照。
- `CC-BY-NC-4.0` 限制必须沿用到训练产物、发布说明和任何下游分发决策。

## 11. 复跑命令与验收条件

在仓库根目录执行：

```powershell
uv run python scripts/wp2_audit_duplexconv_release.py --root D:\dataset\audio\Full_Duplex\qualialabsAI__DuplexConv --revision 0bb99da7ab7a2f6f86d6b23df92c9383e711d09a --report-json reports\duplexconv_release_audit.json --write-layout
```

本机机械盘冷扫描约 9 分 30 秒；系统文件缓存命中后的最终硬门复核约 37 秒。通过条件：

1. `status == "passed"` 且 `errors == []`；
2. `file_count == size_verified_file_count == 194`；
3. `audio_archive_count == 189`、`paired_count == 93709`；
4. 两侧孤儿计数均为 0；
5. 三类 `asr_shape_errors` 均为 0；
6. 数据根目录 JSON 与仓库 JSON 的 SHA-256 完全一致。

本轮上述六项全部通过。
