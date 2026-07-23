# DuplexConv 发布物 gz/md 只读窥探回传

状态：**只读探测完成，训练侧发布物清单暂不具备“完整发布版”冻结条件**

探测日期：2026-07-20（Asia/Shanghai）

本机发布物根目录：`D:\dataset\audio\Full_Duplex\qualialabsAI__DuplexConv`

> 快照范围：本文固定记录 2026-07-20 当时可见的发布物，仅作为历史取证材料。
> 后续下载进度已经变化，当前库存状态以 `CLAUDE.md` 的 DuplexConv 条目为准。

## 0. 结论先行

本机快照可以稳定读取，但它只覆盖 `Edu` 教育场景，且音频与注释数量不相等：

- 外层共有 46 个普通文件，总大小 `337,394,599,212` 字节，即 `337.394599212 GB`
  或 `314.223206799 GiB`。
- 音频侧有 43 个未压缩 tar 包，每包恰好 500 个 WAV，共 **21,500** 个唯一音频成员。
- `Edu\jsons.tar.gz` 内有 **22,050** 个唯一 JSON 成员，全部可以解压流式读取并通过
  `json` 解析。
- 音频会话标识集合是注释会话标识集合的真子集：
  - 音频与注释配对：**21,500**；
  - 只有音频：**0**；
  - 只有注释：**550**。
- 550 个无音频注释从 `Edu--026566.json` 开始，最后为 `Edu--027260.json`，合计
  **12.184188154 小时**。
- 可配对部分为 **21,500 会话、483.733509491 小时**；整个本机注释包为
  **22,050 会话、495.917697645 小时**。
- 数据卡声明完整发布物包含 93,709 个音频、2,000.21 小时、教育与闲聊两类场景。
  本机快照尚未出现 `none_Edu` 闲聊目录；可配对音频数量约为数据卡声明的 22.9434%，
  可配对时长约为 24.1841%。
- 真实 JSON 结构与数据卡中的部分字段说明存在明显差异。转写文本位于
  `labels.txt`；`state` 是可缺省字段；根对象中没有数据卡列出的 VAD 字段。
- 真实发布物含 2、3、4 轨会话，不能把“双通道”写成全局硬约束。
- 当前仓库的 `wp2_freeze_splits.py` 通过 `rglob("*.json")` 找会话，在本发布物上会得到
  0 个会话；当前 `smoothconv.py` 解析器也无法直接解析这套嵌套 `asr` 结构。

因此建议把冻结分为两层：

1. **发布物完整性清单**：当前状态记为 `partial` 或 `blocked`，明确登记缺少闲聊场景和
   550 个 JSON 对应音频。
2. **本机可训练配对子集清单**：若用户批准使用当前局部快照，可显式冻结 21,500 个配对
   会话，并把 550 个排除项单独固化、计数和哈希；不得静默丢弃。

本文只新增探测文档，没有修改发布物、`PREREG.md`、冻结配置、划分文件、解析器或训练代码。

## 1. 探测边界与证据等级

### 1.1 已执行的只读操作

- 递归枚举发布物外层路径、文件大小、时间戳和扩展名。
- 完整读取本机 `README.md` 和 `.gitattributes`。
- 对 `README.md`、`.gitattributes`、`Edu\jsons.tar.gz` 计算 SHA-256。
- 流式枚举 `jsons.tar.gz` 内全部 22,050 个 tar 成员，不落盘解包。
- 逐个解析全部 22,050 个 JSON，统计根字段、句级字段、嵌套字段、标签值、类型和异常。
- 完整枚举 43 个音频 tar 的全部成员头，不读取音频主体，不落盘解包。
- 对每个音频分片的首个 WAV 读取 RIFF 头，另读取一个 3 轨样本和一个 4 轨样本。
- 对音频成员名集合与 JSON 成员名集合做精确集合比较。
- 生成若干仅由成员名、成员大小和会话标识构成的规范化元数据指纹。

### 1.2 未执行的高成本操作

- 没有对 337.338 GB 的 43 个音频 tar 逐包计算完整内容 SHA-256。
- 没有逐个读取全部 21,500 个 WAV 的 RIFF 头。
- 没有把任何 tar 或 gzip 成员解压到磁盘。
- 没有试听音频，也没有把转写正文写进本文。
- 没有访问训练侧确认集或任何因果评估侧数据。

音频格式结论在本文中会明确标为“45 个头部样本观察”；训练侧冻结脚本仍应全量核验每个
WAV 头。成员数量、成员名称和音频/注释集合关系属于全量枚举结果。

### 1.3 现有回传与本轮增量

仓库已有 `reports/duplexconv_selfcheck.json`，生成于 2026-07-17。该报告已正确发现：

- 43 个 `.tar`；
- 1 个 `.gz`；
- 1 个 `README.md`；
- 首个音频 tar 内是 WAV；
- `jsons.tar.gz` 内是 JSON。

旧自检只窥探首批 30 个成员，并因 `_iter_items()` 不认识根级 `asr` 嵌套结构而报告：

```text
ValueError('无法在 JSON 中定位分段列表（segments/utterances/...）')
```

本轮增量完成了全部 JSON、全部音频 tar 成员头、集合配对和异常分布统计。

## 2. 外层发布物结构

### 2.1 实际目录树

```text
qualialabsAI__DuplexConv\
├── .gitattributes
├── README.md
└── Edu\
    ├── jsons.tar.gz
    └── audios\
        ├── Edu_0001.tar
        ├── Edu_0002.tar
        ├── ...
        ├── Edu_0042.tar
        └── Edu_0043.tar
```

本机没有观察到以下发布物：

```text
none_Edu\
none_Edu*\...
Social_Chat\
```

数据卡把 `edu`/`Edu` 前缀定义为教育场景，把 `none_Edu` 前缀定义为闲聊场景。当前快照
因此只能登记为 `Edu` 场景局部快照。

### 2.2 外层文件类型和大小

| 类型 | 文件数 | 说明 |
| --- | ---: | --- |
| `.tar` | 43 | `Edu\audios\Edu_0001.tar` 至 `Edu_0043.tar` |
| `.gz` | 1 | `Edu\jsons.tar.gz`，内部实际是 gzip 压缩的 tar |
| `.md` | 1 | 根目录 `README.md` |
| 无扩展名 | 1 | 根目录 `.gitattributes` |
| 合计 | 46 | 全部为普通文件 |

大小汇总：

| 范围 | 字节数 |
| --- | ---: |
| 43 个音频 tar 合计 | 337,337,681,920 |
| `jsons.tar.gz` | 56,907,357 |
| `README.md` | 7,431 |
| `.gitattributes` | 2,504 |
| 整个快照合计 | 337,394,599,212 |

### 2.3 已计算的内容哈希

| 相对路径 | 字节数 | SHA-256 |
| --- | ---: | --- |
| `README.md` | 7,431 | `a75a294a2211e86db64fe2c57f92b3a0f60f197f6c9db82640c457be60e7489c` |
| `.gitattributes` | 2,504 | `9e75dd981de037ec3769f24f790e126bc5a160b6871f510214e68dc70649aeeb` |
| `Edu/jsons.tar.gz` | 56,907,357 | `1e6451ed46ce7ad03b3baf53efc240b7fdd9a752e6ca3cbac3a78fd8f2693240` |

43 个音频 tar 的完整内容哈希应由正式冻结脚本计算。建议串行读取机械盘，并将已完成结果
写入可断点续跑的临时状态；不要用多个进程在同一机械盘上并发随机读取。

### 2.4 数据卡元数据与大文件实化状态

`README.md` 顶部元数据：

| 字段 | 值 |
| --- | --- |
| `language` | `zh` |
| `license` | `cc-by-nc-4.0` |
| `pretty_name` | `DuplexConv` |
| `tags` | `speech`、`conversational-speech`、`chinese` |

数据卡写明数据仅用于学术与研究用途，并要求遵循 CC BY-NC 4.0。训练清单适合保存
`license_id`、`readme_sha256` 和用途范围，便于下游训练任务进行许可证核验。

`.gitattributes` 把 `.gz`、`.tar.*`、`.tar`、`.wav` 等大文件类型配置为 Git LFS。当前本机
文件已经实化：

- `jsons.tar.gz` 的首字节是 gzip 魔数 `1f 8b`；
- 43 个音频 tar 均能被 tar 解析器直接打开；
- 文件大小为数 GB，未观察到文本形式的 LFS 指针占位。

正式冻结脚本仍应在计算哈希前检测首行
`version https://git-lfs.github.com/spec/v1`，发现未实化指针时立即失败。

## 3. 音频 tar 分片清单

### 3.1 全局结构

- 分片名严格覆盖 `Edu_0001.tar` 至 `Edu_0043.tar`。
- 43 个包均为未压缩 tar。
- 每个包恰好有 500 个普通文件成员。
- 音频成员合计 21,500 个，成员名全部唯一。
- 所有成员名均匹配 `^Edu--\d{6}\.wav$`。
- 没有目录、符号链接、硬链接或其他 tar 成员类型。
- 成员编号存在自然缺口，不能通过整数连续区间生成会话标识。

### 3.2 43 个分片的精确边界

| 分片 | 外层字节数 | 成员数 | 首成员 | 末成员 |
| --- | ---: | ---: | --- | --- |
| `Edu_0001.tar` | 7,799,214,080 | 500 | `Edu--000001.wav` | `Edu--000615.wav` |
| `Edu_0002.tar` | 7,814,389,760 | 500 | `Edu--000616.wav` | `Edu--001230.wav` |
| `Edu_0003.tar` | 8,294,952,960 | 500 | `Edu--001231.wav` | `Edu--001861.wav` |
| `Edu_0004.tar` | 7,584,215,040 | 500 | `Edu--001862.wav` | `Edu--002467.wav` |
| `Edu_0005.tar` | 8,070,563,840 | 500 | `Edu--002468.wav` | `Edu--003086.wav` |
| `Edu_0006.tar` | 8,034,068,480 | 500 | `Edu--003087.wav` | `Edu--003700.wav` |
| `Edu_0007.tar` | 7,885,015,040 | 500 | `Edu--003701.wav` | `Edu--004312.wav` |
| `Edu_0008.tar` | 8,555,407,360 | 500 | `Edu--004313.wav` | `Edu--004926.wav` |
| `Edu_0009.tar` | 7,393,013,760 | 500 | `Edu--004928.wav` | `Edu--005553.wav` |
| `Edu_0010.tar` | 7,440,578,560 | 500 | `Edu--005554.wav` | `Edu--006158.wav` |
| `Edu_0011.tar` | 7,800,422,400 | 500 | `Edu--006159.wav` | `Edu--006762.wav` |
| `Edu_0012.tar` | 7,911,475,200 | 500 | `Edu--006763.wav` | `Edu--007369.wav` |
| `Edu_0013.tar` | 8,007,874,560 | 500 | `Edu--007370.wav` | `Edu--007973.wav` |
| `Edu_0014.tar` | 7,617,873,920 | 500 | `Edu--007974.wav` | `Edu--008594.wav` |
| `Edu_0015.tar` | 8,022,609,920 | 500 | `Edu--008595.wav` | `Edu--009233.wav` |
| `Edu_0016.tar` | 7,608,002,560 | 500 | `Edu--009234.wav` | `Edu--009841.wav` |
| `Edu_0017.tar` | 8,201,093,120 | 500 | `Edu--009843.wav` | `Edu--010455.wav` |
| `Edu_0018.tar` | 7,317,340,160 | 500 | `Edu--010456.wav` | `Edu--011078.wav` |
| `Edu_0019.tar` | 7,612,467,200 | 500 | `Edu--011079.wav` | `Edu--011706.wav` |
| `Edu_0020.tar` | 7,966,894,080 | 500 | `Edu--011707.wav` | `Edu--012311.wav` |
| `Edu_0021.tar` | 7,560,140,800 | 500 | `Edu--012313.wav` | `Edu--012925.wav` |
| `Edu_0022.tar` | 8,036,096,000 | 500 | `Edu--012926.wav` | `Edu--013550.wav` |
| `Edu_0023.tar` | 7,968,808,960 | 500 | `Edu--013551.wav` | `Edu--014155.wav` |
| `Edu_0024.tar` | 7,472,896,000 | 500 | `Edu--014156.wav` | `Edu--014769.wav` |
| `Edu_0025.tar` | 7,541,288,960 | 500 | `Edu--014770.wav` | `Edu--015387.wav` |
| `Edu_0026.tar` | 8,177,203,200 | 500 | `Edu--015388.wav` | `Edu--015998.wav` |
| `Edu_0027.tar` | 8,255,406,080 | 500 | `Edu--015999.wav` | `Edu--016632.wav` |
| `Edu_0028.tar` | 7,947,059,200 | 500 | `Edu--016633.wav` | `Edu--017238.wav` |
| `Edu_0029.tar` | 8,074,332,160 | 500 | `Edu--017239.wav` | `Edu--017861.wav` |
| `Edu_0030.tar` | 7,972,321,280 | 500 | `Edu--017862.wav` | `Edu--018482.wav` |
| `Edu_0031.tar` | 7,819,653,120 | 500 | `Edu--018483.wav` | `Edu--019126.wav` |
| `Edu_0032.tar` | 7,693,424,640 | 500 | `Edu--019127.wav` | `Edu--019749.wav` |
| `Edu_0033.tar` | 7,609,466,880 | 500 | `Edu--019750.wav` | `Edu--020361.wav` |
| `Edu_0034.tar` | 8,369,879,040 | 500 | `Edu--020362.wav` | `Edu--020976.wav` |
| `Edu_0035.tar` | 7,440,353,280 | 500 | `Edu--020978.wav` | `Edu--021596.wav` |
| `Edu_0036.tar` | 7,846,338,560 | 500 | `Edu--021597.wav` | `Edu--022236.wav` |
| `Edu_0037.tar` | 7,735,255,040 | 500 | `Edu--022237.wav` | `Edu--022839.wav` |
| `Edu_0038.tar` | 8,289,576,960 | 500 | `Edu--022840.wav` | `Edu--023461.wav` |
| `Edu_0039.tar` | 8,089,958,400 | 500 | `Edu--023462.wav` | `Edu--024072.wav` |
| `Edu_0040.tar` | 7,522,283,520 | 500 | `Edu--024073.wav` | `Edu--024695.wav` |
| `Edu_0041.tar` | 8,120,791,040 | 500 | `Edu--024696.wav` | `Edu--025321.wav` |
| `Edu_0042.tar` | 7,413,319,680 | 500 | `Edu--025322.wav` | `Edu--025946.wav` |
| `Edu_0043.tar` | 7,444,357,120 | 500 | `Edu--025947.wav` | `Edu--026565.wav` |

### 3.3 WAV 头部样例

本轮读取了每个分片首成员的 WAV 头，并额外读取 3 轨、4 轨样本：

| 音频成员 | 所在分片 | 轨道数 | 采样宽度 | 采样率 | 编码 | 帧数 | 时长 |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: |
| `Edu--000001.wav` | `Edu_0001.tar` | 2 | 2 字节 | 48,000 Hz | PCM | 1,121,280 | 23.360 s |
| `Edu--000198.wav` | `Edu_0001.tar` | 3 | 2 字节 | 48,000 Hz | PCM | 2,110,464 | 43.968 s |
| `Edu--002801.wav` | `Edu_0005.tar` | 4 | 2 字节 | 48,000 Hz | PCM | 1,529,856 | 31.872 s |

45 个头部样本均观察到：

- 线性 PCM，无压缩；
- 16 位采样；
- 48 kHz；
- WAV 轨道数与 JSON 的 `nTrack` 相等；
- WAV 头推导时长与 JSON 的 `timeLenInSec` 相等；
- 代表性样本满足 `文件字节数 = 44 + 帧数 × 轨道数 × 2`。

这些格式特征适合用作冻结脚本的全量核验目标。脚本应把 `channels`、`sample_width_bytes`、
`sample_rate`、`frames`、`duration_s` 写入每会话记录，不能仅依赖数据卡。

## 4. `jsons.tar.gz` 的容器结构

### 4.1 gzip 层

- 文件名：`Edu\jsons.tar.gz`
- 文件大小：`56,907,357` 字节
- 魔数：`1f 8b`
- SHA-256：
  `1e6451ed46ce7ad03b3baf53efc240b7fdd9a752e6ca3cbac3a78fd8f2693240`
- gzip 解压后 tar 大小：`438,927,360` 字节
- tar 内普通文件有效载荷合计：`422,004,799` 字节
- gzip 相对整个 tar 的压缩率约 87.0%

### 4.2 tar 层

- 普通文件成员：22,050
- 其他成员类型：0
- 重复成员名：0
- 成员名正则不匹配：0
- 成员顺序：按六位数字会话编号严格递增
- 首成员：`Edu--000001.json`
- 末成员：`Edu--027260.json`
- 会话编号最小值：1
- 会话编号最大值：27,260
- 该整数范围内缺失编号：5,210 个
- 前 30 个缺失编号：
  `19, 20, 23, 24, 34, 38, 44, 47, 50, 60, 63, 64, 81, 84, 87, 95, 96, 101, 103, 107, 109, 113, 122, 126, 135, 137, 148, 164, 167, 176`

JSON 成员大小分布：

| 统计量 | 字节数 |
| --- | ---: |
| 最小 | 1,383 |
| 25% 分位 | 7,167 |
| 中位数 | 13,335.5 |
| 75% 分位 | 25,340.5 |
| 95% 分位 | 54,799.05 |
| 最大 | 157,038 |
| 均值 | 19,138.54 |

tar 成员权限统一为 `0600`，成员所有者元数据统一为 `uid=10131`、`gid=504`、
`uname=cywang`、`gname=all`。这些打包机元数据不适合参与训练语义或会话标识，只宜作为
可选取证字段。

### 4.3 JSON 文本编码样例

首个 JSON 成员观察到：

- UTF-8，无字节顺序标记；
- 两空格缩进；
- 字段之间有换行；
- 文件末尾没有换行。

正式清单应解析 JSON 后进行规范化，不能依赖空白、缩进或原始字段顺序。

## 5. 真实 JSON 根结构

### 5.1 根字段全集

全部 22,050 个 JSON 的根键集合完全相同：

```text
asr | fs | nTrack | timeLenInSec
```

| 字段 | 出现数 | 类型 | 观察值或含义 |
| --- | ---: | --- | --- |
| `nTrack` | 22,050 | `int` | 2、3、4 |
| `timeLenInSec` | 22,050 | `float` | 会话时长，秒 |
| `fs` | 22,050 | `int` | 全部为 48,000 |
| `asr` | 22,050 | `list` | 外层长度始终等于 `nTrack` |

`nTrack` 分布：

| 轨道数 | 全注释包会话数 | 有音频配对会话数 | 无音频注释数 |
| ---: | ---: | ---: | ---: |
| 2 | 21,761 | 21,218 | 543 |
| 3 | 275 | 268 | 7 |
| 4 | 14 | 14 | 0 |
| 合计 | 22,050 | 21,500 | 550 |

全注释包会话时长分布：

| 统计量 | 秒 |
| --- | ---: |
| 最小 | 10.000 |
| 25% 分位 | 28.591984 |
| 中位数 | 55.312 |
| 75% 分位 | 108.588 |
| 95% 分位 | 238.124 |
| 最大 | 598.176 |
| 均值 | 80.966155 |
| 总计 | 495.917697645 小时 |

按轨道数统计的会话时长：

| 轨道数 | 小时 |
| ---: | ---: |
| 2 | 487.764745561 |
| 3 | 7.662520984 |
| 4 | 0.490431100 |

所有会话均满足 `len(asr) == nTrack`，所有轨道的句子列表均非空。

### 5.2 根结构的脱敏样例

以下样例保留真实字段名、类型和嵌套关系，文本值已替换：

```json
{
  "nTrack": 2,
  "timeLenInSec": 23.36,
  "fs": 48000,
  "asr": [
    [
      {
        "startInMs": 3664,
        "endInMs": 17280,
        "LID": "cn",
        "labels": {
          "gender": "男声",
          "age": "青年",
          "emotion": "<开放文本标签>",
          "accent": "标准普通话",
          "paralinguistic": "无",
          "txt": "<转写文本>"
        },
        "snr": 32.6006,
        "mos": 2.3734209537506104,
        "AED_dasheng": [
          {
            "Speech": 0.9249386191368103
          },
          {
            "<音频事件类别>": 0.0
          },
          {
            "<音频事件类别>": 0.0
          }
        ],
        "speaker": {
          "numSpeakers": 1,
          "multiSpeaker": false,
          "segments": [
            {
              "speaker": "SPEAKER_00",
              "startSec": 3.664,
              "endSec": 10.264
            }
          ]
        },
        "state": "<|complete|>"
      }
    ],
    [
      {
        "startInMs": 1092,
        "endInMs": 2792,
        "LID": "cn",
        "labels": {
          "txt": "<转写文本或事件文本>"
        },
        "snr": -44.0072,
        "mos": 3.0580461025238037,
        "AED_dasheng": [
          {
            "<音频事件类别>": 0.3108343482017517
          },
          {
            "<音频事件类别>": 0.0
          },
          {
            "<音频事件类别>": 0.0
          }
        ],
        "speaker": {
          "numSpeakers": 0,
          "multiSpeaker": false,
          "segments": []
        }
      }
    ]
  ]
}
```

轨道编号由 `asr` 外层列表位置给出。句子对象内部没有 `channel` 或 `channelIndex` 字段。

## 6. 句级结构与覆盖率

### 6.1 三种主要句级结构

全部 22,050 个 JSON 共有 397,192 条句级记录，观察到三种句级键集合：

| 结构 | 数量 | 占全部句级记录 |
| --- | ---: | ---: |
| 标准字段并含 `state` | 320,783 | 80.762704% |
| 标准字段但缺 `state` | 76,397 | 19.234274% |
| 隐私删改占位 | 12 | 0.003021% |

含 `state` 的标准字段集合：

```text
startInMs
endInMs
LID
labels
snr
mos
AED_dasheng
speaker
state
```

缺 `state` 的标准结构只有前八项。

隐私删改占位结构只有：

```json
{
  "startInMs": 0,
  "endInMs": 0,
  "sensitiveRedacted": true
}
```

上例只表示键结构；实际时间值应从原记录保留。

### 6.2 句级字段频率和类型

| 字段 | 出现数 | 类型 |
| --- | ---: | --- |
| `startInMs` | 397,192 | 全部 `int` |
| `endInMs` | 397,192 | 全部 `int` |
| `LID` | 397,180 | 全部 `str` |
| `labels` | 397,180 | 全部 `dict` |
| `snr` | 397,180 | 全部 `float` |
| `mos` | 397,180 | 全部 `float` |
| `AED_dasheng` | 397,180 | 全部 `list` |
| `speaker` | 397,180 | 全部 `dict` |
| `state` | 320,783 | 全部 `str` |
| `sensitiveRedacted` | 12 | 全部 `bool` |

数据卡列出的句级直接字段 `txt`、`privacyFlag`、`asrRes` 在全部 397,192 条记录中均未出现。

### 6.3 `state` 的真实值

| 原始值 | 数量 | 占有标签记录 |
| --- | ---: | ---: |
| `<\|complete\|>` | 208,394 | 64.964166% |
| `<\|incomplete\|>` | 72,511 | 22.604377% |
| `<\|backchannel\|>` | 39,032 | 12.167727% |
| `<\|wait\|>` | 846 | 0.263730% |
| 合计 | 320,783 | 100% |

另有 76,409 条记录没有 `state`，占全部句级记录的 19.237296%。其中包括 76,397 条标准
记录和 12 条隐私删改占位。

归一化时应显式映射：

```text
<|complete|>    -> complete
<|incomplete|>  -> incomplete
<|backchannel|> -> backchannel
<|wait|>        -> wait
缺字段           -> null
```

不能直接把缺字段映射为 `wait`、`incomplete` 或任何其他类别。

### 6.4 `labels` 的三种键集合

| `labels` 键集合 | 数量 | 占全部句级记录 |
| --- | ---: | ---: |
| `gender, age, emotion, accent, paralinguistic, txt` | 313,415 | 78.907682% |
| 只有 `txt` | 83,739 | 21.082751% |
| `candidates, usageMetadata, modelVersion, responseId, txt` | 26 | 0.006546% |
| 没有 `labels` 的隐私占位 | 12 | 0.003021% |

`labels.txt` 在 397,180 条非隐私占位记录中都存在，类型全部为字符串；其中 11,863 条为空
或只含空白，占全部句级记录的 2.986717%。这 11,863 条全部缺 `state`。

`state` 与 `labels` 结构的交叉分布：

| `state` | 丰富标签 | 仅 `txt` | 模型响应元数据 | 隐私占位 | 合计 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `<\|complete\|>` | 203,196 | 5,178 | 20 | 0 | 208,394 |
| `<\|incomplete\|>` | 70,906 | 1,603 | 2 | 0 | 72,511 |
| `<\|backchannel\|>` | 38,499 | 529 | 4 | 0 | 39,032 |
| `<\|wait\|>` | 814 | 32 | 0 | 0 | 846 |
| 缺字段 | 0 | 76,397 | 0 | 12 | 76,409 |

26 条“模型响应元数据”记录的 `labels` 内嵌了 `gemini-2.5-pro` 响应结构，同时仍有非空
`txt`。训练清单应给这些记录添加 `label_schema=model_response_metadata` 标志，避免把
`candidates` 等上游模型元数据送入训练样本。

### 6.5 语言标识

| `LID` 值 | 数量 |
| --- | ---: |
| `cn` | 364,919 |
| `unknown` | 27,338 |
| `en` | 4,923 |
| 缺字段 | 12 |

数据集总体语言为中文，但句级 `LID` 不能冻结为单一 `cn`。`unknown` 和 `en` 均为实际观测
值。

### 6.6 `speaker` 结构

全部 397,180 条非隐私占位记录的 `speaker` 键集合完全一致：

```text
numSpeakers | multiSpeaker | segments
```

`numSpeakers` 分布：

| 值 | 数量 |
| ---: | ---: |
| 0 | 24,660 |
| 1 | 364,518 |
| 2 | 7,999 |
| 3 | 3 |

`multiSpeaker` 分布：

| 值 | 数量 |
| --- | ---: |
| `false` | 389,178 |
| `true` | 8,002 |

`speaker.segments` 总计 727,408 项，每项键集合均为：

```text
speaker | startSec | endSec
```

类型分别为 `str`、`float`、`float`。`speaker` 局部名称分布：

| 值 | 数量 |
| --- | ---: |
| `SPEAKER_00` | 716,910 |
| `SPEAKER_01` | 10,491 |
| `SPEAKER_02` | 7 |

数据卡已经提示这些名称是单条句子内的局部说话人标识，不能当作跨句、跨轨或跨会话的全局
身份。

### 6.7 `AED_dasheng` 结构

- 397,180 条非隐私占位记录均含该字段。
- 每个列表长度都为 3。
- 1,191,540 个列表元素全部为字典。
- 共观察到 360 个不同事件键，属于开放类别空间。
- 高频键包括 `Speech`、`Narration, monologue`、`Female speech, woman speaking`、
  `Speech synthesizer`、`Clicking`、`Silence`、`Conversation`、`Sigh` 等。

冻结脚本可以记录列表长度、元素类型和事件键全集哈希；训练解析器不宜把 360 个事件键写死
成一个短枚举。

### 6.8 时间边界

- `startInMs` 和 `endInMs` 全部为整数。
- 没有负时间。
- 没有 `endInMs < startInMs`。
- 有 157 条记录的 `endInMs` 比 `timeLenInSec × 1000` 大 1 ms 以上。
- 其中 65 条超出 10 ms。
- 最大超出量为 `16.0208333333 ms`。
- 没有超出 100 ms 的记录。

这看起来符合约 16 ms 的边界量化误差。清单应保留原始时间并登记
`end_overrun_ms`；正式训练转换可在另行批准后使用小容差裁剪。清单生成阶段不应直接修改
原标注。

## 7. 配对关系与局部快照缺口

### 7.1 精确集合关系

以去掉扩展名后的完整基名作为会话标识：

```text
audio_ids = 所有 tar 内 *.wav 的 stem
json_ids  = jsons.tar.gz 内所有 *.json 的 stem
```

全量集合比较结果：

| 集合 | 数量 |
| --- | ---: |
| `audio_ids` | 21,500 |
| `json_ids` | 22,050 |
| 交集 | 21,500 |
| `audio_ids - json_ids` | 0 |
| `json_ids - audio_ids` | 550 |

550 个 JSON 独有项：

- 首项：`Edu--026566`
- 末项：`Edu--027260`
- 数量：550
- 时长：12.184188154 小时
- 轨道数：543 个 2 轨，7 个 3 轨
- 句级记录：9,547
- 带 `state`：7,782
- 缺 `state`：1,765

这些编号内部仍有自然缺口，不能把 `026566..027260` 的每个整数都视为存在。

### 7.2 可配对训练候选统计

| 指标 | 可配对部分 |
| --- | ---: |
| 会话 | 21,500 |
| 时长 | 483.733509491 小时 |
| 轨道 | 43,296 |
| 句级记录 | 387,645 |
| 2 轨会话 | 21,218 |
| 3 轨会话 | 268 |
| 4 轨会话 | 14 |
| `<\|complete\|>` | 203,311 |
| `<\|incomplete\|>` | 70,821 |
| `<\|backchannel\|>` | 38,039 |
| `<\|wait\|>` | 830 |
| 缺 `state` | 74,644 |
| 丰富 `labels` | 305,792 |
| 仅 `labels.txt` | 81,817 |
| 模型响应元数据结构 | 25 |
| 隐私删改占位 | 11 |
| 空或空白 `labels.txt` | 11,584 |

“可配对”只表示音频和注释基名一致。正式训练资格还需要全量 WAV 头、字段合法性、内容哈希、
隐私占位、空文本和具体训练任务要求的核验。

### 7.3 数据卡声明与本机观察对照

| 项目 | 数据卡声明 | 本机注释包 | 本机音频可配对部分 |
| --- | --- | --- | --- |
| 会话或音频数 | 93,709 | 22,050 | 21,500 |
| 总时长 | 2,000.21 h | 495.917698 h | 483.733509 h |
| 场景 | 教育、闲聊 | 只有 `Edu` | 只有 `Edu` |
| 最小时长 | 8.0 s | 10.0 s | 未全量读取 WAV 头 |
| 最大时长 | 618.3 s | 598.176 s | 未全量读取 WAV 头 |
| 音频/注释配对 | 每会话配对 | 多 550 个 JSON | 21,500 个已配对 |

当前快照应被清单标记为局部下载或局部发布物。本文不推断缺失内容的上游文件名，也不把
550 个尾部注释自动解释为某个具体缺失分片。

## 8. 数据卡与真实结构的差异

| 数据卡描述 | 全量实测 |
| --- | --- |
| 根对象含 `vadFrmLenInMs` | 22,050 个根对象均无此字段 |
| 根对象含 `vadFlagPerFrmPerTrack` | 22,050 个根对象均无此字段 |
| 句级直接含 `txt` | 0 条；文本位于 `labels.txt` |
| 句级含 `privacyFlag` | 0 条 |
| 句级含 `asrRes` | 0 条 |
| `state` 是句级字段 | 320,783 条存在，76,409 条缺失 |
| 每会话音频与 JSON 配对 | 本机有 550 个 JSON 无对应音频 |
| 下载后按场景顶层文件夹组织 | 本机只有 `Edu`，音频和注释分别打包 |
| 双方共享统一数据设计 | 字段语义相近，实际容器和字段路径仍需专用解析 |

冻结脚本应把本机实测结构作为当前快照契约，并把 `README.md` 内容哈希作为来源证据。未来
重新下载时，只要数据卡哈希、外层文件表、成员指纹或字段指纹发生变化，就应重新探测并生成
新版本清单。

## 9. 当前仓库代码需要注意的接口差异

### 9.1 `wp2_freeze_splits.py`

当前逻辑：

```python
dc_sessions = sorted(p.stem for p in dc_dir.rglob("*.json"))
```

发布物外层没有散装 JSON，因此该表达式返回空列表。DuplexConv 会话发现应改为流式读取
`Edu/jsons.tar.gz` 的成员名，并在冻结前与全部音频 tar 成员名做集合配对。

### 9.2 `src/floor_circuit/data/smoothconv.py`

当前解析器与真实 DuplexConv 的关键差异：

| 当前假设 | 真实 DuplexConv |
| --- | --- |
| 根部有 `segments/utterances/annotations/...` 列表 | 根部是 `asr[track][sentence]` |
| 句子内有通道字段 | 通道由 `asr` 外层索引给出 |
| 时间字段为 `start/end/...` | `startInMs/endInMs` |
| 文本字段为 `text/transcript/...` | `labels.txt` |
| 标签值为 `complete/incomplete/...` | `<\|complete\|>` 等带包裹符号的值 |
| 可复用 SmoothConv 的平面条目迭代 | 需要 DuplexConv 专用嵌套迭代器 |

因此训练清单冻结脚本适合先实现独立的“容器盘点与会话索引”层。解析成统一训练 schema 的
步骤可以复用归一化输出字段，但不能继续沿用当前 `_iter_items()`。

## 10. 训练侧清单冻结契约建议

以下内容是基于本轮观察提出的脚本契约草案，还没有写入冻结配置或 `PREREG.md`。

### 10.1 建议分成三个产物

1. `release_inventory.json`
   - 描述外层 46 个文件、内容哈希、大小、场景覆盖和完整性状态；
   - 描述 43 个 tar 与一个 tar.gz 的成员计数和成员清单哈希；
   - 当前应写 `snapshot_status: "partial"`。
2. `training_manifest.jsonl` 或 `training_manifest.parquet`
   - 每行一个音频与注释均存在的会话；
   - 当前候选行数为 21,500；
   - 逐行保留音频分片、成员名、JSON 成员名、时长、轨道数、标签覆盖和质量标志。
3. `exclusions.jsonl`
   - 每行一个未进入训练清单的会话；
   - 当前至少包含 550 个 `json_without_audio`；
   - 还可按具体训练任务登记隐私占位、空文本或缺标签排除理由。

### 10.2 推荐的发布物清单骨架

```json
{
  "schema_version": 1,
  "dataset": "DuplexConv",
  "license_id": "CC-BY-NC-4.0",
  "snapshot_scope": "Edu_only_local_partial",
  "snapshot_status": "partial",
  "freeze_status": "blocked",
  "source_root": "D:/dataset/audio/Full_Duplex/qualialabsAI__DuplexConv",
  "domains_present": [
    "Edu"
  ],
  "domains_declared_but_absent": [
    "none_Edu"
  ],
  "outer_files": {
    "count": 46,
    "bytes": 337394599212
  },
  "audio_archives": {
    "count": 43,
    "member_count": 21500
  },
  "annotation_archive": {
    "path": "Edu/jsons.tar.gz",
    "sha256": "1e6451ed46ce7ad03b3baf53efc240b7fdd9a752e6ca3cbac3a78fd8f2693240",
    "member_count": 22050
  },
  "pairing": {
    "paired": 21500,
    "audio_only": 0,
    "json_only": 550
  }
}
```

Windows 本机路径可记录为来源元数据，清单内部所有相对路径建议统一使用 `/`。这样可以避免
PowerShell、Python 和后续 Linux 计算节点产生路径分隔符差异。

### 10.3 推荐的单会话记录骨架

```json
{
  "session_id": "Edu--000001",
  "domain": "Edu",
  "audio": {
    "archive": "Edu/audios/Edu_0001.tar",
    "member": "Edu--000001.wav",
    "member_size_bytes": 4485164,
    "channels": 2,
    "sample_width_bytes": 2,
    "sample_rate": 48000,
    "frames": 1121280,
    "duration_s": 23.36
  },
  "annotation": {
    "archive": "Edu/jsons.tar.gz",
    "member": "Edu--000001.json",
    "member_size_bytes": 5019,
    "nTrack": 2,
    "timeLenInSec": 23.36,
    "utterance_count": 5,
    "state_counts": {
      "complete": 0,
      "incomplete": 0,
      "backchannel": 0,
      "wait": 0,
      "missing": 0
    }
  },
  "quality_flags": []
}
```

上例中的 `state_counts` 数字应由脚本从真实会话计算，示例中的零仅作字段占位。

### 10.4 建议的执行顺序

1. 枚举外层普通文件，规范化相对路径，记录大小。
2. 计算 `README.md`、`.gitattributes`、`jsons.tar.gz` 和全部音频 tar 的 SHA-256。
3. 只读枚举所有 tar 成员，拒绝绝对路径、`..`、链接、重复成员和非预期扩展名。
4. 用成员基名构造 `audio_ids` 与 `json_ids`，计算交集和两侧差集。
5. 逐个解析 JSON，验证根键、字段类型、`len(asr) == nTrack` 和轨道非空。
6. 逐个读取 WAV 头，验证 PCM、采样宽度、采样率、轨道数、帧数和时长。
7. 生成逐会话候选行，并把所有缺陷写为显式质量标志。
8. 生成排除清单，确保每个发布物会话恰好落入“训练候选”或“排除”之一。
9. 对规范化清单计算 SHA-256，先发布到临时路径，所有断言通过后再原子改名。
10. 把小型摘要写入仓库 `reports/`；大型逐会话清单写入
    `D:\data_storage\The_Floor_Control_Circuit`。
11. 用户批准局部快照处置后，再生成 `configs/splits/duplexconv.json` 并按仓库流程回填
    预注册指纹。

### 10.5 建议的硬失败条件

- 外层内容哈希与已冻结快照不一致。
- tar 内存在绝对路径、路径穿越、链接、重复成员或未知成员类型。
- 音频或注释成员名不符合约定正则。
- 同一个 `session_id` 映射到多个音频或多个 JSON。
- JSON 无法解析。
- 根键结构或字段类型发生未登记变化。
- `nTrack != len(asr)`。
- 任一轨道类型不为列表，或轨道为空。
- 时间为负，或 `endInMs < startInMs`。
- 音频轨道数与 `nTrack` 不一致。
- 音频采样率、采样宽度或编码不在获批范围。
- 存在音频无注释。
- 以“完整发布版”为目标时，存在注释无音频、缺失声明场景或总量明显低于数据卡。

### 10.6 建议的显式告警或排除标志

```text
json_without_audio
state_missing
text_empty
sensitive_redacted
label_schema_model_response_metadata
language_unknown
language_en
multi_speaker_utterance
three_track_session
four_track_session
end_overrun_within_small_tolerance
```

3、4 轨会话属于真实发布结构，适合标志化处理。若某个训练器只支持双通道，应在训练任务
配置中写明选择或排除规则，并冻结对应会话集合；清单层不要静默截断轨道。

### 10.7 完整性状态建议

建议区分：

```text
release_complete
paired_local_subset
```

当前观察值：

```text
release_complete.status = blocked
paired_local_subset.count = 21500
paired_local_subset.status = awaiting_explicit_approval
```

如果选择等待补齐发布物，冻结脚本应退出且不发布最终清单。如果选择当前配对子集，最终清单
必须携带 `Edu_only_local_partial` 范围声明、550 项排除清单哈希和缺失场景声明。

## 11. 本轮规范化元数据指纹

以下指纹方便核对未来脚本输出。它们只覆盖规范化路径、成员名、成员大小或会话标识，不能替代
外层 tar 的完整内容哈希。

### 11.1 规范化规则

音频成员清单的每行：

```text
<外层tar相对路径>\t<成员名>\t<成员字节数>\n
```

JSON 成员清单的每行：

```text
Edu/jsons.tar.gz\t<成员名>\t<成员字节数>\n
```

会话标识清单的每行：

```text
<session_id>\n
```

所有行均按相应元组升序排列，使用 UTF-8 和 LF。

### 11.2 指纹值

| 对象 | 行数 | 规范化字节数 | SHA-256 |
| --- | ---: | ---: | --- |
| 音频成员元数据 | 21,500 | 1,043,332 | `732208ed48d2a325f58d67026b9c55230dfc7a2c226f1b7ce573b1bae6eb2a17` |
| JSON 成员元数据 | 22,050 | 873,742 | `91275d8b021cbf747917082d3852a88901e786db7c58268032f89c632eff9cc9` |
| 配对会话标识 | 21,500 | 258,000 | `ad935a93b39e66f6123efc63dfca1d4813a2019480066d8a98d37f4309a3849a` |
| JSON 独有会话标识 | 550 | 6,600 | `4f170173532b006c1eab6875074f48a3d9b5e02aecee558e70782ba694398899` |
| 外层相对路径与大小 | 46 | 1,566 | `b93888da7560685a527dbe9a3d8588e5a77c91db3a9dc622de9acf431ccf94e2` |

若正式脚本采用不同规范化格式，应在清单中登记格式版本和字节级规则，避免把不同序列化结果
误判为发布物漂移。

## 12. 已知异常位置

### 12.1 12 条隐私删改占位

```text
Edu--001029.json#track=1,sentence=6
Edu--004537.json#track=0,sentence=4
Edu--007634.json#track=1,sentence=2
Edu--008827.json#track=0,sentence=3
Edu--011096.json#track=0,sentence=1
Edu--015789.json#track=1,sentence=0
Edu--016532.json#track=1,sentence=3
Edu--018806.json#track=0,sentence=4
Edu--022381.json#track=1,sentence=0
Edu--024889.json#track=0,sentence=10
Edu--025518.json#track=1,sentence=5
Edu--026655.json#track=1,sentence=9
```

前 11 条位于音频可配对集合；最后 1 条位于 JSON 独有集合。

### 12.2 26 条模型响应元数据标签结构

```text
Edu--000320.json#track=1,sentence=19
Edu--001249.json#track=0,sentence=7
Edu--003706.json#track=0,sentence=16
Edu--003708.json#track=1,sentence=39
Edu--004090.json#track=1,sentence=7
Edu--004472.json#track=1,sentence=15
Edu--007418.json#track=0,sentence=6
Edu--008362.json#track=1,sentence=25
Edu--010942.json#track=1,sentence=3
Edu--010946.json#track=0,sentence=6
Edu--012402.json#track=1,sentence=1
Edu--014469.json#track=0,sentence=2
Edu--015030.json#track=1,sentence=9
Edu--015577.json#track=0,sentence=27
Edu--016090.json#track=1,sentence=2
Edu--016671.json#track=1,sentence=4
Edu--016900.json#track=0,sentence=8
Edu--017673.json#track=0,sentence=4
Edu--018952.json#track=0,sentence=17
Edu--019528.json#track=0,sentence=7
Edu--020503.json#track=0,sentence=2
Edu--021056.json#track=0,sentence=8
Edu--021625.json#track=2,sentence=8
Edu--022683.json#track=0,sentence=12
Edu--026545.json#track=0,sentence=14
Edu--026702.json#track=1,sentence=0
```

前 25 条位于音频可配对集合；最后 1 条位于 JSON 独有集合。

## 13. 冻结前需要用户裁决的最小问题

训练侧脚本可以直接按本文实现探测和候选清单生成，但最终“冻结”仍需选择快照范围：

1. 等待补齐闲聊场景及 550 个缺音频注释对应资产，再冻结完整发布物；
2. 显式批准 `Edu_only_local_partial`，冻结当前 21,500 个配对会话，并永久登记 550 个
   `json_without_audio` 排除项和缺失闲聊场景。

在该裁决发生前，建议脚本只产出 `candidate` 清单和审计摘要，拒绝写入最终
`configs/splits/duplexconv.json`，也不更新预注册指纹。
