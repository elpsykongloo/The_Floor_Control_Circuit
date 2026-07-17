# 刺激质检报告（S1，双轨协议）

## 最小对（complete vs incomplete，响度配平）：20 对，通过率 95.00%
- 未过：响度 1，采样率 0，削波 0（时长为前缀关系，不判死，见 duration_ratio）

## 变换对（原版 vs F0 拉平，时长 ±5% + 响度）：40 对，通过率 97.50%
- 未过：时长 0，响度 1

### 最小对未过明细

| id          | lang   | a                                                                                | b                                                                                  | sr_ok   | clip_ok   |   duration_diff_pct |   duration_ratio |   lufs_diff | loudness_ok   | duration_checked   | duration_ok   | pass   |
|:------------|:-------|:---------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|:--------|:----------|--------------------:|-----------------:|------------:|:--------------|:-------------------|:--------------|:-------|
| s1_zh_00_08 | zh     | D:\data_storage\The_Floor_Control_Circuit\stimuli\s1\zh\s1_zh_00_08\complete.wav | D:\data_storage\The_Floor_Control_Circuit\stimuli\s1\zh\s1_zh_00_08\incomplete.wav | True    | True      |              25.974 |          0.74026 |     1.24865 | False         | False              | True          | False  |

### 变换对未过明细

| id                          | lang   | a                                                                                | b                                                                                       | sr_ok   | clip_ok   |   duration_diff_pct |   duration_ratio |   lufs_diff | loudness_ok   | duration_checked   | duration_ok   | pass   |
|:----------------------------|:-------|:---------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------|:--------|:----------|--------------------:|-----------------:|------------:|:--------------|:-------------------|:--------------|:-------|
| s1_zh_00_08_complete_f0flat | zh     | D:\data_storage\The_Floor_Control_Circuit\stimuli\s1\zh\s1_zh_00_08\complete.wav | D:\data_storage\The_Floor_Control_Circuit\stimuli\s1\zh\s1_zh_00_08\complete_f0flat.wav | True    | True      |                   0 |                1 |     1.52267 | False         | True               | True          | False  |
