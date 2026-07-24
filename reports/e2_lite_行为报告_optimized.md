# E2-lite 行为报告（PREREG #34；探索性）

- 已分析运行：260；缺失：0
- 注入：L29，h ← h + α·s_v·v̂（s_v = 训练行投影标准差）

## 主方向剂量-反应（配对差 vs baseline）

### probe_a-4
- agent_speech_frac：Δ=-0.1208 [-0.1398,-0.1022]（n=20，+0/−20）
- overlap_frac：Δ=-0.0152 [-0.0224,-0.0086]（n=20，+2/−16）
- median_response_latency_s：Δ=+0.2038 [-0.1720,+0.6259]（n=13，+7/−6）
- response_rate：Δ=-0.1107 [-0.1492,-0.0741]（n=19，+0/−16）
- interruption_rate_per_min_user_speech：Δ=-4.6346 [-7.0236,-1.8231]（n=19，+2/−15）
- yield_rate_400ms：Δ=+0.1625 [-0.0901,+0.4077]（n=13，+6/−4）
- yield_rate_1000ms：Δ=+0.1707 [-0.1205,+0.4103]（n=13，+9/−3）

### probe_a-2
- agent_speech_frac：Δ=-0.0823 [-0.1077,-0.0561]（n=20，+2/−18）
- overlap_frac：Δ=-0.0080 [-0.0150,-0.0026]（n=20，+4/−14）
- median_response_latency_s：Δ=+0.1287 [-0.1917,+0.4743]（n=15，+7/−7）
- response_rate：Δ=-0.0654 [-0.1075,-0.0278]（n=19，+2/−12）
- interruption_rate_per_min_user_speech：Δ=-4.0961 [-6.2652,-2.0033]（n=19，+2/−14）
- yield_rate_400ms：Δ=+0.0259 [-0.1616,+0.2185]（n=16，+6/−8）
- yield_rate_1000ms：Δ=+0.1214 [-0.0560,+0.2848]（n=16，+8/−4）

### probe_a+2
- agent_speech_frac：Δ=+0.1033 [+0.0650,+0.1443]（n=20，+16/−4）
- overlap_frac：Δ=+0.0139 [+0.0069,+0.0216]（n=20，+14/−4）
- median_response_latency_s：Δ=+0.1191 [-0.0877,+0.3430]（n=17，+8/−7）
- response_rate：Δ=+0.0626 [+0.0086,+0.1174]（n=19，+11/−3）
- interruption_rate_per_min_user_speech：Δ=+7.7999 [+3.7135,+11.4121]（n=19，+15/−2）
- yield_rate_400ms：Δ=-0.0415 [-0.1863,+0.1114]（n=17，+5/−8）
- yield_rate_1000ms：Δ=-0.0917 [-0.2767,+0.0822]（n=17，+8/−6）

### probe_a+4
- agent_speech_frac：Δ=+0.3000 [+0.2277,+0.3767]（n=20，+20/−0）
- overlap_frac：Δ=+0.0587 [+0.0367,+0.0805]（n=20，+18/−1）
- median_response_latency_s：Δ=+0.1181 [-0.1322,+0.3900]（n=16，+11/−5）
- response_rate：Δ=+0.0518 [-0.0146,+0.1206]（n=19，+10/−7）
- interruption_rate_per_min_user_speech：Δ=+13.0087 [+5.8822,+22.0499]（n=19，+15/−3）
- yield_rate_400ms：Δ=-0.1661 [-0.3482,+0.0246]（n=18，+5/−12）
- yield_rate_1000ms：Δ=-0.2240 [-0.4074,-0.0585]（n=18，+2/−12）

## α 单调性（Spearman）

- agent_speech_frac：ρ=+0.871（p=7.52e-26，n=80）
- overlap_frac：ρ=+0.749（p=1.31e-15，n=80）
- median_response_latency_s：ρ=-0.000（p=9.97e-01，n=61）
- response_rate：ρ=+0.490（p=6.97e-06，n=76）
- interruption_rate_per_min_user_speech：ρ=+0.638（p=5.82e-10，n=76）
- yield_rate_400ms：ρ=-0.253（p=4.37e-02，n=64）
- yield_rate_1000ms：ρ=-0.393（p=1.33e-03，n=64）

## 差分均值方向与随机对照

- diffmeans_a-4：agent_speech_frac Δ=-0.1606 [-0.1799,-0.1420]（n=20）
- diffmeans_a+4：agent_speech_frac Δ=-0.1306 [-0.1512,-0.1079]（n=20）
- random_r0_a-4：agent_speech_frac Δ=+0.0090 [-0.0278,+0.0436]（n=20）
- random_r0_a+4：agent_speech_frac Δ=-0.0143 [-0.0413,+0.0127]（n=20）
- random_r1_a-4：agent_speech_frac Δ=+0.0104 [-0.0182,+0.0391]（n=20）
- random_r1_a+4：agent_speech_frac Δ=+0.0293 [-0.0018,+0.0546]（n=20）
- random_r2_a-4：agent_speech_frac Δ=-0.0086 [-0.0414,+0.0237]（n=20）
- random_r2_a+4：agent_speech_frac Δ=+0.0102 [-0.0223,+0.0446]（n=20）
