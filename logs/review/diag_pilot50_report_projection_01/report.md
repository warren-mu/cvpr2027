# Pilot50 原记录投影

本报告仅消费封存成功结果及原 comparison；不重跑训练/native、不授予论文结果或正式训练冻结。学习判定调用原函数并与原 summary 逐值一致；辅助计数与耗时预测不改变判据。

原 decision：`learning_change_detected_review_common_config_freeze`；stop_and_report=`false`；common_config_frozen=`false`。

## 原学习判定

窗口为每次更新的 Look 比率/奖励再等权平均十次，不按窗口回合数加权。

| arm | seen 1–10 | seen 41–50 | reward 1–10 | reward 41–50 | abs Δseen | abs Δreward | 两项均<.05 |
|---|---|---|---|---|---|---|---|
| T0 | 0.000000 | 0.000000 | 0.558449 | 0.740716 | 0.000000 | 0.182267 | False |
| T4 | 0.000000 | 0.000000 | 0.562657 | 0.719141 | 0.000000 | 0.156483 | False |

两臂两个判定窗口的 seen 均为 0；即使奖励变化超过阈值，也不能据此宣称学会观察。

## Greedy holdout：0 / 50

固定训练留出场景；每对两个 world 各一回合，不是最终测试集。总体成功率按所有回合汇总，未作类别宏平均。

| arm | update | class | episodes | success | success/episodes | reward/episodes | seen/episodes |
|---|---|---|---|---|---|---|---|
| T0 | 0 | direct_flip | 10 | 7 | 0.700000 | 0.700000 | unknown |
| T0 | 0 | direct_stay | 12 | 9 | 0.750000 | 0.750000 | unknown |
| T0 | 0 | look_flip | 8 | 4 | 0.500000 | 0.500000 | 0.000000 |
| T0 | 0 | look_stay | 10 | 2 | 0.200000 | 0.200000 | 0.000000 |
| T0 | 0 | all episodes | 40 | 22 | 0.550000 | 0.550000 | unknown |
| T0 | 50 | direct_flip | 10 | 6 | 0.600000 | 0.595000 | unknown |
| T0 | 50 | direct_stay | 12 | 8 | 0.666667 | 0.666667 | unknown |
| T0 | 50 | look_flip | 8 | 4 | 0.500000 | 0.475000 | 0.000000 |
| T0 | 50 | look_stay | 10 | 3 | 0.300000 | 0.290000 | 0.000000 |
| T0 | 50 | all episodes | 40 | 21 | 0.525000 | 0.516250 | unknown |
| T4 | 0 | direct_flip | 10 | 7 | 0.700000 | 0.700000 | unknown |
| T4 | 0 | direct_stay | 12 | 9 | 0.750000 | 0.750000 | unknown |
| T4 | 0 | look_flip | 8 | 4 | 0.500000 | 0.500000 | 0.000000 |
| T4 | 0 | look_stay | 10 | 2 | 0.200000 | 0.200000 | 0.000000 |
| T4 | 0 | all episodes | 40 | 22 | 0.550000 | 0.550000 | unknown |
| T4 | 50 | direct_flip | 10 | 4 | 0.400000 | 0.400000 | unknown |
| T4 | 50 | direct_stay | 12 | 9 | 0.750000 | 0.750000 | unknown |
| T4 | 50 | look_flip | 8 | 4 | 0.500000 | 0.500000 | 0.000000 |
| T4 | 50 | look_stay | 10 | 2 | 0.200000 | 0.200000 | 0.000000 |
| T4 | 50 | all episodes | 40 | 19 | 0.475000 | 0.475000 | unknown |

## 每 10 步四类原始快照

这些是第 10/20/30/40/50 次更新本身；不是前十步累计。完整原 class 字典、每一步及日志 SHA 均保存在 JSON。

| arm | update | class | episodes | success | reward sum | seen | premature count | premature/episodes | unparsable | decisions | valid observe | valid actions |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T0 | 10 | direct_flip | 8 | 6 | 6.000000 | unknown | unknown | unknown | 0 | 8 | 0 | 8 |
| T0 | 10 | direct_stay | 31 | 22 | 21.900000 | unknown | unknown | unknown | 0 | 33 | 2 | 33 |
| T0 | 10 | look_flip | 19 | 8 | 8.000000 | 0 | 19 | 1.000000 | 0 | 19 | 0 | 19 |
| T0 | 10 | look_stay | 6 | 4 | 4.000000 | 0 | 6 | 1.000000 | 0 | 6 | 0 | 6 |
| T0 | 20 | direct_flip | 11 | 11 | 11.000000 | unknown | unknown | unknown | 0 | 11 | 0 | 11 |
| T0 | 20 | direct_stay | 19 | 19 | 19.000000 | unknown | unknown | unknown | 0 | 19 | 0 | 19 |
| T0 | 20 | look_flip | 19 | 8 | 8.000000 | 0 | 19 | 1.000000 | 0 | 19 | 0 | 19 |
| T0 | 20 | look_stay | 15 | 10 | 9.950000 | 0 | 15 | 1.000000 | 0 | 16 | 1 | 16 |
| T0 | 30 | direct_flip | 7 | 7 | 7.000000 | unknown | unknown | unknown | 0 | 7 | 0 | 7 |
| T0 | 30 | direct_stay | 20 | 20 | 20.000000 | unknown | unknown | unknown | 0 | 20 | 0 | 20 |
| T0 | 30 | look_flip | 14 | 7 | 7.000000 | 0 | 14 | 1.000000 | 0 | 14 | 0 | 14 |
| T0 | 30 | look_stay | 23 | 21 | 20.900000 | 0 | 23 | 1.000000 | 0 | 25 | 2 | 25 |
| T0 | 40 | direct_flip | 9 | 9 | 9.000000 | unknown | unknown | unknown | 0 | 9 | 0 | 9 |
| T0 | 40 | direct_stay | 27 | 25 | 24.900000 | unknown | unknown | unknown | 0 | 29 | 2 | 29 |
| T0 | 40 | look_flip | 12 | 7 | 6.450000 | 0 | 12 | 1.000000 | 0 | 23 | 11 | 23 |
| T0 | 40 | look_stay | 16 | 16 | 15.950000 | 0 | 16 | 1.000000 | 0 | 17 | 1 | 17 |
| T0 | 50 | direct_flip | 7 | 7 | 7.000000 | unknown | unknown | unknown | 0 | 7 | 0 | 7 |
| T0 | 50 | direct_stay | 23 | 23 | 22.900000 | unknown | unknown | unknown | 0 | 25 | 2 | 25 |
| T0 | 50 | look_flip | 18 | 12 | 11.150000 | 0 | 18 | 1.000000 | 0 | 35 | 17 | 35 |
| T0 | 50 | look_stay | 16 | 16 | 15.650000 | 0 | 10 | 0.625000 | 0 | 23 | 7 | 23 |
| T4 | 10 | direct_flip | 6 | 6 | 6.000000 | unknown | unknown | unknown | 0 | 6 | 0 | 6 |
| T4 | 10 | direct_stay | 32 | 32 | 32.000000 | unknown | unknown | unknown | 0 | 32 | 0 | 32 |
| T4 | 10 | look_flip | 16 | 8 | 8.000000 | 0 | 16 | 1.000000 | 0 | 16 | 0 | 16 |
| T4 | 10 | look_stay | 10 | 6 | 6.000000 | 0 | 10 | 1.000000 | 0 | 10 | 0 | 10 |
| T4 | 20 | direct_flip | 10 | 10 | 10.000000 | unknown | unknown | unknown | 0 | 10 | 0 | 10 |
| T4 | 20 | direct_stay | 14 | 14 | 14.000000 | unknown | unknown | unknown | 0 | 14 | 0 | 14 |
| T4 | 20 | look_flip | 22 | 11 | 11.000000 | 0 | 22 | 1.000000 | 0 | 22 | 0 | 22 |
| T4 | 20 | look_stay | 18 | 18 | 17.900000 | 0 | 18 | 1.000000 | 0 | 20 | 2 | 20 |
| T4 | 30 | direct_flip | 10 | 10 | 10.000000 | unknown | unknown | unknown | 0 | 10 | 0 | 10 |
| T4 | 30 | direct_stay | 24 | 24 | 24.000000 | unknown | unknown | unknown | 0 | 24 | 0 | 24 |
| T4 | 30 | look_flip | 12 | 6 | 6.000000 | 0 | 12 | 1.000000 | 0 | 12 | 0 | 12 |
| T4 | 30 | look_stay | 18 | 18 | 17.900000 | 0 | 18 | 1.000000 | 0 | 20 | 2 | 20 |
| T4 | 40 | direct_flip | 10 | 10 | 10.000000 | unknown | unknown | unknown | 0 | 10 | 0 | 10 |
| T4 | 40 | direct_stay | 26 | 26 | 26.000000 | unknown | unknown | unknown | 0 | 26 | 0 | 26 |
| T4 | 40 | look_flip | 12 | 6 | 6.000000 | 0 | 12 | 1.000000 | 0 | 12 | 0 | 12 |
| T4 | 40 | look_stay | 16 | 16 | 16.000000 | 0 | 16 | 1.000000 | 0 | 16 | 0 | 16 |
| T4 | 50 | direct_flip | 4 | 4 | 4.000000 | unknown | unknown | unknown | 0 | 4 | 0 | 4 |
| T4 | 50 | direct_stay | 22 | 16 | 16.000000 | unknown | unknown | unknown | 0 | 22 | 0 | 22 |
| T4 | 50 | look_flip | 18 | 9 | 9.000000 | 0 | 18 | 1.000000 | 0 | 18 | 0 | 18 |
| T4 | 50 | look_stay | 20 | 20 | 20.000000 | 0 | 20 | 1.000000 | 0 | 20 | 0 | 20 |

## 辅助解析 / 观察率

均为原整数分子/分母在窗口内相加后求比；与学习判定的更新等权平均不同。分母为 0 时保留 unknown。解析失败用逻辑决策数作分母；观察动作同时列逻辑决策与有效动作两个分母。

| arm | window | scope | unparsable | decisions | unparsable/decisions | observe | valid actions | observe/decisions | observe/valid |
|---|---|---|---|---|---|---|---|---|---|
| T0 | early | all_classes | 0 | 662 | 0.000000 | 22 | 660 | 0.033233 | 0.033333 |
| T0 | early | Look | 0 | 343 | 0.000000 | 0 | 343 | 0.000000 | 0.000000 |
| T0 | early | direct_flip | 0 | 56 | 0.000000 | 0 | 56 | 0.000000 | 0.000000 |
| T0 | early | direct_stay | 0 | 263 | 0.000000 | 22 | 261 | 0.083650 | 0.084291 |
| T0 | early | look_flip | 0 | 191 | 0.000000 | 0 | 191 | 0.000000 | 0.000000 |
| T0 | early | look_stay | 0 | 152 | 0.000000 | 0 | 152 | 0.000000 | 0.000000 |
| T0 | late | all_classes | 0 | 844 | 0.000000 | 204 | 844 | 0.241706 | 0.241706 |
| T0 | late | Look | 0 | 510 | 0.000000 | 185 | 510 | 0.362745 | 0.362745 |
| T0 | late | direct_flip | 0 | 56 | 0.000000 | 0 | 56 | 0.000000 | 0.000000 |
| T0 | late | direct_stay | 0 | 278 | 0.000000 | 19 | 278 | 0.068345 | 0.068345 |
| T0 | late | look_flip | 0 | 327 | 0.000000 | 148 | 327 | 0.452599 | 0.452599 |
| T0 | late | look_stay | 0 | 183 | 0.000000 | 37 | 183 | 0.202186 | 0.202186 |
| T4 | early | all_classes | 0 | 656 | 0.000000 | 16 | 655 | 0.024390 | 0.024427 |
| T4 | early | Look | 0 | 348 | 0.000000 | 0 | 348 | 0.000000 | 0.000000 |
| T4 | early | direct_flip | 0 | 50 | 0.000000 | 0 | 50 | 0.000000 | 0.000000 |
| T4 | early | direct_stay | 0 | 258 | 0.000000 | 16 | 257 | 0.062016 | 0.062257 |
| T4 | early | look_flip | 0 | 194 | 0.000000 | 0 | 194 | 0.000000 | 0.000000 |
| T4 | early | look_stay | 0 | 154 | 0.000000 | 0 | 154 | 0.000000 | 0.000000 |
| T4 | late | all_classes | 0 | 656 | 0.000000 | 16 | 656 | 0.024390 | 0.024390 |
| T4 | late | Look | 0 | 340 | 0.000000 | 12 | 340 | 0.035294 | 0.035294 |
| T4 | late | direct_flip | 0 | 56 | 0.000000 | 4 | 56 | 0.071429 | 0.071429 |
| T4 | late | direct_stay | 0 | 260 | 0.000000 | 0 | 260 | 0.000000 | 0.000000 |
| T4 | late | look_flip | 0 | 182 | 0.000000 | 0 | 182 | 0.000000 | 0.000000 |
| T4 | late | look_stay | 0 | 158 | 0.000000 | 12 | 158 | 0.075949 | 0.075949 |

## 实测耗时与工程预测

steady 是 T0/T4 更新 2–50 的 49 次均值；T2 仅更新 2–5 的 4 次，单列敏感性。total_seconds 是完整更新；update_seconds 只是 critic+actor。

| arm | n | first total s | steady total s | generation s | scoring s | critic s | actor s | 200 update-only GPUh |
|---|---|---|---|---|---|---|---|---|
| T0 | 49 | 180.103264 | 108.683805 | 11.022824 | 31.111656 | 32.699100 | 33.850225 | 6.037989 |
| T4 | 49 | 165.805651 | 138.741822 | 6.860207 | 48.685522 | 32.427675 | 50.768418 | 7.707879 |
| T2 | 4 | 160.994265 | 117.891054 | 6.972882 | 42.660326 | 33.414872 | 34.842975 | 6.549503 |

| arm | source auth s | model init s | holdout (update:seconds) | original parent duration s | checkpoint save s |
|---|---|---|---|---|---|
| T0 | 217.455430 | 82.001759 | 0:27.139443, 20:15.886829, 40:4.042154, 50:5.906124 | 5983.600000 | unknown |
| T4 | 214.872909 | 69.735863 | 0:13.714617, 20:10.675017, 40:4.136804, 50:4.077234 | 7397.600000 | unknown |
| T2 | 245.286928 | 79.884265 |  | 1024.600000 | unknown |

200-step、T0/T2/T3/T4 × 3 seeds 的 update-only 情景：**82.475209 GPUh**。假设 T2:=T0、T3:=T4；不是 12 次实跑、不是资源上界。

仅将 T2 换成其 4 次 steady 更新的敏感性情景：**84.009750 GPUh**；T3 仍用 T4 proxy。

两种情景均不含启动、额外评测、保存、gate、清理、失败重试或计费倍率；未假定未来评测频率。actor 保存没有独立计时，不能用剩余时长冒充。实际占卡和扣费以原 canonical PBS 记录为准。

## 原证据

- `results/smoke_e2e_pilot_comparison_01.json`
- `results/smoke_e2e_pilot_T0_s0_01_retry4.json`
- `results/smoke_e2e_pilot_T4_s0_01_retry4.json`
- `results/smoke_e2e_pilot_T2_s0_01_retry1.json`
