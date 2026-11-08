# 实验登记表 v3.2（预注册计划；修改需人类确认。任务说明在 tasks/，运行状态记在 tasks/README.md）

命名：`<阶段>_<臂>_s<种子>`。环境统一为 AI2-THOR 关系指称导航（离散视点，以 `select` 结束），观察后端由 D1 决定，全部实验同一后端。优先级 P3 > P2 > P1 > P0。

## T00 可行性（不进论文）
`t00_thor_checks`、`t00_thor_throughput`、`t00_vagen_smoke`

## D1 门槛实验（去留 #1，11/9 晚）
| 组 | 内容 | 训练？ | 优先级 |
|---|---|---|---|
| pilot_pairs | 四种情况各 ≥20 个世界对：有效性记录全部字段 + 人工核验截图 | 否 | P3 |
| obs_never / obs_always / obs_fixed{2,4} / obs_trigger | 免训练观察策略，套在 base 与 ctrl_ppo 检查点上评测 | 否 | P3 |
| diag_base / diag_ppo / diag_vagen | 主要指标与成功率–观察成本曲线 | 仅评测 | P3 |

## 主实验（11/11–13；同步数、同 rollout 数；T0、T1、T2、T4 必做 3 种子，T3、ctrl_vagen 应做）
| 组 | 内容 |
|---|---|
| main_T0 | PPO，统一接口，无配对数据 |
| main_T1 | 配对世界数据，各分支独立 GAE（**关键消融：同数据同标签，去掉跨回合关系**） |
| main_T2 | 同世界"先看/直接承诺"分叉对比（C2PO 式，我们的改编；rollout 数与 T3 对齐） |
| main_T3 | 信息集信用（观察决策的跨世界优势） |
| main_T4 | 信息集信用 + 证据后一致性（完整方法） |
| ctrl_vagen | VAGEN-Full（含结构化真值评价与重复惩罚），统一接口 |

## 消融（D3–D4；预算紧时 1 种子并在表注说明）
| 组 | 变化 |
|---|---|
| abl_noStayPairs | 去掉 Look→Stay 对（检验"看过就换目标"捷径） |
| abl_trainRate{0.25,0.75} | 训练时世界变化频率 |
| abl_lambda | 观察代价 λ 扫描（生成成本–成功曲线） |
| abl_cmaes | 编辑来源换成 CMA-ES 搜索基线（可选） |

## 评测（D4）
| 组 | 内容 |
|---|---|
| eval_<臂>_rate{0,25,50} | 全部检查点在冻结测试世界对上、三档变化频率下统一评测 |

## 可选（G2 出现信号后）
| 组 | 内容 |
|---|---|
| opt_glance | GLANCE 复现（同接口） |
| opt_7b | Qwen2.5-VL-7B 复现 main_T4 与 main_T2 |

## 汇总（analyst，经 run_logged）
`agg_obs`（--prefix obs_）、`agg_diag`（--prefix diag_）、`agg_main`（--prefix main_）、`agg_abl`（--prefix abl_）、`agg_eval`（--prefix eval_）
