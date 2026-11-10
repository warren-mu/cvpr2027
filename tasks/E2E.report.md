## 2026-11-10：第6步试跑完成，主实验尚未提交

T0、T4 各完整完成 50 次更新，T2 完成规定的 5 次更新；三臂原 parent 均 exit0、errors/cleanup 为空，最终仅保存 bf16 actor。T0/T4 的 0/20/40/50 次固定场景 greedy 评测均完整，每次覆盖 20 对的两个 world，共 40 回合。完整训练、评测与保存链已跑通，但本次结果尚不能作为正式 T04/T05 验收或论文结果。

两臂训练 Look 奖励均上升，但选择前看到锚点的比例仍为 0，留出总体成功率没有提升。原预定判据返回 `stop_and_report=false`，原因是奖励变化超过 0.05；这不等于学会观察，也不授权自动冻结或修改主配置。200 步 × 12 runs 仅训练更新本体的代理估计已为 **82.475209 GPU·小时**，无法与当前共享账本的已记账/预约量同时容纳于 120 GPU·小时总额。主实验未提交，等待人类在 20:00 决定预算；不自行减少步数、运行数或改变方法。

主证据为[原学习比较](../results/smoke_e2e_pilot_comparison_01.json)、[已日志化投影报告](../logs/review/diag_pilot50_report_projection_01/report.md)及其[完整 JSON](../logs/review/diag_pilot50_report_projection_01/report.json)。逐更新分类、每 10 步四类快照、解析/动作分母和所有原 SHA 保存在上述报告；本节只摘录已有数值。

四类缩写：DF＝DirectFlip（直接翻转），DS＝DirectStay（直接保持），LF＝LookFlip（观察翻转），LS＝LookStay（观察保持）；Flip/Stay 指两 world 的正确选择是否改变。

### 学习窗口与留出评测

按冻结定义，每次更新先计算 Look 回合的 seen 比例和平均 task reward，再对 1–10 或 41–50 的十次更新等权平均；不改用窗口回合数加权。seen 要求本回合看到锚点的步骤严格早于实际 parsed terminal select；动作是否合法另行计数。

| 臂 | seen 1–10 | seen 41–50 | task reward 1–10 | task reward 41–50 | seen 绝对变化 | reward 绝对变化 | 两项均 <0.05 |
|---|---:|---:|---:|---:|---:|---:|---|
| T0 | 0 | 0 | 0.558449 | 0.740716 | 0 | 0.182267 | 否 |
| T4 | 0 | 0 | 0.562657 | 0.719141 | 0 | 0.156483 | 否 |

不能把 seen 为 0 写成“观察动作始终为 0”：晚窗 Look 的有效观察动作/逻辑决策，T0 为 **185/510 = 0.362745**，T4 为 **12/340 = 0.035294**；早窗均为 0。两臂两个窗口的 Look 不可解析输出计数均为 0。观察动作已有发生，但在这些判定窗口内尚未转化为选择前看到锚点的证据；动作率只是辅助指标，不替代原学习判据。

以下为同一固定训练留出场景、greedy、每 world 一回合的成功率；总体按全部 40 回合汇总，不作类别宏平均，也不是最终测试集成绩。

| 类别 | T0 更新0：成功/回合（率） | T0 更新50：成功/回合（率） | T4 更新0：成功/回合（率） | T4 更新50：成功/回合（率） |
|---|---:|---:|---:|---:|
| DF | 7/10（0.700000） | 6/10（0.600000） | 7/10（0.700000） | 4/10（0.400000） |
| DS | 9/12（0.750000） | 8/12（0.666667） | 9/12（0.750000） | 9/12（0.750000） |
| LF | 4/8（0.500000） | 4/8（0.500000） | 4/8（0.500000） | 4/8（0.500000） |
| LS | 2/10（0.200000） | 3/10（0.300000） | 2/10（0.200000） | 2/10（0.200000） |
| 总体 | 22/40（0.550000） | 21/40（0.525000） | 22/40（0.550000） | 19/40（0.475000） |

这是一轮固定 seed0 的短试跑；训练奖励增加而固定留出总体未提升，不能据两臂相对次序做夜间超参搜索或宣称泛化改进。原比较保留 `common_config_frozen=false`、`common_config_mutated=false`。

### 实测更新耗时与 200 步 / 12 runs 规划

T0/T4 稳态均值来自各自原更新 **2–50，共 49 次**；完整一步采用 `total_seconds`，包含生成、评分、critic 和 actor，不把仅 critic+actor 的 `update_seconds` 当作全步。T2 只有更新 2–5 的 4 个样本，单列敏感性，不能当成同等长度的稳态测量。均为同型号 H200、每作业 1 GPU、12 CPU、相同模型和原固定输入。

| 臂 / 依据 | 实测样本数 | 全步均值秒 | 生成秒 | 评分秒 | critic 秒 | actor 秒 | 200 步 update-only 秒 | 200 步 update-only GPU·小时 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| T0，更新2–50 | 49 | 108.683805 | 11.022824 | 31.111656 | 32.699100 | 33.850225 | 21736.761014 | 6.037989 |
| T4，更新2–50 | 49 | 138.741822 | 6.860207 | 48.685522 | 32.427675 | 50.768418 | 27748.364364 | 7.707879 |
| T2，更新2–5，仅短样本敏感性 | 4 | 117.891054 | 6.972882 | 42.660326 | 33.414872 | 34.842975 | 23578.210753 | 6.549503 |

各分项沿原计时边界，不强行把分项之和配平到全步。去除首步后的均值只用于工程预测，不能宣称已经运行 200 步。

| 臂 | 来源认证秒 | 模型初始化秒 | holdout0 秒 | holdout20 秒 | holdout40 秒 | holdout50 秒 |
|---|---:|---:|---:|---:|---:|---:|
| T0 | 217.455430 | 82.001759 | 27.139443 | 15.886829 | 4.042154 | 5.906124 |
| T4 | 214.872909 | 69.735863 | 13.714617 | 10.675017 | 4.136804 | 4.077234 |
| T2 | 245.286928 | 79.884265 | 不在规定范围 | 不在规定范围 | 不在规定范围 | 不在规定范围 |

规定的 12 runs 为 **T0/T2/T3/T4 × 3 seeds**。暂用明确代理假设 **T2:=T0、T3:=T4**，每次 200 步、单卡，更新本体合计 **82.475209 GPU·小时**；只将 T2 换成其 4 次实测均值的敏感性情景为 **84.009750 GPU·小时**，T3 仍未实测。这两项都不是实跑费用或保证上界，不含来源认证、模型初始化、未来评测频率、保存、gate、清理、失败重试和 PBS 倍率。actor 保存没有独立计时，未用父运行剩余时长冒充；未来评测频率和全流程额外开销尚不能从这 50 步自动外推。

### 固定试跑总体与正式训练候选

试跑使用固定 **83 对**原 T02 与正式 cache 证书，保留全部 Stay，Flip 交换及整场景留出不变。永久留出场景为 FloorPlan15、FloorPlan204、FloorPlan301；三臂使用同一 manifest、冻结 holdout 和 common configuration。额外来源再认证 709593 超时未 PASS；经明确授权采用证书入口，该额外认证仍记作 `not_awaited` / `passed=null`，没有改写为新的 native 通过。

| 类别 | 试跑总体 | 训练部分 | 固定留出 |
|---|---:|---:|---:|
| DF | 11 | 6 | 5 |
| DS | 32 | 26 | 6 |
| LF | 22 | 18 | 4 |
| LS | 18 | 13 | 5 |
| 合计 | 83 | 63 | 20 |

见[证书总体结果](../results/smoke_e2e_pilot_pool_83_certificates_01.json)、[固定 manifest](../data/world_pairs/e2e_pilot_pool_83_certificates_01.json)、[永久场景留出](../data/world_pairs/train_holdout_scenes.json)。正式 T04 候选经排除该 holdout、原 Stay 位移过滤和原 A/B 平衡后，现有计数为 **DF6、DS2、LF18、LS0**，见[补数计数原记录](../results/gen_train_look_inventory_first_unit_01.json)；正式训练池仍未授权或冻结，不回写试跑的 83 对口径。

### 第二轮测试候选与拒绝漏斗

S01–S07 的原 40 场景来源均已终态，独立 checkpoint 与日志覆盖完整；“终态”不等于 40 场景全部成功，原 deadline、`incomplete` 和 `worker_failed` 均保留。原配置去重选择器对第二轮 S01–S07 单独得到 **44 个候选**；A/B 加 S01–S07 的累计快照为 **95 个候选**。44 不是在累计95之外可再相加的增量，场景数也不跨类别求和。这些均是 T02 前的已独立构造候选，不是完整 T02 通过或最终测试冻结。

| 类别 | 第二轮单独去重配置 | 第二轮该类场景数 | 累计去重配置 | 累计该类场景数 |
|---|---:|---:|---:|---:|
| DF | 5 | 5 | 18 | 12 |
| DS | 19 | 19 | 41 | 25 |
| LF | 4 | 4 | 16 | 10 |
| LS | 16 | 14 | 20 | 15 |
| 配置合计 | 44 | 不跨类相加 | 95 | 不跨类相加 |

见[第二轮完整已记录诊断](../logs/review/diag_test_round2_final_candidates_funnel_01/report.json)、[累计 selected05](../logs/review/diag_current_test_selected_05/report.json)及[原最终审计](../results/diag_test_round2_final_candidates_funnel_audit_01.json)（1 文件、0 FAIL）。新汇总仅重用原记录，不作新的 native 科学验收。

下面按 **原提议类别** 分组，分母是原 journal 尝试数，不是去重候选数。每个单独检查列出原观察到的通过/失败/未知；Direct 不套 Look 的隐藏、前缀、一动作证据等门槛。

| 原提议类别 | 尝试数 | 检查 | 通过 | 失败 | 未知 |
|---|---:|---|---:|---:|---:|
| DF | 317 | 两 world 物理 | 160 | 155 | 2 |
| DS | 33 | 两 world 物理 | 24 | 8 | 1 |
| LF | 311 | 两 world 物理 | 225 | 85 | 1 |
| LF | 311 | 两 world 锚点隐藏 | 49 | 0 | 262 |
| LF | 311 | 前缀差异 ≤3 | 43 | 6 | 262 |
| LF | 311 | 首证据 ≥250 像素 | 30 | 13 | 268 |
| LF | 311 | 两 world 可解 | 8 | 35 | 268 |
| LF | 311 | 两 world witness | 7 | 0 | 304 |
| LS | 4107 | 两 world 物理 | 2631 | 1377 | 99 |
| LS | 4107 | 两 world 锚点隐藏 | 657 | 951 | 2499 |
| LS | 4107 | 前缀差异 ≤3 | 539 | 1069 | 2499 |
| LS | 4107 | 首证据 ≥250 像素 | 93 | 413 | 3601 |
| LS | 4107 | 两 world 可解 | 27 | 479 | 3601 |
| LS | 4107 | 两 world witness | 188 | 0 | 3919 |

物理检查实际先执行，但后续六项表是原记录的检查结果，**不是因果归因或互斥首个拒绝步骤**；未测/缺字段保留未知，不能补作通过。原状态按提议类别记录为 DF 5 passed/312 rejected，DS 5/28，LF 4/307；LS 16 按原类 passed、14 cross-over 到 DS、4077 rejected。因此不能把按提议类别的通过数直接当成最终类别库存，也不能将单项 gate 的通过数当成整对通过。

原拒因计数可以重叠：DF 常见为 w0/w1 physical 85/70、起点可见但 Direct 证据失败87；DS 的 target uniqueness/ratio 为18；LF 的 target uniqueness/ratio 为176、w0/w1 physical 为69/16、一动作证据不可用35；LS 相应为1023、833/544、一动作证据不可用482，另有起点可见但 Direct 证据失败646及真实 Teleport 错误等。完整原理由、示例、未知与非因果 first-observed 表均保留在上述报告，不将这些计数相加作为互斥失败总数。

### 夜间运行、预算与存储

下表保留全部 14 个相关原作业，按所属账本分别计费；Q 未启动取消以原零分配证据为准，不算实际基础设施失败。T0/T4 各经历两次开头黑帧，第三次实际尝试完成；Q 回退没有增加实际尝试次数。所有失败结果、门检、原 PNG、队列取消保护与审计均保留，详细 UUID 和证据路径见 [STATUS](../STATUS.md)。

| 运行 | PBS | 终态 / 去向 | 排队秒 | 实际分配秒 |
|---|---|---|---:|---:|
| 额外来源认证 | 709593 | 内部时限结束，未 PASS | 295 | 5174 |
| T4 工程短测 | 709803 | 完整1次更新及最终 actor 保存 | 34 | 844 |
| T0 初次 | 709866 | 开头黑帧、零训练 → 709908 | 402 | 26 |
| T4 初次 | 709869 | 开头黑帧、零训练 → 709913 | 327 | 20 |
| T0 pin 排队 | 709908 | 满20分钟且确认Q/未分配后取消 → 709961 | Q取消，原链保留 | 0 |
| T4 pin 排队 | 709913 | 满20分钟且确认Q/未分配后取消 → 709964 | Q取消，原链保留 | 0 |
| T0 第二次实际尝试 | 709961 | 开头黑帧、零训练 → 709979 | 63 | 17 |
| T4 第二次实际尝试 | 709964 | 开头黑帧、零训练 → 709996 | 32 | 23 |
| T0 pin 排队 | 709979 | 满20分钟且确认Q/未分配后取消 → 710029 | Q取消，原链保留 | 0 |
| T4 pin 排队 | 709996 | 满20分钟且确认Q/未分配后取消 → 710037 | Q取消，原链保留 | 0 |
| T0 第三次实际尝试 | 710029 | 完整50步、全部规定评测、最终 actor | 43 | 5987 |
| T4 第三次实际尝试 | 710037 | 完整50步、全部规定评测、最终 actor | 21 | 7408 |
| T2 初次 | 709941 | 开头黑帧、零训练 → 710007 | 612 | 23 |
| T2 重提 | 710007 | 完整5步、最终 actor；边界已R，未取消 | 930 | 1028 |

实际分配秒采用原 PBS canonical，不能与 parent `_meta.duration_sec` 或更新计时互换；历史错误记账调用保留，但预算只采用原校验接受的 canonical，未重复计入。T0/T4 的最终 walltime 请求均为 4小时20分，完整 50 步不变，且两臂均在美东 05:00 前实际进入训练。本夜未改奖励、学习率、KL、λ、数据或模型，没有超参搜索。

原预算去重后的主试跑累计实际为 **5.416388888888889 / 12 GPU·小时**，T2 独立账本累计实际为 **0.29194444444444445 / 0.5 GPU·小时**。共享总账在 **710213 提交之前**为已记账/预约 **47.65055555555554 / 120 GPU·小时**，其中包含预约上限，不能全称实际消耗或当成此后实时余额。即便只考虑 12-run 的 82.475209 小时更新本体，也已无法与该共享占用同时装入当前总额；主实验未提交，等待人类 20:00 的预算决定，不静默压缩步数或运行数。

预算来源为[原预算快照](../logs/review/diag_train_look_first_unit_retry1_review_plan_01/budget_state.json)，最终两臂费用见[原 canonical](../results/diag_pilot710029_710037_accounting_01.json)。本次[原审计](../results/diag_e2e_final_report_audit_01.json)覆盖20个新封口或报告引用结果、0 FAIL，保留 P5/P7 警告；本次新封口结果及报告引用范围的原审计无 FAIL；旧未封口配额记录和 `accounting01` 缺结果日志的失败记录继续保留，不声称全仓历史问题已清零。

最新已核 scratch 快照为 **588.3 GiB / 1 TiB、3,116,252 个文件**；这是已核查询快照，不是未来写入峰值或容量保证。本报告首推时 FIG01 保持暂停；此后已完成五对原帧导出并复制到 Mac，见[FIG01 原结果](../results/diag_fig01_firststep_frames_01.json)与 STATUS 的后续记录。这不改变本节试跑结果。trainLook 补数当前新提交 **710213（hopper30，同参数 FP315）**，仍遵循指定节点 20 分钟未启动才取消回退的规则；它不是本试跑通过记录，也未据此增加正式训练库存。

---

## 最终夜间 Look 诊断（固定 inventory05，59 对）

截至 inventory05，按原 proposal_identity 去重的独立构造库存为150对：DirectFlip33、DirectStay58、LookFlip41、LookStay18。本次完整视角诊断覆盖其中全部59对Look；原36和50快照原样保留。

| 原规则下的诊断结果 | 对数 |
|---|---:|
| 确认触发锚点不可见且画面通道差 >3 | 24 |
| 完整原生视角表未触发该项 | 28 |
| 原严格检查失败或导出未完成，无法判定 | 7 |

[最终59对逐对报告](../logs/review/diag_look_table_census_latest_01/report.md)保留全部失败、未知及重试来源。未知不算未触发；未触发这一项也不授予完整T02、缓存或训练池验收。两个严格距离失败的CPU像素谓词诊断仍单列，没有提升主表状态。

首个Look E2E失败对FloorPlan203_00002在后续D4视角出现239个通道差>3的像素，最大差77，集中在右下角地面（x241–254、y235–254）。45个非锚点物体位姿完全一致，变化区域位于地面内部，邻近离屏w1锚点Box的投影；差异通道均在w1变暗。证据支持局部光照变化，但不能仅凭现有RGB区分阴影或反光、或确认唯一光路。[原图与差异](../logs/review/diag_e2e_look_non_anchor_01/original_frames_and_difference.png)。

本轮视角诊断实际计费3.342222222 GPU·小时，涵盖23次PBS分配及全部黑帧重试；复用的原E2E707904不重复计费。[逐作业排队与费用](../logs/review/diag_look_latest_accounting_report_01/report.md)。原16 GPU·小时子预算及60共享预算未变。

科学阈值、任务结构和固定五对Look E2E输入均未变。固定Look E2E仍停在第1步；新Look池上第2–5步及新KL设置的T0/T3训练尚未运行。G01阶段报告已单列150对构造库存与未达到的正式数量、cap和平衡目标；没有冻结训练池。等待人类决定后续规则处理。

---

以下为此前固定快照和阶段记录，按各自截止范围解读。

# E2E 报告：历史 Direct 流程完成，Look 补跑仍受完整视角规则阻塞

## 当前状态更新（2026-11-09 夜间）

历史 100° Direct 流程已完成步骤 1–5。人类已批准在固定 60° Look 池上补跑 T0/T3，且 T05 §1b 已冻结；当前阻塞是原定五对中的首对在完整视角导出时触发原 Look non_anchor_divergent 规则，尚未得到该固定池的完整表，后续步骤未在新池上运行。不会自行替换样本或改变阈值。

旧 Direct T0/T3 使用奖励端 KL；它们不是新 §1b 的同配置基线。新共同配置为 actor KL loss (low_var_kl, 0.001)，use_kl_in_reward=false，G 只含任务奖励；原 GAE → 证据前替换 → 整批 masked whiten 一次，critic 仍用原回报。t05_frozen_training_cpu_01 已完成 CPU 回归，但不代表新配置 GPU 训练或证据前逻辑已验证。须先在同一合格 Look 池补跑 T0 五步，再跑 T3 五步。原历史费用段落仅代表其当时状态；累计费用以原 canonical PBS 记录和 STATUS 为准。

首对 floorplan203_00002 的失败发生在后续可达位置 v_5c39249b3450164be27f:h0，不是起始共享前缀：原起点 x=-4.5，失败视角 x=-4.25。原帧 000135/000138 中，锚点 Box_8e5b2c6b 在两世界的掩码均为 0；239 个像素的最大通道差 >3，全图最大差 77，集中在右下角 x=241–254、y=235–254 的一块连续地板内部区域，未落在候选物体或分割边界上。45 个非锚点物体的位置和旋转严格相同。差异块在二维投影上邻接/重叠 w1 锚点包围盒的画面边缘，w0 锚点投影则在画面左外；该区域在 w1 整体变暗。现有证据支持局部光照变化的解释，未显示非锚点位姿漂移，但仅凭 RGB、物体掩码和投影不能确定是锚点阴影、反光或具体光路。来源为 results/diag_e2e_look_non_anchor_01.json、results/diag_e2e_look_anchor_projection_01.json，原图在 logs/review/diag_e2e_look_non_anchor_01/original_frames_and_difference.png。这与旧 Direct 对的椅子微位移诊断是不同的失败。

固定36对完整视角诊断现已完成：16对触发隐藏锚点画面差、18对完整导出未触发、2对严格距离元数据校验失败仍列未知；来源身份认证无错误。报告在 logs/review/diag_look_table_census_morning_01/report.md。两对严格距离失败的另行CPU原始证据诊断覆盖各74个D4视角，目标像素谓词均未触发，但完整T02仍失败，主表未知不升级。构造验收没有覆盖所有D4视角，完整表会发现起点/见证以外的原Look规则失败。随后独立验收的新G01世界对另外记录，不回写36分母。新增诊断17次PBS实际计费2.2280555555555557 GPU·小时，含黑帧重试，排队表见 logs/review/diag_look_morning_accounting_report_01/report.md；原707904证据复用不重复计费。固定Look E2E输入没有更换，步骤2–5仍未在新Look池运行。

以下早期阶段记录、失败和耗时保留。

# E2E 报告（第 1 步失败，后续步骤停止）
截至 2026-11-09 00:16 UTC。固定输入为人类确认 Claude 已审过的 FP203 五对 Direct 样本；未挑选子集继续。

第 1 步未通过：707395 在 hopper-24 的单卡真实出图门禁通过，首对 floorplan203_00038 导出 D=4 视角表时由 src/envs/pair_table.py::classify_views 抛出 non_anchor_divergent。此判据是两个世界的锚点掩码都为 0，但原始 RGB 最大通道差 >3。它不是非锚点物体位姿差异的直接测量；当前不能把原因归为物理漂移或阴影。尚未形成完整表，也未开始 20 状态的缓存/实时一致性抽查。失败子进程实际耗时 61.170814926037565 秒，保留 412 个动作和 413 个帧记录，清理无异常、非黑帧。原始证据在 scratch 的 smoke_e2e/smoke_e2e_views_01/707395.hopper-m-02/export_0000；结果/日志已保留，阈值未变。

第 2–5 步未运行，因此没有模型评测、T0/T3 每步耗时或 200 步训练时长依据。已通过的 55 项 CPU 表接口测试只是软件检查，不能替代真实视角表验收。现有输入均为 τ=0：证据前逻辑未覆盖。按任务的“每步失败就停下”要求，E2E 在这里停止，等审查后决定修复路径；T01v2 采集继续。

运行入口：scripts/run_logged.py 包装 python3 -u -m src.tools.smoke_e2e_views，参数见 results/smoke_e2e_views_01.json 的 _meta.command。提交记录为 results/smoke_e2e_submit_views_01.json；真实运行、门禁、CPU 契约测试分别为 results/smoke_e2e_views_01.json、results/smoke_e2e_views_01_gate.json、results/smoke_e2e_contract_cpu_01.json。计费来源 results/t00b_monitor31_accounting.json；PBS 分配 GPU 小时乘已核实 queue_charge_rate=1，非独立项目账单核对。

| PBS 作业 | GPU | 排队时间 | 实际占用 | 计费 GPU·小时 |
|---|---:|---|---|---:|
| 707395.hopper-m-02 | 1 | 14 分 23 秒（863 秒） | 1 分 24 秒 | 0.023333333333333334 |

E2E 已用 0.023333333333333334 / 4 GPU·小时。下一次 GPU 运行的用量尚不能确定：需先定位真实导出失败，当前没有后续提交。审计 t01v2_e2e_launch_audit_01：742 文件、0 FAIL。

## 人类批准补充 8 后续（2026-11-09）
原第1步失败完整保留。人类现已批准 Direct 的隐藏锚点差异状态作为分叉、不丢弃；修订代码后将重新导出同一固定五对，然后按步骤继续，不删除失败输入。

首对失败发生在已有帧 000409/000412。锚点 TissueBox 和两本候选 Book 的掩码均为 0；158 个像素的通道差超过 3，最大差 177，集中在画面右下方两把椅子及餐桌边缘。两把椅子的已记录位置分别相差约 0.122 mm、0.052 mm，分割边界也有变化。因此主要差异并不落在被锚点压住的候选本体上，更符合非候选物体的细微位姿及边缘渲染差异；仅凭现有两帧不能确认锚点阴影的因果贡献。来源：results/smoke_e2e_existing_difference_01.json 及其 run_logged 日志；未新增工具、未另行渲染。


## Supplement8/9 continuation: E2E step1 PASS

PBS707497 on hopper-26 completed the same five reviewed 100-degree Direct pairs, D=4, native checks1–4 and the approved20-state fresh-process cache smoke. Exit0, no errors, no cleanup errors. Formal T02 200-state acceptance remains unclaimed.

| Pair | Exported states | Worker seconds | Hidden-anchor divergent Direct states |
|---|---:|---:|---:|
| floorplan203_00038 | 73 | 268.8341847109841 | 4 |
| floorplan203_00039 | 73 | 268.2378776760306 | 10 |
| floorplan203_00041 | 73 | 268.1510090410011 | 7 |
| floorplan203_00080 | 73 | 268.00961209391244 | 2 |
| floorplan203_00081 | 73 | 270.50699310505297 | 1 |

These Direct divergent states retain their own world images under supplement8; they are counted, not filtered. Rendering uses supplement9 paused/frozen construction and retains readback deviations.
PBS accounting:0.4675 GPUh and67 seconds queue wait for707497. Failed707395 and black-gate707488 remain preserved;707488→707497 is recorded in the existing retry evidence.
Sources: results/smoke_e2e_views_02_r8335e243.json, results/smoke_e2e_views_accounting_02.json, results/smoke_e2e_contract_cpu_02.json (64 tests passed).
Step2 is next; steps3–5 remain unrun. T0/T3 reward confirmed by human: select correct1/wrong0, observe−0.05, invalid0, no format bonus. The five pairs are direct/tau0: **证据前逻辑未覆盖**; Look requires a later step5 run.


## Step2 PASS; original-model evaluation next

The CPU witness replay passed for both worlds of all five approved Direct pairs (results/smoke_e2e_replay_01.json). The canonical table interface suite passed64 tests (results/smoke_e2e_contract_cpu_02.json). This used no GPU. All selected pairs remain tau0: 证据前逻辑未覆盖. Next is step3 original Qwen2.5-VL-3B with obs_never/obs_always/diag_base, one episode per pair/world.

## Step3 PASS; T0/T3 training smoke next

PBS 707555 on hopper-27 completed the original Qwen/Qwen2.5-VL-3B-Instruct evaluation with exit 0. The fixed five approved 100-degree Direct pairs were evaluated in both worlds: obs_never, obs_always and diag_base each completed 10 episodes, 30 episodes in total. All episodes contain the selected object and success outcome; the retained decision traces include public observations, generated actions/token IDs, rewards, state transitions and table temporal proofs. No execution or cleanup errors were recorded. Model loading took 58.24130738002714 seconds; the run_logged interval was 372.1 seconds.

The source is results/smoke_e2e_eval_03.json, with detailed records in logs/runs/smoke_e2e_eval_03/episodes.jsonl and the matching run log/meta. This is an end-to-end smoke check, not a formal T03 acceptance or paper result. All selected pairs have tau=0: **证据前逻辑未覆盖**. Pre-evidence credit and paired-generation deduplication remain unmeasured; no T3 correctness claim is made. Steps4–5 had not started when this section was written.

Two prior attempts stopped at the black-frame gate before model evaluation; their failure records are retained. The successful attempt and both failures have complete final PBS allocation accounting:

| PBS job | Outcome | Queue wait (seconds) | Allocated GPUs | PBS walltime | PBS allocation GPU-hours |
|---|---|---:|---:|---|---:|
| 707539.hopper-m-02 | black-frame gate, exit86 | 244 | 1 | 00:00:19 | 0.005277777777777778 |
| 707548.hopper-m-02 | black-frame gate, exit86 | 525 | 1 | 00:00:21 | 0.005833333333333334 |
| 707555.hopper-m-02 | step3 PASS, exit0 | 106 | 1 | 00:06:14 | 0.10388888888888889 |

Accounting sources: results/smoke_e2e_eval_black_accounting_01.json, results/smoke_e2e_eval_black_accounting_02.json and results/smoke_e2e_eval_accounting_03_retry1.json. These report allocated GPUs multiplied by measured PBS walltime, not an independently reconciled project credit bill. The first accounting command, smoke_e2e_eval_accounting_03, failed on the login node because its non-login PATH did not contain qstat; its failed log/meta are retained, and the retry loaded the established PBS PATH. This accounting failure used no GPU.

Next: step4 T0, then step5 T3, with the human-confirmed explicit_task_zero reward policy. The current Direct-only step5 cannot cover pre-evidence logic; a later Look-pair run remains required.

## Step4 T0 PASS after colocated-worker compatibility repair

The first T0 allocation707594 failed before any update because pinned verl required the same immediate base class for colocated actor/ref and critic wrappers. The shared seed/observation-hook base fixed that interface; upstream actor/critic update implementations and training settings were preserved. The failed run remains recorded (0.08361111111111111GPUh). The real pinned-worker CPU regression passed before retry.

Retry707612 onhopper-21 passed all5updates,64episodes per update, critic_offload=false, explicit_task_zero rewards, no format bonus. Final checkpoint is on scratch; producer/run_logged exit0,errors[],cleanup_errors[]. Actual PBS allocation0.3675GPUh,queue206s. Model initialization115.94174984202255seconds is separate from the update timings below.

| Update | Generation s | Scoring s | Critic update s | Actor update s | Total s | Syntax parse failures / physical outputs |
|---:|---:|---:|---:|---:|---:|---:|
|1|20.95697337493766|61.86384365800768|33.788221759023145|60.68973210698459|177.29877089895308|0/64|
|2|10.03317175898701|33.341926231049|36.998113421024755|38.78125666594133|119.1544680770021|0/64|
|3|8.911296078003943|34.50496250006836|41.01187270495575|39.00891356705688|123.43704485008493|0/64|
|4|8.629455079091713|31.69141386600677|40.93558763200417|39.66004744300153|120.91650402010418|0/64|
|5|8.747530417982489|33.540992914000526|36.28870978101622|38.34881312190555|116.92604623490479|0/64|

The original model step3 separately recorded0/10 syntax parse failures under each of obs_never,obs_always anddiag_base;40 forced observations inobs_always are excluded from the model-output denominator. Every T0 update recorded0/64 syntax parse failures and0 syntactically valid-but-unavailable actions. Rates use the actual raw response with parse_action; they are not inferred from reward or success. Sources: results/smoke_e2e_T0_report_02.json and its authenticated original journals.

Largest recorded CUDA allocator peak allocated142252924416bytes; peak reserved163231825920bytes. These are the colocated PPO worker allocator statistics, not sampled whole-device nvidia-smi occupancy. The reserved statistic exceeds physical capacity and must not be interpreted as physical GPU consumption. This smoke succeeded with the critic on GPU; it does not prove all main-training settings fit.

Planning only: the five measured updates average131.5465668162098seconds; multiplying that mean by200 gives7.308142600900545hours, excluding all setup and checkpoint overhead. This extrapolation assumes the same Direct/tau0 short trajectories and is not a measured200-step run or a reliable bound for Look training. Each training uses one GPU; independent repeats can be packed only after their actual memory/concurrency and budget checks.

All current pairs areDirect/tau0: **证据前逻辑未覆盖**. T3 next uses the same reviewed100-degree smoke pool and common config, adding paired sampling and Q_commit scoring. It cannot certify pre-evidence credit until a separately authorized Look smoke is implemented and run.


## Step5 PASS; Direct-only E2E steps1–5 completed

PBS 707642.hopper-m-02 on hopper-30 completed all 5 real T3 PPO updates, using the same common configuration and fixed five reviewed 100-degree Direct pairs as T0. Each update contains 64 episodes / 32 sampled pairs. The producer and run_logged exited0; errors and cleanup_errors are empty. Both actor and critic checkpoints were saved on scratch. Critic offload remained false. This establishes the bounded Direct smoke pipeline; it is not formal T03/T04/T05 acceptance or a paper result.

| Update | Generation seconds | Scoring seconds | Actor+critic update seconds | Total seconds | Raw-model syntax failures / outputs |
|---|---:|---:|---:|---:|---:|
| 1 | 19.4622292999411 | 66.92308282596059 | 95.81631059700157 | 182.20162272290327 | 0/64 |
| 2 | 11.18495745700784 | 61.85113084106706 | 64.1086006090045 | 137.1446889070794 | 0/64 |
| 3 | 7.965900354087353 | 48.72496871300973 | 64.91962409997359 | 121.61049316707067 | 0/64 |
| 4 | 8.076449931017123 | 48.77525428298395 | 64.08328905899543 | 120.93499327299651 | 0/64 |
| 5 | 8.110607679001987 | 48.52600291301496 | 64.13653016707394 | 120.7731407590909 | 0/64 |

The syntax parser is the existing parse_action applied to recorded raw responses. These are per-update training-policy outputs, kept separate from original-model greedy Step3 strategies. Canonical but unavailable actions are counted separately in results/smoke_e2e_training_report_01.json; they are not syntax failures. Shared physical sample IDs are counted once in the physical-output denominator while both logical world decisions remain recorded.

| Update | T0 candidate-scoring RPCs | T3 candidate-scoring RPCs | T3 candidate action scores | T0 scoring seconds | T3 scoring seconds |
|---|---:|---:|---:|---:|---:|
| 1 | 0 | 4 | 128 | 61.86384365800768 | 66.92308282596059 |
| 2 | 0 | 4 | 128 | 33.341926231049 | 61.85113084106706 |
| 3 | 0 | 4 | 128 | 34.50496250006836 | 48.72496871300973 |
| 4 | 0 | 4 | 128 | 31.69141386600677 | 48.77525428298395 |
| 5 | 0 | 4 | 128 | 33.540992914000526 | 48.52600291301496 |

Q_commit used actual teacher-forced probabilities for every complete public select action plus EOS, normalized over the complete public candidate set. Scoring_seconds includes candidate scoring, actor/reference log-probability recomputation, critic values, advantages and related logging. Its T0/T3 difference is not a separately measured isolated Q_commit time. The source-recorded total-time ratios T3/T0 for updates1–5 are [1.0276530502670231, 1.1509823435110411, 0.9852025647144047, 1.0001529092578565, 1.0329019465556655]. Both jobs used an H200, on different nodes, so this smoke comparison is not a causal estimate of algorithm overhead.

Model/Ray initialization was 65.45255886495579 seconds, separately from the update timings. The run_logged interval was 1064.5 seconds and includes setup/finalization. Update1 reported CUDA allocator peak allocated 142252924416 bytes and peak reserved 163229728768 bytes. These are counters inside the colocated PPO process, not sampled whole-device physical occupancy; reserved bytes must not be interpreted as physical GPU use.

**证据前逻辑未覆盖。** All reviewed inputs are tau=0; no pre-evidence T3 advantage replacement or shared-prefix generation path was exercised. S1 variance remains unavailable, and the five-update run did not reach the S2 evaluation interval20. A Look-pair Step5 run remains required before claiming that coverage.

| PBS job | GPUs | Queue wait seconds | PBS walltime | PBS allocation GPU-hours |
|---|---:|---:|---|---:|
| 707642.hopper-m-02 | 1 | 161 | 00:17:46 | 0.2961111111111111 |

These are PBS allocation hours, not an independent project credit-bill reconciliation. Sources: results/smoke_e2e_T3_01.json, results/smoke_e2e_training_report_01.json, results/smoke_e2e_T3_accounting_01.json, and the matching run logs plus ppo/update_*.jsonl journals. The retained run diff contains only STATUS.md progress notes; training source/configuration were unchanged. The source audit has no FAIL and retains its dirty-tree warning. The warning is not removed or treated as a clean checkout.
