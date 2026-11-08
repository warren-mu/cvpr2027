# Related work 种子清单 v3.1（2026-11-07 三次查新；scout 必须逐篇打开确认并生成校验过的 BibTeX）

标记：✅ 已打开原文/官方页核实要点 ｜ ☑ 仅核实标题与出处 ｜ ⚠ 来自外部评审、尚未核实
同期 = 2026-09-15 之后公开（CVPR 2027 规则：需引用讨论，可行时比较，不会仅因此被拒）

## A. 最直接的近邻（必须在 Intro 与 Related Work 中正面区分）
| 工作 | 出处 | 已做的事 | 与我们的差异 | 核实 |
|---|---|---|---|---|
| **SeekVLN: Seek Before You Move** | arXiv 2609.37353（2026-09-29，同期） | VLM 导航智能体在关键证据缺失时仍自信行动；从同一状态分叉"主动取证/直接导航"，C2PO 按后续导航收益给取证决策记功；R2R-CE / RxR-CE + 真机 | 不改变世界：无法监督"目标该不该变"；信用基于实际世界。我们用与当前信息一致的反事实世界对分配信用，并监督证据后的目标更新 | ✅（页面摘要） |
| Grounding on Request: Learning Modality Control in VLMs | arXiv 2610.33841（同期）；github.com/Maximeswd/Grounding-on-Request | 图文冲突下的 Flip/Stay 评测；Counterfactual Twin-Margin 成对目标；CLEVR 系 Ground-CF 等静态数据，另有新闻视频帧 | 静态；无动作、无信息获取 | ✅（README） |
| Red-Teaming VLAs via Quality Diversity Environment Generation (Q-DEG) | arXiv 2610.14415（同期） | 在 LIBERO-Spatial 上对物体位置/相机/光照/背景做 CMA-ES + QD 搜索，MIQP 修复语义约束；困难环境成功轨迹自模仿微调；局限中提到可把成功/失败轨迹组成偏好数据（与我们"原世界/编辑世界"的配对不同，不能把我们写成其 future work 的实现） | 低层 VLA、无推理智能体；纯 CMA-ES 是其论文中的对照方法，我们若使用须称"CMA-ES 搜索基线" | ✅ |
| Encoded but Not in Control: Grounding Gap in VL Robot Policies | arXiv 2610.06235（同期） | 语言干预显示 VLA 大多执行原目标；分析表征与注意力；无训练修复；全可见场景 | 世界干预、部分可观测、训练修复 | ✅ |
| DEER-3D: Error-Driven Scene Editing for 3D Grounding in LLMs | arXiv 2511.14086（ECCV 2026） | 谓词分解定位错误 → Clone-Replace-Modify 最小编辑 → 反事实 QA 迭代微调；ScanRefer 等 | 静态 3D grounding，非交互 | ✅ |
| DUEL: Adversarial Self-Play for Multimodal Reasoning | arXiv 2605.24794 | 同一 VLM 初始化 Challenger/Solver，生成真陈述与最小扰动负陈述，RL 自博弈 | 编辑文字而非世界；静态判别 | ✅ |
| RedVLA: Physical Red Teaming for VLA Models | arXiv 2604.22591（NeurIPS 2026） | 保持可行的物理风险场景，执行反馈迭代优化，跨模型迁移 | 安全红队，非视觉决策学习 | ☑ |
| VAGEN | arXiv 2510.16907（NeurIPS 2025） | 世界模型推理奖励（状态估计/转移）；**已用结构化真值评价与重复惩罚缓解模板化**；Bi-Level GAE | 我们的代码底座与强基线（含其缓解措施） | ✅ |
| GLANCE: Visual-Linguistic Curiosity for VLM Agents | arXiv 2605.03782 | 预测-观测差异作好奇心奖励，VAGEN 同套环境，Qwen2.5-VL-3B | 探索，不区分何时该看；无干预 | ✅ |

## B. 环境设计 / 自博弈（方法渊源，不包装成新范式）
| 工作 | 要点 | 更正说明 | 核实 |
|---|---|---|---|
| SPADE | arXiv 2608.19197；同一 LLM 当环境设计者与解题者 | 设计者奖励是**有/无特权提示的回报差（regret）**加难度带，不只是"让解题者失败"（v2 写法有误） | ✅ |
| PAIRED（NeurIPS 2020）/ ACCEL（ICML 2022） | regret 驱动的无监督环境设计 | — | ☑ |
| SimWorld Studio | arXiv 2605.09423；UE5 场景生成，按智能体掌握程度调整难度 | 已根据智能体能力调整环境（v2 低估了） | ✅ |
| LIBERO-Plus | CVPR 2026；VLA 多维扰动鲁棒性 | 外部评审指出其 6.2 节含扩展数据微调实验，**不是"只评测"**（v2 写法有误，scout 复核） | ⚠ |
| Vision-Zero | arXiv 2509.25541；图片博弈自我进化 | — | ☑ |

## C. 静态"看图敏感度"与反事实监督
DVRP/Thinking with Deltas（2601.06801）、VIGIL（2606.26387）、DeFacto（2509.20912）、Visionary-R1（2505.14677）、More Thinking Less Seeing（2505.21523）——均为单轮静态 ✅

## D. 部分可观测 / 探索 / 反事实
| 工作 | 要点 | 核实 |
|---|---|---|
| C²Nav: Compare Before You Commit (zero-shot VLN) | arXiv 2609.15142；scout 核实与"先比较再承诺"的关系 | ☑ |
| Learning Counterfactual World Models under Partial Observability | arXiv 2609.05834（同期）；RSSM 对动作序列的反事实可分性；无 VLM、无场景干预、无信息获取 | ✅ |
| Counterfactual VLN | NeurIPS 2020；导航中训练原始与反事实环境下的行为稳定性 | ☑ |
| Ask-for-help RL for embodied MLLM agents | arXiv 2504.00907 | ☑ |
| AI2-THOR Room Rearrangement | 检测场景变化的具身任务（scout 补齐出处） | ⚠ |

## 待补搜（scout）
- "embodied / navigation" + "counterfactual scene edit" + "training"
- "VLM agent" + "active perception / when to look" + "scene change"
- 2026-09-15 之后引用 VAGEN、Q-DEG、Grounding on Request 的工作
