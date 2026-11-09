# AGENTS.md：给 Codex 的项目规则

## 项目
CVPR 2027 投稿：*Don't Reward Lucky Guesses: Counterfactual World Pairs Teach Embodied Agents When to Look and When to Change Their Minds*。
- 研究合同（做什么、不做什么）：`docs/RESEARCH_CONTRACT.md`
- 指标定义与世界对校验：`docs/METRICS_SPEC.md`
- 预注册的实验分组和命名：`experiments/REGISTRY.md`
- 截止（美东）：论文注册 11-11 06:59；正文 11-17 06:59；补充材料 11-24 06:59

## 分工
- **你（Codex）**：在 GPU 机器上写代码、跑实验、产出结果。
- **Claude**：写任务说明（`tasks/`），审查你推送的代码和结果（`reviews/`），复核统计，做规划。
- **人类**：拍板、管预算，在你和 Claude 之间传话。

## 每个任务的流程
1. `git pull`，读 `tasks/<ID>.md`（任务说明和验收标准）。只做这个任务，不扩大范围。
2. 写代码：可复用的逻辑放 `src/`，配置放 `configs/`，启动脚本放 `experiments/`。
3. **所有产出数字的运行都用 run_logged 包装**，先提交代码再运行：
   ```bash
   python3 scripts/run_logged.py --name <名字> [--seed N] -- <命令>
   ```
   - 命名：`<阶段>_<环境>_<方法>_s<种子>`，例如 `main_nav_T0_s0`。
   - 被包装的程序必须把结果写到 `results/<名字>.json`，格式为 `{"metrics": {...}, "config": {...}}`。
   - 程序最后必须在标准输出打印一行 `FINAL_METRICS_JSON: <与 metrics 完全相同的 JSON>`。
   - 多种子汇总也用 run_logged 包装：`python3 scripts/run_logged.py --name agg_<组> -- python3 scripts/summarize_results.py --prefix <组>_ --out results/agg_<组>.json`
4. 跑完运行 `python3 scripts/audit_provenance.py`，必须没有 FAIL。
5. 提交并推送：代码、configs、`results/*.json`、`logs/runs/` 下的 `.log`、`.meta.json`、`.diff`，以及小图。
   **不要提交**：checkpoint、模型权重、`third_party/`、任何超过 50MB 的文件。
6. 写 `tasks/<ID>.report.md`：做了什么、用的命令、结果文件路径、失败和异常、没做完的部分、下一步预估要多少 GPU·小时。推送后停下，等审查。
7. Claude 会在 `reviews/<ID>.md` 里写 PASS 或 FAIL 以及修改要求。FAIL 就照着改，改完重复第 4–6 步。拿到 PASS 之前不开始下一个任务，除非任务说明允许并行。

## 铁律
1. **不编造、不估算、不"补齐"任何结果。** 没跑出来就写没跑出来。
2. `results/`、`logs/`、`paper/generated/`、`figures/` 下的文件只能由脚本或命令输出生成，不要手写或手改。
3. 失败的运行也要保留并提交（run_logged 会把旧结果归档到 `results/_archive/`）。不许删除不利的结果。
4. 不改 `docs/RESEARCH_CONTRACT.md`、`docs/METRICS_SPEC.md`、`experiments/REGISTRY.md`，也不改校验和统计脚本（`run_logged.py`、`audit_provenance.py`、`summarize_results.py`、`export_numbers.py`、`check_paper.py`、`verify_bib.py`）。觉得需要改，写进报告，由人类决定。
5. 看到结果后，不改指标定义、不换评测集、不挑种子。种子按 REGISTRY 预先定好的跑。
6. 同一组对比里，所有方法用同样的超参预算、评测集和种子，硬件也要一致（在报告里写明卡型）。
7. 单个任务预计超过 20 GPU·小时，或者需要同时用多台机器，先在报告里写明并停下，等人类批准。
8. 删除文件、使用 sudo（安装渲染相关的系统库除外）、需要密码或密钥的操作，先问人类。不要把任何密钥写进仓库。

## 环境
- conda 环境 `cvpr2027`（Miniforge），由 `scripts/setup_env.sh` 一键重建。新装的任何包都要补进这个脚本。
- VAGEN 放在 `third_party/VAGEN`（不提交），在 `setup_env.sh` 里固定 commit。
- AI2-THOR 用 CloudRendering（Vulkan）做无头渲染。

## 算力：NUS Hopper（PBS）
- 每张卡是 H200（约 141GB 显存），配 12 个 CPU 核、225GB 内存。用项目 `CFP04-SF-125` 的 Express 队列（提交方式见 `hpc autox`）。
- **按人类要求（2026-11-08），所有 GPU 作业都使用 premium。** 具体的队列名或参数，以 `hpc autox`、`qstat -Qf` 和集群说明为准。如果 premium 的计费倍率和普通 Express 不同，要在 STATUS.md 里写明倍率，预算一律按实际扣费的 GPU·小时计算。
- 这个项目的 GPU·小时余额由项目成员共用。每份报告都要写明本任务用掉多少 GPU·小时，以及对应的 PBS 作业号。
- 登录节点只做轻量的事：改代码、git、提交和查看作业、几秒钟的检查。所有 GPU 工作都通过 PBS 作业完成。
- walltime 按需要设紧。交互式作业用完马上结束，不许让卡空等。
- 作业名统一以 `cvpr27_` 开头。不碰其他人、其他项目成员的作业。
- 我们的多次训练彼此独立，每次只用 1 张卡。需要并行时，把几次训练打包进同一个多卡作业，用 `scripts/gpu_queue.py` 在作业内部分配卡，不做多节点训练。
- **排队策略（2026-11-08 按实际排队记录制定，目标是尽早开跑）**：
  - 今天的实测：1 卡作业 8/8 在 1–3 分钟内启动；2 卡作业 5 个一个都没启动过（最久 14.5 小时）；4 卡等 15–270 分钟；8 卡等 2.5 小时到 10 小时以上；指定 host 的 1 卡作业等了 75 分钟。
  - **默认用 1 卡作业**（每人同时最多 3 个 1–2 卡作业）。能按场景、按训练拆开的工作，都拆成 1 卡单元。
  - **不申请 2 卡，也不申请 8 卡及以上**，除非人类另行批准。
  - 1 卡的 3 个名额都占满、还需要更多并行时，用 3–7 卡作业（每人同时最多 2 个），在作业内部用 `scripts/gpu_queue.py` 打包多个 1 卡单元。
  - walltime 按实测加 30% 余量，不要多报：短作业更容易被调度器插空启动。
  - 不加 host 限制。**例外（2026-11-09 16:15，人类同意）**：作业因出图检查发现黑帧而退出后，它的重提作业可以指定一台今天已验证正常、且没有已知黑帧 GPU 的节点（目前是 hopper-22、hopper-28、hopper-30，按顺序轮换；新验证正常的节点可以加进来，记入 STATUS.md）。原因：黑帧 GPU 上的作业几秒钟就退出，这张卡马上又空出来，调度器很容易把下一个作业再分到同一张卡上，只靠自动重提会反复撞上。指定节点的作业提交 20 分钟后仍未启动，就取消并改为不指定节点重提。首次提交仍不指定节点。每份报告列出黑帧节点、GPU UUID 和指定节点作业的排队时间。
  - 每份报告附一张排队时间表：作业号、卡数、提交到启动的分钟数。
- H200 和 H100 是同一款芯片，图形硬件很少，渲染性能大约只有集成显卡的水平。所以 AI2-THOR 的渲染速度必须实测，不能因为 CUDA 正常就默认仿真也够快。
