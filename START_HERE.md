# 服务器上第一次启动 Claude Code 后，把下面整段粘贴进去

你是这个项目的 lead（总控）。项目目录就是当前目录，服务器是 8×RTX A6000（主机 AS-4125GS-TNRT），可能有实验室其他成员共用。

按顺序做，每一步做完在 STATUS.md 里更新并简短汇报给我：

1. 读 CLAUDE.md、docs/RESEARCH_CONTRACT.md、docs/PLAYBOOK.md、docs/METRICS_SPEC.md、experiments/REGISTRY.md、STATUS.md。
2. 运行 `bash scripts/server_preflight.sh`（只读体检），然后告诉我：哪些 GPU 正被别人占用、/mnt/data2 剩余空间、驱动/CUDA 版本、conda 是否可用、外网是否通、AI2-THOR 无头渲染的前提是否具备。**在我确认能用哪几张卡之前，不要启动任何 GPU 任务。**
3. 我确认可用的卡之后：在 third_party/ 下 git clone VAGEN（github.com/mll-lab-nu/VAGEN），**严格按它的 README** 建一个独立 conda 环境（名字 cvpr2027，不要改动 base 或别的环境），安装依赖。
4. 安装后先做 AI2-THOR 的无头渲染冒烟：启动一个场景、渲染一帧、保存到 logs/thor_smoke.png；再验证 METRICS_SPEC §1 需要的三件事——`SetObjectPoses` 传完整物体列表后能否确定性恢复（同一状态恢复两次，帧逐像素一致）、能否拿到实例分割掩码、单个物体能否被稳定挪动。把结论写进 STATUS.md 的 Blockers 或 Decisions。
5. 用 `scripts/gpu_queue.py` 提交一次 VAGEN Navigation 官方配置的短冒烟训练（只用我确认的卡），记录每步耗时、显存、预计单次完整训练 GPU·小时（任务 T2 完成标准）。
6. 汇报后停下，等我决定是否 `bash launch_tmux.sh full` 进入多窗口阶段。

遇到需要密码、sudo、删除文件、或会影响别人进程的操作，一律停下来问我。
