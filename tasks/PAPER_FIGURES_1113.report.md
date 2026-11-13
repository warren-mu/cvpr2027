# 论文插图选择（2026-11-13）

[离线图库目录](../paper/fig_src/training_gallery_1113/) 有 6 张引言图、6 张方法图、8 个真实推理对比和 24 组双世界回放。下载目录并打开 index.html 可查看原图、切换图层、勾选编号。原先选定的 FloorPlan203 Box 和 FloorPlan7 Bread 配对用于 INTRO-01、METHOD-01。英文论文面板的 SVG 与 PNG 同时提供。

定性对比 Q01–Q02 取自 T0/T4 第 50 次更新的训练 rollout，Q03–Q08 取自 DIAG02 原始模型 3B、7B、32B 的固定策略。每例附原日志定位、世界顺序、动作及对应输入哈希。后六例的观察顺序由脚本指定，最终选择由模型给出。案例按可解释的行为差异或一致性选择，供论文展示。

世界对回放取自通过 T02 和缓存复核的非留出训练场景。24 个配置来自 12 个场景、14 种锚点，Direct Flip 5、Direct Stay 7、Look Flip 5、Look Stay 7。每组固定三个观察动作，两个世界各存起点及三步，合计 192 张标号输入。提供未加标号 RGB、三个实例掩码、标号位置、哈希、位姿及按真实坐标生成的俯视示意。选择轨迹时使用视角表中的可见性，仅用于演示。

Hopper 上的运行入口为 `experiments.fig_training_gallery`，通过 `scripts/run_logged.py` 记录。17 项回归通过，1 组冒烟及 4 个后续分片通过，[原始来源审计](../results/diag_fig_training_gallery_audit_01.json) 无 FAIL。取回后逐文件核对 SHA、颜色、尺寸和掩码，定性面板另外核对与原模型历史的对应。GPU 0 小时，无 PBS 作业、排队或模型新推理。Hopper scratch 截面（新加坡 11-14 05:53）为 865.1 GB / 1.0 TB，3,895,878 个文件 / unlimited；home 配额表的错误保留。后续排版可直接使用选定编号的 SVG、PNG 或原帧，追加 GPU 需求为 0。
