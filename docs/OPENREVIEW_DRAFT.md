# OpenReview 注册草稿 v3.1（11-10 23:59 AoE 前由人类提交）

标题和摘要在 11-16 正文截止前都还能改；**注册后作者名单不能再增删**。先用诚实草稿注册，结果出来后更新。

## 标题（候选）
1. Don't Reward Lucky Guesses: Counterfactual World Pairs Teach Embodied Agents When to Look and When to Change Their Minds
2. Flip, Stay, or Look: Information-Consistent Counterfactual Worlds for Evidence-Seeking Embodied Agents

## 摘要草稿（不含任何数字；结果出来后替换最后两句）
Embodied vision-language agents should gather evidence when their observations cannot yet determine the right goal, and then change or keep that goal once the evidence arrives. Reinforcement learning makes this hard to teach: when a learning signal is computed only in the world that happened to be realized, an agent that commits without looking can be rewarded whenever its guess happens to be right. We construct pairs of simulated worlds that are consistent with exactly the same observation history but may call for different goals, and require behavioral divergence only after discriminating evidence has entered the agent's input. These pairs let us assign credit to evidence-seeking decisions over the worlds the agent cannot yet tell apart, and supervise how the goal should be updated once it can — switching when the evidence calls for it and staying when it does not. Every pair is checked for deterministic restoration, physical validity, a unique target in each world, an executable success witness, and pixel-level information consistency. We compare against training on the same pairs without cross-world terms, same-world look-versus-commit branching, and fixed and heuristic observation policies, reporting success as a function of observation cost across different rates of world change.

## 主题领域
Embodied vision and robotics; Agentic AI; Physical AI; Reasoning & Reinforcement Learning
