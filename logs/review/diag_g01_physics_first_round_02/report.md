# 已有物理记录诊断

仅解释原日志，不重新验收、不推断碰撞因果。物理细分的分母为原物理检查失败的计次提议；锚点和其他物体超限可以重叠。未测量的 w1 或缺失字段保留未知。原生调用按原 action 引用去重，包含不计次的放置失败，与提议分母分开。

| 批次/提议 | 计次提议 | 物理失败 | 锚点 >2cm | 其他 >1cm | 二者重叠 | 已测其余 / 尚未知 | 原生失败调用 / 有引用调用 |
|---|---:|---:|---:|---:|---:|---:|---:|
| A/direct_flip | 509 | 375 | 296 | 225 | 147 | 0 / 1 | 7708 / 8498 |
| A/look_stay | 371 | 208 | 63 | 162 | 19 | 0 / 2 | 2058 / 2626 |
| B/direct_flip | 425 | 323 | 130 | 264 | 80 | 0 / 9 | 13299 / 13903 |
| B/look_stay | 130 | 73 | 30 | 44 | 4 | 2 / 1 | 1385 / 1601 |

“已测其余”仅指两项位移均已测量且未超限，仍可能因其他原物理原因失败；“尚未知”指无已观测超限且至少一项未知。原生调用的状态未独立重验 action 文件。

| 批次/提议 | 常见其他物体类型 | world/object 事件 | 实际落定距离可用事件 | 请求位置距离可用事件 |
|---|---|---:|---:|---:|
| A / direct_flip | Apple | 86 | 16 | 86 |
| A / direct_flip | Towel | 64 | 25 | 64 |
| A / direct_flip | HandTowel | 59 | 22 | 59 |
| A / direct_flip | Pillow | 110 | 10 | 110 |
| A / direct_flip | Bed | 48 | 3 | 48 |
| A / look_stay | Towel | 71 | 0 | 71 |
| A / look_stay | Pillow | 139 | 9 | 139 |
| A / look_stay | HandTowel | 68 | 0 | 68 |
| A / look_stay | Bed | 57 | 3 | 57 |
| A / look_stay | Pen | 17 | 1 | 17 |
| B / direct_flip | Towel | 197 | 18 | 197 |
| B / direct_flip | HandTowel | 193 | 23 | 193 |
| B / direct_flip | Tomato | 33 | 15 | 33 |
| B / direct_flip | Apple | 3 | 2 | 3 |
| B / direct_flip | BaseballBat | 3 | 2 | 3 |
| B / look_stay | Towel | 35 | 5 | 35 |
| B / look_stay | HandTowel | 29 | 4 | 29 |
| B / look_stay | Microwave | 6 | 2 | 6 |
| B / look_stay | Statue | 2 | 1 | 2 |
| B / look_stay | Fork | 1 | 1 | 1 |

同一世界同一物体跨两个基准只计一次；w0/w1 是不同物理观察。实际距离只使用同一世界内两个原落定位姿；请求位置距离另列，不能替代实际距离。类型和请求距离只使用已绑定原场景、Controller 路径及 PID 的快照。缺失快照原引用见 physics.json。
