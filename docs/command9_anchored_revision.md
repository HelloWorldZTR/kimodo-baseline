# Command9 v3：相对幅度与起始锚点

原版 v2 已保存在 `outputs/command9_main_20261024_smoke/`，包括原始生成、报告和校验过的 `code_snapshot/`。本修复不改 main `7a23d47` 的命令、时长、参考姿态、文本、种子、评分函数或归一化范围。

v2 的标量命令与单帧目标并不等价：strike 用参考肩位置加命令，却测量手腕行程；kick 用前脚掌位置加命令，却测量踝的增益；jump/squat 以参考高度构造目标，却未约束实际生成的起始高度。squat 的目标还只覆盖一个动作帧，评分却用五帧均值。

v3 使用原生 EEF/Hips 条件补足这些基准，不添加完整姿态、IK、去噪优化、目标裁剪或后处理。每个条件在所有 100 次去噪中持续输入。动作时间上的约束如下，帧号均从 0 开始、30 Hz：

| 任务 | 初始/准备锚点 | 目标 | 回位锚点 |
|---|---|---|---|
| strike | 帧 0、15：右手腕为参考初始手腕 | 帧 30：手腕 forward=初始手腕 forward+d；高度/横向取参考肩 | 帧 45、58、59：回到初始手腕位置 |
| kick | 帧 0：右踝为参考初始踝 | 帧 45：踝 forward=初始踝 forward+d，横向不变；前脚掌仍为绝对高度 1 m | 帧 88、89：回到初始踝位置 |
| jump | 帧 0：骨盆高度 h | 帧 48：骨盆高度 h+d | 帧 97、98：回到 h |
| squat | 帧 0–6：骨盆高度 h，对应前五个评分时刻 | 帧 58–65：骨盆高度 h−d，覆盖第 39–43 个评分帧的五帧窗口 | 帧 115–122：回到 h，覆盖最后五个评分时刻 |

EEF 的手端/前脚掌附属目标按固定原生骨长与旋转计算；初始、事件和回位使用同一旋转。骨盆水平位置、朝向仍沿用原配置，没有为大命令额外移动骨盆。这些是选中关节及原生附带 root 的锚点，并非固定全身起始姿态。

walk、back_walk、sidestep、jog、reach 的 condition 与 v2 完全相同。v3 对四个问题任务提供更多条件帧，因此与 v2 的对比是条件适配方式的对比，不能标为“相同条件数量下”的模型改进。

固定初始目标仍不能保证生成值精确命中；单帧 jump 目标也不能限制其他时刻的高度，strike 的事件窗口由原评分自行检测。高命令在固定骨盆下可能超出原生肢体可达范围，保留原始误差，不裁剪命令或调整评分。

协议：`benchmark_sources/command9_anchored_v3_20261024/protocol.json`，ID `01c8eb82bbeac069902103ecde18c8297b899e62eeda82f8e96427ac1d0f46fc`。smoke 使用与 v2 相同的 243 个样本身份，输出到 `outputs/command9_anchored_v3_20261024_smoke/`。

## 复现

```bash
python -m kimodo_benchmark.revise_conditions \
  --protocol benchmark_sources/command9_main_20261024/protocol.json \
  --output benchmark_sources/command9_anchored_new

python -m kimodo_benchmark.generate \
  --protocol benchmark_sources/command9_anchored_v3_20261024/protocol.json \
  --output outputs/command9_anchored_v3_20261024_smoke/generated \
  --maximum-commands 3 --device cuda:0 --text-encoder-device cuda:0

CUDA_VISIBLE_DEVICES='' python -m kimodo_benchmark.score \
  --generated outputs/command9_anchored_v3_20261024_smoke/generated \
  --output outputs/command9_anchored_v3_20261024_smoke/reports --allow-partial

python scripts/plot_command_response.py \
  --reports outputs/command9_anchored_v3_20261024_smoke/reports \
  --before outputs/command9_main_20261024_smoke/reports \
  --output outputs/command9_anchored_v3_20261024_smoke/comparison
```

26 项 CPU/原生 API 检查通过，包括 120 个受影响命令的原生目标/掩码、五帧窗口与精确 d 差值，以及 1,824 个未受影响命令的条件一致性。动作页现在全程显示目标标记，在受约束的动作帧填实，其他时刻为空心；显示方式不影响模型条件。

## 实测结果

配对 243 样本 smoke 已完成，GPU 已释放。strike MAE 0.343→0.029 m，squat 0.218→0.015 m；kick 0.208→0.185 m，仅略有改善；jump 0.121→0.158 m，变差。后两项在指定目标帧之前发生更大的峰值，当前尚未完全解决。详见 [配对结果与响应图](../outputs/command9_anchored_v3_20261024_smoke/SMOKE_REVIEW.md)。五个未改动任务的 135 条输出与旧版逐数值一致。
