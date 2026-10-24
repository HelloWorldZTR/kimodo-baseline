# Kimodo 九任务 benchmark（raw，单事件关键帧）

状态（2026-10-24）：已实现并同步到 betail；仅执行 CPU 单元测试、原生 condition API 检查和文件预检。**没有运行本次 benchmark 的模型推理或正式评分。** 之前的无约束 walking smoke 不等同于本次九任务验证。

本地仓库：`/Users/andy/Develop/humogen/kimodo-baseline`。远端：`/home/psirobot/projects/kimodo-baseline`，conda 环境 `kimodo`。模型文件位置与下载说明见 [inference_betail.md](inference_betail.md)。

## 冻结的比较口径

本轮只做 raw 输出，不启用原生 postprocessing，也不添加 IK、优化损失、动作筛选或不可达目标裁剪。模型为 `Kimodo-SOMA-RP-v1.2`，原生 DDIM 100 步，separated CFG `[2, 2]`，batch size 1，初始 heading 为 0。使用下载好的原版 Llama + 两级 LLM2Vec adapter，保持原生 bfloat16 文本编码。

命令、三条原始 prompt、时长、参考姿态和 reach workspace 均来自已冻结的 OmniControl 协议。七任务沿用 2026-10-21 版本；strike/kick 使用 2026-10-22 的目标代理与 2026-10-23 的初始 root 坐标系评分。不从当前 frankenmotion checkout 重新采样或重新计算 workspace。

| 来源 | 身份 SHA256 |
|---|---|
| 原九任务 protocol | `cccee66b326263fcb8ca73fe45513e06c769bd18dab8d727a659e75e643b5ea8` |
| strike/kick generation protocol | `fb2d7e81fca272031b00f41c9df75ff5ef5f81598eab0de87e2c74ee6c76a9b6` |
| strike/kick evaluation | `d3af3800a0907d81d8411666422c31a31c8637e392c418f9b7d2ad81fd0fce7f` |
| reach workspace | `8df745f08b937ecff8e9abb85ad6e79af3a544c0bd9a5b23613d12d86b0ad17d` |
| 本轮 Kimodo composite protocol | `88477e5a7d0147d4648129a375b512a3596572fbcaf712d185e42d336dc7c434` |

reach 有 2,360 个命令，其他八任务各 30 个；每个命令使用 3 条 prompt × seeds `1928/1938/1948`，总计 **23,400** 个样本。相同任务、seed、时长的命令使用相同初始噪声；每次原生调用前重置 RNG，文本缓存命中与否不影响噪声。

## 命令到 condition

F 表示 benchmark 坐标 `(forward, left, up)`，K 表示 Kimodo `(X,Y,Z)`；转换为 `K=(F.y,F.z,F.x)`，单位米。以下帧号均为 **从 0 开始的 30 FPS 原生帧**。`h` 是冻结参考姿态的初始 pelvis 高度，`ref[j]` 是其 F 坐标，`S(u)=3u²−2u³`。

| 任务 | 原生 condition | 命令转换 | 原生帧数 / 评分帧数 |
|---|---|---|---|
| walk | dense `Root2DConstraintSet` | K-XZ = `(0, v·t)` | 90 / 60 |
| back_walk | dense `Root2DConstraintSet` | K-XZ = `(0, −v·t)` | 90 / 60 |
| sidestep | dense `Root2DConstraintSet` | K-XZ = `(−d·S(t/D), 0)`；`D=29/20` 秒，最后两个原生帧精确到终点 | 45 / 30 |
| jog | dense `Root2DConstraintSet` | K-XZ = `(0, v·t)` | 75 / 50 |
| reach | 第 66 帧 `RightHandConstraintSet` | F 手腕目标 = `(cx,cy,h+cz)`；只约束 hold 窗口首帧，后续保持由文本引导 | 73 / 49 |
| jump | 第 48 帧 `EndEffectorConstraintSet(joint_names=["Hips"])` | K pelvis = `(0,h+d,0)` | 99 / 66 |
| squat | 第 61 帧 `EndEffectorConstraintSet(joint_names=["Hips"])` | K pelvis = `(0,h−d,0)` | 123 / 82 |
| strike | 第 30 帧 `RightHandConstraintSet` | F 手腕目标 = `ref[17]+(d,0,0)`，17 为右肩关节 | 60 / 40 |
| kick | 第 45 帧 `RightFootConstraintSet` | F 前脚掌目标 = `(ref[11].x+d,ref[11].y,1.0)`；据原生固定足部偏移反推踝点 | 90 / 60 |

实现依据仓库的 `kimodo/demo/generation.py`：EEF 的有效帧直接进入原生 constraint，未对两个事件之间插值生成约束。这里也仅保留一个事件帧，没有额外添加准备、回收、落地或初始全身姿态关键帧。

原生 EEF 并非只写一个 XYZ：RightHand 会约束手腕与中指末端位置、手腕 global rotation；RightFoot 会约束踝与前脚掌位置、踝 global rotation。该事件帧还包含 smooth-root XZ、root 高度和 heading。手部旋转由原生手腕→中指末端向量最短旋转对齐冻结前臂方向；足部旋转对齐冻结踝→前脚掌方向。Hips 使用单位旋转。手/足事件帧的 root XZ 为 0、高度为 h、heading 为 0；jump/squat 只改变该帧高度。

因此原生 369 维条件中，手/足事件只有一帧的 **17 个通道**，Hips 事件只有一帧的 **14 个通道**，其他姿态通道均未约束。Root2D 每帧仅 **2 个 XZ 通道**，不固定全程高度或 heading。代码会独立核对实际原生 mask。

模板中未选关节只是原生 API 的输入容器，不会成为全身约束。当前构造只用于 raw 的 masked channels；以后若加入依赖完整 FK 一致姿态的 postprocessing，需要另做适配，不能直接打开开关。

## 输出与评分

每个样本保存原生 30 FPS、77 关节 XYZ 与 local rotations、实际 root/smooth root，以及送入模型的 normalized condition、bool mask、原始/原生清洗后的 prompt、命令和 seed。另按解剖名称映射到 22 关节 F 坐标，在准确的 `i/20` 秒采样。

原生帧数为 `ceil((T−1)×30/20)+1`；20 FPS 插值索引为 `1.5·i`，不通过拉伸端点改变速度，末尾额外半帧不进入评分。关节映射中 8 为 RightFoot（踝），11 为 RightToeBase（前脚掌），17 为 RightArm（肩），21 为 RightHand（手腕）。

评分直接调用冻结 task 源码，使用固定初始 pelvis/heading 坐标系，保留身体整体平移。strike 的指标是事件窗口内手腕前向行程；kick 是踝相对初始踝的最大前向增益。它们的 condition 目标是预先约定的代理，命中目标不等于命令误差必然为零。jump/squat 仍由整段动作测量，单帧达到指定高度也不保证五帧均值指标。

逐样本算绝对误差，再汇总 MAE/nMAE；宏平均等权计九个任务。reach 继续报告冻结的质量诊断、workspace 分组、real-supported/gap、容差成功率与联合质量成功率。足滑沿用 OmniControl 的 20 Hz toe-height/velocity 规则。`native_primary_error_m` 单独衡量原生 smooth-root XZ 或单帧 primary target 的误差，不能把它当作跨模型的任务指标。

报告包括 `results.json`、`summary.csv`、`samples.csv`、每任务曲线 NPZ/PNG 与离线 HTML 动作页。标量任务固定选低/中/高命令，reach 按距冻结 workspace center 的半径选择，展示全部 9 个 prompt/seed 组合，不选最佳 seed。HTML 用原生 30 FPS 播放。

不足全量的结果必须显式 `--allow-partial`，且每个命令必须有完整的 3×3 组合；partial 输出不会给出正式 macro nMAE。protocol、源码、模型文件、运行环境、几何和样本内容均有身份或哈希检查。更换生成配置或源码要使用新输出目录；重复运行同一命令自动复用已校验样本，损坏文件会报错，不静默跳过。

## 已准备好的文件

`benchmark_sources/command9/` 已在本地和 betail 导出，包含 protocol、三个原始来源 JSON 和 `scoring_source/`。它与模型、生成结果一样不进入 Git，迁移机器时应一起复制，或使用下方命令重建。

评分快照只覆盖替换后的 strike/kick 及其依赖，保留原 reach。冻结 reach 源码把 YAML 解析到 `shared_motion/config/...`，导出器在该处复制同一份已校验 YAML，未修改规则内容。加载器直接导入九个 task 模块，避免上游 umbrella registry 加载四个不在本轮协议内的任务和未冻结配置。

在 betail 重建（输出目录必须不存在）：

```bash
conda activate kimodo
cd ~/projects/kimodo-baseline
python -m kimodo_benchmark.export \
  --base-protocol ../OmniControl-baseline/outputs/command9_20261021/protocol.json \
  --limb-protocol ../OmniControl-baseline/outputs/command_limb_fixed_target_20261022/protocol.json \
  --limb-evaluation ../OmniControl-baseline/outputs/command_limb_initial_frame_eval_20261023/evaluation.json \
  --base-source ../OmniControl-baseline/benchmark_sources/frankenmotion \
  --limb-source ../OmniControl-baseline/outputs/command_limb_initial_frame_eval_20261023/scoring_source \
  --output benchmark_sources/command9_new
```

仅检查文件，不导入 torch、不加载模型：

```bash
CUDA_VISIBLE_DEVICES='' python -m kimodo_benchmark.generate \
  --protocol benchmark_sources/command9/protocol.json --preflight
```

增加 `--verify-assets` 会读取模型文件核对/记录 SHA256，仍不推理。运行时默认资产位置为 `~/checkpoints/kimodo`，可用 `--assets` 指定。

## 后续运行命令（本次未执行）

先做每任务 3 个命令、共 243 个样本的 smoke；**以下 generate 命令会占用 GPU，需等实际开始评估时再执行。**

```bash
python -m kimodo_benchmark.generate \
  --protocol benchmark_sources/command9/protocol.json \
  --output outputs/command9_raw_smoke --maximum-commands 3 \
  --device cuda:0 --text-encoder-device cuda:0

CUDA_VISIBLE_DEVICES='' python -m kimodo_benchmark.score \
  --generated outputs/command9_raw_smoke \
  --output outputs/command9_raw_smoke_report --allow-partial
```

全量生成和评分分开执行：

```bash
python -m kimodo_benchmark.generate \
  --protocol benchmark_sources/command9/protocol.json \
  --output outputs/command9_raw_full \
  --device cuda:0 --text-encoder-device cuda:0

CUDA_VISIBLE_DEVICES='' python -m kimodo_benchmark.score \
  --generated outputs/command9_raw_full --output outputs/command9_raw_full_report
```

生成支持 `--tasks reach strike kick` 等筛选；去掉筛选后可在同目录续跑。报告用 matplotlib，可通过 `pip install -e '.[command9]'` 安装可选报告依赖；现有 betail kimodo 环境已具备本次测试所需依赖。

## CPU 验证

```bash
CUDA_VISIBLE_DEVICES='' python -m unittest discover -s tests -p 'test_command9.py' -v
CUDA_VISIBLE_DEVICES='' python tests/native_command9.py
```

15 项单元/合成数据测试覆盖协议、坐标/时间、条件、评分、种子缓存、文件损坏检查、断点续跑和报告；3 项原生 API 测试核对全部 2,600 个命令、native mask/值、归一化/批次接口以及生成参数/RNG 接线。原生检查只加载小型 skeleton 资产及合成统计量，没有加载 denoiser 或 Llama。真实模型的九任务输出与性能仍待之后的 GPU smoke 验证。
