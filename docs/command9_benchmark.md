# Kimodo 九任务 benchmark（raw，单事件关键帧）

新增的 v3 相对幅度/起始锚点修复及配对 smoke，见 [修复说明](command9_anchored_revision.md)。本页保留 v2 单事件条件的原始说明。

状态（2026-10-24）：已按 frankenmotion main `7a23d470f7b234a46bb5b2d7660e268faebf9728` 更新，21 项 CPU/原生 API 检查通过；betail 已完成 243 样本真实模型 smoke、评分及文件核验，GPU 已释放。详见 [smoke 验收](../outputs/command9_main_20261024_smoke/SMOKE_REVIEW.md)。旧 23,400 样本组合协议保留在 `benchmark_sources/command9/`，详见 [旧版协议](command9_legacy_protocol.md)。两版结果不能直接视为同一测试面板。

本地仓库：`/Users/andy/Develop/humogen/kimodo-baseline`。远端：`/home/psirobot/projects/kimodo-baseline`，conda 环境 `kimodo`。模型文件位置与下载说明见 [inference_betail.md](inference_betail.md)。

## 冻结的比较口径

本轮只做 raw 输出，不启用原生 postprocessing，也不添加 IK、优化损失、动作筛选或不可达目标裁剪。模型为 `Kimodo-SOMA-RP-v1.2`，原生 DDIM 100 步，separated CFG `[2, 2]`，batch size 1，初始 heading 为 0。使用下载好的原版 Llama + 两级 LLM2Vec adapter，保持原生 bfloat16 文本编码。

命令和评分直接调用该 main 提交的 task 源码与 evaluation_plans；输入取 `config/data/real.yaml` 指向的 `real_kick_v2_20261023`。参考姿态取当前 real-train 的中位命令样本（ID 打破平局）的第一帧 FK；reach 取当前 workspace 指定参考。原始 AMASS 和 skeleton 哈希都在导出时核对。没有复用旧协议的参考姿态来冒充当前版本。

| 身份 | 值 |
|---|---|
| main 提交 | `7a23d470f7b234a46bb5b2d7660e268faebf9728` |
| workspace | `1fe3744b1a6dbbdf9ee8c05047dd4edb555a174a71e1edbed98127c12621ef91` |
| Kimodo protocol | `c47faf352cc46d82f7186c96bc4c55bd2dc3bdd615720b0374e41309ff1e5f7e` |

| 任务 | main 命令范围 | 命令数 |
|---|---|---:|
| walk | 0.48–1.88 m/s | 30 |
| back_walk | 0.49–1.39 m/s | 30 |
| sidestep | 0.52–1.02 m | 30 |
| jog | 1.16–3.64 m/s | 30 |
| reach | x: −0.12–0.64，y: −0.70–0.33，z: −0.19–0.94 m | 1,704 |
| jump | 0.13–0.51 m | 30 |
| squat | 0.31–0.64 m | 30 |
| strike | 0.33–0.88 m | 30 |
| kick | 0.46–1.14 m | 30 |

范围是归一化/可接受命令边界；实际标量网格由 main 在真实训练支持和边界的交集内，按 10 bins × 3 points 生成。冻结训练标签中 9 条 kick 和 1 条 strike 越界，仍保留并由元数据报告，不夹取标签或修改数据。reach 使用 1,664 个 reference-verified 目标加 40 个训练 anchor。

每个命令使用 3 条原始 prompt × seeds `1928/1938/1948`，全量 **17,496** 样本。smoke 对每任务选 3 个等索引间隔命令，共 **243** 样本；它不提供正式九任务宏平均。相同任务、seed、时长的命令共享初始噪声，每次原生调用前重置 RNG；文本缓存命中不影响噪声。

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

`benchmark_sources/command9_main_20261024/` 已在本地和 betail 导出，包含 protocol、SOURCE.json 与 scoring_source。`benchmark_sources/main_7a23d47/` 是通过 git archive 从 origin/main 导出的原始输入快照，SOURCE.json 记录提交与全部文件 SHA256。两者与生成结果一样不进入 Git，迁移机器时一起复制。

评分快照包括 main 的全部九任务定义、配置、real release 和 workspace。上游源码按 `shared_motion/config/...` 解析部分配置，因此提供字节相同的配置别名，没有修改规则。评分与命令设计都验证源文件 SHA256。

使用已准备的原始快照重新导出（新输出目录必须不存在；仅 CPU/FK）：

```bash
conda activate kimodo
cd ~/projects/kimodo-baseline
CUDA_VISIBLE_DEVICES='' python -m kimodo_benchmark.export_main \
  --main-source benchmark_sources/main_7a23d47 \
  --skeleton /home/psirobot/projects/frankenmotion/outputs_amass/transfer_charlie_20261008/snapshot/outputs_amass/franken_eleven_20261003/skeleton.npz \
  --amass-root /mnt/sda2/dataset/amass \
  --output benchmark_sources/command9_main_new
```

仅检查文件，不导入 torch、不加载模型：

```bash
CUDA_VISIBLE_DEVICES='' python -m kimodo_benchmark.generate \
  --protocol benchmark_sources/command9_main_20261024/protocol.json --preflight
```

增加 `--verify-assets` 会读取模型文件核对/记录 SHA256，仍不推理。运行时默认资产位置为 `~/checkpoints/kimodo`，可用 `--assets` 指定。

## 生成与评分命令

smoke 的命令如下。generate 会占用 GPU；score 仅使用 CPU。当前 smoke 路径为 `outputs/command9_main_20261024_smoke/`。

```bash
python -m kimodo_benchmark.generate \
  --protocol benchmark_sources/command9_main_20261024/protocol.json \
  --output outputs/command9_main_20261024_smoke/generated --maximum-commands 3 \
  --device cuda:0 --text-encoder-device cuda:0

CUDA_VISIBLE_DEVICES='' python -m kimodo_benchmark.score \
  --generated outputs/command9_main_20261024_smoke/generated \
  --output outputs/command9_main_20261024_smoke/reports --allow-partial
```

全量尚未运行；以后明确需要时再执行：

```bash
python -m kimodo_benchmark.generate \
  --protocol benchmark_sources/command9_main_20261024/protocol.json \
  --output outputs/command9_main_full \
  --device cuda:0 --text-encoder-device cuda:0

CUDA_VISIBLE_DEVICES='' python -m kimodo_benchmark.score \
  --generated outputs/command9_main_full --output outputs/command9_main_full_report
```

生成支持 `--tasks reach strike kick` 等筛选；去掉筛选后可在同目录续跑。报告用 matplotlib，可通过 `pip install -e '.[command9]'` 安装可选报告依赖；现有 betail kimodo 环境已具备本次测试所需依赖。

## CPU 验证

```bash
CUDA_VISIBLE_DEVICES='' python -m unittest discover -s tests -p 'test_command9*.py' -v
CUDA_VISIBLE_DEVICES='' KIMODO_BENCHMARK_BUNDLE=benchmark_sources/command9_main_20261024 \
  python tests/native_command9.py
```

18 项单元/合成数据测试覆盖旧版兼容、新版 main 逐命令采样/元数据一致性、坐标/时间、评分、缓存/种子、恢复和报告；3 项原生 API 测试检查当前全部 1,944 个命令的 condition mask/值、归一化接口与生成接线。原生 API 检查仅加载小型 skeleton 和合成统计量。真实模型 smoke 也已完成；它验证流程，不代表所有任务的动作质量或全量性能合格。

脚本入口亦可使用 `scripts/export_command_benchmark.py`（当前 main 快照）、`scripts/generate_command_benchmark.py`、`scripts/score_command_benchmark.py`，参数与对应模块一致。
