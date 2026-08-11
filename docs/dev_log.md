# 开发日志 — EvoVILA-Seg 时空分割扩展
> 创建时间：2026-08-05 | 最后更新：2026-08-05
> 关联实现指南：docs/implementation.md
> ⚠️ 本文件只追加，不删除。每一次代码修改都必须追加新的日志条目。

## 项目概览

| 项目 | 内容 |
|------|------|
| 研究方向 | 在保持 VILA 原有能力的前提下，扩展语言条件图像/视频时空分割 |
| 实现策略 | 基于 VILA main 的 opt-in adapter；S0 先独立实现解码器 |
| 框架 | PyTorch（复用现有 `evovila` 环境） |
| Git 仓库 | 本地 EvoVILA-Seg 分支；保留 origin/upstream |
| 推送范围 | 本轮不推送 |

## 实现进度

| 模块 | 文件 | 状态 | 完成时间 | 备注 |
|------|------|------|---------|------|
| 需求与实现指南 | `docs/user_requirements.md`, `docs/implementation.md` | ✅ Done | 2026-08-05 | S0 已确认；S1-S4 延后 |
| S0 public package | `llava/evo_seg/__init__.py` | ✅ Done | 2026-08-05 | 不导入 SAM2/VILA |
| S0 解码器 | `llava/evo_seg/decoder.py` | ✅ Done | 2026-08-05 | 15 个契约/解码器测试通过 |
| S0 损失 | `llava/evo_seg/losses.py` | ✅ Done | 2026-08-05 | BCE/Dice/objectness/query-swap/temporal |
| S0 capability | `llava/evo_seg/capability.py` | ✅ Done | 2026-08-05 | 显式 opt-in、计时、状态清理 |
| S0 测试与 smoke | `tests/`, `scripts/evo_seg/` | ✅ Done | 2026-08-05 | 30 passed；synthetic smoke 通过 |
| S0 契约 | `llava/evo_seg/contracts.py` | ✅ Done | 2026-08-05 | 绝对环境路径下编译通过 |
| S0 契约测试 | `tests/test_evo_seg_contracts.py` | ✅ Done | 2026-08-05 | 10 passed |
| S0 解码器测试 | `tests/test_evo_seg_decoder.py` | ✅ Done | 2026-08-05 | query sensitivity、padding、梯度通过 |
| S0 损失测试 | `tests/test_evo_seg_losses.py` | ✅ Done | 2026-08-05 | 6 个负控测试通过 |
| S0 capability 测试 | `tests/test_evo_seg_capability.py` | ✅ Done | 2026-08-05 | 4 passed |
| S0 配置 | `configs/evo_seg/s0_decoder.yaml` | ✅ Done | 2026-08-05 | 仅合成维度和 CPU 默认值 |
| S0 smoke | `scripts/evo_seg/smoke_decoder.py` | ✅ Done | 2026-08-05 | 无权重 forward/backward 通过 |
| VILA 基线契约 | `tests/test_evo_seg_baseline.py` | ✅ Done | 2026-08-05 | text/image/multi-image/video |
| 仓库卫生 | `.gitignore` | ✅ Done | 2026-08-05 | 排除 cache/权重/结果目录 |
| README | `README.md` | ✅ Done | 2026-08-05 | S0 状态、环境与验证命令 |
| S0 milestone | 全部 S0 文件 | ✅ Done | 2026-08-05 | 30 passed；review 通过 |
| S1 fusion scope | `llava/model/fusion_observer.py` | ✅ Done | 2026-08-05 | ContextVar；无 dense import |
| S1 provenance | `llava/evo_seg/provenance.py` | ✅ Done | 2026-08-05 | fused rows、padding、multi-token gather |
| S1 provenance tests | `tests/test_evo_seg_provenance.py` | ✅ Done | 2026-08-05 | 6 passed |
| S1 VILA hook | `llava/model/llava_arch.py` | ✅ Done | 2026-08-05 | 两个 native `_embed`；默认路径不变 |
| S1 VILA adapter | `llava/evo_seg/vila_adapter.py` | ✅ Done | 2026-08-05 | frozen query extraction、dense bridge、分项计时 |
| S1 integration tests | `tests/test_evo_seg_vila_adapter.py` | ✅ Done | 2026-08-05 | 9 passed；全套 45 passed |
| S2 RGB/SAM2 boundary | `llava/evo_seg/sam2_adapter.py` | 🚧 In progress | 2026-08-05 | lazy local build；真实资产 smoke 待完成 |
| S2 boundary tests | `tests/test_evo_seg_sam2_adapter.py` | ✅ Done | 2026-08-05 | 7 passed；全套 53 passed |
| S2 real provider smoke | `scripts/evo_seg/smoke_sam2_image.py` | ✅ Done | 2026-08-05 | 完成上方 In progress 快照中的 provider gate；完整 VILA 图像链路待验证 |
| S2 provider regression | `tests/test_evo_seg_sam2_adapter.py` | ✅ Done | 2026-08-05 | 8 个 boundary 测试；全套 54 passed |
| S2 composed image path | `llava/evo_seg/image_pipeline.py`, `scripts/evo_seg/smoke_image_segmentation.py` | 🚧 In progress | 2026-08-05 | 已找到本地 VILA1.5-3B；先修正显式冻结语义 |
| S2 image plumbing gate | `llava/evo_seg/image_pipeline.py`, `scripts/evo_seg/smoke_image_segmentation.py` | ✅ Done | 2026-08-05 | 完成上方 In progress 快照；真实 VILA+SAM2 与 retention 通过 |

## 开发日志

### 2026-08-05 — 初始化 S0

- **完成内容**：从 `main@0f1426e` 建立 `EvoVILA-Seg` worktree；整理用户约束和函数级实现指南；将图像统一为视频的 `T=1` 契约。
- **遇到的问题**：初版结果只含 `[B,N,D]` 对象表示，无法支持后续相邻帧时序一致性。
- **解决方案**：在实现指南中增加 `[B,N,T,D]` `frame_embeddings`，并明确 objectness 的有效帧归约和 query-swap 的有效区域。

### 2026-08-05 — S0 契约完成

- **完成内容**：新增 `llava/evo_seg/contracts.py`，实现不可变 request 选项、统一图像/视频批次校验、结果形状和无效帧零值校验。
- **遇到的问题**：`conda run -n evovila` 指向不存在的公共环境别名。
- **解决方案**：改用 `/9950backfile/chenjiahui/.conda/envs/evovila/bin/python`；该环境 Python 3.10.14、PyTorch 2.4.1，静态编译通过。

### 2026-08-05 — S0 契约测试完成

- **完成内容**：新增 `tests/test_evo_seg_contracts.py`，覆盖 T=1/T>1、padding、请求冻结和无效结果帧。
- **遇到的问题**：直接调用 `pytest` 启动脚本时 worktree 未加入 `sys.path`。
- **解决方案**：统一使用 `python -m pytest`；契约测试 10 passed。

### 2026-08-05 — S0 解码器完成

- **完成内容**：新增 `llava/evo_seg/decoder.py`，实现多 token language query、对象到语言/空间 cross-attention、动态时空位置编码、anchor mask logits、逐帧/全局对象表示和 objectness。
- **遇到的问题**：初版测试没有精确按每个样本的 frame mask 索引，且常数平移 query 会被 LayerNorm 消除。
- **解决方案**：修正负控测试；`python -m pytest -q tests/test_evo_seg_contracts.py tests/test_evo_seg_decoder.py` 结果为 15 passed。

### 2026-08-05 — S0 损失完成

- **完成内容**：新增 `llava/evo_seg/losses.py`，实现分辨率对齐的 BCE/Dice、有效帧 objectness、同图 query-swap margin、逐帧 temporal consistency 以及加权总 loss。
- **遇到的问题**：无；缺失 target 或启用 query-swap 却缺少 swapped logits 时均 fail closed。
- **解决方案**：新增 6 个 loss/negative-control 测试；S0 合计测试结果为 21 passed。

### 2026-08-05 — S0 capability 完成

- **完成内容**：新增 `llava/evo_seg/capability.py`，将 decoder 包装为 request-scoped opt-in 公共入口，记录 decoder/total timing，并提供无媒体状态的清理接口；包入口同步导出 S0 public API。
- **遇到的问题**：无；禁用请求不会调用 decoder，包导入未引入 `sam2`。
- **解决方案**：新增 capability 的 4 个隔离测试；S0 当前测试结果为 25 passed。

### 2026-08-05 — S0 smoke 完成

- **完成内容**：新增 `configs/evo_seg/s0_decoder.yaml` 和 `scripts/evo_seg/smoke_decoder.py`，输出 commit/config hash、shape、finite、gradient、可选模块导入和组件计时。
- **遇到的问题**：直接运行脚本时 Python 默认路径不含仓库根目录。
- **解决方案**：smoke 入口在导入前显式加入仓库根目录；CPU synthetic smoke 通过，`sam2` 未导入。

### 2026-08-05 — S0 主动审查与基线契约完成

- **完成内容**：收紧不可变 sample IDs、同 device tensor 契约、mixed floating dtype 投影、无同步 masked mean 和 provider 模块注册；新增 VILA 原始 text/single-image/multi-image/video draft prompt contract 测试及根目录 `.gitignore`。
- **遇到的问题**：初版 SAM2 导入测试在同一进程内断言不够强；测试 cache 未被根目录规则排除。
- **解决方案**：改用独立 Python 进程验证 `import llava.evo_seg` 不产生 `sam2`，并显式忽略 cache/权重/结果；S0 全套 30 passed。

### 2026-08-05 — README 同步完成

- **完成内容**：在上游 README 顶部新增 EvoVILA-Seg S0 状态、环境和快速验证命令，不改动原 VILA 文档内容。
- **遇到的问题**：无。
- **解决方案**：明确 S0 尚未集成 SAM2 或修改 VILA model code，避免把后续 gate 描述成已完成能力。

### 2026-08-05 — S0 最终验证完成

- **完成内容**：重跑 30 个 no-weight 测试、S0 synthetic smoke、全部新增 Python 文件静态编译和 Git 空白检查；核对新 worktree 分支/HEAD 与 `main@0f1426e` 一致。
- **遇到的问题**：无；原 worktree 的 `docs/user_requirements.md` 与 `long_rl` 脏改动仍原样保留。
- **解决方案**：smoke 结果写入仓库外 `/tmp/evovila-seg-smoke.0DNV2X.json`；finite/gradient 均为 true，`sam2` 未导入，decoder/total CPU 时间为 317.797 ms（单次 smoke，仅用于路径验证）。

### 2026-08-05 — S1 fusion observer 完成

- **完成内容**：新增 `llava/model/fusion_observer.py`，提供 request-local `ContextVar` scope 和 `None` 默认读取路径。
- **遇到的问题**：无；模块不导入 `llava.evo_seg`、SAM2 或任何 dense component。
- **解决方案**：静态编译和 Git 空白检查通过，等待 provenance recorder 接入 VILA `_embed`。

### 2026-08-05 — S1 provenance 完成

- **完成内容**：新增 `llava/evo_seg/provenance.py`，实现 fused source rows、左右 padding、valid lengths、显式原始 token mask 到完整 query state batch 的映射。
- **遇到的问题**：初版测试夹具给第二个样本标记了不存在的第四个 valid fused token。
- **解决方案**：修正夹具为右 padding；provenance 单测 6 passed，未加载权重或数据。

### 2026-08-05 — S1 VILA fusion hook 完成

- **完成内容**：在 repository-native 标准与 top-down `_embed` 中接入同一个 request-local observer；仅 observer 激活时记录 text/image/video 来源、原始 token 位置、媒体展开与截断结果。
- **遇到的问题**：原始 `_embed` 在移除 padding 后仍按未压缩 `input_ids` 索引；为保持默认行为，不能顺手改变 ordinary path 的历史语义。
- **解决方案**：observer 为 `None` 时保留原分支；只有 opt-in scope 使用压缩后的 token IDs/原始位置映射。无权重测试覆盖左右 padding、多图、视频和 top-down 返回契约。

### 2026-08-05 — S1 VILA adapter 完成

- **完成内容**：新增 `DenseFeatureBatch` 与 `VILASegmentationAdapter`；冻结执行 VILA teacher forcing，按显式原始 token mask 提取完整多 token query states，再连接注入式 dense provider 和 S0 capability。
- **遇到的问题**：包入口直接导出 adapter 时，模块级导入 `llava.model.fusion_observer` 会先执行重量级 `llava.model.__init__`，破坏默认导入隔离。
- **解决方案**：fusion scope 仅在显式 adapter 调用内惰性解析；独立进程验证 `import llava.evo_seg` 不加载 `llava.model` 或 SAM2，普通 `llava.model.llava_arch` 导入也不加载 `llava.evo_seg`。

### 2026-08-05 — S1 集成验证完成

- **完成内容**：新增 9 个 adapter/native harness 测试，覆盖默认 `_embed` 返回、左右 padding、截断、media `-1` 映射、多图/视频顺序、top-down、frozen hidden states、disabled request 和四段计时。
- **遇到的问题**：fake media 每个展开块长度不同，初版测试的预期 fused length 计算错误；top-down 私有 helper 还需返回 selection map/prob 三元组。
- **解决方案**：按 native deque 消费顺序修正夹具并覆盖真实返回结构；`python -m pytest -q tests/test_evo_seg_*.py` 最终为 45 passed，无权重或数据下载。

### 2026-08-05 — S0/S1 milestone 提交

- **完成内容**：将已验证的 S0/S1 扩展、测试和文档提交为 `b625184`（`feat: add opt-in segmentation foundation`）。
- **遇到的问题**：暂存检查发现 5 个新文件末尾存在额外空白行。
- **解决方案**：只做 EOF 格式修正后重新执行 `git diff --cached --check`；未包含 cache、权重、数据、结果或原工作树的本地改动，未推送远端。

### 2026-08-05 — S2 lazy SAM2 image boundary

- **完成内容**：新增严格的 CPU uint8 `RGBFrameBatch`、本地-only `SAM2BuildOptions`、惰性官方 image predictor builder、dense image embedding provider 和 coarse-mask refinement；VILA adapter 新增显式 `dense_input`，不复用 VILA-normalized media。
- **遇到的问题**：当前 `evovila` 环境没有 `sam2` 包或本地源码/checkpoint；官方主线文档要求 PyTorch 2.5.1+，而现有环境为 PyTorch 2.4.1，仓库依赖还固定为 2.3.0，直接安装可能破坏 VILA 基线。
- **解决方案**：不安装、不升级、不下载；使用遵循官方 `set_image_batch`/`predict_batch` API 的 fake predictor 验证 lazy import、冻结、padding、状态清理和 refinement。S2 保持 In progress，等待独立兼容性决策和真实资产 smoke。

### 2026-08-05 — S2 boundary regression

- **完成内容**：新增 7 个 SAM2 boundary 测试，并增加 VILA media 与 raw dense input 隔离测试。
- **遇到的问题**：无；SAM2 默认 builder 在本地源码缺失时 fail closed，且失败前不会导入 SAM2。
- **解决方案**：`python -m pytest -q tests/test_evo_seg_*.py` 为 53 passed；仍未下载权重或数据。

### 2026-08-05 — S2 真实 SAM2 provider smoke 完成

- **完成内容**：发现机器上已有独立 SAM2 环境、官方源码和 tiny checkpoint；新增 `SAM2ImageFeatureProvider.initialize()` 与真实 CUDA smoke，分别测量模型初始化、图像编码、带重编码的掩码细化和峰值显存。所有源码和 checkpoint 均通过 CLI 从 Git 外部传入。
- **遇到的问题**：先前认为 `evovila` 环境缺少可用 SAM2；进一步检查发现虽然该环境未安装 SAM2 包，但可从现有本地源码树惰性加载，且无需升级其 PyTorch。独立 SAM2 环境缺少 VILA 的 `transformers`、`deepspeed` 等依赖，只适合 provider 隔离验证。
- **解决方案**：在 `evovila` 环境和独立 SAM2 环境分别执行同一真实 provider smoke，均通过。`evovila`/A800 单次结果为初始化 6842.09 ms、encoder 358.69 ms、refinement-with-reencode 800.69 ms、峰值 601.12 MiB；独立 SAM2 环境单次结果为 7178.49/401.75/352.12 ms。输出特征为 `[1,1,256,64,64]`，refined mask 为 `[1,1,1,96,128]`，数值有限、SAM2 全冻结、请求状态已清理。该结论完成 S2 provider gate，但完整 VILA+decoder+SAM2 链路仍需真实 VILA checkpoint smoke。

### 2026-08-05 — S2 provider 提交前验证

- **完成内容**：在最终工作树上运行全部 EvoVILA-Seg 无权重回归、静态编译、Git 空白检查和真实 SAM2 provider smoke。
- **遇到的问题**：pytest 仅报告现有依赖的 17 条 deprecation/future warnings，无测试失败。
- **解决方案**：回归结果为 54 passed；`py_compile` 与 `git diff --check` 通过。最终 A800 单次 smoke 为初始化 4763.94 ms、encoder 317.05 ms、refinement-with-reencode 345.60 ms、峰值 601.12 MiB；lazy import、finite、frozen 和 state-clear 检查均为 true。

### 2026-08-05 — S2 完整图像链路兼容性审计

- **完成内容**：定位到 Git 外部已有的完整 VILA1.5-3B checkpoint，并在 `evovila` 环境、offline 模式和单张 A800 上通过仓库原生 `llava.load` 完成加载；模型类型为 `LlavaLlamaModel`，LLM hidden size 为 2560，vision input size 为 384。
- **遇到的问题**：原生 loader 返回 `eval` 模型，但约 31.5 亿 VILA 参数仍为 `requires_grad=True`。现有 adapter 虽使用 `torch.no_grad()`，尚未完整落实需求文档中首阶段冻结 VILA 参数的不变量。
- **解决方案**：先在显式 `VILASegmentationAdapter` 构造边界冻结并置为 eval，再实现 composed image pipeline 和真实 checkpoint smoke。普通 `llava.load`、`forward`、`generate` 与 `generate_content` 不构造该 adapter，默认路径保持不变。

### 2026-08-05 — 显式 VILA adapter 冻结语义修正

- **完成内容**：`VILASegmentationAdapter` 对显式传入的 `torch.nn.Module` VILA 执行 `eval()` 和 `requires_grad_(False)`；测试模型新增真实参数并检查冻结后权重值不变。
- **遇到的问题**：无；非 module 的轻量 callable harness 仍保持兼容。
- **解决方案**：冻结仅发生在扩展 adapter 构造期间，不修改普通 VILA loader 或默认请求路径。

### 2026-08-05 — S2 composed image pipeline 实现

- **完成内容**：新增 `ImageSegmentationOutput` 与 `VILAImageSegmentationPipeline`，在一个显式 image-only 入口内组合 VILA query extraction、SAM2 dense encoding、空间 decoder 和 SAM2 mask refinement；coarse/refined mask 保持独立可检查。
- **遇到的问题**：dense encoder 与 refiner 若不是同一个 provider，可能产生配置或权重不一致。
- **解决方案**：构造时要求对象身份一致；disabled、video、T>1 和 batch mismatch 均在执行组件前 fail closed，并提供统一的 request-state 清理入口。

### 2026-08-05 — S2 image pipeline 测试收集修正

- **完成内容**：修正参数化负控测试的参数名称。
- **遇到的问题**：当前 pytest 配置将 `request` 视为保留 fixture 名，导致测试在收集阶段失败。
- **解决方案**：改名为 `request_value`，不改变测试场景或实现行为。

### 2026-08-05 — Opt-in CUDA 分组件计时修正

- **完成内容**：在 VILA query、SAM2 image encoder、空间 decoder 和 composed refinement 的计时边界增加对应设备同步。
- **遇到的问题**：仅用 `perf_counter` 包围异步 CUDA launch 会低估组件耗时，不能用于真实执行路径报告。
- **解决方案**：同步只存在于显式 segmentation adapter/pipeline；CPU 路径和普通 VILA 请求不增加 CUDA 同步。

### 2026-08-05 — VILA adapter 总计时边界复核

- **完成内容**：移除同步前的冗余 elapsed 赋值，并在 adapter total timer 启动前同步输入设备。
- **遇到的问题**：独立调用 adapter 时，先前排队的 CUDA 工作可能被计入 extension total。
- **解决方案**：total 与各组件使用一致的同步起止边界。

### 2026-08-05 — S2 真实 VILA+SAM2 图像 smoke 入口

- **完成内容**：新增 `smoke_image_segmentation.py`，通过外部 CLI 路径 offline 加载 VILA 与 SAM2，构造或读取 RGB image，显式选择唯一 multi-token query span，并运行 composed image pipeline。
- **遇到的问题**：随机初始化 decoder 的输出只能验证 plumbing；把它当作分割质量结论会误导后续训练判断。
- **解决方案**：JSON 固定标记 `randomly_initialized_plumbing_only`，同时检查 ordinary VILA 前后 final-token logits 精确一致、VILA/SAM2 冻结、SAM2 状态清理、lazy import、finite shape，并分别记录 VILA vision encoder、LLM、SAM2 encoder、mask decoder 和 refinement 时间；image 路径的 video propagation 显式为 N/A。

### 2026-08-05 — S2 真实图像 plumbing gate 通过

- **完成内容**：在 `evovila` 环境与 A800 上 offline 加载本地 VILA1.5-3B、SAM2.1 Hiera Tiny 和 synthetic RGB image，执行 ordinary VILA baseline、显式 composed image pipeline 与 extension 后 ordinary VILA retention。
- **遇到的问题**：VILA loader 的 token embedding resize 初始化耗时较长；首次 ordinary forward 还包含 CUDA warm-up，因此不能与后续 steady-state forward 直接比较延迟。
- **解决方案**：初始化与每个执行组件分开报告。单次结果：VILA 初始化 81132.43 ms、SAM2 初始化 2505.23 ms、VILA query total 60.39 ms（vision encoder 14.31 ms、LLM 22.91 ms）、SAM2 encoder 198.16 ms、mask decoder 157.71 ms、refinement-with-reencode 260.18 ms、pipeline total 701.70 ms、峰值显存 6649.73 MiB。query 的 3 个原始 token 位置 `[46,47,48]` 映射到 fused `[244,245,246]`；dense/coarse/refined shape 分别为 `[1,1,256,64,64]`、`[1,1,1,64,64]`、`[1,1,1,96,128]`。ordinary VILA 前后 logits 完全相同，最大绝对差为 0.0。decoder 仍为随机初始化，本 gate 不评价 mask 质量。

### 2026-08-05 — S2 smoke 生命周期复核

- **完成内容**：真实 smoke 在 pipeline 返回后显式调用统一清理入口，并检查 capability diagnostics 与 SAM2 predictor image state 均已清空。
- **遇到的问题**：无。
- **解决方案**：同时要求 SAM2 在 ordinary baseline 前不存在、在显式初始化后存在，使 lazy-import 断言双向闭合。

## 运行说明

### 环境准备

```bash
conda activate /9950backfile/chenjiahui/.conda/envs/evovila
```

使用现有环境；S0 不安装额外依赖、不下载权重或数据集。

### S0 快速验证

```bash
python -m pytest -q tests/test_evo_seg_*.py
python -m py_compile llava/evo_seg/*.py scripts/evo_seg/smoke_decoder.py
git diff --check
```

- **参数说明**：测试命令不需要模型或数据路径；`py_compile` 只做语法和导入级静态编译；`git diff --check` 检查空白错误。
- **运行后会发生什么**：构造合成的图像/视频特征，验证契约、解码器、损失、显式 capability 开关和 SAM2 导入隔离。
- **输出什么**：pytest 输出通过/失败摘要；smoke JSON 默认输出到 stdout，只有显式指定仓库外路径时才写文件。

### S0 无权重 smoke

```bash
python scripts/evo_seg/smoke_decoder.py
```
- **参数说明**：`--config` 指定合成配置，默认 `configs/evo_seg/s0_decoder.yaml`；`--output` 可选，必须是仓库外的 `smoke.json` 路径。
- **运行后会发生什么**：构造 T=2 的 padded video batch，调用显式 `enabled=true` 的 capability，执行 forward/backward 并检查 SAM2 未被导入。
- **输出什么**：标准输出为 JSON；若给出 `--output`，同时写入仓库外 JSON 文件。

### S1 无权重集成验证

```bash
python -m pytest -q tests/test_evo_seg_provenance.py tests/test_evo_seg_vila_adapter.py
python -m py_compile llava/model/fusion_observer.py llava/model/llava_arch.py llava/evo_seg/provenance.py llava/evo_seg/vila_adapter.py
```

- **参数说明**：使用 fake VILA/dense provider，不需要模型、SAM2 或数据路径。
- **运行后会发生什么**：执行 native `_embed` provenance、显式 query span 提取和 opt-in decoder bridge；disabled path 不调用 VILA 或 dense provider。
- **输出什么**：pytest 输出 15 个 S1 测试的通过/失败摘要；静态编译无输出即成功。

### S2 SAM2 boundary 验证

```bash
python -m pytest -q tests/test_evo_seg_sam2_adapter.py
python -m py_compile llava/evo_seg/sam2_adapter.py
```

- **参数说明**：使用 fake predictor；`configs/evo_seg/s2_image.yaml` 中的源码、checkpoint 和图像路径保持为空，真实运行时必须由仓库外路径覆盖。
- **运行后会发生什么**：验证 raw RGB、lazy build、encoder freeze、无效帧清零、predictor 状态清理和 mask-prompt refinement。
- **输出什么**：8 个 boundary 测试；不会安装/import SAM2，也不会产生模型或图像文件。

### S2 真实 SAM2 image provider smoke

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/evo_seg/smoke_sam2_image.py \
  --source-root /path/to/local/sam2 \
  --checkpoint /path/to/local/sam2.1_hiera_tiny.pt \
  --device cuda:0
```

- **参数说明**：`--source-root` 指向 Git 外部的官方 SAM2 checkout；`--checkpoint` 指向已有本地权重；`--device` 是 `CUDA_VISIBLE_DEVICES` 映射后的设备；`--config` 默认使用 `configs/evo_seg/s2_image.yaml`；`--output` 可选且必须位于仓库外。
- **运行后会发生什么**：显式初始化冻结的 SAM2，构造一张 synthetic RGB image，真实执行 image encoder 与 coarse-mask refinement；不会加载 VILA、下载资产或训练参数。
- **输出什么**：stdout 输出含 commit/config hash、环境、tensor shape、finite/frozen/state-clear 检查、分组件时间和峰值显存的 JSON。refinement 当前会再次编码图像，因此单独标记为 `sam2_mask_refinement_with_reencode`。

### S2 真实 VILA+SAM2 image pipeline smoke

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/evo_seg/smoke_image_segmentation.py \
  --vila-model /path/to/local/VILA1.5-3b \
  --sam2-source-root /path/to/local/sam2 \
  --sam2-checkpoint /path/to/local/sam2.1_hiera_tiny.pt \
  --device cuda:0
```

- **参数说明**：三个资产参数均要求已有的 Git 外部本地路径；`--image` 可选，省略时使用 deterministic synthetic RGB；`--config` 默认使用 `configs/evo_seg/s2_image.yaml`；`--output` 可选且必须位于仓库外。
- **运行后会发生什么**：强制 Hugging Face offline 模式，加载冻结 VILA/SAM2，显式提取 multi-token query states，执行 SAM2 dense encoding、随机初始化 spatial decoder 和 SAM2 refinement，并比较 extension 前后 ordinary VILA final-token logits。
- **输出什么**：stdout 输出 JSON，包含 provenance positions、各 tensor shape、lazy/frozen/finite/state-clear/retention 检查、VILA vision encoder、LLM、SAM2 encoder、mask decoder、refinement 与峰值显存。该 smoke 只批准 plumbing，不批准 mask 质量。

### 2026-08-05 — S2 image plumbing 提交前最终验证

- **完成内容**：在最终工作树重跑全部 EvoVILA-Seg 无权重回归、扩展与 VILA hook/smoke 文件静态编译和 Git 空白检查，并复核第二次真实 VILA+SAM2 image pipeline smoke。
- **遇到的问题**：pytest 仅报告现有依赖的 17 条 deprecation/future warnings，无测试失败；随机初始化 spatial decoder 仍不具备可评价的 mask 质量。
- **解决方案**：无权重回归为 61 passed，`py_compile` 与 `git diff --check` 均通过。第二次 A800 smoke 的分项结果为 VILA 初始化 81168.00 ms、SAM2 初始化 2387.58 ms、VILA query total 60.53 ms（vision encoder 14.02 ms、LLM 23.02 ms）、SAM2 image encoder 186.70 ms、mask decoder 37.32 ms、refinement-with-reencode 250.87 ms、pipeline total 555.68 ms、峰值显存 6649.73 MiB。query 原始位置 `[46,47,48]` 映射到 fused `[244,245,246]`；dense/coarse/refined shape 分别为 `[1,1,256,64,64]`、`[1,1,1,64,64]`、`[1,1,1,96,128]`。ordinary VILA retention 精确通过（最大 final-token logit 差 0.0），VILA/SAM2 均冻结，SAM2 lazy import、finite 输出和两类请求状态清理均通过。本次结论仅批准 S2 image plumbing，不批准 mask 质量或训练效果。

### 2026-08-05 — S3 predicted-anchor 视频传播设计冻结

- **完成内容**：审计本机官方 SAM2.1 `build_sam2_video_predictor`、`init_state`、`add_new_mask`、正反向 `propagate_in_video` 和 `reset_state` API；将 S3 收敛为固定 predicted-anchor 推理契约。
- **遇到的问题**：原实现指南只写了 `[T,H,W]` 单样本占位签名，没有定义 anchor 选择、batch/padding、多对象顺序、反向传播、官方 path-only 视频输入桥接或异常清理；若让 decoder 对所有帧预测后再取 anchor，也会产生不必要的 dense encoding。
- **解决方案**：VILA 继续观察完整视频，dense provider 和空间 decoder 只处理每个样本的一个 anchor 帧；默认第一个有效帧，也允许显式有效索引，动态选择延后。预测 logits 在零阈值处二值化后作为唯一 SAM2 mask prompt，按样本建立 request-local state 并在需要时双向传播，重建 `[B,N,T,H,W]` 且 padding 为零。S3 专用 builder 让 image wrapper 与 video predictor 共享同一冻结 SAM2 权重；官方 path-only 输入通过自动删除的临时 JPEG 目录桥接，并分开记录 I/O、初始化、prompt 和传播时间。S3 仍为随机 decoder plumbing，不启动训练或评价 mask 质量。

### 2026-08-05 — S3 propagator 与 video pipeline 实现

- **完成内容**：`sam2_adapter.py` 新增视频传播选项/结果契约、video-capable image predictor builder 和共享权重 `SAM2VideoMaskPropagator`；新增 `video_pipeline.py`，组合完整 VILA 视频 query、单 anchor dense decoding 与 SAM2 双向传播。
- **遇到的问题**：官方 video predictor 只接收 MP4 bytes 或 JPEG 目录，且返回对象顺序和 compact frame index，需要在恢复 padded batch 前显式校验；传播异常时仍必须清除 predictor state 和临时文件。
- **解决方案**：逐样本将有效 RGB 帧写入自动删除的临时 JPEG 目录，将 source/compact frame index 双向映射，并严格校验对象 ID、返回 shape 和所有有效帧覆盖。所有 state 都在 `finally` 中先 `reset_state` 再清空；pipeline 在执行组件前拒绝 disabled/image/T=1/无效 anchor 请求。两个新实现文件已通过 `py_compile` 和 `git diff --check`。

### 2026-08-05 — S3 无权重传播回归完成

- **完成内容**：新增 `s3_video.yaml` 和 11 个 video pipeline/propagator 测试，覆盖默认/显式 anchor、单 anchor dense encoding、multi-object 顺序恢复、padded batch、正反向传播、共享 provider、video builder 冻结、lazy import 与异常清理。
- **遇到的问题**：无；fake video predictor 刻意以反向 object ID 顺序返回，并在故障用例中从 generator 内抛错，以验证适配层不是依赖理想返回顺序或成功路径才清理。
- **解决方案**：`python -m pytest -q tests/test_evo_seg_video_pipeline.py` 为 11 passed；所有 predictor state 字典被清空，request-local 临时目录均已删除，invalid/disabled 请求未执行 VILA 或 SAM2。

### 2026-08-05 — S3 image/video adapter 边界拆分

- **完成内容**：将 S3 video builder、传播选项/结果契约和 propagator 从 `sam2_adapter.py` 移入独立 `sam2_video_adapter.py`；video pipeline、测试和 smoke 改为只在显式视频入口导入它。
- **遇到的问题**：初版虽然不影响 ordinary VILA，但 S2 image pipeline 导入 `sam2_adapter.py` 时也会解析所有视频临时 I/O 和传播定义，使图像/视频能力边界不够清晰。
- **解决方案**：`sam2_adapter.py` 恢复为纯 S2 image provider/refiner，S3 通过单独模块复用其 local-only build 校验、共享 provider 和设备同步 helper。运行行为、共享权重关系和官方 API 调用序列不变。

### 2026-08-05 — S3 真实 VILA+SAM2 视频 plumbing gate 通过

- **完成内容**：在 `evovila` 环境和单张 A800 上 offline 加载本地 VILA1.5-3B 与 SAM2.1 Hiera Tiny，对 3 帧 synthetic moving-object video 分别执行默认首帧 anchor 和显式中间帧 anchor 的完整 pipeline；两次均只把随机 decoder 的 predicted binary mask 作为 SAM2 prompt。
- **遇到的问题**：本机 SAM2 的可选 `_C` CUDA 后处理扩展未编译，官方 predictor 发出 warning 后自动跳过小孔填充；主 image encoder、mask prompt、memory propagation 和输出恢复均正常。smoke 运行在尚未提交的 S3 worktree 上，base commit 为 `bab23c4`，因此 JSON 需要显式记录 dirty-worktree 状态以避免 provenance 歧义。
- **解决方案**：默认 anchor=0 的真实 forward-only smoke 和 anchor=1 的真实 bidirectional smoke 均通过；后者分项为 VILA 初始化 84757.23 ms、SAM2 初始化 2438.54 ms、VILA query total 127.63 ms（vision encoder 25.08 ms、LLM 39.87 ms）、SAM2 anchor encoder 183.24 ms、mask decoder 41.56 ms、临时 frame I/O 32.29 ms、video state 初始化 136.67 ms、predicted-anchor prompt 131.05 ms、forward propagation 100.31 ms、reverse propagation 58.73 ms、SAM2 video total 462.06 ms、pipeline total 848.62 ms，峰值显存 6771.03 MiB。raw/coarse/propagated shape 为 `[1,3,3,96,128]`、`[1,1,1,64,64]`、`[1,1,3,96,128]`；query 原始 `[49,50,51]` 映射到 fused `[643,644,645]`。两次 ordinary VILA retention 均精确通过（最大 final-token logit 差 0.0），VILA/SAM2 冻结、SAM2 lazy import、finite 输出、image/video/capability state cleanup 均通过。decoder 仍为随机初始化，本 gate 不评价 mask 质量。

### S3 真实 VILA+SAM2 predicted-anchor video smoke

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/evo_seg/smoke_video_segmentation.py \
  --vila-model /path/to/local/VILA1.5-3b \
  --sam2-source-root /path/to/local/sam2 \
  --sam2-checkpoint /path/to/local/sam2.1_hiera_tiny.pt \
  --device cuda:0
```

- **参数说明**：三个模型资产参数都要求已有的 Git 外部本地路径；`--video` 可选，支持 MP4 或 JPEG 目录，省略时生成 deterministic 3-frame RGB；`--anchor-index` 可选，省略时使用第一个有效帧；`--config` 默认 `configs/evo_seg/s3_video.yaml`；`--output` 可选且必须位于仓库外。
- **运行后会发生什么**：强制 offline 模式，VILA 读取完整视频与 multi-token query，decoder 只预测一个固定 anchor mask；该预测 mask 经零阈值二值化后作为唯一 SAM2 prompt，并按需正反向传播。运行前后比较 ordinary VILA final-token logits，不下载资产、不训练参数。
- **输出什么**：stdout JSON 包含 commit/dirty-worktree/config provenance、anchor 策略、shape、finite/frozen/lazy-import/state-clear/retention 检查、VILA vision/LLM、anchor encoder、decoder、frame I/O、state init、prompt、forward/reverse propagation、总时间与峰值显存；随机 decoder 输出只用于 plumbing。

### 2026-08-05 — S3 提交前最终验证

- **完成内容**：在 image/video adapter 拆分和文档同步后的最终工作树，重跑全部 EvoVILA-Seg no-weight tests、所有 extension/VILA hook/smoke 静态编译、包级/image/video 模块的外部 SAM2 import isolation 和 Git 空白检查。
- **遇到的问题**：pytest 仅保留现有依赖的 17 条 deprecation/future warnings，无失败；真实视频 smoke 的 `_C` 可选后处理 warning 已作为环境限制单独记录，不影响 gate 结论。
- **解决方案**：最终 no-weight suite 为 72 passed；`py_compile`、独立进程 import isolation 和 `git diff --check` 均通过。S3 真实 forward-only 与 bidirectional smoke 结果保持有效，未下载资产、未启动训练、未生成仓库内结果文件。

### 2026-08-05 — S3 暂存审查完成

- **完成内容**：仅暂存 8 个 S3 代码、配置、测试和文档文件；核对新增内容不含本机资产绝对路径、权重、数据、结果、缓存或凭据，remote 与 `long_rl` submodule 关系保持不变。
- **遇到的问题**：`git diff --cached --check` 发现 `sam2_video_adapter.py` 末尾多一个空白行。
- **解决方案**：仅移除该 EOF 空白并重新执行 staged whitespace/content 审查；不改变运行行为。

### 2026-08-05 — S4a mask data 与 retention contract 设计冻结

- **完成内容**：在 S2/S3 plumbing 通过后审计现有 `GroundingBatch`、loss、数据准备占位和 retention 要求；将训练前置工作拆为 S4a metadata contract 与需另行批准的 S4b training recipe。
- **遇到的问题**：仓库没有 `idea_report.md` Part 3、真实训练结果或数据 manifest，现有占位 schema 缺少稳定 media ID、采样帧索引、固定 anchor、target ID、same-media query-swap 声明、empty/no-object 语义与 source-media split leakage 检查。empty query 也不能伪装成普通 `GroundingBatch`，因为现有非空 query invariant 会正确拒绝它。
- **解决方案**：设计独立、无文件 I/O 的 `training_contracts.py`：逐样本固定 frame/mask/presence/anchor 对齐；positive/no-object 可进入后续优化，empty-query 只做 fail-closed 评估；query swap 只能连接同媒体不同真实目标；train/val 必须分别覆盖 swap/no-object/empty-query；retention probe set 必须覆盖 ordinary VILA 的 text、single-image、multi-image、video，并绑定 frozen baseline revision。此阶段不实现 loader、optimizer 或训练 recipe。

### 2026-08-05 — S4a metadata contracts 实现

- **完成内容**：新增 `training_contracts.py`，实现不可变 `MaskTrainingRecord`、`QuerySwapPair`、`MaskTrainingManifest`、`RetentionProbe` 与 `RetentionProbeSet`，以及不可变覆盖统计。
- **遇到的问题**：empty-query 控制若混入优化 batch 会违反已冻结的非空 query token invariant；同一 media 的不同 query 若允许各自选择帧或 anchor，会让 query-swap 不再是受控变量。
- **解决方案**：empty-query 明确标为 `optimization_eligible=False`；manifest 要求同一 `media_id` 共享 media path/type、sampled frame indices、anchor 和 split。模块只验证元数据及绝对路径字符串，不读取文件，不导入 torch/VILA/SAM2，也不改变任何现有执行路径。

### 2026-08-05 — S4a control eligibility 测试夹具修正

- **完成内容**：修正 positive/no-object/empty-query eligibility 测试的 record 索引。
- **遇到的问题**：首轮目标测试为 15 passed、1 failed；`records[0::2]` 只返回 positive 和 no-object 两项，却被解包为三项，失败发生在 contract 断言前。
- **解决方案**：改为显式索引 records 0、2、3，不改变 contract 或测试场景。

### 2026-08-05 — S4a 提交前契约审查收紧

- **完成内容**：补充 exact media path 的跨 split 泄漏检查，并要求 query-swap 两侧同时具有不同 query text 与 target ID；同步增加回归测试和实现文档约束。
- **遇到的问题**：仅依赖稳定 `media_id` 无法拦截同一媒体路径被错误重标 ID 后分入 train/val；不同 target 若使用完全相同 query，不能形成有意义的 swapped-query 负控。
- **解决方案**：manifest 同时按 `media_id` signature 和 exact `media_path` 检查 split；query-swap 在相同 media/frame/anchor 基础上拒绝相同 query text。仍只验证元数据，不打开任何外部文件。

### 2026-08-05 — S4a 提交前最终验证

- **完成内容**：运行 S4a 目标测试、全部 EvoVILA-Seg no-weight 回归、扩展/VILA hook/smoke 静态编译、独立进程导入隔离和 Git 空白检查。
- **遇到的问题**：完整 pytest 仅报告既有依赖的 17 条 deprecation/future warnings，无测试失败；流式工具的短等待窗口曾截断 pytest 末尾显示，因此使用持续轮询重新取得完整退出码与摘要。
- **解决方案**：目标测试为 16 passed；完整回归为 88 passed、17 warnings；`py_compile`、`llava.model`/external SAM2 import isolation 和 `git diff --check` 均通过。未读取数据、加载权重、运行 GPU 或启动训练。

### 2026-08-06 — S4b T1 图像 overfit 实现与运行

- **完成内容**：按确认的方案实现 S4b 训练栈并跑 T1 RefCOCO overfit。数据侧：
  `scripts/evo_seg/build_s4b_manifests.py` 从 RefCOCO parquet + COCO2017 栅格化目标掩码，构建
  RefCOCO train/val manifest（train 1,391 条 / 60 swap，val 1,401 条 / 85 swap），并从 Ref-YT-VOS
  构建视频 manifest（train 466 条 / 76 swap，val 490 条 / 78 swap），全部通过冻结的
  `MaskTrainingManifest` 校验；镜像的每帧联合掩码使用调色板像素值，构建器先推断像素→目标映射。
- **训练侧**：`llava/evo_seg/training.py` 增加 LoRA（最后 8 层 q/v，fp32 参数，仅训练作用域生效）、
  `[SEG]` 可训练注入 embedding（不改冻结词表）、`GroundingProjector`（保留完整多 token query 状态并
  追加 `[SEG]` token）；`vila_adapter.py` 增加可微 `extract_query_states_training`（普通路径仍 no_grad）；
  `scripts/evo_seg/train_s4b.py` 实现 T1 训练循环：冻结 VILA 基座 + 可训练 decoder/[SEG]/projector/LoRA/
  SAM2 mask decoder，loss 为 BCE(+正样本加权)/Dice/objectness + SAM2 精修 mask 的 BCE+Dice，
  训练前后跑 text/single-image/multi-image/video 四条 retention probe（精确 logits）。
- **遇到的问题**：首轮 500 步随机抽样未过拟合（IoU≈0）；定位为 ① pooling 把多 token query 语义压成
  同质表示（同图不同 query 余弦 0.96），改为保留完整 query states + `[SEG]` token；② 小目标正负像素
  不平衡导致模型塌缩到全背景，加入 `bce_positive_weight=5` 后单样本 anchor IoU 峰值达到 0.5+。
- **解决方案**：配置 `configs/evo_seg/s4b_t1_image.yaml` 固定 16 样本、800 步、lr 3e-4；正式 T1 已在
  1×A800 后台运行。所有训练产物（manifest/mask/checkpoint/summary/retention）都在
  `evo_artifacts/datasets/s4b` 与 `evo_artifacts/results/s4b`，仓库外。
- **验证**：新增 `tests/test_evo_seg_training_policy.py`（5 passed），全套 no-weight 回归 93 passed、
  `py_compile`、导入隔离、`git diff --check` 均通过；冒烟训练中 VILA 基座可训参数为 0，
  四条 retention probe 训练前后 `max_abs_diff=0.0`。

### 2026-08-06 — S4b T1 多卡加速与正式 overfit 结果

- **完成内容**：`train_s4b.py` 增加 torchrun 8×A800 分布式训练（每 rank 独立采样、梯度 all-reduce）
  与 tqdm 实时进度条（loss/bce/dice/iou/ETA）。800 步从单卡 ~70 分钟降到 8 卡 ~5 分钟；
  1600 步约 6.5 分钟，每步 8 样本（8 rank × batch 1），8 卡各占 ~7.3GB 显存。
- **遇到的问题**：`seg_embedding` 参数初始在 CPU 导致 nccl all-reduce 报
  "No backend type associated with device type cpu"，已移到对应 CUDA 设备。
- **解决方案**：正式 T1 配置固定 16 个 RefCOCO 正样本、1600 步、lr 3e-4、`bce_positive_weight=5`，
  SAM2 mask decoder 可训。结果（`evo_artifacts/results/s4b/t1_image_overfit_v4/summary.json`）：
  train anchor IoU 峰值 0.585、末 200 步均值 0.571（200/200 步 ≥0.5）、loss 2.43→1.93、
  val IoU 0.037→0.067；四条 retention probe 训练前后精确相等（max_abs_diff=0.0）。
- **验证**：8 卡 DDP 4 步冒烟与 1600 步正式跑均通过；提交 `7e82f4d` 包含
  training.py/losses.py/vila_adapter.py/build_s4b_manifests.py/train_s4b.py/配置/测试/文档，
  仓库内无本机绝对路径，未 push。

### 2026-08-06 — S4b T2 视频 anchor 训练实现

- **完成内容**：`build_s4b_manifests.py` 视频分支为 train 正样本按目标像素提取并保存二值掩码
  （调色板联合掩码 → 目标层），`mask_paths` 指向这些二值掩码；`train_s4b.py` 增加 `_video_sample`
  loader（采样帧 PIL 序列 → VILA 多帧输入 + `[SEG]`、SAM2 多帧特征、目标 [B,N,T,H,W]/presence），
  图像/视频共用训练循环，SAM2 精修只作用于 anchor 帧（T=1 切片）。新增
  `configs/evo_seg/s4b_t2_video.yaml`（temporal consistency loss 0.1）。
- **遇到的问题**：仓库配置将 retention 路径清空后，T2 冒烟因空 image_paths 报 IndexError；
  已改为训练时通过 `--retention-image-paths/--retention-video-dir` 显式传入。
- **解决方案**：T2 冒烟（4 步、8 卡）通过，anchor IoU 第 3 步已到 0.47，retention 精确；
  正式 T2 1600 步在 8×A800 后台运行。提交 `3b40842`（视频 loader/二值掩码/T2 配置）。

### 2026-08-06 — S4b T2 正式视频 anchor overfit 结果

- **完成内容**：正式 T2 完成（1600 步、8×A800、~20.5 分钟），结果见
  `evo_artifacts/results/s4b/t2_video_overfit/summary.json`：train anchor IoU 峰值 0.676、
  末 200 步均值 0.545（200/200 步 ≥0.5）、loss 2.38→1.85、dice 0.48；
  val（held-out Ref-YT-VOS 视频）IoU 约 0.24-0.31，明显优于 T1 图像的 0.067；
  四条 retention probe 训练前后精确相等（max_abs_diff=0.0）；峰值显存 10.2GB/卡。
- **遇到的问题**：视频每步含多帧，单步 1.7 it/s 慢于图像 4.3 it/s，1600 步约 20 分钟，可接受。
- **解决方案**：anchor 帧 IoU 稳定上升且 val 泛化为正，说明 decoder+SAM2 微调栈对视频有效；
  query-swap / no-object 控制与正式 J&F 评测留到 T3（小规模正式训练）按已冻结契约执行。

### 2026-08-06 — S4b T3 混合训练实现并启动（tmux 实时日志）

- **完成内容**：`train_s4b.py` 增加 task=mixed（按 `mix_video_ratio` 混抽图像/视频，`no_object_ratio`
  混入 no-object 负控，`swap_prob` 触发同媒体 query-swap margin loss）、双 val manifest 评测、
  `TrainingLogger` 每步结构化落盘 `train.log` + `train_history.jsonl`（step/loss/bce/dice/objectness/
  temporal/swap/iou/lr/gnorm/eta/source/sample），tqdm 终端进度条。新增
  `configs/evo_seg/s4b_t3_mixed.yaml`（4000 步、lr 2e-4、query_swap 0.1、temporal 0.1、
  bce_positive_weight 5、SAM2 mask decoder 可训）。
- **遇到的问题**：随机 query 在 prompt 内分词与单独分词不一致导致 span 定位失败；指令模板改为
  `...referring expression:\n{query}`（`\n` 在 Llama 分词器是独立 token），`_query_span_mask`
  增加按 `\n`/`:` 边界的稳健回退。
- **解决方案**：T3 混合 8 步冒烟通过（双 val 评测、swap/no-object 路径正常、retention 精确）。
  正式 T3 已在 tmux 会话 `s4b_t3` 内以 8×A800 启动（4000 步，约 2 小时），输出目录
  `evo_artifacts/results/s4b/t3_mixed_v1`；提交 `fb3bad2`。

### 2026-08-10 — S4b T4 全量数据基线训练完成 + 稳定性修复（8×A800）

- **完成内容**：T4 全量基线训练（`configs/evo_seg/s4b_t4_mixed.yaml`，10000 步）最终完成，
  结果见 `evo_artifacts/results/s4b/t4_mixed_v3/summary.json`：final train anchor IoU **0.514**、
  final loss **1.32**、峰值显存 **~14GB/卡**；四条 retention probe 训练前后全部精确相等
  （max_abs_diff=0.0，VILA 文本/单图/多图/视频能力零损伤）。正式 val 评测与反捷径评测
  （no-object 幻觉率 + query-swap）于 17:23 以 8 shard 并行启动，结果待汇总。
- **遇到的问题（一串事故链，三个 bug 叠加）**：
  1. 坏视频帧：Ref-YT-VOS 某视频第 5 帧为 1×1 损坏帧，`np.stack` 直接崩；non-finite 跳过
     只包住 forward 未包住数据加载。
  2. resume 特有 bug：循环变量 `adapter` 覆盖模块级 `VILASegmentationAdapter`（Python 变量泄漏）。
  3. fp16 数值不稳定：VILA 前向 fp16 下 projector 偶发溢出 → `query_states` 非有限 → 训练步
     跳过（老 run 17% 浪费）；且检测到 NaN 梯度后仍执行 `optimizer.step()`，把 NaN 写进
     LoRA 参数造成污染；eval 块未保护（坏 val 样本直接崩整个 run）；`logger` 在非 rank-0 为
     None 未判空。
- **解决方案**：分割栈（projector/decoder/loss）全部切 fp32（VILA 前向保持 fp16）、
  NaN 梯度跳过该步 optimizer、logger 判空、eval 块 try/except、query/dense states `nan_to_num`
  兜底、视频帧统一 resize。修复后跳过率从 17% 降到 ~2-3%。提交：`ef30fc4`（帧归一化+坏样本
  跳过）、`8838854`（resume 变量泄漏）、`a255581`（nan_to_num+eval 保护）、`4710f7b`
  （fp32 分割栈+NaN 梯度跳过+logger 判空）。周期 checkpoint + resume（`5b6250e`）在 T4
  崩于 step 6019 时从 step-5000 checkpoint 续跑，实战验证有效。

### 2026-08-10 — M2 数据全量就绪：合并 manifest + 类别感知 no-object 负样本

- **完成内容**：RefCOCO/RefCOCO+/RefCOCOg 全量 manifest 与合并训练集
  （`image_merged_all/train.jsonl`：**321,333** 图，refcoco 120,626 + refcocoplus 120,193 +
  refcocog 80,514）、**167,655** 条类别感知 no-object 负样本（train N=3/图、val N=5/图，
  查询类别不在目标图 COCO 类别集合中，合并 train+val 两份 instances 标注避免漏类别）、
  46,998 对 query-swap。修复 sample_id 冲突（三个变体原先都硬编码 `refcoco.*`，合并时互相
  覆盖导致只剩 14 万条）。提交：`c720348`（负样本生成器+训练集成）、`9719864`（合并脚本+
  反捷径评测协议+sample_id 前缀）、`feed5b0`（T5 合并数据训练配置）。

### 2026-08-10 — S4b T5 合并数据主模型训练启动（论文主模型）

- **完成内容**：`configs/evo_seg/s4b_t5_merged.yaml`（10000 步、lr 2e-4、lora_lr 4e-4、
  mix_video_ratio 0.25、no_object_ratio 0.25、swap_prob 0.25、bce_positive_weight 8.0）在 tmux
  `s4b_t5_v1` 以 8×A800 启动，输出 `evo_artifacts/results/s4b/t5_merged_v1`。池子：
  image_pos=321,327 / video_pos=6,345 / image_neg=167,658 / video_neg=3,438 / pairs=46,998。
  稳态 ~2.25s/it（节点负载 ~300 导致慢于 T4 的 0.6s/step），ETA ~6h。这是论文要写的
  anti-shortcut 主模型：全量三个图集 + 系统负样本 + query-swap。

### 2026-08-11 — S4b T5 NaN 根因定位与 bf16 修复（关键）

- **症状**：T5 训练（s4b_t5_merged.yaml）从 step 2337 起间歇 NaN 梯度，~step 5594 起每步
  NON-FINITE 导致 optimizer 被跳过、参数冻结在 ~5586；且 NaN 跳过路径 `continue` 绕过
  checkpoint 保存，6000 步 checkpoint 未落盘。
- **定位**：`--detect-anomaly` + 从 step-4000 checkpoint resume 在 step 4001（no-object 样本）
  复现，traceback 指向 Llama 最终 RMSNorm `hidden_states * rsqrt(variance + eps)` 的
  `MulBackward0` 返回 NaN。机制：VILA 残差流在 fp16（max ~65504）下对某些样本溢出成 Inf，
  `variance=Inf` → `rsqrt=0` → `Inf×0=NaN` → 全图梯度 NaN。VILA 的 QuantLinearTE 本来就
  以 bf16 I/O（FP8 量化），训练/评测却用 fp16 autocast 包裹，既冗余又易溢出。
- **修复（提交 `bcfab63`）**：train/eval 前向统一 `model.to(bf16)` + `torch.autocast(dtype=bf16)`，
  视觉输入改 bf16；`losses._masked_mean` 先清零无效项再加权（避免 NaN×0 污染均值）；
  NaN 梯度日志精确到参数名；新增 `--detect-anomaly` 调试开关。
- **验证**：resume-4000 复现修复前 step 4001 即 NaN，修复后干净跑过 4225+（含 no-object/swap，
  0 NON-FINITE）；eval_s4b / eval_antishortcut bf16 冒烟通过（3 样本，指标与 fp16 一致）。
