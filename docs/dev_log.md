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
- **输出什么**：7 个 boundary 测试；不会安装/import SAM2，也不会产生模型或图像文件。
