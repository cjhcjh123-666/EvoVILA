# EvoVILA-Seg 架构 v2（对标 LISA / Sa2VA / GLUS / LENS）

## 结论
核心架构（[SEG] token -> SAM2 mask decoder + VILA-LoRA）与 LISA/Sa2VA 同款，
不是"错"，问题在 4 个执行层面：SAM2 太弱、数据太少、LLM 适配可更强、视频时序未训。

## v2 增量（已实现并推送）
| # | 项 | 状态 |
|---|---|---|
| 1 | SAM2.1-hiera-large（最高官方档，接口已验证兼容） | ✅ 已接 v2 配置 |
| 2 | 数据：RefCOCO/+/g + Ref-YT-VOS + MeViS v2（新构建 41.5k 记录/43k swap 对） | ✅ MeViS manifest 已构建 |
| 3 | LoRA rank64/全 32 层/7 投影 + 视觉塔(SigLIP) LoRA rank16 | ✅ 代码+配置 |
| 4 | 门控全量微调（lora.full_rank: true，rank=hidden，seg 作用域内 delta，基座零损伤） | ✅ 代码+配置 |
| 5 | 视频 memory 训练（SAM2 memory attention 端到端） | ⏳ 接口已确认，需 GPU 验证后接入 |
| 6 | 帧策略（GLUS 式 context+query、训练/推理对齐、关键帧选择） | ⏳ 下一工作包 |
| 7 | 自建数据集（Ref-SAV 式） | ⏸ 等 v2 跑起来再构建 |
| 8 | LENS 式多 context-query + GRPO RL | ⏳ 下一工作包 |

## 数据现状
- 已有：RefCOCO/+/g（图像 32.1 万）、Ref-YT-VOS（视频 1.3 万）、MeViS v2（视频 4.1 万）
- 缺失（需下载）：ADE20k / COCO-Stuff / ReasonSeg / gRefCOCO / SA-1B / VQA
- 本机另发现：SAM3（/9950backfile/zhangyafei/sam3/sam3.pt，需新适配器）、VideoLISA-3.8B（HF 缓存，视频架构参照）

## 对标精度（RefCOCO/+/g cIoU；视频 Ref-YT-VOS J&F）
- LISA-7B: 74.1/62.4/66.4 ｜ Sa2VA-1B: 79.6/73.6/77.7（视频 71.3@4B）
- LENS-3B: 81.2 avg ｜ PaliGemma-3B-448: 75.6/69.8/70.2 ｜ GLUS-S: 66.6 J&F

## 运行
- v2 训练：`bash scripts/evo_seg/run_s4b_lisa3b_v2.sh [--resume]`
- 想全量微调：`configs/evo_seg/s4b_lisa3b_v2.yaml` 里开 `lora.full_rank: true`（每卡约 +13GB）
