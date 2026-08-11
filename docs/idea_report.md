<!--
修改时间：08-10_14:00
修改内容：初稿。基于 EvoVILA-Seg 当前实现与实验状态，规划 CVPR 2027 论文的定位、贡献、实验与基线。
上一状态：不存在（新文件）
-->

# EvoVILA-Seg: 无捷径的指代分割（Anti-Shortcut Referring Segmentation）

> 目标会议：CVPR 2027（全文截稿约 2026-11 中旬）。写作流程走 ResearchPilot G 阶段，
> 初稿方向供讨论，未冻结。

## 1 问题定义

指代分割（referring segmentation）要求模型根据一句语言描述，在图像/视频中输出对应物体的像素级
mask。现有 LLM-based 方法（LISA / GLaMM / VideoLISA 一族）有一个共通的已知弱点：

> 模型容易"看图猜 mask"而不是"听懂语言"——退化成视觉捷径（visual shortcut）：
> 对同一图像的不同查询给出近乎相同的 mask；对"不存在物体"的查询仍然幻觉出一个 mask。

这类方法大多用单个 `[SEG]` token 承载指代语义，丢掉了多 token query 的细粒度信息，是语义塌缩
（semantic collapse）的根源。本项目针对这一弱点提出显式的反捷径（anti-shortcut）训练与评测。

## 2 方法（当前实现）

架构（`llava/evo_seg/`，VILA1.5-3B 基座，全部 opt-in，普通 VILA 路径不变）：

1. **多 token query 对齐**：保留完整 query 隐状态（multi-token query states）+ `[SEG]` 标记，
   经 `GroundingProjector` 送入 query-conditioned spatial decoder；不采用单 `[SEG]` token 池化。
   （动机：同一图像不同 query 的 pooled 表示余弦相似度 0.96，正是语义塌缩的直接证据。）
2. **显式反捷径训练信号**（一等公民，非可选）：
   - query-swap：同图不同目标的查询互换，margin loss 强制 query 真正区分目标；
   - no-object：查询描述不存在的物体，objectness loss 要求输出"无目标"；
   - empty-query：空查询，仅作 fail-closed 评估；
   - temporal consistency（视频）。
3. **统一图像/视频**：图像视为 T=1 视频；共享 decoder；视频用 predicted-anchor + SAM2 双向传播。
4. **SAM2 作为可选的 mask 引擎**：粗 mask → SAM2 精修（图像）/传播（视频）；普通路径零依赖。
5. **能力保留不变量**：训练前后四条 retention probe（text/image/multi-image/video）final-token
   logits 精确相等（当前 max_abs_diff=0.0）。

## 3 论文贡献（草案）

1. 诊断性发现：单 token 指代表示导致语义塌缩，多 token query 对齐是必要而非锦上添花。
2. 反捷径训练协议（query-swap / no-object / empty-query 作为结构化训练与评测信号）——
   顺带贡献一个 **swap/no-object 评测协议**（objectness 正确率、空 mask 正确率），
   现有 SOTA 基线在该协议上会显式暴露失败模式。
3. 统一图像-视频的轻量架构（3B 基座 + 小 decoder + 可选 SAM2），带逐组件效率拆分。

## 4 实验计划

### 4.1 数据
- 训练：RefCOCO / RefCOCO+ / RefCOCOg（train 全量，manifest 已具备 120k 规模）；Ref-YT-VOS train。
- 评测：RefCOCO/+/g val + testA/B（cIoU、gIoU）；Ref-YT-VOS val（J&F，SAM2 传播）；
  可扩展 MeViS、SA-V。
- 反捷径评测集：从 val 构建 no-object / query-swap 子集（现有契约已保证结构）。

### 4.2 基线
- LISA-7B / GLaMM / PixelLM / SEEM（图像）；VideoLISA（视频，本地有完整仓库可跑）；
  SAM2+LLM prompting 的上界/下界参考。

### 4.2.1 基座策略（用户确认 2026-08-10）
- **3B 为主角**：全部主实验、消融、效率表都在 VILA1.5-3B 上做（反捷径 + 轻量双卖点）。
- **7B 测精度上限**：若 3B 已能刷到 SOTA，7B 作为 scaling 证据，证明方法没有被 3B 基座卡住；
  若 3B 差 SOTA 一口气，7B 补精度表的缺口。7B 只需跑主 benchmark + 反捷径协议。
- 实现影响：decoder/ projector 由 `model.llm.config.hidden_size` 自适应，训练栈可复用；
  需下载 VILA1.5-7b（约 14GB，进入 M3/M4 前申请）。

### 4.3 消融
- 指代表示：单 token vs 多 token（语义塌缩定量对比）。
- 训练信号：query-swap / no-object / empty-query 逐个开关。
- 组件：SAM2 refine 开关、temporal loss、anchor 策略、LoRA 层数、decoder 分辨率。
- 效率：vision encoder / LLM / decoder / SAM2 分项 latency 与显存（AGENTS.md 要求）。

## 5 现状与差距

- 当前基线（T3，RefCOCO 4000 步）：val coarse cIoU 0.096、refined 0.006；Ref-YT-VOS J&F 0.04。
- 诊断结论：粗 mask 太弱（25% 样本为空），SAM2 精修在弱粗 mask 下是负收益；
  **精修代码本身正确**（合成测试 IoU=1.0），问题在粗模型质量。
- T4 全数据（120k 图 + 13k 视频，10000 步）正在 8×A800 训练（2026-08-10 启动，含周期
  checkpoint + resume），预计 2-3 小时，将给出全量基线。
- 关键差距：no-object 训练样本目前每 split 仅 1 条（`image_neg=1`），需扩展为系统性负样本；
  RefCOCO+/g manifest 正在构建。

## 6 里程碑

- M1 基建与诊断（本周，进行中）：T4 全量基线、refine 诊断、manifest 补齐。
- M2 数据（~1 周）：RefCOCO/+/g 全量 manifest、系统性 no-object 负样本、评测协议。
- M3 方法调优（2-3 周）：把 coarse cIoU 从 0.1 打到 0.6+ 量级，refine 策略重建。
- M4 消融 + 对标（2-3 周）：VideoLISA 等基线、反捷径评测、效率表。
- M5 写作（3-4 周）：G 阶段流程 + AAAI 审稿模拟自审。

## 7 开放问题

- 是否引入更强的基座（VILA1.5-7B / NVILA）或保持 3B 主打效率。
- 反捷径评测协议是否作为独立 benchmark 投稿（配套数据/代码）。
- 与 shared-spatial-grounding 分支（空间特征来源审计）的结论如何并入。
