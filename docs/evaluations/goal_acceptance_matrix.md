# `goal.md` 验收矩阵

> 更新日期：2026-09-16
> 当前基线：`main = 5a075d4`
> 离线验证：`478 passed, 2 deselected`
> 真实模型：`live_openai` 此前 2 项通过
> 真实双回路：最终 60 分钟 20 个事件、20/20 对话成功、`dialogue.failed=0`、backlog 峰值 1/最终 0、死信 0、worker 无错误、stop 前 health ready=true；`failure_classification` 仅 1 次 semantic provider timeout

状态说明：

- **已满足**：当前范围内有实现、自动化证据或已完成的真实验证；
- **部分满足**：主链已实现，但仍缺少真实模型、长期运行、UI 或完整边界证据；
- **未满足**：核心实现或验收证据仍缺失。

## 9.1 认知行为

| ID | 标准 | 状态 | 当前证据与缺口 |
| --- | --- | --- | --- |
| COG-01 | 七类认知模块可独立启停、替换和降级 | 已满足 | 模块注册表、健康状态、按主体 disabled_modules 已接入事件处理 |
| COG-02 | 身份、价值、能力、限制、目标都有版本与证据 | 部分满足 | 领域状态和规则问答已实现；身份、价值、能力在 60 分钟真实运行中的表达仍未专项评测 |
| COG-03 | 对话稳定使用情景、语义、关系、程序性、叙事记忆 | 部分满足 | Workspace/Retrieval 已接入；真实模型对情景、语义、关系、程序性、叙事记忆的长期使用质量仍受模型波动影响 |
| COG-04 | 区分事实、推断、偏好、假设、冲突和未知 | 部分满足 | 已有事实/推断/偏好/未知/冲突；假设类型和通用表达仍不足 |
| COG-05 | 线索、间隔、干扰、巩固、衰减、不确定表达 | 已满足 | 记忆生命周期测试通过；真实开放域表达仍受限 |
| COG-06 | 认知结论可追溯到事件或系统先验 | 部分满足 | provenance graph 已覆盖 event/contribution/memory/relationship/narrative/emotion/mood/action request/decision/result/tool result；非事件证据（system prior/file fragment）节点的完整 provenance 仍未收口 |

## 9.2 数据正确性

| ID | 标准 | 状态 | 当前证据与缺口 |
| --- | --- | --- | --- |
| DATA-01 | 重复事件/贡献不重复改变状态 | 已满足 | 幂等测试通过 |
| DATA-02 | 事件序列重放后得到业务等价状态 | 已满足 | Replay 测试通过 |
| DATA-03 | 跨 mind 读取被拒绝，心智内归属清晰 | 已满足 | 隔离、取消、主体归属测试通过 |
| DATA-04 | 状态、记忆和关系更新保留历史证据 | 已满足 | 版本化状态和记忆版本测试通过 |
| DATA-05 | 所有索引可删除并重建 | 部分满足 | provenance 图索引具备 manifest、rebuild_all、delete_all、verify 完整生命周期；字段/时间索引可重建；向量索引仍未接入 |
| DATA-06 | 删除清除权威、缓存、导出、派生索引 | 已满足 | 删除传播到情绪、心境、主动意图和状态重放链路 |

## 9.3 对话与行动

| ID | 标准 | 状态 | 当前证据与缺口 |
| --- | --- | --- | --- |
| EXEC-01 | 回答只使用 Workspace | 已满足 | 规则和真实模型均经 Workspace 入口 |
| EXEC-02 | 无证据/低置信度不以确定语气表达 | 部分满足 | Review + claim 级字段已实现；最终 60 分钟 20/20 对话成功、0 次 generate 失败；真实模型长期校准仍待持续评测 |
| EXEC-03 | 计划包含依赖、预算、失败分支、取消点 | 已满足 | 计划/进度/校验测试通过 |
| EXEC-04 | 工具动作经过权限、风险、幂等检查 | 已满足 | LLM 动作判断、治理、幂等测试通过 |
| EXEC-05 | 高风险动作等待确认由 LLM 判断 | 部分满足 | 动作确认模型已接入；真实高风险工具有限 |
| EXEC-06 | 工具成功、失败、超时、取消、部分完成形成标准事件 | 已满足 | action.result 标准结果测试通过 |

## 9.4 可靠性

| ID | 标准 | 状态 | 当前证据与缺口 |
| --- | --- | --- | --- |
| REL-01 | 模型超时/非法结构不破坏状态 | 已满足 | 失败隔离、修复调用和状态保护测试通过 |
| REL-02 | 单模块失败不阻塞独立模块 | 已满足 | Engine 隔离测试通过 |
| REL-03 | 崩溃后识别未完成运行并恢复/终止 | 已满足 | 真实子进程强杀测试：pending event 重启后恢复入队并成功 drain；running RunRecord 重启后明确标记 INTERRUPTED |
| REL-04 | 取消传播到模型、工具、worker | 已满足 | 模型/动作取消测试已有；新增运行中 worker 取消集成测试，RunLifecycle.request_cancel 可传播到 RunContext 和模块 |
| REL-05 | 本地事件、查询、写入有延迟指标 | 已满足 | 新增 MetricsRegistry 百分位能力与 `scripts/measure_local_latency.py`；20 次本地采样 p50/p95/p99：ingest 0.0174/0.0196/0.0196s，query 0.00122/0.00144/0.00145s，state_write 0.00432/0.00495/0.00510s |
| REL-06 | 备份恢复校验事件数、版本和索引 | 已满足 | backup/restore/migrate 测试通过 |

## 9.5 可替换性与可运维性

| ID | 标准 | 状态 | 当前证据与缺口 |
| --- | --- | --- | --- |
| OPS-01 | 替换模型、存储、索引不修改 core | 已满足 | Protocol 边界和适配器已建立 |
| OPS-02 | 默认测试离线，真实模型单独运行 | 已满足 | `.env` 隔离和 live marker 已生效 |
| OPS-03 | CLI、HTTP、WebUI、worker 共用应用服务 | 已满足 | CLI、HTTP、WebUI、worker 复用同一应用服务；主动消息和情绪/心境操作已接入 WebUI |
| OPS-04 | 可查看健康、积压、用量、失败链、降级项 | 部分满足 | Health 已细化到 module/model/task/provider；WebUI 长期健康、积压和降级历史展示仍不完整 |
| OPS-05 | 配置有默认值、校验、密钥边界、迁移 | 部分满足 | 配置校验和 schema 迁移已实现；供应商/多环境配置仍有限 |

## 9.6 主动性、情绪与真人感

| ID | 标准 | 状态 | 当前证据与缺口 |
| --- | --- | --- | --- |
| ACT-01 | 快慢回路异步，慢失败不阻塞快速交互 | 已满足 | 5/30/60 分钟真实运行快速路径始终返回；最终 60 分钟 20/20 无 fast error，worker 未因 APITimeoutError 退出，backlog 最终为 0 |
| ACT-02 | 无滚动历史仍保持身份、关系、目标连续性 | 已满足 | 无滚动上下文重启连续性测试通过 |
| ACT-03 | 外部事件、目标、关系、情绪能形成动机 | 已满足 | boredom 在 5/30/60 分钟真实运行中形成 social_connection 动机；最终 60 分钟共 5 条动机和 5 条 Mailbox |
| ACT-04 | 主动表达、提醒、询问、提议或沉默 | 已满足 | 主动表达真实验证通过，模型动态生成自然语言消息；最终 60 分钟产生 5 条主动 Mailbox；长期主观体验仍待盲测 |
| ACT-05 | 情绪影响行为但不改事实/身份 | 已满足 | 情绪仅影响社交 review 阈值和主动动机；事实 grounding 不受影响 |
| ACT-06 | 主动意图支持接纳、延迟、合并、取消、过期、重评 | 已满足 | 意图生命周期测试通过 |
| ACT-07 | 正例及时、负例沉默、重复一致性 | 已满足 | 确定性正负例矩阵通过；最终 60 分钟 5 条 `social_connection` Mailbox，冷却和幂等未产生等价重复 |

## 9.7 实时情绪系统

| ID | 标准 | 状态 | 当前证据与缺口 |
| --- | --- | --- | --- |
| EMO-01 | 快速情绪反应 + 慢速重评估 | 已满足 | `fast_reaction` + LLM affect 已实现；最终 60 分钟真实双回路中慢速评估、失败恢复、worker/backlog 保持稳定 |
| EMO-02 | 情绪维度完整 | 已满足 | valence/arousal/control/certainty/intensity/object/cause/decay 已落地 |
| EMO-03 | 情绪衰减与 MoodState 累积 | 已满足 | decay/accumulate/Workspace 读取衰减测试通过 |
| EMO-04 | Workspace 注入当前情绪和心境 | 已满足 | `WorkspaceFixedContext.emotion` 已实现 |
| EMO-05 | 情绪可调节社交 review，但不放宽事实 grounding | 已满足 | ReviewPolicy 单元测试覆盖边界 |
| EMO-06 | boredom 形成 social_connection 动机与主动聊天 | 已满足 | 5/30/60 分钟真实运行均产生主动 Mailbox；最终 60 分钟 5 条 social_connection；动态 LLM 表达真实验证通过 |
| EMO-07 | 用户可以查看、纠正、导出、删除、关闭情绪系统 | 已满足 | CLI/HTTP/WebUI 查看；纠正、导出、删除、关闭/重新开启均有入口和测试；删除传播已覆盖 |
| EMO-08 | 情绪行为可审计、可取消、可重放、幂等 | 已满足 | 冷却/幂等测试已有；新增跨重启 affect/mood/intention/Mailbox 一致性测试 |

## 状态汇总

| 状态 | 数量 |
| --- | ---: |
| 已满足 | 35 |
| 部分满足 | 9 |
| 未满足 | 0 |

## 真实双回路最终结果（2026-09-16）

### 最终 60 分钟（`main = 02e1ed6`）

- 场景：20 条用户消息，每 180 秒；boredom 消息在 0、5、35 分钟；
- `dialogue.started=20`、`assistant.message=20`、`dialogue.failed=0`；
- `fast_success=20/20`，无 fast error；
- 主动：5 条 `social_connection` 动机、5 条 `ProactiveIntention`、5 条 Mailbox；
- `max_backlog=1`，最终 `backlog=0`，`dead_letters=0`；
- worker 无错误类型；stop 前 health ready=true，最终 models/modules 为 healthy；
- 降级：`module_degradation_counts={semantic.llm_extractor: 3}`，`model_unhealthy_counts={proactive: 1}`，触发原因均为 `APITimeoutError`；
- `failure_classification`：1 次 `semantic.llm_extractor / provider_timeout_or_no_response`；
- 结论：对话主链 20/20 未被模型波动打断；单次 semantic timeout 被隔离，未产生死信、backlog 或状态破坏。

### 最终 30 分钟（`main = a7dabdb`）

- 10/10 fast 无异常；`dialogue.started=10`、`assistant.message=9`、`dialogue.failed=1` 为 `GroundingRejected`；
- `failure_classification total_failures=0`；`module_degradation_counts={}`；
- `max_backlog=1`，最终 `backlog=0`，`dead_letters=0`；
- 该 run 的唯一 `dialogue.failed` 是合理的 `GroundingRejected`：回答新增了未被证据支持的推断“听起来是平平稳稳的一天”；claim 级审查按要求拒绝，不作为系统故障。
- 另一次 60 分钟预跑中定位到 input evidence 与低置信 workspace item 共享证据时的 grounding 误伤，已在 `02e1ed6` 修复并加单元测试；最终 60 分钟无 generate 失败且 `dialogue.failed=0`。

## 阶段 41 可靠性验证（2026-09-16）

- 真实子进程强杀：`scripts/reliability_probe.py kill-pending` 后重启，pending event 被恢复、入队并成功 drain；
- 真实子进程强杀：`kill-running` 后重启，未完成 RunRecord 被明确标记 `INTERRUPTED`；
- worker 取消：运行中模块观察到 `RunContext` 取消并返回 CANCELLED，RunRecord 落为 `CANCELLED`；
- 跨重启：`EmotionState`、`MoodState`、`ProactiveIntention`、Mailbox 在重启后保持一致；
- 覆盖测试：`test/test_phase41_reliability.py`。

## 多进程 worker 与本地延迟指标（2026-09-16）

- 多进程 worker：3 个进程按 subject shard 并发 drain 9 个事件；每个 subject 的 3 个事件保序处理，state.version=3，恰好 9 条 state.reduced，无 pending/dead letters；
- 并发修复：`FileEventStore` 增加跨进程 append 锁，`append_many` 不再用进程内旧缓存全量重写事件日志；`FileProcessJournal` 使用 `BlockingFileLock`，避免 recovery/claim 竞争直接失败；
- 延迟采样（20 samples）：event_ingest p50/p95/p99 = 0.0174/0.0196/0.0196s；query = 0.00122/0.00144/0.00145s；state_write = 0.00432/0.00495/0.00510s；
- 覆盖：`scripts/measure_local_latency.py`、`test/test_local_latency_metrics.py`、`test/test_phase41_reliability.py` 多进程用例。

## 阶段 43.1 provenance 索引（2026-09-16）

- 新增 `core/provenance.py`：节点 EVENT / CONTRIBUTION / MEMORY / EMOTION / MOOD，边 CAUSED_BY / DERIVED_FROM / SUPERSEDES；
- 新增 `indexes/provenance.py` 与 `FileProvenanceStore`：可从事件日志与记忆文件重建，支持删除后重建和 `source_events` 追溯；
- 集成测试：事件、贡献、记忆、快速情绪、心境均可入图，记忆/情绪链可回到源事件；
- 覆盖：`test/test_provenance_graph.py`。

## 阶段 43.2 provenance 生命周期（2026-09-16）

- relationship / narrative / action request / action decision / action result / tool result 节点入图；
- provenance store 维护 manifest，支持 `rebuild_all`、`delete_all`、`verify`，并清理 stale graph；
- 修复多进程 state 读写竞争：`FileStateRepository` 增加按 subject 跨进程锁，避免 PermissionError 重试造成重复 `state.reduced`；
- 覆盖：`test/test_provenance_graph.py` 与 `test/test_phase41_reliability.py` 多进程用例。

## 当前仍未收口的验证项

- 真实浏览器 WebUI 自动化；
- 超过 60 分钟的情绪衰减、心境累积、沉默负例和主观体验盲测；
- 向量索引、非事件证据 provenance（system prior/file fragment 等）；
- 真实高风险工具、动作确认和失败降级场景；
- 真实模型长期校准、低置信度表达和未知/假设类型（最终 60 分钟仍有 1 次 semantic provider timeout）；
- WebUI 中情绪/心境的纠正、导出、删除全流程可视化。
