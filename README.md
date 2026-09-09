# 自我认知 Agent

自我认知 Agent 是一个面向长期运行的本地智能体系统。它把交互、认知、记忆、关系、情感、目标和行动组织成可持续演化的心智状态，让 Agent 不依赖滚动聊天记录，也能记得经历、理解自己、认识用户、形成目标，并在后续对话和行动中调用这些认知。

## 实现的功能

### 长期连续的认知

- 将用户消息、模型回答、认知结果、计划、行动和工具结果写入统一事件流。
- 使用版本化状态保存身份、价值、能力、限制、目标、偏好和当前状态。
- 每次模型调用只读取当前事件与受预算约束的认知空间，长期连续性由事件、状态和记忆提供。
- 支持事件重放、状态恢复、幂等处理和不同 `mind_id` 之间的数据隔离。

### 七类认知模块

- 情景认知：记录具体经历、人物、时间、环境、行动和结果。
- 语义认知：提取事实、偏好、名称、概念和稳定模式。
- 身份与价值认知：维护 Agent 的身份、原则、能力、限制和目标。
- 关系认知：记录与不同用户之间的共享经历、边界、承诺和关系状态。
- 叙事认知：把分散事件组织成任务、时期、身份和关系叙事。
- 元认知：识别未知、冲突、不确定性、失败原因和认知修正。
- 情感认知：形成带对象、强度、原因和衰减状态的情感评估。

### 对话、规划与行动

- 根据当前任务动态检索状态、记忆、关系、冲突和证据，生成有限的 `WorkspacePacket`。
- 对话回答携带声明、披露判断和证据引用，可追溯到原始事件或模型响应。
- 支持创建目标、生成计划、查看进度，以及暂停、继续、完成和取消目标。
- 支持动作提议、价值与风险判断、人工确认、幂等执行和标准结果事件。
- 对话、规划、动作、元认知和情感 Agent 可以共用同一个 OpenAI 模型。

### 记忆与用户控制

- 保存情景、语义、关系、程序性和叙事记忆，并记录来源、置信度、显著性、稳定性和可提取性。
- 支持线索召回、间隔强化、干扰、巩固、衰减、归档、纠正和遗忘。
- 可以查看记忆、修正认知、导出个人数据、预览删除范围并执行删除。
- 删除会传播到事件、状态、记忆、处理记录和可重建索引，防止已删除内容被重放恢复。

### 双回路与主动意图

- 快速行为回路负责当前对话和行动，不等待慢速认知任务完成。
- 慢速认知回路更新状态、记忆、关系、情感和叙事，并形成主动意图。
- 主动意图记录动机、证据、优先级、预算、有效期、停止条件和状态版本。
- 支持主动意图的接纳、延迟、合并、取消、过期、重复抑制和保持沉默。

### 本地运行与数据维护

- 提供 WebUI、HTTP API 和命令行三种入口，共用同一套应用服务。
- 提供健康状态、运行记录、调用链、积压、延迟和模型用量查看。
- 数据保存在本地文件系统，支持校验和备份、恢复、索引重建和 schema 迁移。
- OpenAI Responses API 与 OpenAI 兼容端点通过 `.env` 配置，无需修改代码。

## 怎么使用

### 1. 安装

需要 Python 3.10 或更高版本，以及 [uv](https://docs.astral.sh/uv/)。

使用 OpenAI 或兼容模型：

```powershell
uv sync --locked --extra openai
Copy-Item .env.example .env
```

不接入模型、直接使用内置规则模型：

```powershell
uv sync --locked
```

### 2. 配置模型

编辑项目根目录的 `.env`：

```dotenv
OPENAI_API_KEY=你的 API Key
OPENAI_MODEL=你的模型名称
OPENAI_BASE_URL=
```

`OPENAI_API_KEY` 和 `OPENAI_MODEL` 必须同时填写。使用 OpenAI 默认地址时让 `OPENAI_BASE_URL` 保持为空；使用兼容服务时填写完整 API 地址。对话、规划、动作、元认知和情感 Agent 会共用这组配置。

运行数据默认保存在 `data/`。需要更换目录时修改：

```dotenv
SC_DATA_DIR=data
```

### 3. 启动 WebUI

```powershell
.\scripts\operations.ps1 serve -DataDir data -Port 8765
```

打开 `http://127.0.0.1:8765`，即可使用：

- 对话与回答证据链；
- 长期记忆；
- 运行记录；
- 动作批准；
- 关系和冲突投影；
- 模型用量；
- Agent 控制状态与服务健康状态。

### 4. 使用命令行

发送一条消息：

```powershell
uv run python -m self_cognition.interfaces.cli user-1 "我最近更喜欢早上学习"
```

查看记忆和当前健康状态：

```powershell
uv run python -m self_cognition.interfaces.cli memories user-1 --data-dir data
uv run python -m self_cognition.interfaces.cli doctor user-1 --data-dir data
```

修正一条认知：

```powershell
uv run python -m self_cognition.interfaces.cli correct user-1 --target-field preferences.study_time --value 早上 --cognition-type preference --data-dir data
```

导出、重放和遗忘个人数据：

```powershell
uv run python -m self_cognition.interfaces.cli export user-1 --data-dir data
uv run python -m self_cognition.interfaces.cli replay user-1 --data-dir data
uv run python -m self_cognition.interfaces.cli forget-dry-run user-1 --data-dir data
uv run python -m self_cognition.interfaces.cli forget user-1 --data-dir data
```

### 5. 调用 HTTP API

服务启动后，可以直接发送请求：

```powershell
$body = @{
    subject_id = "user-1"
    message = "根据你记住的内容，我更适合什么时候学习？"
} | ConvertTo-Json

Invoke-RestMethod `
    -Uri "http://127.0.0.1:8765/chat" `
    -Method Post `
    -ContentType "application/json" `
    -Body $body
```

主要接口：

| 功能 | 接口 |
| --- | --- |
| 对话 | `POST /chat` |
| 记忆 | `GET /memories`、`POST /memories/correct` |
| 目标与计划 | `GET/POST /goals`、`GET /goals/{id}/progress` |
| 目标控制 | `POST /goals/{id}/pause`、`resume`、`complete`、`cancel` |
| 运行记录 | `GET /runs`、`GET /runs/{id}`、`POST /runs/{id}/cancel` |
| 动作批准 | `GET /approvals`、`POST /approvals/{id}/approve` |
| 自我与关系 | `GET /self-model`、`/relationships`、`/conflicts` |
| 用户数据 | `POST /export`、`/forget/dry-run`、`/forget` |
| 状态恢复 | `GET /replay` |
| 运行观测 | `GET /health`、`/metrics`、`/usage`、`/traces` |

请求未指定时使用 `default-mind` 和 `user-1`；也可以在查询参数或 JSON 中传入 `mind_id` 与 `subject_id`。

### 6. 备份、恢复和迁移

```powershell
.\scripts\operations.ps1 backup -DataDir data -Archive backups\data.zip
.\scripts\operations.ps1 restore -Archive backups\data.zip -Target restored-data
.\scripts\operations.ps1 migrate -DataDir legacy-data -Target migrated-data
```

恢复和迁移会校验数据、清单与 schema，并重建派生索引。`restore` 和 `migrate` 的目标目录必须尚不存在。

## 文件结构

```text
.
├─ .env.example                         # 模型与运行配置示例
├─ pyproject.toml                       # Python 包与可选依赖
├─ uv.lock                              # 锁定依赖版本
├─ scripts/
│  ├─ operations.ps1                   # 安装、启动、备份、恢复和迁移入口
│  ├─ maintain_data.py                 # 数据维护命令
│  └─ evaluate_stage29.py              # 能力评测入口
├─ src/self_cognition/
│  ├─ bootstrap.py                     # 应用组装与模型配置入口
│  ├─ settings.py                      # .env 与运行设置
│  ├─ application/                     # 对话、目标、行动、纠正、导出和遗忘用例
│  ├─ core/                            # 事件、状态、记忆、关系、计划和行动契约
│  ├─ cognition/                       # 七类认知模块
│  ├─ blackboard/                      # 认知贡献校验与状态归并
│  ├─ memory/                          # 记忆编码、检索、巩固和生命周期
│  ├─ workspace/                       # 有限认知空间的检索与构建
│  ├─ executive/                       # 对话、规划、动作和编排
│  ├─ runtime/                         # 事件总线、调度、运行、取消和恢复
│  ├─ infrastructure/
│  │  ├─ llm/                         # OpenAI 认知、对话、规划和动作适配器
│  │  └─ persistence/                 # 文件事件、状态、记忆、运行和备份实现
│  ├─ tools/                           # 工具注册、能力描述与执行边界
│  ├─ observability/                   # 健康、指标、日志和调用链
│  ├─ workers/                         # 认知 worker 与定时调度
│  └─ interfaces/
│     ├─ cli.py                        # 命令行入口
│     └─ http/                         # HTTP API 与静态文件服务
├─ webui/                              # 本地对话与管理界面
├─ docs/                               # 隐私说明与能力评测数据
├─ test/                               # 自动化测试
└─ data/                               # 运行后生成的本地数据目录
   ├─ events/                          # 事件日志
   ├─ states/                          # 版本化主体状态
   ├─ memories/                        # 长期记忆
   ├─ runs/                            # 运行与恢复记录
   ├─ governance/                      # 用户控制和批准记录
   ├─ processing/                      # 处理状态、outbox 和死信
   ├─ indexes/                         # 可重建索引
   ├─ exports/                         # 用户数据导出
   └─ blobs/                           # 文件与大型结果
```
