# 🐍 Data-Agent Python Backend

基于 **FastAPI + LangGraph + PostgreSQL (pgvector) + SQLGlot** 构建的企业级问数智能体（Text2SQL & Python Data Agent）后端服务。

## 🌟 核心架构特性

- **LangGraph 类型化状态图编排**：以 `AgentState` 全局状态中心串联意图重写、双通道知识召回、外键图谱剪枝、可行性研判、多步任务规划（Planner）、HITL 人机协同审批、SQL 自愈重试、Python 沙箱深度计算与流式商业报告生成。
- **A2A (Agent-to-Agent) 流式通信协议**：基于 `FastAPI` + `StreamingResponse` (SSE) 与 JSON-RPC 2.0，实时推送节点卡片状态快照与 Token 级流式报告；结合 `MemorySaver` 实现 `interrupt_before` 人工挂起与无损恢复。
- **三级只读 SQL 安全防线与自愈闭环**：前置采用 `SQLGlot` AST 语法树静态解析拦截 `DROP / DELETE / UPDATE / INSERT / ALTER` 等写操作，物理层使用 SQLite URI 只读锁（`?mode=ro`）；执行异常时自动携带错误堆栈回跳重试（最多 3 次）。
- **Docker / Subprocess 双模隔离沙箱**：通过标准输入（`stdin`）安全注入 JSON 查询结果集，在隔离环境中执行 `Pandas / NumPy` 二次统计分析，支持超时熔断与无 Docker 环境平滑降级。
- **动态 Token 防爆采样**：在综合报告生成节点（`ReportGeneratorNode`）前置对超长结果集实施动态采样（超过 20 行自动安全截断并保留总行数元信息），杜绝上下文窗口溢出。

## 📁 目录结构

```text
data-agent-backend-python/
├── app/
│   ├── a2a/                              # A2A JSON-RPC 路由与 SSE 流式事件推送
│   ├── config.py                         # 环境变量配置管理 (Pydantic Settings)
│   ├── graph/
│   │   ├── nodes/                        # LangGraph 核心编排节点
│   │   │   ├── evidence_recall.py        # 独立问题重写 + 业务口径与历史 QA 双路召回
│   │   │   ├── scheme_recall.py          # 表结构与列描述向量召回
│   │   │   ├── table_relation.py         # 0.4 动态阈值过滤 + 外键拓扑补齐 + Mix Selector 剪枝
│   │   │   ├── feasibility_assessment.py # 需求可行性门禁评估
│   │   │   ├── planner.py                # 多步分析计划拆解与驳回重规划
│   │   │   ├── human_feedback.py         # HITL 人工审批状态接入与回退控制
│   │   │   ├── plan_execute.py           # current_step 步骤指针推进与动态调度中枢
│   │   │   ├── sql_generator.py          # SQL 生成与错误堆栈反思修复
│   │   │   ├── sql_execute.py            # SQLGlot AST 只读校验 + SQLite 物理只读执行
│   │   │   ├── python_generator.py       # Python 统计分析脚本生成
│   │   │   ├── python_execute.py         # Docker / Subprocess 隔离沙箱执行
│   │   │   ├── python_analyze.py         # Python 统计产物商业化解读
│   │   │   └── report_generator.py       # 多步证据汇总与流式 Markdown 商业报告生成
│   │   ├── edges.py                      # 四大核心条件路由边函数
│   │   ├── spec.py                       # 节点名与状态键统一枚举契约
│   │   ├── state.py                      # AgentState 全局状态模型定义
│   │   └── builder.py                    # StateGraph 图拓扑组装与编译
│   ├── prompts/                          # 17 套 .st 结构化提示词模板与管理器
│   ├── retrieval/                        # PostgreSQL + pgvector 向量检索器与外键元数据仓库
│   └── shared/                           # Docker / Subprocess 双模 Python 安全沙箱执行器
├── data/
│   ├── database.sql                      # PostgreSQL 知识库与向量表初始化 SQL
│   └── dev_20240627/                     # BIRD Benchmark 评测数据库与元数据描述
├── tests/                                # 44 项单元测试与集成测试套件
├── main.py                               # FastAPI 应用启动入口
├── pyproject.toml                        # uv / PEP 621 项目依赖声明
└── .env.example                          # 环境变量配置模板
```

## 🚀 快速启动

### 1. 安装依赖

推荐使用 `uv`（也可使用标准 `venv + pip`）：

```bash
# 使用 uv
uv sync

# 或使用标准 virtualenv
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 2. 配置环境变量

复制 `.env.example` 并填入你的 API Key：

```bash
cp .env.example .env
```

配置说明：
- `OPENAI_API_KEY`: OpenAI 兼容格式的大模型 API 密钥。
- `OPENAI_BASE_URL`: 模型 API 地址（如阿里云百炼 DashScope、DeepSeek、OpenAI 等）。
- `DATABASE_URL`: PostgreSQL 连接串（需启用 `vector` 插件并导入 `data/database.sql`）。

### 3. 启动服务

```bash
uvicorn main:app --host 0.0.0.0 --port 9933 --reload
```

## 🧪 运行测试

```bash
pytest tests/ -m "not e2e" -v
```
