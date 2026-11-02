# 🐍 Data-Agent Python Backend

基于 **FastAPI + LangGraph + LangChain** 构建的 Text2SQL 智能体后端服务。

## 🌟 核心特性

- **LangGraph 状态图驱动**：完全对齐 Kotlin/Spring AI Alibaba Graph 的编排链路，包含从 Query 重写、双通道知识召回、任务规划（Planner）、HITL 人机协同，到 SQL 迭代纠错、Python 沙箱高阶计算与商业分析报告生成。
- **A2A (Agent-to-Agent) 协议流式传输**：采用 Server-Sent Events (SSE) 协议实时向前端推送节点状态卡片、打字机 Markdown 流与交互事件。
- **Docker / Subprocess 双模安全沙箱**：隔离运行大模型生成的 Python 二次分析计算代码，具备标准输入安全注入、异常回溯与自动降级保护。
- **动态 Token 防爆机制**：在数据交付报告节点（ReportGeneratorNode）前置实施智能采样保护（多于 20 行数据自动取样），彻底杜绝超长 Context Window 导致的大模型超时与费用暴涨。

## 📁 目录结构

```text
data-agent-backend-python/
├── app/
│   ├── a2a/                  # A2A 协议与 SSE 流式事件路由
│   ├── config.py             # 配置管理 (Pydantic BaseSettings)
│   ├── graph/
│   │   ├── nodes/            # 14 个独立 LangGraph 核心编排节点
│   │   │   ├── evidence_recall.py     # 知识召回 (Query Rewrite + PGVector + Schema)
│   │   │   ├── graph_recall.py        # 关系图谱关联表扩展与剪枝 (Mix Selector)
│   │   │   ├── feasibility_assessment.py # 可行性评估
│   │   │   ├── planner.py             # 任务拆解与执行计划生成
│   │   │   ├── plan_execution.py      # 步骤循环状态控制器
│   │   │   ├── sql_generator.py       # SQL 生成
│   │   │   ├── sql_execute.py         # SQLite 本地执行与 DisplaySpec 构建
│   │   │   ├── sql_error_fixer.py     # SQL 自我纠错循环 (反思重试)
│   │   │   ├── python_generator.py    # Python 计算与可视化脚本生成
│   │   │   ├── python_execute.py      # Docker / Subprocess 沙箱执行引擎
│   │   │   ├── python_analyze.py      # Python 统计结果商业化解读
│   │   │   └── report_generator.py    # 综合报告流式生成 (结构化汇总与 ECharts 渲染)
│   │   ├── edges/            # 条件分支路由 (HumanFeedback, PlanExecutor)
│   │   ├── state.py          # OverAllState 图状态机模型
│   │   └── workflow.py       # StateGraph 图编排组装
│   ├── prompts/              # StringTemplate 统一提示词库与管理器
│   ├── sandbox/              # Python 执行沙箱 (Docker / 本地隔离)
│   └── service/              # 业务服务层
├── data/
│   ├── database.sql          # PostgreSQL 知识库初始化 SQL
│   └── dev_20240627/         # BIRD Benchmark SQLite 数据库集合与元数据
├── tests/                    # 单元与端到端集成测试集
├── main.py                   # FastAPI 服务入口
├── pyproject.toml            # uv / pip 依赖管理配置
└── .env.example              # 环境变量配置模板
```

## 🚀 快速启动

### 1. 安装依赖

推荐使用现代 Python 包管理器 `uv`（也可以使用标准 `venv + pip`）：

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
- `OPENAI_API_KEY`: 兼容 OpenAI 格式的 API 密钥。
- `OPENAI_BASE_URL`: 模型中转地址或官方地址（支持阿里 DashScope、DeepSeek 等）。
- `DATABASE_URL`: PostgreSQL 连接串（需预先导入 `data/database.sql`）。

### 3. 运行服务

```bash
# 激活环境后启动
uvicorn main:app --host 0.0.0.0 --port 9933 --reload
```

## 🧪 运行测试

```bash
pytest tests/ -v
```

