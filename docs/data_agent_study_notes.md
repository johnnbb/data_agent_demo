# 🧠 Data-Agent Text2SQL 智能体研读与复现指南

> **文档性质**：本地私有学习记录与架构拆解笔记（已配置 Git 局部忽略，不会受 `git checkout` 切换版本影响）。  
> **项目定位**：基于 `spring-ai-alibaba/DataAgent` 拆解复现的企业级 Text2SQL 智能体教程。  
> **复现目标**：深入理解 **StateGraph 图编排**、**双通道 RAG**、**自我纠错** 与 **HITL 人机协同**，为后续迁移至 **Python (LangGraph + FastAPI)** 提供完整蓝图。

---

## 目录
1. [项目全景与核心架构](#1-项目全景与核心架构)
2. [源码演进路线图（Git Commit 逐级拆解）](#2-源码演进路线图git-commit-逐级拆解)
3. [核心设计模式剖析（以当前 Toy 分支为例）](#3-核心设计模式剖析以当前-toy-分支为例)
4. [Text2SQL 核心实战链路（阶段进阶）](#4-text2sql-核心实战链路阶段进阶)
5. [Kotlin 到 Python (LangGraph) 1:1 迁移与复现指南](#5-kotlin-到-python-langgraph-11-迁移与复现指南)
6. [本地 Git 漫游与开发避坑备忘](#6-本地-git-漫游与开发避坑备忘)

---

## 1. 项目全景与核心架构

传统单次 Prompt 生成 SQL（Zero-shot Text2SQL）在企业真实复杂场景下准确率往往不足 40%，核心痛点是：
- **Schema 过载与歧义**：上百张表、千级字段，超出上下文窗口且容易匹配错误字段。
- **业务暗语与隐式规则**：例如“有效订单”是指状态为 1 且未退款，单纯看表结构大模型根本无法得知。
- **缺乏纠错能力**：SQL 语法错误或执行异常时直接失败，无法自我修复。
- **复杂计算受限**：单纯 SQL 很难做复杂统计、时间序列回归或高级可视化。

### 宏观架构设计
本项目通过 **StateGraph 有向状态图** 将智能体拆解为端到端闭环链路：

```text
[用户自然语言提问]
       │
       ▼
┌──────────────────────────────────────────────┐
│ 1. 知识双通道检索 (Dual-channel Retrieval)     │
│    ├── 结构化 Schema 召回 (表结构/字段/外键)     │
│    └── 非结构化 Evidence 召回 (业务术语/计算规则)│
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│ 2. 关系推导与任务规划 (Relation & Planning)    │
│    ├── 表拓扑关系自动构建 (Table Relation Graph) │
│    ├── 规划器拆解多步子任务 (Planner Node)        │
│    └── 可行性评估 (Feasibility Check)          │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│ 3. 人机协同审核 (HITL: Human-in-the-Loop)     │
│    └── 用户确认/修改规划路径 (Confirm & Feedback)│
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│ 4. 执行与自愈纠错闭环 (Execution & Self-Correction)│
│    ├── SQL 生成与沙箱执行 (SQL Execute Node)    │
│    │    └── 遇错自动反馈回溯修正                 │
│    └── Python 沙箱高级计算 (Docker Sandbox)    │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│ 5. 多维决策报告与可视化 (Multi-dimensional Report)│
│    └── 整合结构化数据表格 + 统计结论 + 图表呈现 │
└──────────────────────────────────────────────┘
```

---

## 2. 源码演进路线图（Git Commit 逐级拆解）

通过分析提交历史，可以看到作者由浅入深设计教学的完整节奏：

| 阶段 | Commit | 说明 | 核心掌握要点 |
| :--- | :--- | :--- | :--- |
| **基础底座** | `5fc4ee6` | 前后端框架 | Spring Boot (Kotlin) + Vue 3 (Vite) 基础搭建 |
| | `913e1fb` | a2a 前后端实例 | A2A (Agent-to-Agent) 协议对接与 SSE 流式事件推送 |
| **图编排入门** | `3cd1c28` | 简单的 graph 编排 | Spring AI Alibaba StateGraph 最简单节点打通 |
| | `eecfe5a` | **多个分支节点 (当前位置)** | 场景路由判断 (Route Node) + 条件分流 (Travel/Study) + 汇合收尾 (WrapUp) |
| | `b78072f` | 人工确认交互 | HITL 核心机制：图的中断挂起 (Suspend) 与用户确认恢复 (Resume) |
| **数据与 RAG** | `da994ee` | 构建 BirdSQL 数据集 | 准备业界权威 Text2SQL 评测数据集 (SQLite 多库) |
| | `a3fc8e4` | 导入向量库和构建数据集 | 向量数据库构建 (PgVector/Milvus)，存储非结构化知识 |
| | `70ab3ab` | evidence-recall | 业务领域知识召回 (Evidence Recall)：针对业务黑话与特定计算口径 |
| | `af5bca6` | recall | 数据库元数据 Schema 召回：大表过滤、表字段相似度召回 |
| **拓扑与规划** | `61b5cd7` | table relation | 表关系推导：利用外键约束和关联关系推导最小联表路径 |
| | `4a5da3e` | planner and feasibility | 任务拆解规划器 (Planner) + 任务是否可被执行的评估机制 |
| | `efd93c3` | plan_execute | 规划步骤状态机调度执行器 |
| **执行与纠错** | `58c081a` | sql execute generate | SQL 生成、在真实数据库执行、捕获报错并反哺自愈修复 |
| | `79213a6` | PythonCode | 动态生成 Python 数据分析脚本，并在沙箱环境中执行运算 |
| | `3a86db7` | report | 综合 SQL 结果与 Python 运算输出，生成最终商业分析决策报告 |

---

## 3. 核心设计模式剖析（以当前 Toy 分支为例）

在提交 `eecfe5a (feat: 多个分支节点)` 中，演示了 StateGraph 的三大核心要素：

### 1. 全局状态（State & KeyStrategy）
- 节点之间不直接传参，所有数据沉淀在 `OverAllState` 中。
- 通过 `KeyStrategy.REPLACE` 声明键更新策略（支持替换、累加、合并等）。
- 状态键：
  - `input`: 用户初始输入
  - `scene`: 路由分类结果（`travel` 或 `study`）
  - `sceneLabel`: 场景中文标识
  - `draft`: 各分支节点生成的初步文本流
  - `finalOutput`: 收尾节点整合后的输出

### 2. 路由分流与条件边（Conditional Edges）
- **`ToySceneRouterNode`**：根据规则/LLM 识别用户意图。
- **`ToySceneBranchEdge`**：返回条件标识，驱动 Graph 根据返回值跳转到对应处理分支：
  ```kotlin
  .addConditionalEdges(
      ToyGraphSpec.Node.ROUTE,
      AsyncEdgeAction.edge_async(toySceneBranchEdge),
      mapOf(
          ToyGraphSpec.Scene.TRAVEL to ToyGraphSpec.Node.TRAVEL_PLAN,
          ToyGraphSpec.Scene.STUDY to ToyGraphSpec.Node.STUDY_PLAN,
      )
  )
  ```

### 3. 多分支汇合（Merge Node）
- `TRAVEL_PLAN` 和 `STUDY_PLAN` 执行完毕后，均指向同一个 `WRAP_UP` 节点。
- `ToyWrapUpNode` 读取状态里的 `draft`，调用大模型将其浓缩为 3 到 5 条行动项，完成流程收敛。

---

## 4. Text2SQL 核心实战链路（阶段进阶）

当你继续切入后续 commit 时，这几个核心业务节点是重点研读对象：

### 1. 双通道召回 (Dual Retrieval)
- **`SchemeReCallNode`**：
  - 输入：用户自然语言查询
  - 逻辑：通过向量检索 + 关键词检索匹配最相关的 Table 与 Column，将包含几百张表的巨大 Schema 剪枝为仅包含 3~5 张相关表的最小 Schema。
- **`EvidenceRecallNode`**：
  - 输入：用户查询重写后的关键词
  - 逻辑：召回业务规则知识库中的条目（例如："活跃用户定义为近 30 天内有登录记录"）。

### 2. 表拓扑关系推导 (`TableRelationNode`)
- 单纯挑出表还不够，大模型经常写出错误的 `JOIN` 关联。
- 该节点基于数据库物理外键和业务外键构建有向/无向图，用最小生成树或最短路径算法计算关联路径，直接告知大模型正确的 `JOIN ON` 链条。

### 3. 动态规划与可行性检查 (`PlannerNode` & `FeasibilityAssessmentNode`)
- 遇到复杂多维分析（如：“先查出销售额前三的门店，再分析其客户年龄分布”），由 Planner 拆解为多个子步骤。
- 可行性检查确保召回的信息足以回答该问题，信息缺失时直接提醒用户补充。

### 4. SQL 纠错与 Python 分析 (`SqlExecuteNode` & `PythonExecuteNode`)
- SQL 执行失败时，将数据库报错信息放入 State，循环返回 SQL 生成节点进行 1 到 3 次自愈重试。
- 数据集返回后，若包含高级统计（方差、增长率、趋势预测），交由 Python 代码生成节点在沙箱中运算。

---

## 5. Kotlin 到 Python (LangGraph) 1:1 迁移与复现指南

Spring AI Alibaba 的 `StateGraph` 在设计理念上与 Python 的 **`LangGraph`** 完全契合。用 Python 重构不仅难度极低，而且更贴合主流大模型生态。

### 技术栈对应映射表

| 能力层 | Kotlin (当前项目) | Python 复现推荐技术栈 | 说明 |
| :--- | :--- | :--- | :--- |
| **状态图编排** | `Spring AI Alibaba StateGraph` | **`LangGraph`** | 概念完全一致：State、Node、Edge、Conditional Edge |
| **状态数据结构** | `OverAllState` (Map) | **`TypedDict` 或 `Pydantic`** | Python 原生强类型注解，支持校验与提示 |
| **LLM 客户端** | `Spring AI ChatClient` | **`LangChain ChatOpenAI` / `instructor`** | 支持结构化输出与流式流转 |
| **API 与流式协议**| Spring Boot WebFlux / A2A | **`FastAPI` + `StreamingResponse`** | 现代异步高性能框架，原生支持 SSE |
| **向量库检索** | Spring AI VectorStore | **`Chroma` / `PGVector` / `Qdrant`** | Python 生态开箱即用 |
| **沙箱代码执行** | `SimplePythonExecutor` | **`docker-py` 或 `E2B Sandbox`** | 安全沙盒运行动态生成的 Python 统计代码 |

### Python (LangGraph) 极简复现骨架示例

```python
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# 1. 定义全局状态 (对应 Kotlin OverAllState)
class AgentState(TypedDict):
    input: str
    scene: str
    draft: str
    final_output: str

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# 2. 路由节点 (对应 ToySceneRouterNode)
def route_node(state: AgentState) -> dict:
    travel_keywords = ["旅游", "旅行", "出游", "攻略", "景点", "美食"]
    user_input = state.get("input", "")
    is_travel = any(k in user_input for k in travel_keywords)
    scene = "travel" if is_travel else "study"
    return {"scene": scene}

# 3. 条件边判断函数 (对应 ToySceneBranchEdge)
def scene_branch_edge(state: AgentState) -> Literal["travel_plan", "study_plan"]:
    return "travel_plan" if state["scene"] == "travel" else "study_plan"

# 4. 专业生成节点
def travel_plan_node(state: AgentState) -> dict:
    response = llm.invoke([
        SystemMessage(content="你是一个旅行规划助手，请生成精炼的旅行攻略。"),
        HumanMessage(content=state["input"])
    ])
    return {"draft": response.content}

def study_plan_node(state: AgentState) -> dict:
    response = llm.invoke([
        SystemMessage(content="你是一个学习教练，请生成精炼的学习计划。"),
        HumanMessage(content=state["input"])
    ])
    return {"draft": response.content}

# 5. 收尾节点 (对应 ToyWrapUpNode)
def wrap_up_node(state: AgentState) -> dict:
    response = llm.invoke([
        SystemMessage(content="请把输入内容整理为 3 到 5 条具体的行动建议。"),
        HumanMessage(content=state["draft"])
    ])
    return {"final_output": response.content}

# 6. 状态图装配 (对应 GraphConfiguration.kt)
builder = StateGraph(AgentState)

builder.add_node("route", route_node)
builder.add_node("travel_plan", travel_plan_node)
builder.add_node("study_plan", study_plan_node)
builder.add_node("wrap_up", wrap_up_node)

builder.add_edge(START, "route")
builder.add_conditional_edges("route", scene_branch_edge, {
    "travel_plan": "travel_plan",
    "study_plan": "study_plan"
})
builder.add_edge("travel_plan", "wrap_up")
builder.add_edge("study_plan", "wrap_up")
builder.add_edge("wrap_up", END)

# 编译图
graph = builder.compile()

# 7. 执行调用
if __name__ == "__main__":
    initial_input = {"input": "帮我做一份杭州西湖周末游玩攻略"}
    for step in graph.stream(initial_input):
        print("步骤执行:", step)
```

---

## 6. 本地 Git 漫游与开发避坑备忘

1. **版本切换不影响此笔记**：
   - 本文档存放于 `docs/` 目录下，且已在 `.git/info/exclude` 中注册。
   - 无论你执行 `git checkout <commit>` 多少次，此笔记都不会被 Git 删除或覆盖。
2. **常用的漫游操作**：
   - 切换到下一阶段（人工确认）：`git checkout b78072f`
   - 切换回最新代码：`git checkout main`
   - 随时查看当前位置与提交图：`git log --graph --oneline -n 15`
3. **在特定版本上动手做实验**：
   - 遇到想尝试修改调试的代码，千万不要在游离 HEAD 上直接改；先拉一个临时分支：  
     `git checkout -b test/toy-branch-experiment`  
     实验完毕后，可随时切回主干或删掉临时分支。
