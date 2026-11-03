from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from app.graph.state import AgentState
from app.graph.spec import NodeName
from app.graph.edges import (
    feasibility_assessment_edge,
    human_feedback_edge,
    plan_executor_edge,
    sql_executor_edge,
)
from app.graph.nodes.evidence_recall import evidence_recall_node
from app.graph.nodes.scheme_recall import scheme_recall_node
from app.graph.nodes.table_relation import table_relation_node
from app.graph.nodes.feasibility_assessment import feasibility_assessment_node
from app.graph.nodes.planner import planner_node
from app.graph.nodes.human_feedback import human_feedback_node
from app.graph.nodes.plan_execute import plan_execute_node
from app.graph.nodes.sql_generator import sql_generator_node
from app.graph.nodes.sql_execute import sql_execute_node
from app.graph.nodes.python_generator import python_generator_node
from app.graph.nodes.python_execute import python_execute_node
from app.graph.nodes.python_analyze import python_analyze_node
from app.graph.nodes.report_generator import report_generator_node


def build_graph(checkpointer: MemorySaver = None, interrupt_before_feedback: bool = True):
    """
    组装 Text2SQL 核心双通道检索、结构链接、可行性研判、任务规划、人工审核、执行调度与 SQL/Python 生成执行状态图（对应 GraphConfiguration.kt）。
    START -> EVIDENCE_RECALL -> SCHEME_RECALL -> TABLE_RELATION -> FEASIBILITY_ASSESSMENT
      -> (条件分支: 数据分析) -> PLANNER -> HUMAN_FEEDBACK
           -> (审核通过) -> PLAN_EXECUTION
                -> (动态调度: PlanExecutorEdge) -> SQL_GENERATION -> SQL_EXECUTION -> (回环) PLAN_EXECUTION
                -> (动态调度: PlanExecutorEdge) -> PYTHON_GENERATION -> PYTHON_EXECUTION -> PYTHON_ANALYSIS -> END
                -> (已完成/无后续) -> END
           -> (驳回重试) -> 回流至 PLANNER (带 feedback 错误提示，最多 3 次)
           -> (超限/取消) -> END
      -> (条件分支: 闲聊/澄清) -> END
    """
    builder = StateGraph(AgentState)

    # 1. 注册核心节点
    builder.add_node(NodeName.EVIDENCE_RECALL.value, evidence_recall_node)
    builder.add_node(NodeName.SCHEMA_RECALL.value, scheme_recall_node)
    builder.add_node(NodeName.TABLE_RELATION.value, table_relation_node)
    builder.add_node(NodeName.FEASIBILITY_ASSESSMENT.value, feasibility_assessment_node)
    builder.add_node(NodeName.PLANNER.value, planner_node)
    builder.add_node(NodeName.HUMAN_FEEDBACK.value, human_feedback_node)
    builder.add_node(NodeName.PLAN_EXECUTION.value, plan_execute_node)
    builder.add_node(NodeName.SQL_GENERATION.value, sql_generator_node)
    builder.add_node(NodeName.SQL_EXECUTION.value, sql_execute_node)
    builder.add_node(NodeName.PYTHON_GENERATION.value, python_generator_node)
    builder.add_node(NodeName.PYTHON_EXECUTION.value, python_execute_node)
    builder.add_node(NodeName.PYTHON_ANALYSIS.value, python_analyze_node)
    builder.add_node(NodeName.REPORT_GENERATION.value, report_generator_node)

    # 2. 编排工作流链路
    builder.add_edge(START, NodeName.EVIDENCE_RECALL.value)
    builder.add_edge(NodeName.EVIDENCE_RECALL.value, NodeName.SCHEMA_RECALL.value)
    builder.add_edge(NodeName.SCHEMA_RECALL.value, NodeName.TABLE_RELATION.value)
    builder.add_edge(NodeName.TABLE_RELATION.value, NodeName.FEASIBILITY_ASSESSMENT.value)

    # 3. 动态条件分支路由：评估为《数据分析》则进入 PLANNER，否则直接终止走向 END
    builder.add_conditional_edges(
        NodeName.FEASIBILITY_ASSESSMENT.value,
        feasibility_assessment_edge,
        {
            NodeName.PLANNER.value: NodeName.PLANNER.value,
            END: END,
        },
    )

    # 4. 规划完成进入人工审核节点
    builder.add_edge(NodeName.PLANNER.value, NodeName.HUMAN_FEEDBACK.value)

    # 5. 人工审核条件路由：通过走向执行控制，驳回回流至规划，超限走向结束
    builder.add_conditional_edges(
        NodeName.HUMAN_FEEDBACK.value,
        human_feedback_edge,
        {
            NodeName.PLAN_EXECUTION.value: NodeName.PLAN_EXECUTION.value,
            NodeName.PLANNER.value: NodeName.PLANNER.value,
            END: END,
        },
    )

    # 6. 计划调度动态条件路由：根据当前步骤工具分发至 SQL_GENERATION、PYTHON_GENERATION、REPORT_GENERATION 或直接走向 END
    builder.add_conditional_edges(
        NodeName.PLAN_EXECUTION.value,
        plan_executor_edge,
        {
            NodeName.SQL_GENERATION.value: NodeName.SQL_GENERATION.value,
            NodeName.PYTHON_GENERATION.value: NodeName.PYTHON_GENERATION.value,
            NodeName.REPORT_GENERATION.value: NodeName.REPORT_GENERATION.value,
            END: END,
        },
    )

    # 7. SQL 生成后流向真实数据库执行，根据执行结果条件分支路由（报错未超限回跳 SQL_GENERATION 自纠错，成功或超限回环至 PLAN_EXECUTION）
    builder.add_edge(NodeName.SQL_GENERATION.value, NodeName.SQL_EXECUTION.value)
    builder.add_conditional_edges(
        NodeName.SQL_EXECUTION.value,
        sql_executor_edge,
        {
            NodeName.SQL_GENERATION.value: NodeName.SQL_GENERATION.value,
            NodeName.PLAN_EXECUTION.value: NodeName.PLAN_EXECUTION.value,
        },
    )

    # 8. Python 生成后流向沙箱执行，分析解读完成后回环至计划调度器 PLAN_EXECUTION
    builder.add_edge(NodeName.PYTHON_GENERATION.value, NodeName.PYTHON_EXECUTION.value)
    builder.add_edge(NodeName.PYTHON_EXECUTION.value, NodeName.PYTHON_ANALYSIS.value)
    builder.add_edge(NodeName.PYTHON_ANALYSIS.value, NodeName.PLAN_EXECUTION.value)

    # 9. 报告生成节点执行完毕，终局闭环，走向 END
    builder.add_edge(NodeName.REPORT_GENERATION.value, END)

    if checkpointer is None:
        checkpointer = MemorySaver()

    interrupt_nodes = [NodeName.HUMAN_FEEDBACK.value] if interrupt_before_feedback else []

    return builder.compile(checkpointer=checkpointer, interrupt_before=interrupt_nodes)
