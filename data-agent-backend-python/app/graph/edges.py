import logging
from langgraph.graph import END

from app.graph.state import AgentState
from app.graph.spec import StateKey, NodeName

logger = logging.getLogger(__name__)


def feasibility_assessment_edge(state: AgentState) -> str:
    """
    可行性研判条件路由边（对应 Kotlin 的 FeasibilityAssessmentEdge.kt）。
    若大模型研判为《数据分析》，放行至 PLANNER_NODE；否则直接走向 END。
    """
    output = state.get(StateKey.FEASIBILITY_RESULT.value, "")
    if "【需求类型】：《数据分析》" in output:
        logger.info("[FeasibilityAssessmentEdge] 需求类型为数据分析，放行进入 PLANNER_NODE 节点")
        return NodeName.PLANNER.value
    else:
        logger.info("[FeasibilityAssessmentEdge] 需求类型非数据分析（或需澄清/闲聊），提前路由至 END 节点")
        return END


def human_feedback_edge(state: AgentState) -> str:
    """
    人工反馈条件路由边（对应 Kotlin 的 HumanFeedbackEdge.kt）。
    根据 HumanFeedbackNode 写入的 HUMAN_NEXT_NODE 状态，
    路由至 PLAN_EXECUTION、回流至 PLANNER 或终止至 END。
    """
    next_node = state.get(StateKey.HUMAN_NEXT_NODE.value, END)
    logger.info(f"[HumanFeedbackEdge] 路由至下一节点: {next_node}")
    return next_node


def plan_executor_edge(state: AgentState) -> str:
    """
    计划调度执行条件路由边（对应 Kotlin 的 PlanExecutorEdge.kt）。
    根据 PlanExecuteNode 计算的 PLAN_NEXT_NODE 状态，
    动态分发至 SQL_GENERATE_NODE、PYTHON_GENERATE_NODE 或终止至 END。
    若步骤包含后续待开发节点（如 REPORT_GENERATOR_NODE），安全降级走向 END。
    """
    next_node = state.get(StateKey.NEXT_NODE.value, END)
    if not next_node or next_node == END:
        logger.info("[PlanExecutorEdge] 步骤已全部完成或无后续节点，路由至 END")
        return END

    supported_tools = {
        NodeName.SQL_GENERATION.value,
        NodeName.PYTHON_GENERATION.value,
        NodeName.REPORT_GENERATION.value,
    }

    if next_node not in supported_tools:
        logger.info(f"[PlanExecutorEdge] 目标节点 '{next_node}' 当前尚未注册或属于最终报告阶段，安全路由至 END")
        return END

    logger.info(f"[PlanExecutorEdge] 动态调度至工具节点: {next_node}")
    return next_node


